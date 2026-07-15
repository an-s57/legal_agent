"""FAISS 向量库 — 自定义 Ollama 嵌入 + 法律文档检索"""
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'#不检查更新
os.environ['HF_HUB_OFFLINE'] = '1'#不联网下载
import torch
import httpx
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

FAISS_DB_PATH = "rag/vectorstore/db_faiss"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


class OllamaEmbeddings(Embeddings):
    """自定义嵌入类，直接调 Ollama 旧版 /api/embeddings 接口。
    绕过 langchain_ollama 不兼容新版 ollama Python 客户端的问题。
    """

    def __init__(
        self,
        model: str = OLLAMA_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            results.append(self.embed_query(text))
        return results

    def embed_query(self, text: str) -> list[float]:
        with httpx.Client(trust_env=False) as client:
            resp = client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()["embedding"]


embedding_model = OllamaEmbeddings()

_faiss_db = None

_reranker_model=None
_reranker_tokenizer=None
RERANKER_MODEL_NAME="BAAI/bge-reranker-base"

#加载模型  只在第一次调用的时候执行 
def _get_reranker():
    global _reranker_model,_reranker_tokenizer
    if _reranker_model is None:
        _reranker_tokenizer=AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        _reranker_model=AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME, torch_dtype=torch.float32)
        #16位改成32位，之前精度溢出了，rerank打分不准确
        _reranker_model.eval()
    return _reranker_model,_reranker_tokenizer

def _get_faiss_db():
    global _faiss_db
    if _faiss_db is None:
        _faiss_db = FAISS.load_local(
            FAISS_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    return _faiss_db
#输出为重排列后的文档列表
def _rerank(query:str,docs:list,top_k:int=5)->list:
    model,tokenizer=_get_reranker()
    pairs=[[query,doc.page_content] for doc in docs]#配对
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)#翻译
        scores=model(**inputs).logits.squeeze(-1)#打分
        ranked_indices = scores.argsort(descending=True)[:top_k]#排序取前5条
        return [docs[i] for i in ranked_indices]


def build_legal_vectorstore(pdf_folder: str):
    """把一个文件夹里的法律 PDF 建成向量库（只需运行一次）"""
    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if not filename.endswith(".pdf"):
            continue
        file_path = os.path.join(pdf_folder, filename)
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = filename
        all_chunks.extend(chunks)

    faiss_db = FAISS.from_documents(all_chunks, embedding_model)
    faiss_db.save_local(FAISS_DB_PATH)
    print(f"向量库建好了，共 {len(all_chunks)} 个文本段")
    return faiss_db

def add_legal_documents(pdf_folder:str):
    """向已有向量库增量添加新PDF，不重新建已有的"""
    faiss_db=_get_faiss_db()
    existing_sources=set()
    for doc in faiss_db.docstore._dict.values():
        existing_sources.add(doc.metadata.get("source",""))
    new_chunks=[]
    for filename in os.listdir(pdf_folder):
        if not filename.endswith(".pdf"):
            continue
        if filename in existing_sources:
            print(f"跳过已存在：{filename}")
            continue
        print(f"正在处理:{filename}")
        file_path=os.path.join(pdf_folder,filename)
        loader=PDFPlumberLoader(file_path)
        documents=loader.load()
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)
        chunks=splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"]=filename
        new_chunks.extend(chunks)
    
    if not new_chunks:
        print("没有新的PDF要添加")
        return faiss_db
    
    faiss_db.add_documents(new_chunks)
    faiss_db.save_local(FAISS_DB_PATH)
    print(f"新增{len(new_chunks)}个文本段，来自{len(set(c.metadata['source'] for c in new_chunks))}个文件")
    return faiss_db

def retrieve_legal_docs(query: str, k: int = 5, top_k: int = 3) -> list[str]:
    """对外暴露的检索接口，返回字符串列表"""
    faiss_db = _get_faiss_db()
    docs = faiss_db.similarity_search(query, k=k)
    try:
        docs=_rerank(query,docs,top_k=top_k)
    except Exception as e:
        print(f"Rerank 失败，使用 FAISS 原始结果: {e}")
        docs=docs[:top_k]

    results = []
    for doc in docs:
        source = doc.metadata.get("source", "未知文件")
        results.append(f"【来源：{source}】\n{doc.page_content}")
    return results
