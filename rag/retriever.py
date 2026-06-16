# retriever.py
import os
import httpx
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

FAISS_DB_PATH = "rag/vectorstore/db_faiss"#向量库的存放路径
OLLAMA_BASE_URL = "http://127.0.0.1:11434"#ollama服务的地址
OLLAMA_EMBED_MODEL = "nomic-embed-text"#转向量的模型


class OllamaEmbeddings(Embeddings):
    """自定义嵌入类，直接调 Ollama 旧版 /api/embeddings 接口
    绕过 langchain_ollama 不兼容新版 ollama Python 客户端的问题"""

    def __init__(self, model: str = OLLAMA_EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入——Ollama 的旧 API 一次只支持一个文本，所以逐条调用"""
        results = []
        for text in texts:
            results.append(self.embed_query(text))
        return results

    def embed_query(self, text: str) -> list[float]:
        """单条文本嵌入（trust_env=False 避免被代理拦截）"""
        with httpx.Client(trust_env=False) as client:
            resp = client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()["embedding"]


embedding_model = OllamaEmbeddings()
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # 支持中文


# ---- 建库用：只需要跑一次 ----
def build_legal_vectorstore(pdf_folder: str):
    """
    把一个文件夹里的法律PDF建成向量库
    """
    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if not filename.endswith(".pdf"):
            continue
        file_path = os.path.join(pdf_folder, filename)
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()#加载文件成文档对象列表
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = filename#标签 显示来源
        all_chunks.extend(chunks)#存储所有文本段

    faiss_db = FAISS.from_documents(all_chunks, embedding_model)#存入向量库
    faiss_db.save_local(FAISS_DB_PATH)#把向量库保存到本地
    print(f"向量库建好了，共 {len(all_chunks)} 个文本段")
    return faiss_db


# ---- 检索用：Agent每次调用 ----
def retrieve_legal_docs(query: str, k: int = 10) -> list[str]:
    """
    对外暴露的检索接口，返回字符串列表
    """
    faiss_db = FAISS.load_local(
        FAISS_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )#加载磁盘上的向量库
    docs = faiss_db.similarity_search(query, k=k)#根据查询找相似文本段，返回文档对象列表
    #rerank
    #构造打分对，chunk和问题
    pairs=[[query,doc.page_content] for doc in docs]
    #模型打分
    scores=reranker.predict(pairs)
    #按分数降序
    sorted_results=sorted(zip(scores,docs),key=lambda x:x[0],reverse=True)
    #取前3
    top_docs=[doc for _,doc in sorted_results[:3]]
    results = []
    for doc in top_docs:
        source = doc.metadata.get("source", "未知文件")#
        results.append(f"【来源：{source}】\n{doc.page_content}")#
    return results
