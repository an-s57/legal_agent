# rag/retriever.py
import os
import httpx
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FAISS_DB_PATH = "rag/vectorstore/db_faiss"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


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
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = filename
        all_chunks.extend(chunks)

    faiss_db = FAISS.from_documents(all_chunks, embedding_model)
    faiss_db.save_local(FAISS_DB_PATH)
    print(f"向量库建好了，共 {len(all_chunks)} 个文本段")
    return faiss_db


# ---- 检索用：Agent每次调用 ----
def retrieve_legal_docs(query: str, k: int = 5) -> list[str]:
    """
    对外暴露的检索接口，返回字符串列表
    """
    faiss_db = FAISS.load_local(
        FAISS_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    docs = faiss_db.similarity_search(query, k=k)
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "未知文件")
        results.append(f"【来源：{source}】\n{doc.page_content}")
    return results
