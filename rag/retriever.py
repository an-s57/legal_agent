"""FAISS 向量库 — 自定义 Ollama 嵌入 + 法律文档检索"""
import os
import time
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')  # 有本地缓存时不检查更新
os.environ.setdefault('HF_HUB_OFFLINE', '1')         # 有本地缓存时不联网下载
import torch
import httpx
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    FAISS_DB_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    RERANKER_MODEL_NAME,
    RERANK_MAX_LENGTH,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_K_BM25,
    RETRIEVAL_K_VECTOR,
    RETRIEVAL_RRF_K,
    RETRIEVAL_TOP_K,
)
from logger import get_logger
from rag.hybrid import hybrid_candidates

logger = get_logger("legal_agent.rag")


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

_reranker_model = None
_reranker_tokenizer = None


def _get_reranker():
    """加载 reranker 模型（首次调用时加载，后续使用缓存）"""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        _reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        try:
            _reranker_model = AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL_NAME, torch_dtype=torch.float32
            )
        except Exception:
            # 离线模式下没有本地缓存时，临时取消离线标志允许联网下载
            _saved_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _saved_hf = os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                _reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    RERANKER_MODEL_NAME, torch_dtype=torch.float32
                )
            finally:
                if _saved_offline: os.environ["TRANSFORMERS_OFFLINE"] = _saved_offline
                if _saved_hf: os.environ["HF_HUB_OFFLINE"] = _saved_hf
        _reranker_model.eval()
        # INT8 动态量化：把 Linear 层权重从 float32 压成 int8，
        # CPU 推理快 2~3 倍；排序只看分数相对大小，重排质量几乎不变。
        _reranker_model = torch.quantization.quantize_dynamic(
            _reranker_model, {torch.nn.Linear}, dtype=torch.qint8
        )
    return _reranker_model, _reranker_tokenizer


def preload_reranker():
    """启动时预加载 reranker，避免首次请求等待。

    新机器没有本地缓存时，允许联网下载模型（跳过离线标志）。
    加载失败时不阻塞启动，降级为首次请求时再加载。
    """
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is not None:
        return
    logger.info("[RAG] 预加载 reranker 模型...")
    try:
        _get_reranker()
        logger.info("[RAG] reranker 模型加载完成")
    except Exception as e:
        logger.warning(
            f"[RAG] reranker 预加载失败（{type(e).__name__}: {e}），"
            f"将在首次检索时重试。如需手动下载："
            f"pip install huggingface_hub && "
            f"python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{RERANKER_MODEL_NAME}')\""
        )

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
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=RERANK_MAX_LENGTH)#翻译
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
    logger.info(f"向量库建好了，共 {len(all_chunks)} 个文本段")
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
            logger.info(f"跳过已存在：{filename}")
            continue
        logger.info(f"正在处理:{filename}")
        file_path=os.path.join(pdf_folder,filename)
        loader=PDFPlumberLoader(file_path)
        documents=loader.load()
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)
        chunks=splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"]=filename
        new_chunks.extend(chunks)
    
    if not new_chunks:
        logger.info("没有新的PDF要添加")
        return faiss_db
    
    faiss_db.add_documents(new_chunks)
    faiss_db.save_local(FAISS_DB_PATH)
    logger.info(f"新增{len(new_chunks)}个文本段，来自{len(set(c.metadata['source'] for c in new_chunks))}个文件")
    return faiss_db

def retrieve_legal_docs(query: str, k: int = RETRIEVAL_K_VECTOR, top_k: int = RETRIEVAL_TOP_K) -> list[str]:
    """对外暴露的检索接口，返回字符串列表"""
    #向量库加载用时
    load_start=time.perf_counter()
    faiss_db = _get_faiss_db()
    load_ms=(time.perf_counter()-load_start)*1000
    logger.info(f"[PERF] vectorstore_load={load_ms:.0f}ms")
    #混合检索：FAISS 向量 + BM25 词面 → RRF 融合（k 作 k_vector 传）
    search_start=time.perf_counter()
    docs = hybrid_candidates(
        query, faiss_db, k_vector=k, k_bm25=RETRIEVAL_K_BM25, rrf_k=RETRIEVAL_RRF_K, top_candidates=RETRIEVAL_CANDIDATES,
    )
    search_ms=(time.perf_counter()-search_start)*1000
    logger.info(f"[PERF] hybrid_search={search_ms:.0f}ms "
                f"k_vector={k} k_bm25={RETRIEVAL_K_BM25} candidates={RETRIEVAL_CANDIDATES}")
    #rerank用时
    rerank_time=time.perf_counter()
    try:
        docs=_rerank(query,docs,top_k=top_k)
        rerank_ms=(time.perf_counter()-rerank_time)*1000
        logger.info(f"[PERF] rerank={rerank_ms:.0f}ms "
                    f"top_k={top_k} fallback=False")
    except Exception as e:
        rerank_ms=(time.perf_counter()-rerank_time)*1000
        logger.info(f"[PERF] rerank={rerank_ms:.0f}ms "
                    f"fallback=True error_type={type(e).__name__}")
        logger.warning(f"Rerank 失败，使用 FAISS 原始结果: {e}")
        docs=docs[:top_k]

    results = []
    for doc in docs:
        source = doc.metadata.get("source", "未知文件")
        results.append(f"【来源：{source}】\n{doc.page_content}")
    return results
