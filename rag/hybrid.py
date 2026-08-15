"""混合检索：FAISS 向量召回 + BM25 词面召回 + RRF 融合。

P3 实验：7 道稳定失败题的短锚句（10~17 字）在纯向量检索里词面匹配弱，
进不了 Top-5。BM25 按词频打分，恰好补这个短板；RRF（Reciprocal Rank
Fusion）只用排名不看分数融合两路，只被一路召回的段也能进候选池。

语料与向量库同源：索引从 faiss_db.docstore 的文本段建，source/page
元数据天然对齐，评测判定（document_matches_evidence）零改动。

用法（评测/生产共用同一实现）：
    from rag.hybrid import hybrid_candidates
    docs = hybrid_candidates(query, faiss_db, k_vector=40, k_bm25=40,
                             rrf_k=60, top_candidates=40)
"""
import threading

import jieba
from rank_bm25 import BM25Okapi

from config import RETRIEVAL_RRF_K as RRF_K  # RRF 融合常数，经验默认 60（经典取值，后续可扫）

_bm25_index = None
_bm25_docs = None
_bm25_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    """jieba 搜索引擎模式分词，过滤空白 token。

    搜索引擎模式会把长词再切细（如「消费者权益保护法」→
    「消费/消费者/权益/保护/法」），对短锚句词面匹配更友好。
    """
    return [w for w in jieba.cut_for_search(text) if w.strip()]


def _get_bm25(faiss_db) -> tuple[BM25Okapi | None, list]:
    """懒加载 BM25 索引（与向量库同源），线程安全缓存。

    注：缓存无失效机制——docstore 变化（增量建库）后需重启进程。
    生产库里文本段固定，P3 阶段可接受。
    """
    global _bm25_index, _bm25_docs
    if _bm25_index is None:
        with _bm25_lock:#加锁
            if _bm25_index is None:#双重检查
                docs = list(faiss_db.docstore._dict.values())#BM25和向量库同源
                if not docs:
                    # 空语料：不建索引（BM25Okapi([]) 会除零），调用方按"无 BM25 路"处理
                    return None, []
                corpus = [_tokenize(d.page_content) for d in docs]#索引文档的分词结果
                _bm25_index = BM25Okapi(corpus)#建索引
                _bm25_docs = docs
    return _bm25_index, _bm25_docs


def _rrf_fuse(
    vector_docs: list,
    bm25_docs: list,
    *,
    rrf_k: int = RRF_K,
    top_candidates: int = 40,
) -> list:
    """按排名融合两路召回，取 top_candidates 返回。

    score(d) = Σ 1/(rrf_k + rank)，排名从 1 开始。只被一路召回的段
    加一项即可进池——这是 BM25 能救短锚句的机制。

    依赖：两路返回的是 docstore 里的同一 Document 引用（langchain
    FAISS 从 docstore 取对象，BM25 从 docstore 建索引），故用 id(doc)
    对齐。语料 156 段很小，by_id 重建开销可忽略。
    """
    scores = {}
    for rank, doc in enumerate(vector_docs, start=1):
        key = id(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    for rank, doc in enumerate(bm25_docs, start=1):
        key = id(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores, key=scores.get, reverse=True)[:top_candidates]
    by_id = {id(doc): doc for doc in vector_docs + bm25_docs}
    return [by_id[doc_id] for doc_id in ranked if doc_id in by_id]


def hybrid_candidates(
    query: str,
    faiss_db,
    *,
    k_vector: int = 40,
    k_bm25: int = 40,
    rrf_k: int = RRF_K,
    top_candidates: int = 40,
) -> list:
    """混合召回：FAISS top k_vector + BM25 top k_bm25 → RRF → top_candidates。

    返回 list[Document]，由调用方继续 rerank / 判定。
    """
    vector_docs = faiss_db.similarity_search(query, k=k_vector)#向量路

    tokens = _tokenize(query)#给问题分词
    bm25_docs = []
    if tokens:
        bm25, docs = _get_bm25(faiss_db)
        if bm25 is not None:
            scores = bm25.get_scores(tokens)#对查询的每个词每个文档算分数
            top_idx = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            bm25_docs = [docs[i] for i in top_idx[:k_bm25] if scores[i] > 0]

    return _rrf_fuse(
        vector_docs, bm25_docs, rrf_k=rrf_k, top_candidates=top_candidates
    )