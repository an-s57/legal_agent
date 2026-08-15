"""混合检索单元测试 — RRF 融合 + hybrid_candidates。

全部纯内存运行：不依赖 FAISS 索引文件、不调 Ollama、不联网。
hybrid_candidates 用假 FAISS 对象（只模拟 similarity_search 和 docstore._dict），
BM25 路用真实的 jieba + rank_bm25 在内存语料上建索引。

注意：rag.hybrid 的 BM25 索引是模块级缓存（无失效机制），
setUp/tearDown 必须重置，避免测试间互相污染。
"""
import unittest
from types import SimpleNamespace

from langchain_core.documents import Document

import rag.hybrid as hybrid


class FakeFaiss:
    """假 FAISS：只实现 hybrid 用到的两个属性。"""

    def __init__(self, docs: list[Document], vector_order: list[int] | None = None):
        self.docstore = SimpleNamespace(_dict={f"doc{i}": d for i, d in enumerate(docs)})
        self._docs = docs
        self._vector_order = vector_order if vector_order is not None else list(range(len(docs)))

    def similarity_search(self, query: str, k: int) -> list[Document]:
        """模拟向量路：按预定顺序返回前 k 个（不真正计算相似度）。"""
        ordered = [self._docs[i] for i in self._vector_order if i < len(self._docs)]
        return ordered[:k]


class RrfFuseTest(unittest.TestCase):
    """_rrf_fuse 纯函数：只按排名融合，不看分数。"""

    def setUp(self) -> None:
        self.a = Document(page_content="A")
        self.b = Document(page_content="B")
        self.c = Document(page_content="C")

    def test_两路都召回的排最前(self) -> None:
        # B 在 vector 排第 2、bm25 排第 1 → 两路得分之和最大，应排第一
        result = hybrid._rrf_fuse([self.a, self.b, self.c], [self.b], rrf_k=60, top_candidates=10)
        self.assertEqual(result[0], self.b)

    def test_只被bm25_一路召回的也能进结果(self) -> None:
        # C 只出现在 bm25 路 → 仍应进结果（这正是 BM25 救短锚句的机制）
        result = hybrid._rrf_fuse([self.a, self.b], [self.c], rrf_k=60, top_candidates=10)
        self.assertIn(self.c, result)

    def test_top_candidates_截断(self) -> None:
        docs = [Document(page_content=f"doc{i}") for i in range(10)]
        result = hybrid._rrf_fuse(docs, docs, rrf_k=60, top_candidates=3)
        self.assertEqual(len(result), 3)

    def test_空输入不崩(self) -> None:
        self.assertEqual(hybrid._rrf_fuse([], []), [])
        self.assertEqual(hybrid._rrf_fuse([self.a], []), [self.a])

    def test_大rrf_k_退化为按名次排序(self) -> None:
        # rrf_k 极大时，分差主要来自名次 → 名次靠前的顺序保持
        result = hybrid._rrf_fuse([self.a, self.b, self.c], [self.a], rrf_k=100000, top_candidates=10)
        self.assertEqual(result[0], self.a)


class HybridCandidatesTest(unittest.TestCase):
    """hybrid_candidates 全链路：假向量路 + 真 BM25 路 + RRF 融合。"""

    def setUp(self) -> None:
        # 语料：只有 d3 含"退货"（d1/d2 完全避开该词）
        self.d1 = Document(page_content="苹果手机购买须知。")
        self.d2 = Document(page_content="商品质量保证说明。")
        self.d3 = Document(page_content="七天无理由退货。")
        # 重置模块级 BM25 缓存（无失效机制，测试间必须清）
        hybrid._bm25_index = None
        hybrid._bm25_docs = None

    def tearDown(self) -> None:
        hybrid._bm25_index = None
        hybrid._bm25_docs = None

    def test_bm25_救回向量漏掉的文档(self) -> None:
        # 向量路只召回 [d1, d2]（故意漏掉 d3，模拟纯向量漏召回短锚句）
        faiss = FakeFaiss([self.d1, self.d2, self.d3], vector_order=[0, 1])
        docs = hybrid.hybrid_candidates(
            "退货", faiss, k_vector=2, k_bm25=10, rrf_k=60, top_candidates=10
        )
        # d3 被 BM25 路救回 → 出现在结果里
        self.assertIn(self.d3, docs)
        # 向量路 rank1 的 d1 仍在 d3 之前（RRF 分 1/61 与 d3 并列，稳定排序保持插入序）
        self.assertLess(docs.index(self.d1), docs.index(self.d3))

    def test_两路都召回的排最前(self) -> None:
        # 向量路 [d1,d2,d3]，BM25 只命中 d3 → d3 两路有分，应排第一
        faiss = FakeFaiss([self.d1, self.d2, self.d3], vector_order=[0, 1, 2])
        docs = hybrid.hybrid_candidates(
            "退货", faiss, k_vector=10, k_bm25=10, rrf_k=60, top_candidates=10
        )
        self.assertEqual(docs[0], self.d3)

    def test_查询无命中_退化为纯向量结果(self) -> None:
        faiss = FakeFaiss([self.d1, self.d2, self.d3], vector_order=[0, 1, 2])
        docs = hybrid.hybrid_candidates(
            "完全不存在的词xyz", faiss, k_vector=10, k_bm25=10, rrf_k=60, top_candidates=10
        )
        # bm25 路空 → 只剩向量路，按向量顺序返回
        self.assertEqual(docs, [self.d1, self.d2, self.d3])

    def test_空语料不崩(self) -> None:
        faiss = FakeFaiss([])
        docs = hybrid.hybrid_candidates(
            "退货", faiss, k_vector=10, k_bm25=10, rrf_k=60, top_candidates=10
        )
        self.assertEqual(docs, [])


if __name__ == "__main__":
    unittest.main()
