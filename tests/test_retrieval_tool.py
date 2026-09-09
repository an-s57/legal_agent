"""legal_rag_search 工具的检索缓存行为测试 — 全离线，mock 掉检索与 Redis 读写。

运行：python -m unittest tests.test_retrieval_tool -v
"""
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CACHE_ENABLED", "0")

from unittest import mock
import unittest

from tools import legal_tools


class LegalRagSearchCacheTest(unittest.TestCase):

    def test_命中缓存_跳过检索(self) -> None:
        retrieve = mock.MagicMock(return_value=["不该被调用"])
        with mock.patch.object(legal_tools, "get_retrieval", return_value="缓存的检索结果"), \
             mock.patch.object(legal_tools, "retrieve_legal_docs", retrieve), \
             mock.patch.object(legal_tools, "set_retrieval") as set_r:
            out = legal_tools.legal_rag_search.invoke({"query": "违法解除 赔偿"})
        self.assertEqual(out, "缓存的检索结果")
        retrieve.assert_not_called()   # 命中 → 不跑真正的检索（省 ~3s rerank）
        set_r.assert_not_called()      # 命中 → 不再写

    def test_未命中_检索并写缓存(self) -> None:
        retrieve = mock.MagicMock(return_value=["第一条", "第二条"])
        with mock.patch.object(legal_tools, "get_retrieval", return_value=None), \
             mock.patch.object(legal_tools, "retrieve_legal_docs", retrieve), \
             mock.patch.object(legal_tools, "set_retrieval") as set_r:
            out = legal_tools.legal_rag_search.invoke({"query": "试用期最长多久"})
        self.assertEqual(out, "第一条\n\n---\n\n第二条")
        retrieve.assert_called_once()
        set_r.assert_called_once_with("试用期最长多久", "第一条\n\n---\n\n第二条")

    def test_空结果也缓存哨兵(self) -> None:
        retrieve = mock.MagicMock(return_value=[])
        with mock.patch.object(legal_tools, "get_retrieval", return_value=None), \
             mock.patch.object(legal_tools, "retrieve_legal_docs", retrieve), \
             mock.patch.object(legal_tools, "set_retrieval") as set_r:
            out = legal_tools.legal_rag_search.invoke({"query": "某冷门词"})
        self.assertEqual(out, "法律文档库中未找到相关内容")
        set_r.assert_called_once_with("某冷门词", "法律文档库中未找到相关内容")


if __name__ == "__main__":
    unittest.main()
