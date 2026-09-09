"""Redis 意图缓存单测 — 全离线：mock redis 客户端，不连接真实 Redis。

运行：python -m unittest tests.test_redis_cache -v
"""
import unittest
from unittest import mock

from cache import redis_client as rc


class TestNormalizeKey(unittest.TestCase):
    def test_strips_space_and_punct(self):
        self.assertEqual(rc.normalize_key(" 试用期最长多久？"), "试用期最长多久")
        self.assertEqual(rc.normalize_key(" 试用期最长多久？! "), "试用期最长多久")

    def test_lowercase(self):
        self.assertEqual(rc.normalize_key("AbC 试用期"), "abc试用期")

    def test_fullwidth_space(self):
        self.assertEqual(rc.normalize_key("试用期\u3000最长"), "试用期最长")

    def test_build_key_deterministic(self):
        self.assertEqual(rc.build_cache_key("试用期最长多久"),
                         rc.build_cache_key("试用期最长多久？"))
        self.assertNotEqual(rc.build_cache_key("试用期最长多久"),
                            rc.build_cache_key("服务期最长多久"))
        self.assertTrue(rc.build_cache_key("x").startswith(rc.PREFIX))


class TestCacheFailOpen(unittest.TestCase):
    @mock.patch.object(rc, "_get_client", return_value=None)
    def test_no_client_returns_none(self, _):
        self.assertIsNone(rc.get_intent_decision("试用期最长多久"))
        self.assertFalse(rc.set_intent_decision("试用期最长多久", {"info_complete": True}))

    @mock.patch.object(rc, "_get_client")
    def test_redis_down_does_not_raise(self, mock_get):
        client = mock.MagicMock()
        client.get.side_effect = Exception("connection refused")
        client.set.side_effect = Exception("connection refused")
        mock_get.return_value = client
        self.assertIsNone(rc.get_intent_decision("问题"))      # fail-open
        self.assertFalse(rc.set_intent_decision("问题", {"info_complete": True}))  # fail-open

    @mock.patch.object(rc, "_get_client")
    def test_corrupt_json_returns_none(self, mock_get):
        client = mock.MagicMock()
        client.get.return_value = b"{not json"
        mock_get.return_value = client
        self.assertIsNone(rc.get_intent_decision("问题"))


class TestRoundTrip(unittest.TestCase):
    @mock.patch.object(rc, "_get_client")
    def test_set_then_get_returns_decision(self, mock_get):
        client = mock.MagicMock()
        stored = {}

        def fake_set(key, val, ex=None):
            stored[key] = val
            return True

        def fake_get(key):
            return stored.get(key)

        client.set.side_effect = fake_set
        client.get.side_effect = fake_get
        mock_get.return_value = client

        decision = {"info_complete": False, "follow_up": "请问是什么时候受伤的？"}
        self.assertTrue(rc.set_intent_decision("我在工地受伤了", decision))
        got = rc.get_intent_decision("我在工地受伤了")
        self.assertEqual(got, decision)
        # key 带 TTL
        args = client.set.call_args
        self.assertEqual(args.kwargs.get("ex"), rc.INTENT_CACHE_TTL)


class TestAnswerCache(unittest.TestCase):
    """回答缓存（1.1B）：与意图缓存隔离的 key 前缀 + 往返 + fail-open。"""

    def test_answer_key_与意图key隔离(self):
        self.assertTrue(rc.build_answer_key("x").startswith(rc.ANSWER_PREFIX))
        self.assertTrue(rc.ANSWER_PREFIX.startswith("answer_cache:v"))  # 带版本号
        self.assertNotEqual(rc.build_answer_key("试用期最长多久"),
                            rc.build_cache_key("试用期最长多久"))

    def test_answer_key_归一化一致(self):
        self.assertEqual(rc.build_answer_key("试用期最长多久"),
                         rc.build_answer_key("试用期 最长 多久？"))

    @mock.patch.object(rc, "_get_client")
    def test_回答往返_带TTL(self, mock_get):
        client = mock.MagicMock()
        stored = {}
        client.set.side_effect = lambda k, v, ex=None: stored.__setitem__(k, v) or True
        client.get.side_effect = lambda k: stored.get(k)
        mock_get.return_value = client

        self.assertTrue(rc.set_answer("试用期最长多久", "最长六个月。"))
        self.assertEqual(rc.get_answer("试用期最长多久？"), "最长六个月。")  # 归一化命中
        self.assertEqual(client.set.call_args.kwargs.get("ex"), rc.ANSWER_CACHE_TTL)

    @mock.patch.object(rc, "_get_client")
    def test_残缺或脏数据返回None(self, mock_get):
        client = mock.MagicMock()
        mock_get.return_value = client
        client.get.return_value = b"{not json"
        self.assertIsNone(rc.get_answer("x"))
        client.get.return_value = b'{"no_answer_key": 1}'
        self.assertIsNone(rc.get_answer("x"))
        client.get.return_value = b'{"answer": ""}'   # 空回答视为未命中
        self.assertIsNone(rc.get_answer("x"))

    @mock.patch.object(rc, "_get_client", return_value=None)
    def test_无客户端_fail_open(self, _):
        self.assertIsNone(rc.get_answer("x"))
        self.assertFalse(rc.set_answer("x", "答案"))

    @mock.patch.object(rc, "_get_client")
    def test_连接异常_fail_open(self, mock_get):
        client = mock.MagicMock()
        client.get.side_effect = Exception("boom")
        client.set.side_effect = Exception("boom")
        mock_get.return_value = client
        self.assertIsNone(rc.get_answer("x"))
        self.assertFalse(rc.set_answer("x", "答案"))


class TestRetrievalCache(unittest.TestCase):
    """检索结果缓存（#3）：key 绑版本+k+top_k、往返、空哨兵、fail-open。"""

    def test_检索key带版本与检索配置(self):
        k = rc.build_retrieval_key("违法解除 赔偿")
        self.assertTrue(k.startswith("retrieval_cache:"))
        self.assertIn("_k", k)     # 编入了 k_vector
        self.assertIn("_tk", k)    # 编入了 top_k
        self.assertNotEqual(k, rc.build_answer_key("违法解除 赔偿"))
        self.assertNotEqual(k, rc.build_cache_key("违法解除 赔偿"))

    def test_检索key归一化一致(self):
        self.assertEqual(rc.build_retrieval_key("违法解除赔偿"),
                         rc.build_retrieval_key("违法解除 赔偿？"))

    @mock.patch.object(rc, "_get_client")
    def test_检索往返(self, mock_get):
        client = mock.MagicMock()
        stored = {}
        client.set.side_effect = lambda k, v, ex=None: stored.__setitem__(k, v) or True
        client.get.side_effect = lambda k: stored.get(k)
        mock_get.return_value = client
        self.assertTrue(rc.set_retrieval("试用期最长多久", "第一条\n\n---\n\n第二条"))
        self.assertEqual(rc.get_retrieval("试用期 最长 多久？"), "第一条\n\n---\n\n第二条")
        self.assertEqual(client.set.call_args.kwargs.get("ex"), rc.RETRIEVAL_CACHE_TTL)

    @mock.patch.object(rc, "_get_client")
    def test_空哨兵也缓存(self, mock_get):
        client = mock.MagicMock()
        stored = {}
        client.set.side_effect = lambda k, v, ex=None: stored.__setitem__(k, v) or True
        client.get.side_effect = lambda k: stored.get(k)
        mock_get.return_value = client
        sentinel = "法律文档库中未找到相关内容"
        self.assertTrue(rc.set_retrieval("某冷门词", sentinel))
        self.assertEqual(rc.get_retrieval("某冷门词"), sentinel)

    @mock.patch.object(rc, "_get_client", return_value=None)
    def test_检索fail_open(self, _):
        self.assertIsNone(rc.get_retrieval("x"))
        self.assertFalse(rc.set_retrieval("x", "结果"))


class TestCacheStats(unittest.TestCase):
    """命中率统计：hit/miss/unavailable/write 计数 + 命中率 + fail-safe。"""

    def setUp(self):
        rc.reset_stats()

    def test_hit_miss_unavailable_计数(self):
        c = mock.MagicMock(); c.get.return_value = b'{"info_complete": true}'
        with mock.patch.object(rc, "_get_client", return_value=c):
            rc.get_intent_decision("q")                       # hit
        c2 = mock.MagicMock(); c2.get.return_value = None
        with mock.patch.object(rc, "_get_client", return_value=c2):
            rc.get_intent_decision("q")                       # miss
        with mock.patch.object(rc, "_get_client", return_value=None):
            rc.get_intent_decision("q")                       # unavailable
        s = rc.get_cache_stats()["intent"]
        self.assertEqual(s["hit"], 1)
        self.assertEqual(s["miss"], 1)
        self.assertEqual(s["unavailable"], 1)                 # Redis 挂了不算 miss
        self.assertEqual(s["hit_rate"], 0.5)

    def test_write计数(self):
        c = mock.MagicMock()
        with mock.patch.object(rc, "_get_client", return_value=c):
            self.assertTrue(rc.set_retrieval("q", "res"))
        self.assertEqual(rc.get_cache_stats()["retrieval"]["write"], 1)

    def test_record_fail_safe(self):
        rc.record("bogus_cache", "hit")     # 未知缓存名 → 静默忽略
        rc.record("intent", "bogus_field")  # 未知字段 → 静默忽略
        self.assertEqual(rc.get_cache_stats()["intent"]["hit"], 0)

    def test_snapshot_含三层与命中率字段(self):
        s = rc.get_cache_stats()
        for layer in ("intent", "answer", "retrieval"):
            self.assertIn(layer, s)
            self.assertIn("hit_rate", s[layer])
            self.assertIsNone(s[layer]["hit_rate"])   # 无样本时命中率为 None


if __name__ == "__main__":
    unittest.main()
