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


if __name__ == "__main__":
    unittest.main()
