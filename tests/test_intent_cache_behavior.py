"""意图缓存"行为"单元测试 — 验证 call_planner 里读/写/跳过缓存的分支是否正确。

与 test_redis_cache.py 的分工：
  - test_redis_cache.py 测"缓存函数本身"（key 归一化、fail-open、读写往返）；
  - 本文件测"缓存被用在对的地方"（命中跳过 LLM、有上下文绝不查缓存等安全约束）。

全离线：mock 掉 planner LLM 与 Redis 读写函数，不连真实 Redis、不调真实 DeepSeek。
运行：python -m unittest tests.test_intent_cache_behavior -v
"""
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")  # 占位，保证导入模块不依赖真实密钥
os.environ.setdefault("CACHE_ENABLED", "0")            # 离线单测不连真实 Redis

from unittest import mock
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agent import legal_agent


class _StubResponse:
    """假 LLM 响应：只带 call_planner 用到的 tool_calls。"""

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def _planner_returning(args):
    """假 planner：invoke() 固定返回给定决策，可统计是否被调用。"""
    planner = mock.MagicMock()
    planner.invoke.return_value = _StubResponse(
        [{"name": "PlannerDecision", "args": args, "id": "stub-1", "type": "tool_call"}]
    )
    return planner


def _first_ask(msg="我在工地受伤了") -> dict:
    """无上下文首问 state（_ctx_free=True 前提：单条 Human、无 AIMessage、空案情摘要）。"""
    return {
        "messages": [HumanMessage(content=msg)],
        "case_summary": "{}",
        "skip_planner": False,
    }


class IntentCacheBehaviorTest(unittest.TestCase):

    def test_命中缓存_跳过LLM(self) -> None:
        cached = {"info_complete": False, "follow_up": "缓存里的追问"}
        planner = _planner_returning({"info_complete": True})
        with mock.patch.object(legal_agent, "get_intent_decision", return_value=cached), \
             mock.patch.object(legal_agent, "set_intent_decision") as set_cache, \
             mock.patch.object(legal_agent, "planner_tool_llm", planner):
            result = legal_agent.call_planner(_first_ask())
        self.assertFalse(result["info_complete"])
        self.assertEqual(result["messages"][-1].content, "缓存里的追问")
        planner.invoke.assert_not_called()   # 命中 → 一次 LLM 都不调（真省钱）
        set_cache.assert_not_called()        # 命中 → 无需再写缓存

    def test_未命中_调用LLM并写回缓存(self) -> None:
        planner = _planner_returning({"info_complete": True})
        with mock.patch.object(legal_agent, "get_intent_decision", return_value=None), \
             mock.patch.object(legal_agent, "set_intent_decision") as set_cache, \
             mock.patch.object(legal_agent, "planner_tool_llm", planner):
            result = legal_agent.call_planner(_first_ask())
        self.assertTrue(result["info_complete"])
        planner.invoke.assert_called_once()  # 未命中 → 老实跑 LLM
        set_cache.assert_called_once()       # 并把结果写回缓存
        self.assertEqual(
            set_cache.call_args[0][1],
            {"info_complete": True, "follow_up": "", "is_general_knowledge": False,
             "is_data_query": False},
        )

    def test_有上下文_绝不查缓存(self) -> None:
        # 带对话历史 + 案情摘要：即便"假装有缓存"也不能去查，必须走 LLM
        state = {
            "messages": [
                HumanMessage(content="我在工地受伤了"),
                AIMessage(content="请问是什么时候受伤的？"),
                HumanMessage(content="昨天"),
            ],
            "case_summary": '{"event": "工地受伤"}',
            "skip_planner": False,
        }
        planner = _planner_returning({"info_complete": True})
        with mock.patch.object(legal_agent, "get_intent_decision") as get_cache, \
             mock.patch.object(legal_agent, "set_intent_decision") as set_cache, \
             mock.patch.object(legal_agent, "planner_tool_llm", planner):
            result = legal_agent.call_planner(state)
        get_cache.assert_not_called()        # 安全命门：带上下文连查都不查
        set_cache.assert_not_called()        # 也不写
        planner.invoke.assert_called_once()  # 直接走 LLM
        self.assertTrue(result["info_complete"])

    def test_缓存内容残缺_保守放行(self) -> None:
        # 缓存 dict 缺 info_complete → .get(..., True) 默认放行，不能误判成"要追问"
        cached = {"follow_up": "只有一个字段"}
        planner = _planner_returning({"info_complete": False, "follow_up": "不该被用到"})
        with mock.patch.object(legal_agent, "get_intent_decision", return_value=cached), \
             mock.patch.object(legal_agent, "planner_tool_llm", planner):
            result = legal_agent.call_planner(_first_ask())
        self.assertTrue(result["info_complete"])
        planner.invoke.assert_not_called()

    def test_客观知识_落缓存带is_general_knowledge(self) -> None:
        # 回答缓存（1.1B）写门控依赖此信号：客观知识问题的意图缓存值须带 is_general_knowledge=True
        planner = _planner_returning({"info_complete": True, "is_general_knowledge": True})
        with mock.patch.object(legal_agent, "get_intent_decision", return_value=None), \
             mock.patch.object(legal_agent, "set_intent_decision") as set_cache, \
             mock.patch.object(legal_agent, "planner_tool_llm", planner):
            legal_agent.call_planner(_first_ask("试用期最长可以约定几个月？"))
        set_cache.assert_called_once()
        stored = set_cache.call_args[0][1]
        self.assertTrue(stored["is_general_knowledge"])

    def test_个人案情_落缓存is_general_knowledge为False(self) -> None:
        # 案情咨询（即便信息完整放行）is_general_knowledge 必须为 False，回答缓存才不会缓存它
        planner = _planner_returning({"info_complete": True})  # 未给该字段 → 默认 False
        with mock.patch.object(legal_agent, "get_intent_decision", return_value=None), \
             mock.patch.object(legal_agent, "set_intent_decision") as set_cache, \
             mock.patch.object(legal_agent, "planner_tool_llm", planner):
            legal_agent.call_planner(_first_ask("上周我买的手机是翻新机想退货"))
        stored = set_cache.call_args[0][1]
        self.assertFalse(stored["is_general_knowledge"])


    def test_数据查询_落缓存带标记并返回指路(self) -> None:
        # Planner 判定 is_data_query=True → 不进法律链路，返回指路提示，缓存带标记
        planner = _planner_returning({"info_complete": True, "is_data_query": True})
        with mock.patch.object(legal_agent, "get_intent_decision", return_value=None), \
             mock.patch.object(legal_agent, "set_intent_decision") as set_cache, \
             mock.patch.object(legal_agent, "planner_tool_llm", planner):
            result = legal_agent.call_planner(_first_ask("8月有多少个会话？"))
        stored = set_cache.call_args[0][1]
        self.assertTrue(stored["is_data_query"])
        self.assertFalse(result["info_complete"])
        self.assertIn("运营问答", result["messages"][-1].content)

    def test_缓存命中数据查询_回放指路提示(self) -> None:
        # 旧缓存里带 is_data_query 标记 → 命中后回放指路提示，不再调 LLM
        cached = {"info_complete": False, "follow_up": legal_agent.DATA_QUERY_REDIRECT,
                  "is_data_query": True}
        planner = _planner_returning({"info_complete": True})
        with mock.patch.object(legal_agent, "get_intent_decision", return_value=cached), \
             mock.patch.object(legal_agent, "planner_tool_llm", planner):
            result = legal_agent.call_planner(_first_ask("8月有多少个会话？"))
        self.assertIn("运营问答", result["messages"][-1].content)
        planner.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
