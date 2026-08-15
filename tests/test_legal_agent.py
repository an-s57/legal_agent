"""Agent 层单元测试 — 历史截断 + Planner 决策。

Planner 测试用自定义 stub 替换 planner_tool_llm（不调用真实 LLM）。
import agent.legal_agent 会构建 LangGraph 图（离线，不联网）。
"""
import os
from unittest import mock

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")  # 占位，只保证导入模块不依赖真实密钥

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agent import legal_agent
from config import MAX_HISTORY_TURNS


class TrimHistoryTest(unittest.TestCase):
    """_trim_history：保留最近 max_turns 轮，超出丢弃（数据库存全量，Agent 层截断）。"""

    def _turns(self, n: int) -> list:
        msgs = []
        for i in range(n):
            msgs.append(HumanMessage(content=f"h{i}"))
            msgs.append(AIMessage(content=f"a{i}"))
        return msgs

    def _contents(self, msgs) -> list[str]:
        return [m.content for m in msgs]

    def test_不超过上限_全保留(self) -> None:
        msgs = self._turns(MAX_HISTORY_TURNS)
        self.assertEqual(self._contents(legal_agent._trim_history(msgs)), self._contents(msgs))

    def test_超过上限_保留最近N轮(self) -> None:
        msgs = self._turns(MAX_HISTORY_TURNS + 1)
        trimmed = legal_agent._trim_history(msgs)
        self.assertEqual(len(trimmed), MAX_HISTORY_TURNS * 2)
        # 最近一轮保留，最旧一轮丢弃
        self.assertEqual(trimmed[0].content, "h1")
        self.assertEqual(trimmed[-1].content, f"a{MAX_HISTORY_TURNS}")

    def test_空历史(self) -> None:
        self.assertEqual(legal_agent._trim_history([]), [])


class _StubResponse:
    """假 LLM 响应：只带 call_planner 用到的 tool_calls 属性。

    不用 AIMessage 构造——langchain 校验 tool_calls.args 必须是 dict，
    无法直接构造"args 为 JSON 字符串"的场景（生产里该场景来自模型适配器透传）。
    """

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _StubPlanner:
    """假 planner：invoke() 返回带 tool_calls 的响应，不碰真实 LLM。"""

    def __init__(self, args):
        self._args = args

    def invoke(self, prompt):
        return _StubResponse(
            [
                {
                    "name": "PlannerDecision",
                    "args": self._args,
                    "id": "stub-1",
                    "type": "tool_call",
                }
            ]
        )


class CallPlannerTest(unittest.TestCase):
    """call_planner：结构化决策解析 + 容错（fail-open）。"""

    def _state(self, user_msg: str = "我在工地受伤了") -> dict:
        return {
            "messages": [HumanMessage(content=user_msg)],
            "case_summary": "{}",
            "skip_planner": False,
        }

    def test_信息不完整_返回追问(self) -> None:
        stub = _StubPlanner(
            {
                "info_complete": False,
                "missing_fields": ["event_time"],
                "follow_up": "请问是什么时候受伤的？",
            }
        )
        with mock.patch.object(legal_agent, "planner_tool_llm", stub):
            result = legal_agent.call_planner(self._state())
        self.assertFalse(result["info_complete"])
        self.assertEqual(result["messages"][-1].content, "请问是什么时候受伤的？")

    def test_信息完整_放行(self) -> None:
        stub = _StubPlanner({"info_complete": True})
        with mock.patch.object(legal_agent, "planner_tool_llm", stub):
            result = legal_agent.call_planner(self._state())
        self.assertTrue(result["info_complete"])

    def test_args为JSON字符串_也能解析(self) -> None:
        # 部分模型把结构化参数作为 JSON 字符串返回（和流式路径一致的处理）
        stub = _StubPlanner('{"info_complete": false, "missing_fields": ["event_time"], "follow_up": "什么时间？"}')
        with mock.patch.object(legal_agent, "planner_tool_llm", stub):
            result = legal_agent.call_planner(self._state())
        self.assertFalse(result["info_complete"])
        self.assertEqual(result["messages"][-1].content, "什么时间？")

    def test_坏参数_fail_open放行(self) -> None:
        # 缺 info_complete 字段 → 解析失败 → 保守放行（不崩）
        stub = _StubPlanner({"follow_up": "只有一个字段"})
        with mock.patch.object(legal_agent, "planner_tool_llm", stub):
            result = legal_agent.call_planner(self._state())
        self.assertTrue(result["info_complete"])

    def test_模型未调用工具_fail_open放行(self) -> None:
        class _NoToolPlanner:
            def invoke(self, prompt):
                return AIMessage(content="我不调用工具")

        with mock.patch.object(legal_agent, "planner_tool_llm", _NoToolPlanner()):
            result = legal_agent.call_planner(self._state())
        self.assertTrue(result["info_complete"])

    def test_skip_planner_直接放行(self) -> None:
        state = self._state()
        state["skip_planner"] = True
        result = legal_agent.call_planner(state)
        self.assertTrue(result["info_complete"])


if __name__ == "__main__":
    unittest.main()
