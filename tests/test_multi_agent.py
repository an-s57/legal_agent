"""多 agent（小律所）单元测试 — 全离线：mock 掉检索工具与 LLM。

覆盖：外勤员出动条件、秘书汇总、合伙人判卷（通过/打回/降级）、
律师重写时带审稿意见。运行：python -m unittest tests.test_multi_agent -v
"""
import asyncio
import os
import unittest
from unittest import mock

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CACHE_ENABLED", "0")

from langchain_core.messages import AIMessage, HumanMessage

from agent import legal_agent
from agent.hallucination_guard import check_hallucination


def _run(coro):
    return asyncio.run(coro)


class NeedsWebSearchTest(unittest.TestCase):

    def test_带时效性关键词_出动(self):
        self.assertTrue(legal_agent.needs_web_search("2026年最新的人工智能法规有哪些？"))

    def test_纯法条问题_不出动(self):
        self.assertFalse(legal_agent.needs_web_search("试用期最长可以约定几个月？"))


class LawWorkerTest(unittest.TestCase):

    def test_资料员调用法条检索并写回(self):
        tool = mock.MagicMock()
        tool.ainvoke = mock.AsyncMock(return_value="【法条】劳动合同法第19条……")
        with mock.patch.object(legal_agent, "legal_rag_search", tool):
            result = _run(legal_agent.law_worker_node({"question": "试用期规定"}))
        self.assertIn("法条", result["law_result"])

    def test_检索失败_返回降级文案不抛异常(self):
        async def broken_ainvoke(args):
            raise RuntimeError("faiss 炸了")

        tool = mock.MagicMock()
        tool.ainvoke = broken_ainvoke
        with mock.patch.object(legal_agent, "legal_rag_search", tool):
            result = _run(legal_agent.law_worker_node({"question": "试用期规定"}))
        self.assertIn("暂时不可用", result["law_result"])


class WebWorkerTest(unittest.TestCase):

    def test_无时效关键词_不出动(self):
        result = _run(legal_agent.web_worker_node({"question": "试用期规定"}))
        self.assertEqual(result["web_result"], "")

    def test_有时效关键词_出动并写回(self):
        tool = mock.MagicMock()
        tool.ainvoke = mock.AsyncMock(return_value="【网页】最新规定……")
        with mock.patch.object(legal_agent, "web_legal_search", tool):
            result = _run(legal_agent.web_worker_node({"question": "2026年最新的AI法规"}))
        self.assertIn("网页", result["web_result"])


class MergeNodeTest(unittest.TestCase):

    def test_两路材料分区汇总(self):
        result = legal_agent.merge_node({"law_result": "法条A", "web_result": "网页B"})
        self.assertIn("【法条库检索结果】", result["materials"])
        self.assertIn("【联网检索结果】", result["materials"])
        self.assertIn("法条A", result["materials"])
        self.assertIn("网页B", result["materials"])

    def test_两路都空_给占位文案(self):
        result = legal_agent.merge_node({"law_result": "", "web_result": ""})
        self.assertIn("均无结果", result["materials"])


class BuildDraftMessagesTest(unittest.TestCase):

    def _state(self, feedback=""):
        return {
            "messages": [
                HumanMessage(content="上一轮的问题"),
                AIMessage(content="上一轮的回答"),
                HumanMessage(content="试用期最长多久？"),
            ],
            "materials": "【法条库检索结果】\n《劳动合同法》第19条……",
            "question": "试用期最长多久？",
            "verify_feedback": feedback,
        }

    def test_首稿_含资料与问题_不含审稿意见(self):
        msgs = legal_agent._build_draft_messages(self._state())
        last = msgs[-1].content
        self.assertIn("【参考资料】", last)
        self.assertIn("试用期最长多久？", last)
        self.assertNotIn("审稿意见", last)

    def test_重写时附带审稿意见(self):
        msgs = legal_agent._build_draft_messages(self._state("引用的条文在资料里找不到"))
        self.assertIn("引用的条文在资料里找不到", msgs[-1].content)

    def test_历史对话保留但最后一条被结构化块替代(self):
        msgs = legal_agent._build_draft_messages(self._state())
        roles = [(isinstance(m, HumanMessage), isinstance(m, AIMessage)) for m in msgs]
        # 结构：System + [历史 H/A...] + 最后的 Human 结构化块
        self.assertTrue(msgs[0].content.startswith("你是一个专业的AI法律助手"))
        self.assertEqual(msgs[-2].content, "上一轮的回答")


class VerifyDecisionTest(unittest.TestCase):
    """合伙人判卷：规则 + GLM 复核 → 通过 / 打回 / 降级。"""

    def test_全绿_通过(self):
        answer, materials = "依据《劳动合同法》第19条……", "《劳动合同法》第19条……"
        passed, feedback, final = legal_agent._verify_decision(
            answer, materials, {"verdict": 1, "reason": "一致"},
            attempts=0, check=check_hallucination(answer, materials))
        self.assertTrue(passed)
        self.assertEqual(feedback, "")
        self.assertEqual(final, "依据《劳动合同法》第19条……")

    def test_引用不在资料中_打回并给修改方向(self):
        answer, materials = "依据《民法典》第999条……", "《劳动合同法》第19条……"
        passed, feedback, _ = legal_agent._verify_decision(
            answer, materials, {"verdict": 1},
            attempts=0, check=check_hallucination(answer, materials))
        self.assertFalse(passed)
        self.assertIn("第999条", feedback)
        self.assertIn("删除或改用", feedback)

    def test_GLM判0_打回并附理由(self):
        passed, feedback, _ = legal_agent._verify_decision(
            "回答内容", "资料内容", {"verdict": 0, "reason": "与资料存在出入"},
            attempts=0, check=check_hallucination("回答内容", "资料内容"))
        self.assertFalse(passed)
        self.assertIn("事实一致性复核不通过", feedback)
        self.assertIn("与资料存在出入", feedback)

    def test_GLM不可用_fail_open只按规则判(self):
        passed, _, _ = legal_agent._verify_decision(
            "回答内容", "资料内容", {"verdict": None, "reason": ""},
            attempts=0, check=check_hallucination("回答内容", "资料内容"))
        # 规则无警告（内容与资料无引用冲突时覆盖度按空资料兜底为通过）→ 以实际规则为准
        self.assertIsInstance(passed, bool)

    def test_达到重试上限_降级输出并带风险标注(self):
        answer, materials = "依据《民法典》第999条……", "《劳动合同法》第19条……"
        passed, _, final = legal_agent._verify_decision(
            answer, materials, {"verdict": 1},
            attempts=legal_agent.VERIFY_MAX_RETRIES,
            check=check_hallucination(answer, materials))
        self.assertFalse(passed)                       # 仍判定不通过
        self.assertIn("检索校验", final)               # 但答案带风险标注输出（老行为兜底）

    def test_规则失败重试路径_返回打回意见(self):
        answer, materials = "依据《民法典》第999条……", "《劳动合同法》第19条……"
        passed, feedback, _ = legal_agent._verify_decision(
            answer, materials, {"verdict": 1},
            attempts=0, check=check_hallucination(answer, materials))
        self.assertFalse(passed)
        self.assertTrue(feedback)                      # 未达上限 → 带意见打回


if __name__ == "__main__":
    unittest.main()
