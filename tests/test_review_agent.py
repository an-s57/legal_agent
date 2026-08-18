"""合规复核 Agent 的离线单测 — 用 stub judge 替换 GLM，不调用任何真实 API。

运行：python -m unittest tests.test_review_agent -v
"""
import unittest
from unittest.mock import MagicMock

from agent.review_agent import (
    format_review_warning,
    review_answer,
)


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class TestReviewAgent(unittest.TestCase):
    def _judge_returning(self, content: str) -> MagicMock:
        judge = MagicMock()
        judge.invoke.return_value = _FakeResp(content)
        return judge

    def test_verdict_0_appends_warning(self):
        judge = self._judge_returning('{"verdict": 0, "reason": "编造了不存在的法条"}')
        result = review_answer("用户问题", "助手回答", "证据：第五十五条…", judge=judge)
        self.assertEqual(result["verdict"], 0)
        self.assertIn("编造", result["reason"])
        warning = format_review_warning(result)
        self.assertIn("合规复核", warning)

    def test_verdict_1_no_warning(self):
        judge = self._judge_returning('{"verdict": 1, "reason": "与法条一致"}')
        result = review_answer("用户问题", "助手回答", "证据：第五十五条…", judge=judge)
        self.assertEqual(result["verdict"], 1)
        self.assertEqual(format_review_warning(result), "")

    def test_verdict_0_with_empty_reason_still_warns(self):
        judge = self._judge_returning('{"verdict": 0, "reason": ""}')
        result = review_answer("问题", "回答", "证据", judge=judge)
        self.assertEqual(result["verdict"], 0)
        self.assertIn("合规复核", format_review_warning(result))

    def test_no_evidence_skips_review(self):
        # 无检索证据时不应调用 judge（fail-open 但不白花钱）
        judge = MagicMock()
        result = review_answer("问题", "回答", "", judge=judge)
        self.assertEqual(result["verdict"], None)
        judge.invoke.assert_not_called()

    def test_judge_failure_fail_open(self):
        judge = MagicMock()
        judge.invoke.side_effect = Exception("API down")
        result = review_answer("问题", "回答", "证据", judge=judge)
        self.assertEqual(result["verdict"], None)
        self.assertEqual(format_review_warning(result), "")

    def test_judge_returns_garbage_fail_open(self):
        judge = self._judge_returning("抱歉，我无法判卷")
        result = review_answer("问题", "回答", "证据", judge=judge)
        self.assertEqual(result["verdict"], None)
        self.assertEqual(format_review_warning(result), "")

    def test_markdown_codeblock_json_parsed(self):
        judge = self._judge_returning('```json\n{"verdict": 0, "reason": "结论相反"}\n```')
        result = review_answer("问题", "回答", "证据", judge=judge)
        self.assertEqual(result["verdict"], 0)
        self.assertIn("结论相反", result["reason"])


if __name__ == "__main__":
    unittest.main()
