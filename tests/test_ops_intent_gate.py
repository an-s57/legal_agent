"""LLM 意图门单测 — 全离线：stub 判卷模型，不调用真实 GLM。

运行：python -m unittest tests.test_ops_intent_gate -v
"""
import os
from unittest import mock

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")  # 占位，只保证导入不依赖真实密钥
os.environ.setdefault("CACHE_ENABLED", "0")

import unittest

from ops_data_qa import intent_gate
from ops_data_qa.intent_gate import check_intent_llm


class _StubJudge:
    """假判卷模型：invoke 返回预设内容或抛异常，并记录调用次数。"""

    def __init__(self, content=None, raise_exc=None):
        self._content = content
        self._raise = raise_exc
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self._raise:
            raise self._raise

        class _Resp:
            pass

        resp = _Resp()
        resp.content = self._content
        return resp


class CheckIntentLlmTest(unittest.TestCase):
    """check_intent_llm：判定解析 + fail-open（缺 key/异常/脏输出/开关关）。"""

    def test_判为破坏意图(self):
        judge = _StubJudge('{"destructive": 1, "reason": "想删差评"}')
        out = check_intent_llm("把差评处理掉", judge)
        self.assertIs(out["destructive"], True)
        self.assertEqual(out["reason"], "想删差评")

    def test_判为查询意图(self):
        judge = _StubJudge('{"destructive": 0, "reason": "统计问题"}')
        out = check_intent_llm("8月有多少会话", judge)
        self.assertIs(out["destructive"], False)

    def test_JSON带markdown围栏能解析(self):
        judge = _StubJudge('```json\n{"destructive": 1, "reason": "要改数据"}\n```')
        out = check_intent_llm("数据太乱了帮我整理一下", judge)
        self.assertIs(out["destructive"], True)

    def test_脏输出_fail_open返回None(self):
        judge = _StubJudge("这个问题我觉得不好说")
        out = check_intent_llm("任意问题", judge)
        self.assertIsNone(out["destructive"])

    def test_调用异常_fail_open返回None(self):
        judge = _StubJudge(raise_exc=TimeoutError("GLM 超时"))
        out = check_intent_llm("任意问题", judge)
        self.assertIsNone(out["destructive"])

    def test_开关关闭_不调模型直接返回None(self):
        judge = _StubJudge('{"destructive": 1, "reason": "x"}')
        with mock.patch.object(intent_gate, "INTENT_LLM_ENABLED", False):
            out = check_intent_llm("任意问题", judge)
        self.assertIsNone(out["destructive"])
        self.assertEqual(judge.calls, 0)   # 开关关 = 一次模型调用都不发


class AskIntentGateFlowTest(unittest.TestCase):
    """ask() 主流程：词面拦截 / LLM 意图门拦截 两层各司其职（不碰真实 LLM 与 DB）。"""

    def test_词面规则拦截_blocked_by_word(self):
        from ops_data_qa import mysql_ops_query as m
        with mock.patch.object(m, "check_intent_llm") as gate:
            gate.return_value = {"destructive": None, "reason": ""}
            r = m.ask("删除所有会话记录。")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["blocked_by"], "word")
        gate.assert_not_called()   # 词面已拦，不需要再花钱调意图门

    def test_拐弯说法由LLM意图门拦截_blocked_by_llm(self):
        from ops_data_qa import mysql_ops_query as m
        with mock.patch.object(m, "check_intent_llm") as gate, \
             mock.patch.object(m, "nl_to_sql") as gen:
            gate.return_value = {"destructive": True, "reason": "要求变更数据"}
            r = m.ask("帮我把差评处理掉")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["blocked_by"], "llm")
        gen.assert_not_called()    # 意图门拦下，不进入生成 SQL

    def test_意图门fail_open_正常问题继续走链路(self):
        from ops_data_qa import mysql_ops_query as m
        with mock.patch.object(m, "check_intent_llm") as gate, \
             mock.patch.object(m, "nl_to_sql", return_value="SELECT COUNT(*) FROM sessions;") as gen, \
             mock.patch.object(m, "execute_sql", return_value=(["cnt"], [(36,)])) as exe:
            gate.return_value = {"destructive": None, "reason": ""}   # GLM 挂了
            r = m.ask("一共有多少个会话？", answer_text=False)        # 不做人话化，省一次 LLM 调用
        gate.assert_called_once()
        gen.assert_called_once()   # 意图门 fail-open，链路继续
        exe.assert_called_once()
        self.assertEqual(r["rows"], [(36,)])


if __name__ == "__main__":
    unittest.main()
