"""审计日志单测 — 临时目录离线跑，不动真实 data/ 目录。

运行：python -m unittest tests.test_ops_audit -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops_data_qa import audit


class AuditTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ops_audit.jsonl"
        patcher = mock.patch.object(audit, "AUDIT_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_record追加一行_令牌只存指纹(self):
        audit.record("secret-token-123", question="8月有多少会话", allowed=True, rows=14)
        rows = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        raw = json.dumps(rows, ensure_ascii=False)
        self.assertNotIn("secret-token-123", raw)      # 原始令牌绝不落盘
        self.assertEqual(len(rows[0]["token"]), 8)     # 只存 8 位指纹
        self.assertIn("ts", rows[0])                   # 带时间戳
        self.assertEqual(rows[0]["rows"], 14)

    def test_recent倒序读取_损坏行跳过(self):
        self.path.write_text(
            '{"半截行被进程杀死\n'
            + json.dumps({"ts": "t1", "question": "第一题"}, ensure_ascii=False) + "\n"
            + json.dumps({"ts": "t2", "question": "第二题"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        entries = audit.recent(10)
        self.assertEqual([e["question"] for e in entries], ["第二题", "第一题"])
        self.assertEqual(audit.recent(1)[0]["question"], "第二题")   # limit 生效

    def test_文件不存在返回空列表(self):
        self.assertEqual(audit.recent(5), [])

    def test_写失败不影响调用方(self):
        # 指向一个必然写不进去的路径：record 必须吞掉异常（fail-open），绝不向上抛
        with mock.patch.object(audit, "AUDIT_PATH", Path(self.tmp.name) / "不存在目录" / "x.jsonl"):
            with mock.patch.object(Path, "mkdir", side_effect=OSError("disk full")):
                audit.record("t", question="x")        # 不抛异常即通过


if __name__ == "__main__":
    unittest.main()
