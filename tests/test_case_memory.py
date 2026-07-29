"""SQLite 会话持久化的最小回归测试。"""
import os
import tempfile
import unittest
from pathlib import Path

# 测试不调用远程 LLM；占位值只保证导入模块时不依赖真实密钥。
os.environ.setdefault("GLM_API_KEY", "test-key")

from memory import case_memory


class CaseMemorySQLiteTest(unittest.TestCase):
    """测试保存一轮对话和案情摘要后，能按 session_id 恢复。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_data_dir = case_memory.DATA_DIR
        self.original_db_path = case_memory.DB_PATH

        case_memory.DATA_DIR = Path(self.temp_dir.name) / "data"
        case_memory.DB_PATH = case_memory.DATA_DIR / "legal_agent.db"
        case_memory.init_db()

    def tearDown(self) -> None:
        case_memory.DATA_DIR = self.original_data_dir
        case_memory.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_save_and_restore_session(self) -> None:
        session_id = "test-session-001"
        expected_summary = {
            "case_type": "消费纠纷",
            "event_time": "昨天",
        }

        case_memory.save_exchange(
            session_id,
            "我花 188 元买到假鞋。",
            "请问是什么时候购买的？",
        )
        case_memory.save_case_summary(session_id, expected_summary)

        restored = case_memory.load_session_from_db(session_id)

        self.assertEqual(
            restored["history"],
            [{"human": "我花 188 元买到假鞋。", "ai": "请问是什么时候购买的？"}],
        )
        self.assertEqual(restored["case_summary"], expected_summary)

    def test_unknown_session_returns_empty_data(self) -> None:
        restored = case_memory.load_session_from_db("missing-session")

        self.assertEqual(restored, {"history": [], "case_summary": {}})

    def test_sessions_are_isolated_and_keep_message_order(self) -> None:
        case_memory.save_exchange("session-a", "A 的第一问", "A 的第一答")
        case_memory.save_exchange("session-b", "B 的第一问", "B 的第一答")
        case_memory.save_exchange("session-a", "A 的第二问", "A 的第二答")

        restored_a = case_memory.load_session_from_db("session-a")
        restored_b = case_memory.load_session_from_db("session-b")

        self.assertEqual(
            restored_a["history"],
            [
                {"human": "A 的第一问", "ai": "A 的第一答"},
                {"human": "A 的第二问", "ai": "A 的第二答"},
            ],
        )
        self.assertEqual(
            restored_b["history"],
            [{"human": "B 的第一问", "ai": "B 的第一答"}],
        )


if __name__ == "__main__":
    unittest.main()
