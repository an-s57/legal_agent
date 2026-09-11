"""运营问答 SQL 安全校验单测 — 纯函数离线跑（不连 MySQL、不调 LLM）。

运行：python -m unittest tests.test_ops_sql_validator -v
"""
import unittest

from ops_data_qa.validator import check_intent, enforce_limit, validate_sql

ALLOWED = {"sessions", "messages", "answer_ratings"}


class ValidateSqlTest(unittest.TestCase):
    """validate_sql：语法树级校验 — 含针对旧正则版漏洞的回归用例。"""

    def test_普通查询通过(self):
        ok, why = validate_sql("SELECT COUNT(*) FROM sessions;", ALLOWED, "ops_demo")
        self.assertTrue(ok, why)

    def test_多表JOIN通过(self):
        sql = ("SELECT s.session_key FROM sessions s "
               "JOIN answer_ratings r ON s.id=r.session_id WHERE r.rating=1")
        ok, why = validate_sql(sql, ALLOWED, "ops_demo")
        self.assertTrue(ok, why)

    def test_逗号多表越权_旧正则的漏报(self):
        # 旧正则只看 FROM 后第一个词，逗号后捎带的越权表看不见 → sqlglot 能看到
        ok, why = validate_sql("SELECT * FROM sessions, mysql.user", ALLOWED, "ops_demo")
        self.assertFalse(ok)
        self.assertIn("白名单外", why)   # 带库名前缀时报"白名单外的库"，纯表名时报"白名单外的表"

    def test_反引号表名能识别(self):
        ok, why = validate_sql("SELECT * FROM `sessions`", ALLOWED, "ops_demo")
        self.assertTrue(ok, why)

    def test_UNION查其他库被拦(self):
        ok, why = validate_sql(
            "SELECT session_key FROM sessions UNION SELECT user FROM mysql.user",
            ALLOWED, "ops_demo")
        self.assertFalse(ok)

    def test_库名前缀越权被拦(self):
        ok, why = validate_sql("SELECT * FROM other_db.sessions", ALLOWED, "ops_demo")
        self.assertFalse(ok)
        self.assertIn("白名单外的库", why)

    def test_SELECT子句里的越权子查询被拦(self):
        ok, why = validate_sql(
            "SELECT (SELECT COUNT(*) FROM mysql.user) AS x FROM sessions",
            ALLOWED, "ops_demo")
        self.assertFalse(ok)

    def test_写操作一票否决(self):
        for sql in ("DELETE FROM sessions",
                    "UPDATE sessions SET case_type='x'",
                    "DROP TABLE sessions",
                    "TRUNCATE TABLE messages",
                    "CREATE TABLE t(id INT)"):
            ok, why = validate_sql(sql, ALLOWED, "ops_demo")
            self.assertFalse(ok, sql)

    def test_多语句被拦(self):
        ok, why = validate_sql("SELECT 1; SELECT 2", ALLOWED, "ops_demo")
        self.assertFalse(ok)
        self.assertIn("多条", why)

    def test_字符串里的分号不误杀(self):
        ok, why = validate_sql("SELECT ';' AS x FROM sessions", ALLOWED, "ops_demo")
        self.assertTrue(ok, why)

    def test_CTE别名放行_真表仍需白名单(self):
        sql = ("WITH t AS (SELECT case_type FROM sessions) "
               "SELECT case_type, COUNT(*) FROM t GROUP BY case_type")
        ok, why = validate_sql(sql, ALLOWED, "ops_demo")
        self.assertTrue(ok, why)

    def test_CTE定义体里的越权表仍被拦(self):
        ok, why = validate_sql(
            "WITH t AS (SELECT * FROM mysql.user) SELECT * FROM t",
            ALLOWED, "ops_demo")
        self.assertFalse(ok)

    def test_危险函数被拦(self):
        for sql in ("SELECT SLEEP(5) FROM sessions",
                    "SELECT LOAD_FILE('/etc/passwd') FROM sessions"):
            ok, why = validate_sql(sql, ALLOWED, "ops_demo")
            self.assertFalse(ok, sql)

    def test_非SQL文本解析失败被拦(self):
        ok, why = validate_sql("帮我查一下有多少会话", ALLOWED, "ops_demo")
        self.assertFalse(ok)

    def test_无表的语句被拦(self):
        ok, why = validate_sql("SELECT 1", ALLOWED, "ops_demo")
        self.assertFalse(ok)
        self.assertIn("未识别到查询的表", why)


class CheckIntentTest(unittest.TestCase):
    """check_intent：意图门卫 — 强破坏词直拦、弱写词看是否有分析词。"""

    def test_强破坏词直接拦(self):
        blocked, why = check_intent("删除所有会话记录。")
        self.assertTrue(blocked)
        blocked, _ = check_intent("清空消息表。")
        self.assertTrue(blocked)

    def test_弱写词无分析词_拦(self):
        blocked, _ = check_intent("把所有评价更新成点赞")
        self.assertTrue(blocked)

    def test_弱写词带分析词_放行(self):
        blocked, _ = check_intent("最近更新的会话有哪些")
        self.assertFalse(blocked)

    def test_纯统计问题放行(self):
        blocked, _ = check_intent("8月有多少个会话？")
        self.assertFalse(blocked)


class EnforceLimitTest(unittest.TestCase):
    """enforce_limit：没写 LIMIT 自动补上限。"""

    def test_没LIMIT自动补(self):
        out = enforce_limit("SELECT * FROM sessions;", 200)
        self.assertTrue(out.upper().endswith("LIMIT 200;"))

    def test_已有LIMIT不动(self):
        out = enforce_limit("SELECT * FROM sessions LIMIT 5;", 200)
        self.assertEqual(out, "SELECT * FROM sessions LIMIT 5;")


if __name__ == "__main__":
    unittest.main()
