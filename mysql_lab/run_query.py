"""单条 SQL 执行器 —— 绕开 Windows 终端打不了中文的问题（SQL 走参数/文件，UTF-8 直达）。

用法：
    python mysql_lab/run_query.py "SELECT COUNT(*) FROM sessions_big WHERE case_type = '刑事咨询'"
    python mysql_lab/run_query.py "EXPLAIN SELECT * FROM sessions_big WHERE case_type = '刑事咨询'"
    python mysql_lab/run_query.py "CREATE INDEX idx_case_type ON sessions_big(case_type)" --root
                                                                                              ↑ 建索引是写操作，要管理账号
"""
import argparse
import sys
import time
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    OPS_DB_ADMIN_PASSWORD,
    OPS_DB_ADMIN_USER,
    OPS_DB_HOST,
    OPS_DB_NAME,
    OPS_DB_PASSWORD,
    OPS_DB_PORT,
    OPS_DB_USER,
)


def connect(as_root: bool = False):
    if as_root:
        if not OPS_DB_ADMIN_PASSWORD:
            sys.exit("先在 .env 配置 OPS_DB_ADMIN_PASSWORD（建索引等写操作需要管理账号）")
        return pymysql.connect(
            host=OPS_DB_HOST, port=OPS_DB_PORT, user=OPS_DB_ADMIN_USER,
            password=OPS_DB_ADMIN_PASSWORD, database=OPS_DB_NAME, charset="utf8mb4",
        )
    return pymysql.connect(
        host=OPS_DB_HOST, port=OPS_DB_PORT, user=OPS_DB_USER,
        password=OPS_DB_PASSWORD, database=OPS_DB_NAME, charset="utf8mb4",
    )


def main():
    parser = argparse.ArgumentParser(description="单条 SQL 执行器（带耗时与对齐输出）")
    parser.add_argument("sql", help="要执行的 SQL，用引号包起来")
    parser.add_argument("--root", action="store_true", help="用管理账号执行（CREATE INDEX 等写操作）")
    args = parser.parse_args()

    conn = connect(as_root=args.root)
    try:
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(args.sql)
            elapsed = time.perf_counter() - t0
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        conn.commit()
    finally:
        conn.close()

    print(f"耗时 {elapsed * 1000:.1f} ms · 返回 {len(rows)} 行\n")
    if not rows:
        return
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)))
    print("-+-".join("-" * w for w in widths))
    for r in rows[:30]:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))
    if len(rows) > 30:
        print(f"...（共 {len(rows)} 行）")


if __name__ == "__main__":
    main()
