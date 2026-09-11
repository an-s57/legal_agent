"""造 100 万行测试数据 —— 索引实验用（不影响 ops_demo 原有评测数据）。

写入目标：ops_demo.sessions_big（结构复制自 sessions，但不在 Agent 白名单里，
text-to-SQL 碰不到它；原 sessions 表一行不动，50 题评测不受影响）。

为什么用 root：造数是写操作，query_user 只读账号干不了（这正是读写分离的意义）。
连接配置在 .env：OPS_DB_ADMIN_USER（默认 root）/ OPS_DB_ADMIN_PASSWORD。

数据分布故意不均匀：90% 劳动纠纷、9% 其他类型、1% 空值——
让"查冷门类型"的索引前后对比足够明显。

用法：
    python mysql_lab/make_big_table.py                # 默认 100 万行（约 1~3 分钟）
    python mysql_lab/make_big_table.py --rows 100000  # 先小规模试跑
    python mysql_lab/make_big_table.py --demo-slow    # 先逐条插 2000 行再批量插，对比两种速度
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OPS_DB_HOST, OPS_DB_NAME, OPS_DB_PORT

CASE_TYPES = ["劳动纠纷", "消费维权", "合同纠纷", "交通事故", "房屋租赁", "婚姻家庭", "刑事咨询"]

BASE_TIME = datetime(2025, 1, 1)


def make_rows(start_id: int, n: int) -> list:
    """生成一批 (session_key, case_type, created_at) 行，id 由自增主键负责。"""
    rows = []
    for i in range(start_id, start_id + n):
        r = random.random()
        if r < 0.90:
            case_type = "劳动纠纷"          # 大头：90%
        elif r < 0.99:
            case_type = random.choice(CASE_TYPES[1:])   # 其他 6 类共 9%
        else:
            case_type = None                # 1% 空值（纯知识问答）
        created = BASE_TIME + timedelta(seconds=random.randint(0, 365 * 24 * 3600))
        rows.append((f"s_big_{i:08d}", case_type, created.strftime("%Y-%m-%d %H:%M:%S")))
    return rows


def main():
    parser = argparse.ArgumentParser(description="造索引实验用的大表")
    parser.add_argument("--rows", type=int, default=1_000_000, help="总行数（默认 100 万）")
    parser.add_argument("--batch", type=int, default=5000, help="每批插入行数")
    parser.add_argument("--demo-slow", action="store_true",
                        help="先逐条插 2000 行计时，再批量插剩余（对比网络往返的代价）")
    args = parser.parse_args()

    user = os.getenv("OPS_DB_ADMIN_USER", "root")
    password = os.getenv("OPS_DB_ADMIN_PASSWORD", "")
    if not password:
        sys.exit("请先在 .env 配置 OPS_DB_ADMIN_PASSWORD（造数需要写权限账号，如 root）")

    conn = pymysql.connect(
        host=OPS_DB_HOST, port=OPS_DB_PORT, user=user, password=password,
        database=OPS_DB_NAME, charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS sessions_big")
            cur.execute("""
                CREATE TABLE sessions_big (
                  id INT PRIMARY KEY AUTO_INCREMENT,
                  session_key VARCHAR(64) NOT NULL,
                  case_type VARCHAR(30) DEFAULT NULL,
                  created_at DATETIME NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            conn.commit()
            print(f"[OK] 已建表 sessions_big，开始插入 {args.rows} 行...")

            done = 0

            if args.demo_slow:
                # 逐条插入：每条 INSERT 都是一趟完整的网络往返 + 一次磁盘事务
                t0 = time.perf_counter()
                for row in make_rows(0, 2000):
                    cur.execute(
                        "INSERT INTO sessions_big (session_key, case_type, created_at) "
                        "VALUES (%s, %s, %s)", row)
                conn.commit()
                slow_ms = (time.perf_counter() - t0) * 1000
                print(f"[对比] 逐条插 2000 行耗时 {slow_ms:.0f}ms（每趟一个网络往返）")
                done = 2000

            # 批量插入：一批打包一次发送
            t0 = time.perf_counter()
            while done < args.rows:
                n = min(args.batch, args.rows - done)
                cur.executemany(
                    "INSERT INTO sessions_big (session_key, case_type, created_at) "
                    "VALUES (%s, %s, %s)",
                    make_rows(done, n),
                )
                conn.commit()
                done += n
                if done % (args.rows // 10 or 1) < args.batch or done == args.rows:
                    print(f"  进度 {done}/{args.rows}")
            total = time.perf_counter() - t0
            rate = (args.rows - (2000 if args.demo_slow else 0)) / total if total > 0 else 0
            print(f"[OK] 批量插入完成，用时 {total:.1f}s（约 {rate:.0f} 行/秒）")

            print("[统计] 各类型行数：")
            cur.execute(
                "SELECT case_type, COUNT(*) FROM sessions_big "
                "GROUP BY case_type ORDER BY 2 DESC")
            for ct, cnt in cur.fetchall():
                print(f"  {str(ct):<8} {cnt}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
