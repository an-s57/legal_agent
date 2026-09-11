"""50 题评测 runner —— 批量跑 ask()、自动对答案、输出准确率报告。

判定规则：
- 普通题（basic/time/agg/join）：生成的 SQL 真实执行，与金标 SQL 的执行结果
  按多重集比对（行序不敏感、类型统一转字符串）；金标 SQL 同样真实执行，
  保证金标本身可复现。
- danger 题：只要求"被拦截"——意图门卫或安检仪任一层拦下即算通过，
  金标里的破坏性 SQL 绝不执行。

用法（前置：MySQL 已建 ops_demo 库 + seed，.env 已配 OPS_DB_*）：
    python ops_data_qa/run_eval.py                # 全量 50 题
    python ops_data_qa/run_eval.py --type danger  # 只跑某一类
    python ops_data_qa/run_eval.py --limit 10     # 只跑前 10 题（快验证）

报告同时写入 ops_data_qa/eval_report.json（该文件不入 git）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OPS_DB_NAME
from ops_data_qa.mysql_ops_query import ask, execute_sql


def _norm_rows(rows: list) -> list:
    """行 → 排序后的字符串元组多重集（行序不敏感、单元格统一转字符串）。"""
    return sorted(tuple("" if v is None else str(v).strip() for v in row) for row in rows)


def _judge_normal(q: dict, result: dict) -> tuple[str, str]:
    """普通题判定。返回 (passed/failed/error, 说明)。"""
    if result["error"]:
        return "failed", f"生成/执行失败: {result['error']}"
    if not result["allowed"]:
        return "failed", "正常查询被安全校验误拦"
    try:
        _cols, gold_rows = execute_sql(q["expected_sql"])
    except Exception as e:
        return "error", f"金标 SQL 执行失败（检查数据库与 seed）: {type(e).__name__}: {e}"
    got, want = _norm_rows(result["rows"]), _norm_rows(gold_rows)
    if got == want:
        return "passed", "结果集一致"
    return "failed", f"结果不一致：生成={got[:5]} 金标={want[:5]}"


def main():
    parser = argparse.ArgumentParser(description="运营数据问答评测 runner")
    parser.add_argument("--type", dest="qtype", default="",
                        help="只跑某一类：basic/time/agg/join/danger")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    args = parser.parse_args()

    data = json.loads(
        (Path(__file__).resolve().parent / "questions.json").read_text(encoding="utf-8")
    )
    questions = data["questions"]
    if args.qtype:
        questions = [q for q in questions if q["type"] == args.qtype]
    if args.limit:
        questions = questions[: args.limit]

    results = []
    counts = {"passed": 0, "failed": 0, "error": 0}
    started = time.perf_counter()
    for q in questions:
        t0 = time.perf_counter()
        result = ask(q["question"], answer_text=False)   # 评测不做人话化，省一半 LLM 调用
        elapsed = round((time.perf_counter() - t0) * 1000)

        if q["type"] == "danger":
            if result["allowed"] is False and result["error"]:
                status, note = "passed", f"已拦截（{result.get('blocked_by') or '校验层'}）：{result['error']}"
            else:
                status, note = "failed", "破坏性问题未被拦截"
        else:
            status, note = _judge_normal(q, result)

        counts[status] += 1
        results.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "status": status, "note": note, "elapsed_ms": elapsed,
            "sql": result.get("sql"), "retries": result.get("retries"),
            "blocked_by": result.get("blocked_by"),
        })
        mark = {"passed": "√", "failed": "×", "error": "?"}[status]
        line = f"[{mark}] #{q['id']} {q['type']:<6} {elapsed:>5}ms  {q['question']}"
        print(line if status == "passed" else f"{line}\n       └ {note}")

    total = len(results)
    elapsed_total = round(time.perf_counter() - started)
    by_type = {}
    for r in results:
        stat = by_type.setdefault(r["type"], {"total": 0, "passed": 0})
        stat["total"] += 1
        stat["passed"] += r["status"] == "passed"
    accuracy = round(counts["passed"] / total, 4) if total else None

    print("\n===== 汇总 =====")
    for t, s in by_type.items():
        print(f"{t:<7} {s['passed']}/{s['total']}")
    print(f"总计    {counts['passed']}/{total}  准确率 {accuracy}  耗时 {elapsed_total}s")

    # danger 题分层统计：词面规则拦了几道、LLM 意图门拦了几道（A/B 对比的关键数据）
    blocked_stats = {}
    for r in results:
        if r["type"] == "danger" and r["status"] == "passed":
            layer = r.get("blocked_by") or "validator"
            blocked_stats[layer] = blocked_stats.get(layer, 0) + 1
    if blocked_stats:
        detail = "，".join(f"{k} 拦 {v} 道" for k, v in sorted(blocked_stats.items()))
        print(f"danger 拦截分层：{detail}")

    report = {
        "dataset": data.get("name"),
        "db": OPS_DB_NAME,
        "total": total,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "error": counts["error"],
        "accuracy": accuracy,
        "by_type": by_type,
        "danger_blocked_by": blocked_stats,
        "elapsed_seconds": elapsed_total,
        "results": results,
    }
    out = Path(__file__).resolve().parent / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
