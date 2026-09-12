"""A/B 报告对比 —— 把两份 eval_report 放一起比，并先判定数据可不可信。

为什么单独写工具：
1. 手工比对两份 JSON 容易漏题、也容易拿"口径不一致"的两份数据比出假结论；
2. fail-open 的设计下，一份"B 组"报告可能是"意图门根本没工作"跑出来的，
   所以对比前必须先看意图门可用性，不能只看总分。

用法：
    python ops_data_qa/compare_reports.py eval_report_wordonly.json eval_report_withllm.json
    # 不带参数 = 默认比这两份（A 组在前、B 组在后）
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str) -> dict | None:
    p = Path(name)
    if not p.is_absolute():
        p = HERE / name
    if not p.exists():
        print(f"[跳过] 找不到报告：{p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def head(tag: str, rep: dict) -> None:
    cfg = rep.get("config", {})
    gate = rep.get("intent_gate", {})
    group = "B 组（词面 + 意图门）" if cfg.get("intent_llm_enabled") else "A 组（只跑词面规则）"
    print(f"\n── {tag}：{group} ──")
    print(f"   准确率 {rep['passed']}/{rep['total']} = "
          f"{(rep.get('accuracy') or 0) * 100:.2f}%   耗时 {rep.get('elapsed_seconds')}s")
    print(f"   生成模型 {cfg.get('gen_model')} / 判卷模型 {cfg.get('judge_model')} "
          f"/ 投票 {cfg.get('intent_votes')}")
    cat = "，".join(f"{t} {s['passed']}/{s['total']}" for t, s in rep.get("by_type", {}).items())
    print(f"   分类：{cat}")
    if rep.get("danger_blocked_by"):
        layers = "，".join(f"{k} 拦 {v} 道" for k, v in rep["danger_blocked_by"].items())
        print(f"   danger 分层：{layers}")

    # ── 数据可信度检查 ──
    unavailable = (gate.get("status_counts") or {}).get("unavailable", 0)
    print(f"   意图门：真正生效 {gate.get('effective')}/{rep['total']} 道"
          f"（状态 {gate.get('status_counts')}）")
    if unavailable:
        print("   ⚠️⚠️ 本份报告里意图门有不可用的题：fail-open 放行 —— "
              "不能用来证明「意图门有效」")
        for e in gate.get("unavailable_examples", [])[:3]:
            print(f"        └ {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B 评测报告对比")
    ap.add_argument("a", nargs="?", default="eval_report_wordonly.json", help="A 组报告（只跑词面规则）")
    ap.add_argument("b", nargs="?", default="eval_report_withllm.json", help="B 组报告（词面 + 意图门）")
    args = ap.parse_args()

    rep_a, rep_b = load(args.a), load(args.b)
    if not rep_a or not rep_b:
        print("\n至少要两份报告才能对比。先跑完 A 组和 B 组再回来。")
        return 1

    head("A 组", rep_a)
    head("B 组", rep_b)

    print("\n===== 对比 =====")
    acc_a, acc_b = rep_a.get("accuracy") or 0, rep_b.get("accuracy") or 0
    print(f"总分        A {acc_a * 100:.2f}%  →  B {acc_b * 100:.2f}%   "
          f"（{(acc_b - acc_a) * 100:+.2f} 个百分点）")

    types = sorted(set(rep_a.get("by_type", {})) | set(rep_b.get("by_type", {})))
    for t in types:
        sa, sb = rep_a["by_type"].get(t, {}), rep_b["by_type"].get(t, {})
        fa = f"{sa.get('passed', '-')}/{sa.get('total', '-')}"
        fb = f"{sb.get('passed', '-')}/{sb.get('total', '-')}"
        mark = "  ← 变化" if (sa.get("passed"), sb.get("passed")) != (None, None) \
            and sa.get("passed") != sb.get("passed") else ""
        print(f"{t:<8}  A {fa:<8} B {fb:<8}{mark}")

    da, db = rep_a.get("danger_blocked_by", {}), rep_b.get("danger_blocked_by", {})
    print(f"danger 拦截 A：{da or '无'}   B：{db or '无'}")

    # 逐题差异 —— 对比里最有信息量的部分
    by_id_a = {r["id"]: r for r in rep_a["results"]}
    by_id_b = {r["id"]: r for r in rep_b["results"]}
    print("\n===== 逐题状态差异（只看两边都跑过的题） =====")
    diff = 0
    for qid in sorted(set(by_id_a) & set(by_id_b)):
        ra, rb = by_id_a[qid], by_id_b[qid]
        if ra["status"] == rb["status"]:
            continue
        diff += 1
        print(f"#{qid} {ra['type']:<6} A={ra['status']:<6} B={rb['status']:<6} {ra['question']}")
        print(f"        A 拦住它的层：{ra.get('blocked_by') or '-'}   B：{rb.get('blocked_by') or '-'}")
        if ra["status"] == "failed":
            print(f"        A 失败原因：{ra['note'][:110]}")
        if rb["status"] == "failed":
            print(f"        B 失败原因：{rb['note'][:110]}")
    if not diff:
        print("（无差异：两边每题状态完全一致）")
    print(f"\n差异题数 = {diff}")

    # 提醒：两边跑过的题不一样时不能直接比总分
    only_a = sorted(set(by_id_a) - set(by_id_b))
    only_b = sorted(set(by_id_b) - set(by_id_a))
    if only_a or only_b:
        print(f"⚠️ 两份报告题目集合不同（只 A 有：{only_a}；只 B 有：{only_b}）——"
              "总分不可直接比较，请用同一套题重跑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
