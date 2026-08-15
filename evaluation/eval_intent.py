"""意图识别（PLANNER 节点）评测 — 测「追问决策」准确率。

复用 agent/legal_agent.py 里的 PLANNER_PROMPT / planner_tool_llm，逐条对
intent_eval_dataset.json 里的消息跑 planner 的真实决策链路（结构化输出，
框架直接返回 PlannerDecision 对象，无 JSON 解析环节；模型未调用工具时
走 fail-open 放行，记为 parse_fail）。

用法（在项目根目录）：
    python -X utf8 evaluation/eval_intent.py
    # 评测临时换模型（不改生产配置）：--model 覆盖底层 LLM，结果文件名带模型标签
    python -X utf8 evaluation/eval_intent.py --model qwen3:8b

产出：stdout 逐条判定 + 汇总；结果 JSON 写到 evaluation/results/。
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage

from llm_client import _make_llm

from agent.legal_agent import PLANNER_PROMPT, PlannerDecision, _format_planner_context, planner_tool_llm

DATASET = Path(__file__).parent / "intent_eval_dataset.json"


def build_planner(model: str | None):
    """按需构造 planner 决策链。

    --model 指定时：用独立 _make_llm(temperature=0, model=...) 绑定同一个
    PlannerDecision 工具，复用生产提示词与工具 schema，只换底层模型。
    --model 缺省：直接用生产 planner_tool_llm，保证评测即线上、零漂移。
    """
    if model:
        return _make_llm(temperature=0, model=model).bind_tools([PlannerDecision])
    return planner_tool_llm


def judge_one(message: str, planner) -> dict:
    """对单条消息跑 planner 决策，返回原始 JSON 或解析失败标记。"""
    recent_history = _format_planner_context([HumanMessage(content=message)])
    prompt = PLANNER_PROMPT.format(
        recent_history=recent_history,
        case_summary="{}",
        user_message=message,
    )
    response = planner.invoke(prompt)
    if response.tool_calls:
        decision = PlannerDecision(**response.tool_calls[0]["args"])
        return {
            "raw": decision.model_dump_json(),
            "json": {
                "info_complete": decision.info_complete,
                "missing_fields": decision.missing_fields or [],
                "follow_up": decision.follow_up or "",
            },
        }
    # 模型未调用工具：等价于旧的"解析失败"，走 fail-open 放行
    return {"raw": response.content or "", "json": None}


def main():
    parser = argparse.ArgumentParser(description="PLANNER 意图识别评测")
    parser.add_argument(
        "--model", default=None,
        help="评测用的底层模型名（默认用生产 planner_tool_llm；指定则独立构造 planner，不改生产配置）",
    )
    args = parser.parse_args()
    planner = build_planner(args.model)
    model_tag = (args.model or "default").replace("/", "_").replace(":", "_")

    data = json.load(open(DATASET, encoding="utf-8"))
    items = data["items"]

    results = []
    parse_fail = 0
    print("=" * 78)
    print(f"评测模型：{model_tag}")
    print(f"{'id':<6} {'意图':<16} {'gold':<6} {'pred':<6} {'判定':<4} 追问内容/说明")
    print("-" * 78)

    for it in items:
        gold_ask = it["should_ask"]
        out = judge_one(it["message"], planner)
        parsed = out["json"]

        if parsed is None:
            parse_fail += 1
            # 解析失败 → 生产里 call_planner 会 fail-open 放行
            pred_ask = False
            ok = (pred_ask == gold_ask)
            note = "JSON解析失败(fail-open→放行)"
        else:
            info_complete = parsed.get("info_complete", True)
            pred_ask = not info_complete
            ok = (pred_ask == gold_ask)
            note = ""
            if pred_ask:
                note = parsed.get("follow_up", "")[:40]

        mark = "✓" if ok else "✗"
        g = "追问" if gold_ask else "放行"
        p = "追问" if pred_ask else "放行"
        print(f"{it['id']:<6} {it['intent']:<16} {g:<6} {p:<6} {mark:<4} {note}")

        results.append({
            "id": it["id"],
            "message": it["message"],
            "intent": it["intent"],
            "gold_should_ask": gold_ask,
            "pred_should_ask": pred_ask,
            "ok": ok,
            "parse_fail": parsed is None,
            "pred_missing_fields": (parsed or {}).get("missing_fields", []),
            "pred_follow_up": (parsed or {}).get("follow_up", ""),
        })

    # ── 汇总 ──
    n = len(items)
    correct = sum(1 for r in results if r["ok"])
    # 误追问：gold 放行 但 pred 追问
    false_ask = [r for r in results if (not r["gold_should_ask"]) and r["pred_should_ask"]]
    # 漏追问：gold 追问 但 pred 放行
    miss_ask = [r for r in results if r["gold_should_ask"] and (not r["pred_should_ask"])]

    print("=" * 78)
    print(f"总数 {n}  |  JSON 解析失败 {parse_fail} ({parse_fail/n*100:.0f}%)")
    print(f"追问决策准确率：{correct}/{n} = {correct/n*100:.1f}%")
    print(f"误追问（该放行却追问）：{len(false_ask)} 条")
    for r in false_ask:
        print(f"    ✗ {r['id']} [{r['intent']}] \"{r['message'][:30]}\" → 追问\"{(r['pred_follow_up'] or '')[:30]}\"")
    print(f"漏追问（该追问却放行）：{len(miss_ask)} 条")
    for r in miss_ask:
        print(f"    ✗ {r['id']} [{r['intent']}] \"{r['message'][:30]}\" (解析失败={r['parse_fail']})")

    # 分意图统计误追问率
    print("-" * 78)
    print("分意图误追问率（该放行的意图里被误追问的比例）：")
    for intent in ["knowledge_query", "chitchat", "non_legal", "complaint"]:
        grp = [r for r in results if r["intent"] == intent]
        if not grp:
            continue
        fa = sum(1 for r in grp if (not r["gold_should_ask"]) and r["pred_should_ask"])
        print(f"    {intent:<16} {fa}/{len(grp)} 被误追问")

    # 写结果（文件名带模型标签，不同模型的评测结果可对比、可存档）
    out_path = ROOT / "evaluation" / "results" / f"intent_eval_{model_tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": data["name"],
        "total": n,
        "parse_fail": parse_fail,
        "parse_fail_rate": round(parse_fail / n, 4),
        "accuracy": round(correct / n, 4),
        "false_ask_count": len(false_ask),
        "miss_ask_count": len(miss_ask),
        "details": results,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{out_path}")


if __name__ == "__main__":
    main()