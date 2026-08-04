"""事实性评测 · 第二步：LLM-as-judge 判卷 —— 答案是否与法条依据在事实上一致。

读第一步 eval_factual_collect.py 的产物，逐题把
{题目, 标准法条依据(article + anchor), 答案主体} 喂给判卷模型，
输出 verdict(0/1) + 一句话理由，按 development / validation 分开统计。

判卷依据只用 gold_evidence 的 article（法条名）和 anchor（法条原文锚句）；
不看 pdf_page——那是检索定位用的，对判卷无意义。

关于"异构 judge"：当前环境只有 DeepSeek key，judge 默认复用项目主模型
deepseek-v4-flash（同模型，但用完全独立的"判卷模式" prompt，降低自评偏置）。
拿到第二家 key 后，用 --judge-model / --judge-api-key 切换实现真异构。

用法：
    python -X utf8 evaluation/eval_factual_judge.py --answers evaluation/results/factual_answers_all_XXXX.json
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import SystemMessage, HumanMessage

from llm_client import llm

RESULTS_DIR = Path(__file__).parent / "results"

JUDGE_SYSTEM = (
    "你是法律答题判卷员。只依据给定的标准法条依据判断助手回答的事实正确性，"
    "不做额外引申，不要求措辞一致。每次只输出一个 JSON。"
)

JUDGE_PROMPT = """题目：{question}

题目对应的正确答案依据（法条原文摘录）：
- 法条：{article}
- 原文片段：{anchor}

助手回答：
{answer}

判卷规则：
1. 核心：助手回答的法律结论是否与"原文片段"规定的规则在事实上一致（允许措辞不同，要求事实要点一致）。
2. 结论与法条相反或明显错误 → 判 0。
3. 漏掉了该题应回答的关键规则 → 判 0。
4. 未引用该法条但结论正确 → 判 1。
5. 内容与题目无关、只说"信息不足无法回答"、或只是复述题目 → 判 0。

只输出一个 JSON，不要输出任何其他文字：
{{"verdict": 1, "reason": "不超过30字的一句话理由"}}
verdict: 1 = 事实正确，0 = 事实错误"""


def _extract_json(text: str) -> dict | None:
    """从可能包含 markdown 代码块或多余文本的 LLM 响应中提取 JSON。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def judge_one_round(item: dict, judge_llm) -> dict:
    """对一道题调一次判卷模型，返回 {verdict, reason, elapsed}。"""
    evidence = item.get("gold_evidence") or {}
    prompt = JUDGE_PROMPT.format(
        question=item.get("question", ""),
        article=evidence.get("article", "（未提供）"),
        anchor=evidence.get("anchor", "（未提供）"),
        answer=item.get("answer", ""),
    )

    last_error = ""
    for attempt in range(2):
        try:
            start = time.perf_counter()
            resp = judge_llm.invoke(
                [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
            )
            elapsed = round(time.perf_counter() - start, 2)
            parsed = _extract_json(resp.content)
            if parsed and "verdict" in parsed:
                return {
                    "verdict": 1 if parsed["verdict"] else 0,
                    "reason": str(parsed.get("reason", ""))[:100],
                    "elapsed": elapsed,
                }
            last_error = f"JSON 解析失败: {str(resp.content)[:80]}"
        except Exception as e:
            last_error = str(e)

    return {"verdict": None, "reason": "", "error": last_error, "elapsed": 0}


def judge_one(item: dict, judge_llm, rounds: int = 3) -> dict:
    """对一道题判 rounds 轮，verdict 取多数投票，降低 judge 随机性。

    LLM 判卷有随机性，单次判定可能因措辞波动；多轮多数能给出更稳的结论。
    返回 {verdict, reason, round_verdicts, consistency, elapsed, error}。
    """
    round_results = [judge_one_round(item, judge_llm) for _ in range(rounds)]
    verdicts = [r["verdict"] for r in round_results if r["verdict"] is not None]

    if not verdicts:
        return {
            "verdict": None,
            "reason": "",
            "error": round_results[0].get("error", "judge 调用失败"),
            "elapsed": round(sum(r.get("elapsed", 0) for r in round_results), 2),
            "round_verdicts": [r.get("verdict") for r in round_results],
            "consistency": f"0/{rounds}",
        }

    correct_votes = sum(1 for v in verdicts if v == 1)
    majority = 1 if correct_votes > len(verdicts) / 2 else 0

    # reason 取"与多数 verdict 一致"的第一条有理由的结果
    reason = ""
    for r in round_results:
        if r["verdict"] == majority and r.get("reason"):
            reason = r["reason"]
            break

    return {
        "verdict": majority,
        "reason": reason,
        "elapsed": round(sum(r.get("elapsed", 0) for r in round_results), 2),
        "round_verdicts": [r.get("verdict") for r in round_results],
        "consistency": f"{correct_votes}/{len(verdicts)}",
    }


def summarize(items: list[dict]) -> dict:
    """统计 verdict 分布与事实正确率。"""
    judged = [x for x in items if x.get("verdict") is not None]
    correct = sum(1 for x in judged if x["verdict"] == 1)
    return {
        "total": len(items),
        "judged": len(judged),
        "correct": correct,
        "accuracy": round(correct / len(judged) * 100, 1) if judged else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="事实性评测 · 第二步：LLM-as-judge 判卷")
    parser.add_argument(
        "--answers", type=Path, required=True,
        help="第一步 collect 产出的答案 JSON 路径",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="判卷模型名（默认复用项目主模型 deepseek-v4-flash）",
    )
    parser.add_argument(
        "--judge-api-key", default=None,
        help="判卷模型的 API Key；与 --judge-model 一起用可实现异构 judge",
    )
    parser.add_argument(
        "--judge-api-base", default=None,
        help="判卷模型的 API Base URL（默认 DeepSeek）",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="每题判卷轮数，verdict 取多数投票（默认 3；越大越稳，越慢）",
    )
    parser.add_argument(
        "--only-ids", default=None,
        help="只判卷指定题号（逗号分隔，如 r12,r14）；不填则判全部",
    )
    parser.add_argument(
        "--output", type=Path,
        help="判卷结果输出路径；不写时自动保存到 evaluation/results/",
    )
    args = parser.parse_args()

    data = json.loads(args.answers.read_text(encoding="utf-8"))
    answers = data.get("answers", data) if isinstance(data, dict) else data
    ok_items = [a for a in answers if a.get("status") == "ok" and a.get("gold_evidence")]
    skipped = [a for a in answers if a.get("status") != "ok" or not a.get("gold_evidence")]

    if args.only_ids:
        id_set = {x.strip() for x in args.only_ids.split(",") if x.strip()}
        ok_items = [a for a in ok_items if a.get("id") in id_set]

    # judge 模型：默认复用项目主模型；有 --judge-* 时构造独立实例（异构）
    if args.judge_model:
        from langchain_openai import ChatOpenAI
        judge_llm = ChatOpenAI(
            model=args.judge_model,
            openai_api_key=args.judge_api_key,
            openai_api_base=args.judge_api_base or "https://api.deepseek.com/v1",
            timeout=60,
            max_retries=1,
        )
        judge_name = args.judge_model
    else:
        judge_llm = llm
        judge_name = llm.model_name

    print("=" * 72)
    print(f"事实性评测 · 第二步：LLM-as-judge 判卷（{len(ok_items)} 题）")
    print(f"judge 模型: {judge_name}")
    print("=" * 72)

    judged_items = []
    for item in ok_items:
        qid = item["id"]
        result = judge_one(item, judge_llm, rounds=args.rounds)
        verdict = result["verdict"]
        if verdict == 1:
            mark = "OK"
        elif verdict == 0:
            mark = "FAIL"
        else:
            mark = "ERR"
        print(
            f"[{qid}] {mark} verdict={verdict} "
            f"votes={result.get('consistency', '?')} "
            f"reason={result.get('reason', result.get('error', ''))[:40]} "
            f"({result['elapsed']}s)"
        )
        judged_items.append({
            "id": qid,
            "question": item.get("question"),
            "split": item.get("split"),
            "difficulty": item.get("difficulty"),
            "answer": item.get("answer"),
            "answer_full": item.get("answer_full"),
            "gold_evidence": item.get("gold_evidence"),
            "tools_used": item.get("tools_used"),
            "verdict": verdict,
            "round_verdicts": result.get("round_verdicts"),
            "consistency": result.get("consistency"),
            "reason": result.get("reason", ""),
            "judge_error": result.get("error", ""),
            "judge_elapsed": result.get("elapsed"),
        })

    all_stats = summarize(judged_items)
    dev_items = [x for x in judged_items if x.get("split") == "development"]
    val_items = [x for x in judged_items if x.get("split") == "validation"]
    dev_stats = summarize(dev_items)
    val_stats = summarize(val_items)

    print("\n" + "=" * 72)
    print("判卷汇总")
    print("=" * 72)
    print(f"全部:     {all_stats['correct']}/{all_stats['judged']} = {all_stats['accuracy']}%  "
          f"(judge 成功 {all_stats['judged']}/{all_stats['total']})")
    print(f"development: {dev_stats['correct']}/{dev_stats['judged']} = {dev_stats['accuracy']}%")
    print(f"validation:  {val_stats['correct']}/{val_stats['judged']} = {val_stats['accuracy']}%")
    if skipped:
        print(f"\n⚠️  跳过 {len(skipped)} 题（collect 报错或无 gold_evidence）：")
        for s in skipped:
            print(f"  - {s['id']}: {s.get('error', '无 gold_evidence')}")
    print("=" * 72)

    default_name = f"factual_judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = args.output or RESULTS_DIR / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "mode": "factual_judge",
        "judge_model": judge_name,
        "answers_source": str(args.answers),
        "stats": {
            "all": all_stats,
            "development": dev_stats,
            "validation": val_stats,
        },
        "details": judged_items,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {output_path}")