"""事实性评测 · 第一步：跑完整 Agent 收集答案。

对每题调 /legal/chat API，走完整链路（planner → 工具检索 → LLM 回答），
把 {答案主体, 幻觉警告, 使用的工具, gold_evidence, 耗时} 落盘 JSON，
供第二步 eval_factual_judge.py 做 LLM-as-judge 事实一致性判卷。

与 evaluate_ksweep.py 维度2 的区别：这里不做"工具选择"判定，
而是把完整答案收集下来逐题判卷——测的是"答得准不准"。

用法：
    python -X utf8 evaluation/eval_factual_collect.py                     # 全部 50 题
    python -X utf8 evaluation/eval_factual_collect.py --split validation  # 只跑验证集
    python -X utf8 evaluation/eval_factual_collect.py --limit 5           # 冒烟测试
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

API_BASE = "http://127.0.0.1:8000"
DEFAULT_QUESTIONS = Path(__file__).parent / "retrieval_k_tuning_50.json"
RESULTS_DIR = Path(__file__).parent / "results"

# agent 在回答末尾追加幻觉校验警告的固定分隔标记（见 hallucination_guard.py）
WARNING_MARK = "\n\n---\n**检索校验**"


def check_server() -> bool:
    """检查 API 服务器是否在线。"""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def split_answer_warning(answer: str) -> tuple[str, str]:
    """把 agent 追加的幻觉校验警告从答案主体中剥离出来。

    返回 (答案主体, 警告文本)。无警告时第二项为空字符串。
    """
    if WARNING_MARK in answer:
        body, warning = answer.split(WARNING_MARK, 1)
        return body.strip(), WARNING_MARK + warning
    return answer, ""


def select_questions(questions: list, split: str, limit: int | None) -> list:
    """按 split 过滤题目，limit 用于冒烟测试。"""
    selected = questions if split == "all" else [q for q in questions if q.get("split") == split]
    return selected[:limit] if limit is not None else selected


def collect_answers(
    questions: list,
    *,
    skip_planner: bool = False,
) -> list[dict]:
    """逐题调 /legal/chat，收集答案。每题独立 session，避免 case_summary 串味。"""
    print("=" * 72)
    print(f"事实性评测 · 第一步：跑 Agent 收集答案（{len(questions)} 题）")
    print(f"skip_planner={skip_planner}")
    print("=" * 72)

    results = []
    error_count = 0

    for q in questions:
        qid = q["id"]
        session_id = f"eval-factual-{qid}"
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{API_BASE}/legal/chat",
                json={
                    "session_id": session_id,
                    "message": q["question"],
                    "skip_planner": skip_planner,
                },
                timeout=120,
            )
            elapsed = round(time.perf_counter() - start, 2)

            if resp.status_code != 200:
                error_count += 1
                results.append({
                    "id": qid,
                    "question": q["question"],
                    "status": "error",
                    "error": f"HTTP {resp.status_code}",
                    "elapsed": elapsed,
                })
                print(f"[{qid}] [HTTP {resp.status_code}] {elapsed}s")
                continue

            data = resp.json()
            answer_full = data.get("answer", "")
            answer_body, warning = split_answer_warning(answer_full)
            tools_used = data.get("tools_used", [])

            results.append({
                "id": qid,
                "question": q["question"],
                "category": q.get("category"),
                "split": q.get("split"),
                "difficulty": q.get("difficulty"),
                "expected_sources": q.get("expected_sources"),
                "gold_evidence": q.get("gold_evidence"),
                "answer": answer_body,
                "answer_full": answer_full,
                "hallucination_warning": warning,
                "tools_used": tools_used,
                "elapsed": elapsed,
                "status": "ok",
            })
            print(f"[{qid}] OK tools={tools_used} {elapsed}s")
        except Exception as e:
            elapsed = round(time.perf_counter() - start, 2)
            error_count += 1
            results.append({
                "id": qid,
                "question": q["question"],
                "status": "error",
                "error": str(e),
                "elapsed": elapsed,
            })
            print(f"[{qid}] [ERROR] {e} {elapsed}s")

    ok_count = len(results) - error_count
    print(f"\n{'─' * 52}")
    print(f"收集完成：OK {ok_count}/{len(results)}，错误 {error_count}")
    print(f"{'─' * 52}")
    return results


def save_json_report(path: Path, report: dict) -> None:
    """创建结果目录并保存 UTF-8 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="事实性评测 · 第一步：跑 Agent 收集答案")
    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_QUESTIONS,
        help="题集 JSON 路径（默认 retrieval_k_tuning_50.json）",
    )
    parser.add_argument(
        "--split", choices=["all", "development", "validation"], default="all",
        help="题集划分（默认 all）",
    )
    parser.add_argument(
        "--limit", type=int,
        help="只取前 N 道题，用于冒烟测试",
    )
    parser.add_argument(
        "--skip-planner", action="store_true",
        help="跳过 Planner 直接进 ReAct（评测题信息通常完整，一般不需要）",
    )
    parser.add_argument(
        "--output", type=Path,
        help="答案 JSON 输出路径；不写时自动保存到 evaluation/results/",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数")

    if not check_server():
        parser.error("API 服务器未启动！请先运行：python main.py")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = select_questions(questions, args.split, args.limit)
    if not selected:
        parser.error(f"在 {args.questions} 中没有找到 split={args.split} 的题")

    answers = collect_answers(selected, skip_planner=args.skip_planner)

    default_name = (
        f"factual_answers_{args.split}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path = args.output or RESULTS_DIR / default_name
    save_json_report(output_path, {
        "timestamp": datetime.now().isoformat(),
        "mode": "factual_collect",
        "config": {
            "questions_path": str(args.questions),
            "split": args.split,
            "limit": args.limit,
            "skip_planner": args.skip_planner,
        },
        "total": len(answers),
        "ok_count": sum(1 for a in answers if a["status"] == "ok"),
        "error_count": sum(1 for a in answers if a["status"] == "error"),
        "answers": answers,
    })