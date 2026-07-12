"""评测脚本 v2 — 两个维度：检索命中率 + 工具选择准确率

维度1「检索命中率」：直接调用 retriever，检查 Top-5 文档是否命中期望的法律文件。
                    测的是 RAG 管道质量（FAISS + Rerank），不经过 LLM，确定性结果。

维度2「工具选择准确率」：调用 /legal/chat API，检查 LLM 是否选了正确的工具。
                    测的是 Agent 的工具决策能力，受 LLM 随机性影响。
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 把项目根目录加入 Python 路径，确保能找到 rag 包
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# 维度1 需要：直接调 retriever 内部函数
from rag.retriever import _get_faiss_db, _rerank

API_BASE = "http://127.0.0.1:8000"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def check_server():
    """检查 API 服务器是否在线"""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# 维度1：检索命中率（不经过 LLM，直接测 RAG 管道）
# ═══════════════════════════════════════════════════════════════

def evaluate_retrieval(questions: list) -> dict:
    """对所有「法条检索」类题目，直接调 FAISS + Rerank，检查 Top-5
    文档的来源是否包含 expected_sources 中的法律文件名。

    这是确定性测试——同样的 query 同样的向量库，结果永远一致。
    """
    retrieval_questions = [
        q for q in questions
        if q["category"] == "法条检索" and q.get("expected_sources")
    ]

    if not retrieval_questions:
        print("⚠️  没有带 expected_sources 的题目，跳过检索评测")
        return {"total": 0, "hit": 0, "accuracy": 0, "details": []}

    print("=" * 60)
    print("维度1：检索命中率（FAISS + Rerank → Top-5 文档来源是否匹配）")
    print("=" * 60)

    faiss_db = _get_faiss_db()
    details = []

    for q in retrieval_questions:
        expected = q["expected_sources"]

        # FAISS 粗排 20 条 → Rerank 精排取 5 条
        docs_20 = faiss_db.similarity_search(q["question"], k=20)
        try:
            docs_top5 = _rerank(q["question"], docs_20, top_k=5)
        except Exception as e:
            print(f"  ⚠️ Rerank 失败，回退到 FAISS Top-5: {e}")
            docs_top5 = docs_20[:5]

        actual_sources = [d.metadata.get("source", "") for d in docs_top5]

        # 检查 Top-5 中是否有任意一个文档来源匹配期望来源（子串匹配）
        hit = any(
            any(exp in actual for actual in actual_sources)
            for exp in expected
        )

        details.append({
            "id": q["id"],
            "question": q["question"],
            "expected_sources": expected,
            "actual_sources": actual_sources,
            "hit": hit,
        })

        mark = "[OK]" if hit else "[FAIL]"
        print(f"\n[{q['id']}] {q['question']}")
        print(f"  期望来源: {expected[0]}")
        for i, src in enumerate(actual_sources):
            print(f"  Top{i+1}: {src}")
        print(f"  命中: {mark}")

    total = len(details)
    hit_count = sum(1 for d in details if d["hit"])
    accuracy = round(hit_count / total * 100, 1) if total else 0

    print(f"\n{'─' * 40}")
    print(f"检索命中率: {hit_count}/{total} = {accuracy}%")
    print(f"{'─' * 40}")

    return {"total": total, "hit": hit_count, "accuracy": accuracy, "details": details}


# ═══════════════════════════════════════════════════════════════
# 维度2：工具选择准确率（调用 API，测 LLM 工具决策）
# ═══════════════════════════════════════════════════════════════

def run_single_question(question: dict, round_idx: int) -> dict:
    """对一道题调一次 API，返回单次结果"""
    session_id = f"eval-{question['id']}-r{round_idx}"

    start = time.time()
    try:
        resp = requests.post(
            f"{API_BASE}/legal/chat",
            json={"session_id": session_id, "message": question["question"]},
            timeout=60,
        )
        elapsed = round(time.time() - start, 2)

        if resp.status_code != 200:
            return {
                "status": "error",
                "error": f"HTTP {resp.status_code}",
                "elapsed": elapsed,
            }

        data = resp.json()
        actual_tools = data.get("tools_used", [])
        expected = question["expect_tool"]
        correct = expected in actual_tools

        return {
            "status": "ok",
            "question": question["question"],
            "expected_tool": expected,
            "actual_tools": actual_tools,
            "correct": correct,
            "answer": data.get("answer", "")[:200],
            "elapsed": elapsed,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "elapsed": round(time.time() - start, 2),
        }


def evaluate_tool_selection(questions: list, rounds: int = 1) -> dict:
    """调 /legal/chat API，统计 LLM 工具选择准确率"""
    print("\n" + "=" * 60)
    print(f"维度2：工具选择准确率（LLM 是否选了正确的工具，每道题 {rounds} 轮）")
    print("=" * 60)

    all_results = []

    for q in questions:
        print(f"\n[{q['id']}] {q['question'][:40]}...")
        print(f"  期望工具: {q['expect_tool']}")

        round_results = []
        for r in range(rounds):
            result = run_single_question(q, r)
            round_results.append(result)

            if result["status"] == "ok":
                mark = "[OK]" if result["correct"] else "[FAIL]"
                print(f"  round {r+1}: {mark} actual={result['actual_tools']} ({result['elapsed']}s)")
            else:
                print(f"  round {r+1}: [ERROR] {result['error']}")

        # 分离成功和失败的 round
        ok_rounds = [r for r in round_results if r["status"] == "ok"]
        error_rounds = [r for r in round_results if r["status"] == "error"]

        if not ok_rounds:
            # 全部 round 都报错 → 排除在准确率统计之外
            question_status = "error"
            majority_correct = False
            consistency = f"0/{rounds} (全部超时/报错)"
        else:
            # 至少有一轮成功
            question_status = "partial" if error_rounds else "ok"

            # 只在成功的 round 里做多数投票
            if rounds > 1:
                correct_count = sum(1 for r in ok_rounds if r.get("correct"))
                majority_correct = correct_count > len(ok_rounds) / 2
                consistency = f"{correct_count}/{len(ok_rounds)}"
            else:
                majority_correct = ok_rounds[0].get("correct", False)
                consistency = "1/1"

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "expected_tool": q["expect_tool"],
            "correct": majority_correct,
            "status": question_status,
            "consistency": consistency,
            "rounds": round_results,
        })

    # 统计：排除全部 error 的题
    total = len(all_results)
    valid_results = [r for r in all_results if r["status"] != "error"]
    error_results = [r for r in all_results if r["status"] == "error"]
    valid_total = len(valid_results)
    error_total = len(error_results)
    correct_num = sum(1 for r in valid_results if r["correct"])
    accuracy = round(correct_num / valid_total * 100, 1) if valid_total else 0

    rag_results = [r for r in valid_results if r["category"] == "法条检索"]
    web_results = [r for r in valid_results if r["category"] == "联网搜索"]
    rag_acc = round(sum(1 for r in rag_results if r["correct"]) / len(rag_results) * 100, 1) if rag_results else 0
    web_acc = round(sum(1 for r in web_results if r["correct"]) / len(web_results) * 100, 1) if web_results else 0

    all_elapsed = []
    for r in all_results:
        for rd in r["rounds"]:
            if rd.get("elapsed"):
                all_elapsed.append(rd["elapsed"])
    avg_time = round(sum(all_elapsed) / len(all_elapsed), 2) if all_elapsed else 0

    print(f"\n{'─' * 40}")
    print(f"工具选择准确率: {correct_num}/{valid_total} = {accuracy}%")
    if error_total:
        print(f"  ⚠️  {error_total} 题因超时/报错被排除，不计入准确率")
    print(f"  法条检索: {rag_acc}%")
    print(f"  联网搜索: {web_acc}%")
    print(f"  平均耗时: {avg_time}s")
    print(f"{'─' * 40}")

    return {
        "total": total,
        "valid_total": valid_total,
        "error_total": error_total,
        "correct": correct_num,
        "accuracy": accuracy,
        "rag_accuracy": rag_acc,
        "web_accuracy": web_acc,
        "avg_response_time": avg_time,
        "details": all_results,
    }


# ═══════════════════════════════════════════════════════════════
# 主评测
# ═══════════════════════════════════════════════════════════════

def evaluate(rounds: int = 1):
    """跑完整评测：维度1（检索命中率）+ 维度2（工具选择准确率）"""
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"加载 {len(questions)} 道测试题")
    print(f"  法条检索: {sum(1 for q in questions if q['category'] == '法条检索')} 题")
    print(f"  联网搜索: {sum(1 for q in questions if q['category'] == '联网搜索')} 题")

    # ── 维度1：检索命中率（不需要服务器，不依赖 LLM） ──
    retrieval_report = evaluate_retrieval(questions)

    # ── 维度2：工具选择准确率（需要服务器 + LLM） ──
    tool_report = None
    if check_server():
        tool_report = evaluate_tool_selection(questions, rounds)
    else:
        print("\n" + "=" * 60)
        print("⚠️  服务器未启动！跳过维度2（工具选择准确率）")
        print("   如需测试工具选择，请先运行：python main.py")
        print("=" * 60)

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("📊 评测汇总")
    print("=" * 60)
    r_acc = retrieval_report['accuracy']
    print(f"维度1 检索命中率:    {r_acc}% ({retrieval_report['hit']}/{retrieval_report['total']})")
    if tool_report:
        t_acc = tool_report['accuracy']
        print(f"维度2 工具选择准确率: {t_acc}% ({tool_report['correct']}/{tool_report['total']})")
    else:
        print(f"维度2 工具选择准确率: (未测 — 服务器离线)")
    print("=" * 60)

    # ── 存结果 ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "rounds": rounds,
        "retrieval": retrieval_report,
        "tool_selection": tool_report,
    }
    RESULTS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已保存至: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法律 Agent 评测脚本 v2")
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="工具选择评测的轮数（默认1，建议正式评测用3）。检索评测无需多轮。"
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="只跑检索命中率评测（不需要服务器，直接测 RAG 管道）"
    )
    parser.add_argument(
        "--tool-only", action="store_true",
        help="只跑工具选择评测（需要先启动服务器）"
    )
    args = parser.parse_args()

    if args.retrieval_only:
        questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        report = evaluate_retrieval(questions)
        RESULTS_PATH.write_text(
            json.dumps({"timestamp": datetime.now().isoformat(), "retrieval": report},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n结果已保存至: {RESULTS_PATH}")
    elif args.tool_only:
        evaluate(rounds=args.rounds)
    else:
        evaluate(rounds=args.rounds)