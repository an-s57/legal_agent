"""评测脚本 v2 — 两个维度：检索命中率 + 工具选择准确率

维度1「检索命中率」：直接调用 retriever，检查 Top-5 文档是否命中期望的法律文件。
                    测的是 RAG 管道质量（FAISS + Rerank），不经过 LLM，确定性结果。

维度2「工具选择准确率」：调用 /legal/chat API，检查 LLM 是否选了正确的工具。
                    测的是 Agent 的工具决策能力，受 LLM 随机性影响。
"""
import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# 把项目根目录加入 Python 路径，确保能找到 rag 包
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# 维度1 需要：直接调 retriever 内部函数
from rag.retriever import RERANKER_MODEL_NAME, _get_faiss_db, _get_reranker, _rerank

API_BASE = "http://127.0.0.1:8000"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
K_TUNING_RESULTS_DIR = Path(__file__).parent / "results"


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


def normalize_text(value: str) -> str:
    """去掉 PDF 提取产生的空白，便于匹配短证据句。"""
    return re.sub(r"\s+", "", value or "")


def timing_stats(values: list[float]) -> dict:
    """返回一组毫秒耗时的平均值、中位数和 P95。"""
    if not values:
        return {"avg": 0, "median": 0, "p95": 0}

    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "avg": round(sum(ordered) / len(ordered), 2),
        "median": round(ordered[len(ordered) // 2], 2),
        "p95": round(ordered[p95_index], 2),
    }


def source_matches(expected_sources: list[str], actual_source: str) -> bool:
    """判断返回文档是否来自任一目标法律文件。"""
    return any(
        expected == actual_source or expected in actual_source
        for expected in expected_sources
    )


def document_matches_evidence(doc, expected_sources: list[str], evidence: dict) -> bool:
    """严格判断一条返回文档是否命中目标文件、PDF 页码和短证据句。"""
    if not evidence:
        return False

    actual_source = doc.metadata.get("source", "")
    if not source_matches(expected_sources, actual_source):
        return False

    expected_pdf_page = evidence.get("pdf_page")
    if expected_pdf_page is not None:
        # 评测集使用人看到的 PDF 页码（从 1 开始），FAISS metadata.page 从 0 开始。
        if doc.metadata.get("page") != expected_pdf_page - 1:
            return False

    anchor = normalize_text(evidence.get("anchor", ""))
    return not anchor or anchor in normalize_text(doc.page_content)


# ═══════════════════════════════════════════════════════════════
# 维度1：检索命中率（不经过 LLM，直接测 RAG 管道）
# ═══════════════════════════════════════════════════════════════

def select_retrieval_questions(
    questions: list,
    split: str = "all",
    limit: int | None = None,
) -> list:
    """筛选本地法条检索题；split 用于隔离开发集与验证集。"""
    selected = [
        q for q in questions
        if q["category"] == "法条检索" and q.get("expected_sources")
    ]
    if split != "all":
        selected = [q for q in selected if q.get("split") == split]
    return selected[:limit] if limit is not None else selected


def evaluate_retrieval(
    questions: list,
    *,
    k: int = 20,
    top_k: int = 5,
    faiss_db=None,
    verbose: bool = False,
) -> dict:
    """直接评测 FAISS + Rerank，不经过 FastAPI、Agent 或远程大模型。

    除来源命中外，若题目提供 gold_evidence，还会检查目标 PDF 页码和短证据句。
    """
    if k < top_k:
        raise ValueError(f"k={k} 不能小于 top_k={top_k}")

    retrieval_questions = select_retrieval_questions(questions)
    if not retrieval_questions:
        print("⚠️  没有带 expected_sources 的题目，跳过检索评测")
        return {
            "k": k,
            "top_k": top_k,
            "total": 0,
            "hit": 0,
            "accuracy": 0,
            "source_hit": 0,
            "source_accuracy": 0,
            "evidence_total": 0,
            "evidence_hit": 0,
            "evidence_accuracy": 0,
            "fallback_count": 0,
            "timing_ms": {},
            "details": [],
        }

    print("=" * 72)
    print(f"检索评测：FAISS k={k} → Rerank Top-{top_k}（{len(retrieval_questions)} 题）")
    print("=" * 72)

    faiss_db = faiss_db or _get_faiss_db()
    details = []

    for q in retrieval_questions:
        expected_sources = q["expected_sources"]
        evidence = q.get("gold_evidence")

        vector_start = time.perf_counter()
        candidate_docs = faiss_db.similarity_search(q["question"], k=k)
        vector_search_ms = (time.perf_counter() - vector_start) * 1000

        rerank_start = time.perf_counter()
        rerank_fallback = False
        try:
            docs_top = _rerank(q["question"], candidate_docs, top_k=top_k)
        except Exception as e:
            rerank_fallback = True
            print(f"  ⚠️ [{q['id']}] Rerank 失败，回退到 FAISS Top-{top_k}: {e}")
            docs_top = candidate_docs[:top_k]
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        total_retrieval_ms = vector_search_ms + rerank_ms

        actual_sources = [doc.metadata.get("source", "") for doc in docs_top]
        source_hit = any(
            source_matches(expected_sources, actual_source)
            for actual_source in actual_sources
        )
        evidence_hit = (
            any(document_matches_evidence(doc, expected_sources, evidence) for doc in docs_top)
            if evidence else None
        )

        actual_documents = []
        for rank, doc in enumerate(docs_top, start=1):
            actual_documents.append({
                "rank": rank,
                "source": doc.metadata.get("source", ""),
                "page_index": doc.metadata.get("page"),
                "evidence_match": document_matches_evidence(doc, expected_sources, evidence)
                if evidence else None,
            })

        details.append({
            "id": q["id"],
            "question": q["question"],
            "expected_sources": expected_sources,
            "gold_evidence": evidence,
            "actual_sources": actual_sources,
            "actual_documents": actual_documents,
            # hit / accuracy 保留旧字段名，兼容原来的汇总逻辑。
            "hit": source_hit,
            "source_hit": source_hit,
            "evidence_hit": evidence_hit,
            "vector_search_ms": round(vector_search_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "total_retrieval_ms": round(total_retrieval_ms, 2),
            "rerank_fallback": rerank_fallback,
            "candidate_count": len(candidate_docs),
            "returned_count": len(docs_top),
        })

        source_mark = "OK" if source_hit else "FAIL"
        evidence_mark = "N/A" if evidence_hit is None else ("OK" if evidence_hit else "FAIL")
        print(
            f"[{q['id']}] source={source_mark} evidence={evidence_mark} "
            f"vector={vector_search_ms:.0f}ms rerank={rerank_ms:.0f}ms "
            f"total={total_retrieval_ms:.0f}ms fallback={rerank_fallback}"
        )
        if verbose:
            print(f"  问题: {q['question']}")
            print(f"  目标: {expected_sources[0]}")
            for document in actual_documents:
                print(
                    f"  Top{document['rank']}: {document['source']} "
                    f"page_index={document['page_index']} "
                    f"evidence_match={document['evidence_match']}"
                )

    total = len(details)
    source_hit_count = sum(1 for detail in details if detail["source_hit"])
    source_accuracy = round(source_hit_count / total * 100, 1) if total else 0
    evidence_details = [detail for detail in details if detail["evidence_hit"] is not None]
    evidence_hit_count = sum(1 for detail in evidence_details if detail["evidence_hit"])
    evidence_accuracy = (
        round(evidence_hit_count / len(evidence_details) * 100, 1)
        if evidence_details else 0
    )
    fallback_count = sum(1 for detail in details if detail["rerank_fallback"])

    timing = {
        "vector_search": timing_stats([detail["vector_search_ms"] for detail in details]),
        "rerank": timing_stats([detail["rerank_ms"] for detail in details]),
        "total_retrieval": timing_stats([detail["total_retrieval_ms"] for detail in details]),
    }

    print(f"\n{'─' * 52}")
    print(f"来源命中率: {source_hit_count}/{total} = {source_accuracy}%")
    if evidence_details:
        print(f"证据命中率: {evidence_hit_count}/{len(evidence_details)} = {evidence_accuracy}%")
    print(f"Rerank 回退: {fallback_count}/{total}")
    print(
        "耗时（平均 / 中位数 / P95）："
        f"vector={timing['vector_search']['avg']:.0f}/{timing['vector_search']['median']:.0f}/{timing['vector_search']['p95']:.0f}ms，"
        f"rerank={timing['rerank']['avg']:.0f}/{timing['rerank']['median']:.0f}/{timing['rerank']['p95']:.0f}ms，"
        f"total={timing['total_retrieval']['avg']:.0f}/{timing['total_retrieval']['median']:.0f}/{timing['total_retrieval']['p95']:.0f}ms"
    )
    print(f"{'─' * 52}")

    return {
        "k": k,
        "top_k": top_k,
        # 这三个字段保留旧命名，表示来源命中率。
        "total": total,
        "hit": source_hit_count,
        "accuracy": source_accuracy,
        "source_hit": source_hit_count,
        "source_accuracy": source_accuracy,
        "evidence_total": len(evidence_details),
        "evidence_hit": evidence_hit_count,
        "evidence_accuracy": evidence_accuracy,
        "fallback_count": fallback_count,
        "timing_ms": timing,
        "details": details,
    }


def run_retrieval_k_sweep(
    retrieval_questions: list,
    k_values: list[int],
    top_k: int,
    verbose: bool = False,
) -> dict:
    """在同一进程中依次评测多个 k，避免每次重复加载 FAISS 和 reranker。"""
    if not retrieval_questions:
        raise ValueError("没有可用于 K 调参的法条检索题")

    unique_k_values = list(dict.fromkeys(k_values))
    if any(k <= 0 for k in unique_k_values):
        raise ValueError("所有 k 都必须是正整数")
    if any(k < top_k for k in unique_k_values):
        raise ValueError(f"所有 k 都必须不小于 top_k={top_k}")

    print("\n" + "=" * 72)
    print(
        f"K 参数对比：{len(retrieval_questions)} 题，"
        f"k={unique_k_values}，固定 top_k={top_k}"
    )
    print("预热阶段不计入正式耗时，用于避免模型首次推理影响第一个 k。")
    print("=" * 72)

    faiss_load_start = time.perf_counter()
    faiss_db = _get_faiss_db()
    faiss_load_ms = (time.perf_counter() - faiss_load_start) * 1000
    index_document_count = getattr(getattr(faiss_db, "index", None), "ntotal", None)
    if index_document_count is not None and any(k > index_document_count for k in unique_k_values):
        raise ValueError(
            f"请求的 k 超过当前 FAISS 文档数 {index_document_count}：{unique_k_values}"
        )

    reranker_load_start = time.perf_counter()
    _get_reranker()
    reranker_load_ms = (time.perf_counter() - reranker_load_start) * 1000

    warmup = {"status": "ok", "vector_search_ms": 0, "rerank_ms": 0}
    try:
        warmup_question = retrieval_questions[0]["question"]
        warmup_vector_start = time.perf_counter()
        warmup_docs = faiss_db.similarity_search(warmup_question, k=max(unique_k_values))
        warmup["vector_search_ms"] = round(
            (time.perf_counter() - warmup_vector_start) * 1000, 2
        )
        warmup_rerank_start = time.perf_counter()
        _rerank(warmup_question, warmup_docs, top_k=top_k)
        warmup["rerank_ms"] = round(
            (time.perf_counter() - warmup_rerank_start) * 1000, 2
        )
    except Exception as e:
        warmup["status"] = "error"
        warmup["error"] = f"{type(e).__name__}: {e}"
        print(f"⚠️  预热失败，正式评测仍会继续：{warmup['error']}")

    runs = []
    for k in unique_k_values:
        report = evaluate_retrieval(
            retrieval_questions,
            k=k,
            top_k=top_k,
            faiss_db=faiss_db,
            verbose=verbose,
        )
        runs.append(report)

    summary = [
        {
            "k": run["k"],
            "source_hit": run["source_hit"],
            "source_total": run["total"],
            "source_accuracy": run["source_accuracy"],
            "evidence_hit": run["evidence_hit"],
            "evidence_total": run["evidence_total"],
            "evidence_accuracy": run["evidence_accuracy"],
            "fallback_count": run["fallback_count"],
            "vector_search_ms": run["timing_ms"]["vector_search"],
            "rerank_ms": run["timing_ms"]["rerank"],
            "total_retrieval_ms": run["timing_ms"]["total_retrieval"],
        }
        for run in runs
    ]

    print("\n" + "=" * 72)
    print("K 参数对比汇总")
    print("=" * 72)
    print("k | 来源命中 | 证据命中 | 平均 rerank | 平均总检索 | 回退")
    for row in summary:
        print(
            f"{row['k']:>2} | "
            f"{row['source_hit']}/{row['source_total']} ({row['source_accuracy']:>5.1f}%) | "
            f"{row['evidence_hit']}/{row['evidence_total']} ({row['evidence_accuracy']:>5.1f}%) | "
            f"{row['rerank_ms']['avg']:>8.0f}ms | "
            f"{row['total_retrieval_ms']['avg']:>8.0f}ms | "
            f"{row['fallback_count']}"
        )
    print("=" * 72)

    return {
        "timestamp": datetime.now().isoformat(),
        "mode": "retrieval_k_sweep",
        "config": {
            "question_count": len(retrieval_questions),
            "question_ids": [question["id"] for question in retrieval_questions],
            "k_values": unique_k_values,
            "top_k": top_k,
            "faiss_document_count": index_document_count,
            "reranker_model": RERANKER_MODEL_NAME,
            "faiss_load_ms": round(faiss_load_ms, 2),
            "reranker_preload_ms": round(reranker_load_ms, 2),
            "warmup": warmup,
        },
        "summary": summary,
        "runs": runs,
    }


def save_json_report(path: Path, report: dict) -> None:
    """创建结果目录并保存 UTF-8 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {path}")


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
        "--questions", type=Path, default=QUESTIONS_PATH,
        help="题集 JSON 路径（默认 evaluation/questions.json）"
    )
    parser.add_argument(
        "--split", choices=["all", "development", "validation"], default="all",
        help="K 参数评测时使用的题集划分（默认 all）"
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=[20],
        help="K 参数评测要比较的 FAISS 粗召回候选数，例如：--k-values 5 10 15 20"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Rerank 后保留的文档数（默认 5）；K 调参时应保持固定"
    )
    parser.add_argument(
        "--limit", type=int,
        help="只取筛选后的前 N 道题，用于小规模冒烟测试"
    )
    parser.add_argument(
        "--output", type=Path,
        help="K 参数评测报告输出路径；不写时自动保存到 evaluation/results/"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印每题 Top 文档来源和页码；默认只打印一行摘要"
    )
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

    if args.retrieval_only and args.tool_only:
        parser.error("--retrieval-only 和 --tool-only 不能同时使用")
    if args.top_k <= 0:
        parser.error("--top-k 必须是正整数")
    if any(k <= 0 for k in args.k_values):
        parser.error("--k-values 中的所有 k 都必须是正整数")
    if any(k < args.top_k for k in args.k_values):
        parser.error("--k-values 中的每个 k 都必须不小于 --top-k")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数")

    if args.retrieval_only:
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
        retrieval_questions = select_retrieval_questions(
            questions,
            split=args.split,
            limit=args.limit,
        )
        if not retrieval_questions:
            parser.error(
                f"在 {args.questions} 中没有找到 split={args.split} 的本地法条检索题"
            )

        report = run_retrieval_k_sweep(
            retrieval_questions,
            k_values=args.k_values,
            top_k=args.top_k,
            verbose=args.verbose,
        )
        report["config"].update({
            "questions_path": str(args.questions),
            "split": args.split,
            "limit": args.limit,
        })

        default_name = (
            f"k_sweep_{args.split}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path = args.output or K_TUNING_RESULTS_DIR / default_name
        save_json_report(output_path, report)
    elif args.tool_only:
        evaluate(rounds=args.rounds)
    else:
        evaluate(rounds=args.rounds)
