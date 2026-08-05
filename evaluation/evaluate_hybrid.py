"""P3 混合检索评测：FAISS + BM25 + RRF 对比纯向量基线。

复用 evaluate_retrieval 的 recall 参数，扫描 k_bm25 × 候选池两档，
并统计 7 道基线稳定失败题的救回情况（P1 结论：r01 r05 r06 r16 r18 r28 r29）。

用法：
    python -X utf8 evaluation/evaluate_hybrid.py                # development 默认网格
    python -X utf8 evaluation/evaluate_hybrid.py --split validation
    python -X utf8 evaluation/evaluate_hybrid.py --limit 2     # 冒烟测试
    python -X utf8 evaluation/evaluate_hybrid.py --grid "20:40,40:60"  # 自定义 k_bm25:候选池
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import RERANKER_MODEL_NAME, _get_faiss_db, _get_reranker
from rag.hybrid import hybrid_candidates
from evaluate_ksweep import (
    evaluate_retrieval,
    save_json_report,
    select_retrieval_questions,
)

QUESTIONS_PATH = Path(__file__).parent / "retrieval_k_tuning_50.json"
RESULTS_DIR = Path(__file__).parent / "results"

# P1 分块实验确认的 7 道稳定失败题（短锚句进不了 Top-5）
BASELINE_FAILS = ["r01", "r05", "r06", "r16", "r18", "r28", "r29"]

# 评测固定参数：k_vector=40 是 K 扫描最优点，top_k=5 是生产值
K_VECTOR = 40
TOP_K = 5
RRF_K = 60
# 默认网格：k_bm25 × 候选池（用户拍板扫 40/60 两档）
DEFAULT_GRID = [
    (20, 40), (20, 60),
    (40, 40), (40, 60),
]


def parse_grid(text: str) -> list[tuple[int, int]]:
    """解析 '20:40,40:60' 形式的 k_bm25:候选池 网格。"""
    grid = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k_bm25, cand = (int(x) for x in part.split(":"))
        except ValueError:
            raise SystemExit(f"网格格式应为 'k_bm25:候选池'，收到：{part!r}")
        grid.append((k_bm25, cand))
    return grid


def rescue_analysis(baseline: dict, hybrid: dict) -> dict:
    """对比基线与 hybrid 的每题 evidence_hit，找救回/丢失。"""
    base_by_id = {d["id"]: d["evidence_hit"] for d in baseline["details"]}
    hybrid_by_id = {d["id"]: d["evidence_hit"] for d in hybrid["details"]}

    rescued, lost, still_fail = [], [], []
    for qid in base_by_id:
        b, h = base_by_id[qid], hybrid_by_id.get(qid)
        if b is False and h is True:
            rescued.append(qid)
        elif b is True and h is False:
            lost.append(qid)
        elif b is False and h is False:
            still_fail.append(qid)
    return {
        "rescued": rescued,
        "lost": lost,
        "still_fail": still_fail,
        "baseline_fails": [qid for qid, v in base_by_id.items() if v is False],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P3 混合检索评测：FAISS+BM25+RRF vs 纯向量基线"
    )
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH,
                        help="题集路径（默认 retrieval_k_tuning_50.json）")
    parser.add_argument("--split", choices=["development", "validation", "all"],
                        default="development", help="题集划分（默认 development）")
    parser.add_argument("--grid", default=None,
                        help="k_bm25:候选池 网格，逗号分隔，如 '20:40,40:60'；默认四档")
    parser.add_argument("--k", type=int, default=K_VECTOR,
                        help=f"向量召回数 k_vector（默认 {K_VECTOR}）")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Rerank 后保留数（默认 {TOP_K}）")
    parser.add_argument("--rrf-k", type=int, default=RRF_K,
                        help=f"RRF 融合常数（默认 {RRF_K}）")
    parser.add_argument("--limit", type=int,
                        help="只取筛选后的前 N 题，用于冒烟测试")
    parser.add_argument("--output", type=Path,
                        help="报告输出路径；默认 evaluation/results/")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每题 Top 文档来源和页码")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数")
    grid = parse_grid(args.grid) if args.grid else DEFAULT_GRID

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = select_retrieval_questions(
        questions, split=args.split, limit=args.limit,
    )
    if not selected:
        parser.error(f"在 {args.questions} 中没有找到 split={args.split} 的法条检索题")

    print(f"预加载 reranker + BM25 索引（冷启动一次）...", flush=True)
    _get_reranker()
    faiss_db = _get_faiss_db()
    # 预热 BM25：jieba 首次分词要加载词典，不进正式指标
    if selected:
        hybrid_candidates(selected[0]["question"], faiss_db,
                          k_vector=args.k, k_bm25=20, rrf_k=args.rrf_k,
                          top_candidates=40)

    print("\n" + "=" * 72)
    print(f"P3 混合检索评测 split={args.split}（{len(selected)} 题）")
    print(f"基线失败题：{BASELINE_FAILS}")
    print("=" * 72)

    # 1) 纯向量基线
    baseline = evaluate_retrieval(
        selected, k=args.k, top_k=args.top_k, faiss_db=faiss_db,
        verbose=args.verbose, recall="vector",
    )

    # 2) hybrid 网格
    runs = []
    print("\n" + "─" * 72)
    print("混合检索配置扫描")
    print("─" * 72)
    print("配置                      | 证据命中        | 平均总检索   | 救回          | 丢失")
    for k_bm25, cand in grid:
        rep = evaluate_retrieval(
            selected, k=args.k, top_k=args.top_k, faiss_db=faiss_db,
            verbose=args.verbose, recall="hybrid",
            k_bm25=k_bm25, rrf_k=args.rrf_k, top_candidates=cand,
        )
        rescue = rescue_analysis(baseline, rep)
        runs.append({
            "k_bm25": k_bm25,
            "top_candidates": cand,
            "rrf_k": args.rrf_k,
            "k_vector": args.k,
            "evidence_hit": rep["evidence_hit"],
            "evidence_total": rep["evidence_total"],
            "evidence_accuracy": rep["evidence_accuracy"],
            "avg_total_retrieval_ms": rep["timing_ms"]["total_retrieval"]["avg"],
            "avg_rerank_ms": rep["timing_ms"]["rerank"]["avg"],
            "rescue": rescue,
            "report": rep,
        })
        name = f"hybrid k_bm25={k_bm25} cand={cand}"
        print(
            f"{name:<24} | "
            f"{rep['evidence_hit']}/{rep['evidence_total']} "
            f"({rep['evidence_accuracy']:>5.1f}%) | "
            f"{rep['timing_ms']['total_retrieval']['avg']:>8.0f}ms | "
            f"{','.join(rescue['rescued']) or '-':<12} | "
            f"{','.join(rescue['lost']) or '-'}"
        )
    print("─" * 72)

    # 汇总落盘
    default_name = (
        f"hybrid_{args.split}_k{args.k}_top{args.top_k}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path = args.output or RESULTS_DIR / default_name
    save_json_report(output_path, {
        "timestamp": datetime.now().isoformat(),
        "mode": "hybrid_sweep",
        "config": {
            "questions_path": str(args.questions),
            "split": args.split,
            "limit": args.limit,
            "k_vector": args.k,
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "grid": grid,
            "reranker_model": RERANKER_MODEL_NAME,
            "baseline_fails": BASELINE_FAILS,
        },
        "baseline": {
            "evidence_hit": baseline["evidence_hit"],
            "evidence_total": baseline["evidence_total"],
            "evidence_accuracy": baseline["evidence_accuracy"],
            "avg_total_retrieval_ms": baseline["timing_ms"]["total_retrieval"]["avg"],
            "report": baseline,
        },
        "hybrid_runs": runs,
    })
    print(f"\n报告已保存：{output_path}")


if __name__ == "__main__":
    main()