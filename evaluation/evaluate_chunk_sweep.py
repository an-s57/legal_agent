"""分块参数对比实验：扫描 chunk_size × chunk_overlap，测证据命中率。

K 参数扫描（evaluate_ksweep.py）换的是粗召回数 k，向量库始终是同一个；
本实验要换向量库——每个 (chunk_size, chunk_overlap) 配置都要重新切块、
重新嵌入、重新建 FAISS。建库是唯一的自变量，评测逻辑 100% 复用
evaluate_ksweep.evaluate_retrieval（相似度搜索 + rerank + 证据判定 + 耗时统计），
避免两处维护同一套判定逻辑。

设计决策：
1. 主指标 = evidence_hit（页码 + 法条原文锚句都落在返回 chunk 里）。
   语料只有 6 个法条 PDF，source_hit 几乎恒为 100%，分不出配置差别。
2. 固定 k=40：K 扫描在 development 上测出的最优点（26/33=78.8%，与 k=50
   持平，k=30 只有 24/33=72.7%）。chunk 实验选的配置最终会和 k=40 一起上线，
   所以固定值用最优 k，而不是陈旧的生产默认 k=30。
3. 两阶段：development 全网格选型 → validation 上只确认「最优 + 基线」，
   防止过拟合 development。
4. 安全：所有配置的向量库建到系统临时目录，绝不碰生产库 rag/vectorstore/db_faiss。

用法：
    # 阶段A：development 全网格（8 个配置）
    python -X utf8 evaluation/evaluate_chunk_sweep.py

    # 阶段B：validation 上确认「基线 + 某最优配置」
    python -X utf8 evaluation/evaluate_chunk_sweep.py \
        --split validation --configs "1000:200,800:160"

    # 冒烟测试：只建一个配置、只评测前 2 题
    python -X utf8 evaluation/evaluate_chunk_sweep.py \
        --configs "1000:200" --limit 2
"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.retriever import (
    RERANKER_MODEL_NAME,
    _get_reranker,
    _rerank,
    embedding_model,
)
from evaluate_ksweep import (
    evaluate_retrieval,
    save_json_report,
    select_retrieval_questions,
)

PDF_FOLDER = Path(__file__).parent.parent / "legal_pdfs"
DEFAULT_QUESTIONS = Path(__file__).parent / "retrieval_k_tuning_50.json"
RESULTS_DIR = Path(__file__).parent / "results"

# 阶段A 全网格：size 扫描（overlap=20%size）+ 在 size=1000 上做 overlap 扫描
DEFAULT_CONFIGS = [
    (500, 100), (800, 160), (1000, 200), (1500, 300), (2000, 400),
    (1000, 0), (1000, 100), (1000, 300),
]
# 当前生产参数，作为对照基线
BASELINE = (1000, 200)
# K 扫描在 development 上测出的最优粗召回数（见 evaluate_ksweep.py 的结果）
K_FIXED = 40
TOP_K_FIXED = 5


def parse_configs(text: str) -> list[tuple[int, int]]:
    """解析 '500:100,800:160' 形式的配置列表。"""
    configs = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            size, overlap = (int(x) for x in part.split(":"))
        except ValueError:
            raise SystemExit(f"配置格式应为 'size:overlap'，收到：{part!r}")
        if overlap >= size:
            raise SystemExit(f"overlap={overlap} 必须小于 chunk_size={size}")
        configs.append((size, overlap))
    return configs


def build_vectorstore(
    pdf_folder: Path,
    chunk_size: int,
    chunk_overlap: int,
    save_dir: Path,
) -> tuple[FAISS, int, float]:
    """按给定 chunk 参数把 legal_pdfs 建成 FAISS 库，存到 save_dir。

    返回 (faiss_db, chunk_count, build_ms)。与 retriever.build_legal_vectorstore
    完全同构，只是 chunk 参数和输出目录可配。
    """
    all_chunks = []
    for filename in sorted(os.listdir(pdf_folder)):
        if not filename.endswith(".pdf"):
            continue
        loader = PDFPlumberLoader(str(pdf_folder / filename))
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = filename
        all_chunks.extend(chunks)

    build_start = time.perf_counter()
    faiss_db = FAISS.from_documents(all_chunks, embedding_model)
    build_ms = round((time.perf_counter() - build_start) * 1000, 2)

    save_dir.mkdir(parents=True, exist_ok=True)
    faiss_db.save_local(str(save_dir))
    return faiss_db, len(all_chunks), build_ms


def run_one_config(
    config: tuple[int, int],
    questions: list,
    *,
    k: int,
    top_k: int,
    pdf_folder: Path,
    temp_root: Path,
    verbose: bool,
) -> dict:
    """建一个配置的向量库并评测，返回该配置的完整报告。

    建库（含嵌入）不计入评测耗时；建好后先做一次预热搜索+rerank，
    让模型首次推理的固定开销不进正式指标。
    """
    chunk_size, chunk_overlap = config
    name = f"cs{chunk_size}_ov{chunk_overlap}"
    print("\n" + "=" * 72)
    print(f"配置 {name}：chunk_size={chunk_size} chunk_overlap={chunk_overlap}")
    print("=" * 72)

    build_start = time.perf_counter()
    faiss_db, chunk_count, build_ms = build_vectorstore(
        pdf_folder, chunk_size, chunk_overlap, temp_root / name
    )
    print(f"[BUILD] {chunk_count} 个文本段，耗时 {build_ms:.0f}ms")

    if chunk_count < k:
        print(f"⚠️  当前配置 chunk 数 {chunk_count} < k={k}，无法评测，跳过")
        return None

    # 预热：让 reranker 首次推理不计入正式评测
    warmup_question = questions[0]["question"]
    warmup_docs = faiss_db.similarity_search(warmup_question, k=k)
    try:
        _rerank(warmup_question, warmup_docs, top_k=top_k)
    except Exception as e:
        print(f"⚠️  预热失败，继续正式评测：{type(e).__name__}: {e}")

    report = evaluate_retrieval(
        questions,
        k=k,
        top_k=top_k,
        faiss_db=faiss_db,
        verbose=verbose,
        rerank=True,
    )

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "build_ms": build_ms,
        "chunk_count": chunk_count,
        "report": report,
    }


def run_sweep(
    configs: list[tuple[int, int]],
    questions: list,
    *,
    k: int,
    top_k: int,
    pdf_folder: Path,
    temp_root: Path,
    verbose: bool,
) -> dict:
    """依次建库并评测每个配置，汇总成对照表。"""
    _get_reranker()  # 预加载 reranker，避免第一个配置被模型加载时间污染

    runs = []
    for config in configs:
        run = run_one_config(
            config, questions, k=k, top_k=top_k,
            pdf_folder=pdf_folder, temp_root=temp_root, verbose=verbose,
        )
        if run is not None:
            runs.append(run)

    summary = []
    print("\n" + "=" * 72)
    print("分块参数对比汇总")
    print("=" * 72)
    print("配置           | 来源命中 | 证据命中 | 建库耗时 | 平均总检索 | 文本段数")
    for run in runs:
        report = run["report"]
        row = {
            "chunk_size": run["chunk_size"],
            "chunk_overlap": run["chunk_overlap"],
            "build_ms": run["build_ms"],
            "chunk_count": run["chunk_count"],
            "source_hit": report["source_hit"],
            "source_total": report["total"],
            "source_accuracy": report["source_accuracy"],
            "evidence_hit": report["evidence_hit"],
            "evidence_total": report["evidence_total"],
            "evidence_accuracy": report["evidence_accuracy"],
            "avg_total_retrieval_ms": report["timing_ms"]["total_retrieval"]["avg"],
        }
        summary.append(row)
        name = f"cs{row['chunk_size']}_ov{row['chunk_overlap']}"
        print(
            f"{name:<14} | "
            f"{row['source_hit']}/{row['source_total']} ({row['source_accuracy']:>5.1f}%) | "
            f"{row['evidence_hit']}/{row['evidence_total']} ({row['evidence_accuracy']:>5.1f}%) | "
            f"{row['build_ms']:>6.0f}ms | "
            f"{row['avg_total_retrieval_ms']:>8.0f}ms | "
            f"{row['chunk_count']}"
        )
    print("=" * 72)

    return {"summary": summary, "runs": runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="分块参数对比实验：扫描 chunk_size × chunk_overlap，测证据命中率"
    )
    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_QUESTIONS,
        help="题集 JSON 路径（默认 retrieval_k_tuning_50.json）",
    )
    parser.add_argument(
        "--pdf-folder", type=Path, default=PDF_FOLDER,
        help="法律 PDF 文件夹（默认 legal_pdfs/）",
    )
    parser.add_argument(
        "--split", choices=["development", "validation", "all"], default="development",
        help="评测的题集划分（默认 development；阶段B 用 validation）",
    )
    parser.add_argument(
        "--configs", default=None,
        help="要扫描的配置，逗号分隔 'size:overlap'，例如 '500:100,1000:200'；"
             "不填则用默认网格",
    )
    parser.add_argument(
        "--k", type=int, default=K_FIXED,
        help=f"FAISS 粗召回候选数（默认 {K_FIXED}，K 扫描测出的最优点）",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K_FIXED,
        help=f"Rerank 后保留的文档数（默认 {TOP_K_FIXED}）",
    )
    parser.add_argument(
        "--limit", type=int,
        help="只取筛选后的前 N 道题，用于小规模冒烟测试",
    )
    parser.add_argument(
        "--output", type=Path,
        help="报告输出路径；不写时自动保存到 evaluation/results/",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印每题 Top 文档来源和页码；默认只打印摘要",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数")

    configs = (
        parse_configs(args.configs)
        if args.configs else list(DEFAULT_CONFIGS)
    )

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = select_retrieval_questions(
        questions, split=args.split, limit=args.limit,
    )
    if not selected:
        parser.error(f"在 {args.questions} 中没有找到 split={args.split} 的法条检索题")

    temp_root = Path(tempfile.mkdtemp(prefix=f"chunk_sweep_{args.split}_"))
    print(f"临时向量库目录：{temp_root}（实验结束可删除，不影响生产库）")

    result = run_sweep(
        configs,
        selected,
        k=args.k,
        top_k=args.top_k,
        pdf_folder=args.pdf_folder,
        temp_root=temp_root,
        verbose=args.verbose,
    )

    default_name = (
        f"chunk_sweep_{args.split}_k{args.k}_top{args.top_k}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path = args.output or RESULTS_DIR / default_name
    save_json_report(output_path, {
        "timestamp": datetime.now().isoformat(),
        "mode": "chunk_sweep",
        "config": {
            "questions_path": str(args.questions),
            "pdf_folder": str(args.pdf_folder),
            "split": args.split,
            "limit": args.limit,
            "k": args.k,
            "top_k": args.top_k,
            "reranker_model": RERANKER_MODEL_NAME,
            "temp_dir": str(temp_root),
            "baseline": list(BASELINE),
            "k_note": "k=40 是 K 扫描在 development 上的最优点（与 k=50 持平，k=30 落后 2 题）",
        },
        "summary": result["summary"],
        "runs": result["runs"],
    })

    if args.split == "development":
        if result["summary"]:
            best = max(
                result["summary"],
                key=lambda row: (row["evidence_accuracy"], -row["build_ms"]),
            )
            print(f"\n阶段B 建议："
                  f"python -X utf8 evaluation/evaluate_chunk_sweep.py "
                  f"--split validation "
                  f"--configs \"{BASELINE[0]}:{BASELINE[1]},{best['chunk_size']}:{best['chunk_overlap']}\"")