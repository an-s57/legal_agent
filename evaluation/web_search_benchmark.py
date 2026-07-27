"""比较当前 DuckDuckGo 接入、新版 ddgs 与 AnySearch 的法律网页搜索结果。

这是独立评测脚本：不调用 FastAPI、LangGraph、LLM、FAISS 或 Reranker，
也不会修改线上 Agent 的 web_legal_search 工具。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

try:
    # 保持与当前生产工具一致，评测的是现有 DuckDuckGo 接入的实际表现。
    from duckduckgo_search import DDGS as LegacyDDGS
except ImportError as exc:  # pragma: no cover - 给第一次运行时更清晰的提示
    raise SystemExit(
        "缺少 duckduckgo-search。请先在虚拟环境中执行：pip install -r requirements.txt"
    ) from exc

try:
    # 仅用于评测的新接入；线上 tools/legal_tools.py 暂不使用它。
    from ddgs import DDGS as ModernDDGS
except ImportError as exc:  # pragma: no cover - 给第一次运行时更清晰的提示
    raise SystemExit(
        "缺少 ddgs。请先在虚拟环境中执行：pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("web_search_benchmark_20.json")
RESULTS_DIR = Path(__file__).with_name("results")
ANYSEARCH_URL = "https://api.anysearch.com/v1/search"
DEFAULT_QUERY_SUFFIX = "法律法规 中国"
PROVIDER_LABELS = {
    "duckduckgo": "DuckDuckGo（旧接入）",
    "ddgs": "DuckDuckGo（ddgs）",
    "anysearch": "AnySearch",
}
DEFAULT_PROVIDERS = tuple(PROVIDER_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较当前 DuckDuckGo 接入、新版 ddgs 与 AnySearch 的法律网页搜索质量和耗时。"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="JSON 题集路径（默认：20 道联网搜索评测题）。",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="每个服务保留的结果数，默认 3；与当前 DuckDuckGo 工具一致。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="单次网络请求超时秒数，默认 15。",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.15,
        help="两次题目请求间的暂停秒数，默认 0.15，避免触发搜索服务限流。",
    )
    parser.add_argument(
        "--providers",
        "--provider",
        dest="providers",
        nargs="+",
        choices=DEFAULT_PROVIDERS,
        default=list(DEFAULT_PROVIDERS),
        help="要运行的服务；默认三者全跑。可写 --providers ddgs anysearch。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 题；首次建议使用 3 做冒烟测试。",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=6,
        help="Markdown 报告最多列出多少道建议人工复核的题，默认 6。",
    )
    parser.add_argument(
        "--no-suffix",
        action="store_true",
        help="不在题目后追加“法律法规 中国”；正式公平对比不要使用此选项。",
    )
    return parser.parse_args()


def load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    try:
        questions = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"找不到题集：{path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"题集 JSON 格式错误：{path}（{exc}）") from exc

    if not isinstance(questions, list) or not questions:
        raise SystemExit("题集必须是非空 JSON 数组。")

    required_fields = {
        "id",
        "category",
        "question",
        "expected_domains",
        "core_term_groups",
        "detail_term_groups",
    }
    for index, item in enumerate(questions, start=1):
        missing = required_fields - set(item)
        if missing:
            raise SystemExit(f"第 {index} 题缺少字段：{', '.join(sorted(missing))}")

    return questions[:limit] if limit is not None else questions


def build_query(question: str, suffix: str | None) -> str:
    return f"{question} {suffix}".strip() if suffix else question.strip()


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_domain(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def domain_matches(host: str, expected_domain: str) -> bool:
    expected = expected_domain.lower().removeprefix("www.")
    return host == expected or host.endswith(f".{expected}")


def timing_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0, "median": 0, "p95": 0}

    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "avg": round(sum(ordered) / len(ordered), 1),
        "median": round(ordered[len(ordered) // 2], 1),
        "p95": round(ordered[p95_index], 1),
    }


def result_text(result: dict[str, str]) -> str:
    return f"{result.get('title', '')} {result.get('snippet', '')}".lower()


def term_group_matches(text: str, alternatives: list[str]) -> bool:
    return any(term.lower() in text for term in alternatives)


def evaluate_results(question: dict[str, Any], results: list[dict[str, str]]) -> dict[str, Any]:
    """按预先标注的官方域名、核心主题和细节词给单个服务结果打分。

    搜索摘要往往只有法律全文开头，不能要求它一定显示某个深处的法条编号。
    因此“核心主题”用于自动通过判定，“细节词”只作为摘要覆盖诊断指标。
    """
    expected_domains = question["expected_domains"]
    matched_domains: list[str] = []

    for result in results:
        host = normalize_domain(result.get("url", ""))
        if host and any(domain_matches(host, expected) for expected in expected_domains):
            matched_domains.append(host)

    combined_text = " ".join(result_text(result) for result in results)
    core_groups = question["core_term_groups"]
    detail_groups = question["detail_term_groups"]
    core_group_matches = [term_group_matches(combined_text, group) for group in core_groups]
    detail_group_matches = [term_group_matches(combined_text, group) for group in detail_groups]
    core_coverage = (
        round(sum(core_group_matches) / len(core_groups) * 100, 1)
        if core_groups else 100.0
    )
    detail_coverage = (
        round(sum(detail_group_matches) / len(detail_groups) * 100, 1)
        if detail_groups else 100.0
    )
    official_hit = bool(matched_domains)
    core_topic_hit = all(core_group_matches)

    return {
        "official_source_hit_at_n": official_hit,
        "matched_official_domains": sorted(set(matched_domains)),
        "core_term_group_matches": core_group_matches,
        "core_topic_coverage_percent": core_coverage,
        "core_topic_hit_at_n": core_topic_hit,
        "detail_term_group_matches": detail_group_matches,
        "detail_preview_coverage_percent": detail_coverage,
        "automatic_pass_at_n": official_hit and core_topic_hit,
    }


def search_duckduckgo(query: str, top_n: int, timeout: float) -> dict[str, Any]:
    """调用与线上工具相同的旧 DuckDuckGo 接入，并统一结果字段。"""
    start = time.perf_counter()
    try:
        # 旧包会发出已改名提示；报告中的“旧接入”标签仍保留这一基线的含义。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with LegacyDDGS(timeout=timeout) as ddgs:
                raw_results = list(ddgs.text(query, max_results=top_n))

        results = [
            {
                "title": safe_text(item.get("title")),
                "url": safe_text(item.get("href")),
                "snippet": safe_text(item.get("body")),
            }
            for item in raw_results[:top_n]
        ]
        return {
            "success": bool(results),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            "error": None if results else "empty_result",
            "provider_reported_search_time_ms": None,
            "results": results,
        }
    except Exception as exc:  # 任何网络异常都记录为结果，而不是中断整轮评测
        return {
            "success": False,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "provider_reported_search_time_ms": None,
            "results": [],
        }


def search_ddgs(query: str, top_n: int, timeout: float) -> dict[str, Any]:
    """调用新版 ddgs；只用于独立评测，不改变生产 Agent。"""
    start = time.perf_counter()
    try:
        with ModernDDGS(timeout=timeout) as ddgs:
            raw_results = list(ddgs.text(query, max_results=top_n))

        results = [
            {
                "title": safe_text(item.get("title")),
                "url": safe_text(item.get("href") or item.get("url")),
                "snippet": safe_text(
                    item.get("body") or item.get("snippet") or item.get("description")
                ),
            }
            for item in raw_results[:top_n]
        ]
        return {
            "success": bool(results),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            "error": None if results else "empty_result",
            "provider_reported_search_time_ms": None,
            "results": results,
        }
    except Exception as exc:
        return {
            "success": False,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "provider_reported_search_time_ms": None,
            "results": [],
        }


def search_anysearch(
    client: httpx.Client,
    query: str,
    top_n: int,
) -> dict[str, Any]:
    """调用 AnySearch REST API，并统一为 title/url/snippet 格式。"""
    start = time.perf_counter()
    try:
        response = client.post(
            ANYSEARCH_URL,
            json={
                "query": query,
                "max_results": top_n,
                "language": "zh-CN",
                "zone": "cn",
            },
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        results = [
            {
                "title": safe_text(item.get("title")),
                "url": safe_text(item.get("url")),
                "snippet": safe_text(
                    item.get("snippet") or item.get("content") or item.get("description")
                ),
            }
            for item in raw_results[:top_n]
            if isinstance(item, dict)
        ]
        return {
            "success": bool(results),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            "error": None if results else "empty_result",
            "provider_reported_search_time_ms": metadata.get("search_time_ms"),
            "results": results,
        }
    except Exception as exc:  # 认证、限流和网络错误均应出现在报告中
        return {
            "success": False,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "provider_reported_search_time_ms": None,
            "results": [],
        }


def add_metrics(question: dict[str, Any], provider_result: dict[str, Any]) -> dict[str, Any]:
    provider_result = dict(provider_result)
    provider_result["metrics"] = evaluate_results(question, provider_result["results"])
    return provider_result


def provider_summary(details: list[dict[str, Any]], provider: str) -> dict[str, Any]:
    provider_details = [detail[provider] for detail in details if provider in detail]
    total = len(provider_details)
    successful = [item for item in provider_details if item["success"]]
    official_hits = sum(item["metrics"]["official_source_hit_at_n"] for item in provider_details)
    core_topic_hits = sum(item["metrics"]["core_topic_hit_at_n"] for item in provider_details)
    automatic_passes = sum(item["metrics"]["automatic_pass_at_n"] for item in provider_details)

    return {
        "total": total,
        "success_count": len(successful),
        "success_rate_percent": round(len(successful) / total * 100, 1) if total else 0,
        "official_source_hit_count": official_hits,
        "official_source_hit_rate_percent": round(official_hits / total * 100, 1) if total else 0,
        "core_topic_hit_count": core_topic_hits,
        "core_topic_hit_rate_percent": round(core_topic_hits / total * 100, 1) if total else 0,
        "detail_preview_coverage_percent": round(
            sum(item["metrics"]["detail_preview_coverage_percent"] for item in provider_details) / total,
            1,
        ) if total else 0,
        "automatic_pass_count": automatic_passes,
        "automatic_pass_rate_percent": round(automatic_passes / total * 100, 1) if total else 0,
        "elapsed_ms": timing_stats([item["elapsed_ms"] for item in provider_details]),
    }


def select_review_items(
    details: list[dict[str, Any]],
    providers: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """只挑出真正有判别价值的题，避免人工逐条看全部搜索结果。"""
    candidates: list[tuple[int, dict[str, Any], str]] = []
    for detail in details:
        available = [provider for provider in providers if provider in detail]
        if len(available) < 2:
            continue

        passes = [detail[provider]["metrics"]["automatic_pass_at_n"] for provider in available]
        successes = [detail[provider]["success"] for provider in available]
        if len(set(passes)) > 1:
            candidates.append((0, detail, "不同服务的自动通过结果不一致"))
        elif len(set(successes)) > 1:
            candidates.append((1, detail, "不同服务的请求成功状态不一致"))
        elif not any(passes):
            candidates.append((2, detail, "所有服务均未自动通过"))

    candidates.sort(key=lambda item: (item[0], item[1]["id"]))
    return [
        {
            "id": detail["id"],
            "question": detail["question"],
            "reason": reason,
            "top_urls": {
                provider: top_url(detail.get(provider))
                for provider in providers
            },
        }
        for _, detail, reason in candidates[:limit]
    ]


def top_url(provider_result: dict[str, Any] | None) -> str:
    if not provider_result:
        return ""
    results = provider_result.get("results", [])
    return safe_text(results[0].get("url")) if results else ""


def percent_text(numerator: int, denominator: int) -> str:
    percent = round(numerator / denominator * 100, 1) if denominator else 0
    return f"{numerator}/{denominator} ({percent}%)"


def build_markdown_report(run: dict[str, Any]) -> str:
    lines = [
        "# 联网法律搜索服务对比报告",
        "",
        f"- 运行时间：{run['run_at']}",
        f"- 题集：`{run['questions_path']}`（{run['question_count']} 题）",
        f"- 统一查询后缀：`{run['query_suffix'] or '无'}`",
        f"- 每题返回数：Top-{run['top_n']}",
        "- 边界：本报告只比较网页搜索服务；不经过 Agent、LLM、FAISS 或 Reranker。",
        "",
        "## 自动指标汇总",
        "",
        "| 服务 | 成功率 | 官方来源命中@N | 核心主题命中@N | 自动通过@N | 细节摘要覆盖（平均） | 平均 / 中位数 / P95 耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for provider in run["providers"]:
        label = PROVIDER_LABELS[provider]
        summary = run["summary"].get(provider)
        if not summary:
            continue
        total = summary["total"]
        timing = summary["elapsed_ms"]
        lines.append(
            f"| {label} | "
            f"{percent_text(summary['success_count'], total)} | "
            f"{percent_text(summary['official_source_hit_count'], total)} | "
            f"{percent_text(summary['core_topic_hit_count'], total)} | "
            f"{percent_text(summary['automatic_pass_count'], total)} | "
            f"{summary['detail_preview_coverage_percent']:.1f}% | "
            f"{timing['avg']:.0f} / {timing['median']:.0f} / {timing['p95']:.0f} ms |"
        )

    lines.extend([
        "",
        "## 指标如何理解",
        "",
        "- **官方来源命中@N**：Top-N 中至少有一个链接属于该题预标注的官方域名。",
        "- **核心主题命中@N**：Top-N 的标题或摘要覆盖该题的核心法律名称、办事渠道或纠纷主题。",
        "- **细节摘要覆盖**：条文编号、金额、地域、具体事实等是否出现在搜索摘要中；只作诊断，不作为通过门槛。",
        "- **自动通过@N**：同时满足官方来源命中与核心主题命中；它是可复核的代理指标，不等同于法律意见正确。",
        "",
        "## 建议人工复核（仅看有分歧或双方失败的题）",
        "",
    ])

    reviews = run["manual_review_candidates"]
    if not reviews:
        lines.append("本轮没有产生需要优先人工复核的题。仍建议随机抽查 2 题，确认官方域名与内容相关性。")
    else:
        lines.extend([
            "| 题号 | 原因 | " + " | ".join(PROVIDER_LABELS[p] + " 首条链接" for p in run["providers"]) + " |",
            "| --- | --- | " + " | ".join("---" for _ in run["providers"]) + " |",
        ])
        for item in reviews:
            urls = " | ".join(item["top_urls"].get(provider, "") for provider in run["providers"])
            lines.append(
                f"| {item['id']} | {item['reason']} | {urls} |"
            )

    lines.extend([
        "",
        "## 结论填写模板",
        "",
        "先根据自动通过率、失败率和 P95 耗时选出候选服务；再查看上表中少量分歧题。`duckduckgo-search` 仅作为当前生产接入基线；若新版 ddgs 仍无明显改善，而 AnySearch 显著更稳定，再考虑替换生产联网工具。",
        "",
    ])
    return "\n".join(lines)


def write_results(run: dict[str, Any]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"web_search_benchmark_{stamp}.json"
    markdown_path = RESULTS_DIR / f"web_search_benchmark_{stamp}.md"
    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown_report(run), encoding="utf-8")
    return json_path, markdown_path


def print_summary(run: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("联网法律搜索服务对比汇总")
    print("=" * 72)
    for provider in run["providers"]:
        label = PROVIDER_LABELS[provider]
        summary = run["summary"].get(provider)
        if not summary:
            continue
        total = summary["total"]
        timing = summary["elapsed_ms"]
        print(
            f"{label}: 成功 {percent_text(summary['success_count'], total)} | "
            f"官方来源 {percent_text(summary['official_source_hit_count'], total)} | "
            f"核心主题 {percent_text(summary['core_topic_hit_count'], total)} | "
            f"自动通过 {percent_text(summary['automatic_pass_count'], total)} | "
            f"细节摘要覆盖 {summary['detail_preview_coverage_percent']:.1f}% | "
            f"耗时 avg/median/P95={timing['avg']:.0f}/{timing['median']:.0f}/{timing['p95']:.0f}ms"
        )


def main() -> None:
    args = parse_args()
    if not 1 <= args.top_n <= 20:
        raise SystemExit("--top-n 必须在 1 到 20 之间。")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须大于 0。")

    load_dotenv(PROJECT_ROOT / ".env")
    questions = load_questions(args.questions, args.limit)
    suffix = None if args.no_suffix else DEFAULT_QUERY_SUFFIX

    anysearch_key = os.getenv("ANYSEARCH_API_KEY")
    providers = list(args.providers)
    if "anysearch" in providers and not anysearch_key:
        raise SystemExit(
            "未读取到 ANYSEARCH_API_KEY。请确认项目根目录 .env 中已配置该变量。"
        )

    print("=" * 72)
    print(f"联网法律搜索对比：{len(questions)} 题，Top-{args.top_n}，服务：{', '.join(providers)}")
    print("说明：每题使用相同查询；不调用 Agent、LLM、FAISS 或 Reranker。")
    print("=" * 72)

    details: list[dict[str, Any]] = []
    anysearch_client = (
        httpx.Client(
            timeout=args.timeout,
            headers={
                "Authorization": f"Bearer {anysearch_key}",
                "Content-Type": "application/json",
            },
        )
        if "anysearch" in providers
        else None
    )

    try:
        for index, question in enumerate(questions, start=1):
            query = build_query(question["question"], suffix)
            detail: dict[str, Any] = {
                "id": question["id"],
                "category": question["category"],
                "question": question["question"],
                "query": query,
                "expected_domains": question["expected_domains"],
                "core_term_groups": question["core_term_groups"],
                "detail_term_groups": question["detail_term_groups"],
                "why": question.get("why", ""),
            }

            if "duckduckgo" in providers:
                detail["duckduckgo"] = add_metrics(
                    question,
                    search_duckduckgo(query, args.top_n, args.timeout),
                )
            if "ddgs" in providers:
                detail["ddgs"] = add_metrics(
                    question,
                    search_ddgs(query, args.top_n, args.timeout),
                )
            if "anysearch" in providers and anysearch_client is not None:
                detail["anysearch"] = add_metrics(
                    question,
                    search_anysearch(anysearch_client, query, args.top_n),
                )

            details.append(detail)
            status_text = []
            for provider in providers:
                result = detail[provider]
                status_text.append(
                    f"{provider}={'PASS' if result['metrics']['automatic_pass_at_n'] else 'CHECK'} "
                    f"{result['elapsed_ms']:.0f}ms"
                )
            print(f"[{index:02d}/{len(questions):02d}] {question['id']}: {' | '.join(status_text)}")

            if index < len(questions) and args.pause_seconds > 0:
                time.sleep(args.pause_seconds)
    finally:
        if anysearch_client is not None:
            anysearch_client.close()

    run = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "questions_path": str(args.questions),
        "question_count": len(questions),
        "top_n": args.top_n,
        "query_suffix": suffix,
        "providers": providers,
        "summary": {provider: provider_summary(details, provider) for provider in providers},
        "manual_review_candidates": select_review_items(details, providers, args.review_limit),
        "details": details,
    }
    json_path, markdown_path = write_results(run)
    print_summary(run)
    print(f"\n原始结果：{json_path}")
    print(f"简短报告：{markdown_path}")


if __name__ == "__main__":
    main()
