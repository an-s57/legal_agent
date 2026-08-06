"""法律工具集 — RAG 检索 + 联网搜索"""
import os
import time

import httpx
from langchain.tools import tool

from rag.retriever import retrieve_legal_docs

ANYSEARCH_URL = "https://api.anysearch.com/v1/search"
WEB_SEARCH_SUFFIX = "法律法规 中国"
WEB_SEARCH_TOP_N = 3
WEB_SEARCH_TIMEOUT_SECONDS = 10.0

@tool
def legal_rag_search(query: str) -> str:
    """
    在法律文档中检索相关法条和案例。
    适合回答法律条文定义、法律概念解释、历史案例引用、某法条的具体规定。
    不适合：查询涉及最新动态、司法解释、时效性强的法律新闻。
    """
    # 基于开发集调参与验证集确认后的线上检索配置：K=40，Top-K=5。
    # 注意：k 会作为 k_vector 传给 hybrid_candidates（FAISS 召回数），
    # K 扫描定案 k=40 是"最小安全候选池"，不能砍小。
    results = retrieve_legal_docs(query, k=40, top_k=5)
    if not results:
        return "法律文档库中未找到相关内容"
    return "\n\n---\n\n".join(results)

# 联网搜索策略：AnySearch 主搜索；缺少密钥、请求异常或结果异常时回退到 ddgs。
def _request_anysearch_payload(query: str) -> object:
    """调用 AnySearch，返回未经解析的 JSON payload。"""
    api_key = os.getenv("ANYSEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 ANYSEARCH_API_KEY")

    search_query = f"{query} {WEB_SEARCH_SUFFIX}"
    start = time.perf_counter()

    with httpx.Client(timeout=WEB_SEARCH_TIMEOUT_SECONDS) as client:
        response = client.post(
            ANYSEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": search_query,
                "max_results": WEB_SEARCH_TOP_N,
                "language": "zh-CN",
                "zone": "cn",
            },
        )
        response.raise_for_status()
        payload = response.json()

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[WEB] provider=anysearch status=ok duration_ms={elapsed_ms:.0f}")
    return payload


def _format_anysearch_results(payload: object) -> list[dict[str, str]]:
    """把 AnySearch 返回的 JSON 转换成项目统一的搜索结果格式。"""
    if isinstance(payload, dict):
        data = payload.get("data", payload)
    else:
        data = {}

    if isinstance(data, dict):
        raw_results = data.get("results", [])
    else:
        raw_results = []

    results = []
    for item in raw_results[:WEB_SEARCH_TOP_N]:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        body_value = (
            item.get("snippet")
            or item.get("content")
            or item.get("description")
            or ""
        )
        body = str(body_value).strip()
        href = str(item.get("url", "")).strip()

        if not href:
            continue

        results.append(
            {
                "title": title,
                "body": body,
                "href": href,
            }
        )

    return results


def _search_with_anysearch(query: str) -> list[dict[str, str]]:
    """调用 AnySearch，并返回项目统一格式的搜索结果。"""
    payload = _request_anysearch_payload(query)
    results = _format_anysearch_results(payload)

    if not results:
        raise RuntimeError("AnySearch 未返回有效搜索结果")

    return results


def _search_with_ddgs(query: str) -> list[dict[str, str]]:
    """调用 ddgs，作为 AnySearch 失败时的备用联网搜索。"""
    from ddgs import DDGS

    search_query = f"{query} {WEB_SEARCH_SUFFIX}"
    start = time.perf_counter()

    with DDGS(timeout=WEB_SEARCH_TIMEOUT_SECONDS) as ddgs:
        raw_results = list(
            ddgs.text(
                search_query,
                max_results=WEB_SEARCH_TOP_N,
            )
        )

    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        body = str(item.get("body", "")).strip()
        href = str(item.get("href", "")).strip()

        if not href:
            continue

        results.append(
            {
                "title": title,
                "body": body,
                "href": href,
            }
        )

    if not results:
        raise RuntimeError("ddgs 未返回有效搜索结果")

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(
        f"[WEB] provider=ddgs status=ok "
        f"duration_ms={elapsed_ms:.0f}"
    )
    return results


@tool
def web_legal_search(query: str) -> str:
    """
    联网搜索最新的法律法规、司法解释和法律新闻。
    适合回答用户查询涉及最新动态、时效性强的内容。
    比如"新规"、"2024"、"2025"等时效性关键词的问题。
    优先使用 AnySearch,AnySearch 不可用时回退到 ddgs。
    """
    try:
        results=_search_with_anysearch(query)
    except (httpx.HTTPError,RuntimeError,ValueError) as anysearch_error:
        print(
            f"[WEB] provider=anysearch status=fallback "
            f"reason={type(anysearch_error).__name__}"
        )
        try:
            results = _search_with_ddgs(query)
        except Exception as ddgs_error:
            print(f"[WEB] provider=ddgs status=error "
                  f"reason={type(ddgs_error).__name__}")

            return "联网搜索服务暂时不可用，请稍后重试或先参考本地法律知识库。"

    formatted = []
    for item in results:
        formatted.append(
            f"【{item['title']}】\n{item['body']}\n链接：{item['href']}"
        )

    return "\n\n---\n\n".join(formatted)
