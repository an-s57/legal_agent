"""法律工具集 — RAG 检索 + 联网搜索"""
from langchain.tools import tool

from rag.retriever import retrieve_legal_docs


@tool
def legal_rag_search(query: str) -> str:
    """
    在法律文档中检索相关法条和案例。
    适合回答法律条文定义、法律概念解释、历史案例引用、某法条的具体规定。
    不适合：查询涉及最新动态、司法解释、时效性强的法律新闻。
    """
    results = retrieve_legal_docs(query)
    if not results:
        return "法律文档库中未找到相关内容"
    return "\n\n---\n\n".join(results)


@tool
def web_legal_search(query: str) -> str:
    """
    联网搜索最新的法律法规、司法解释和法律新闻。
    适合回答用户查询涉及最新动态、时效性强的内容。
    比如"新规"、"2024"、"2025"等时效性关键词的问题。
    """
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(query + " 法律法规 中国", max_results=3)
            )
        if not results:
            return "联网搜索未找到相关结果"
        formatted = []
        for r in results:
            formatted.append(
                f"【{r['title']}】\n{r['body']}\n链接：{r['href']}"
            )
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"搜索失败：{str(e)}"
