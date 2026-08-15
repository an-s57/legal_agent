"""MCP server — 把法律检索工具暴露给任何 MCP 客户端（Claude Desktop / Cursor 等）。

复用 tools/legal_tools.py 里已有的两个 LangChain 工具，零重复实现，
工具语义、检索配置（k=40 / top_k=5）、联网回退策略与主项目完全一致。

通信走 stdio：协议消息（JSON-RPC）走 stdout，日志走 stderr——
这正是 logger.py 里"日志输出到 stderr"设计的原因，避免日志污染协议流。

启动：python mcp_server.py            # 默认 stdio 传输
      fastmcp dev mcp_server.py       # 启动网页调试器（MCP Inspector）
"""
from fastmcp import FastMCP

from logger import get_logger
from tools.legal_tools import legal_rag_search as _rag_search
from tools.legal_tools import web_legal_search as _web_search

logger = get_logger("legal_agent.mcp")

mcp = FastMCP("legal-agent")


@mcp.tool()
def legal_rag_search(query: str) -> str:
    """在法律文档中检索相关法条和案例。
    适合回答法律条文定义、法律概念解释、历史案例引用、某法条的具体规定。
    不适合：查询涉及最新动态、司法解释、时效性强的法律新闻。
    """
    return _rag_search.invoke({"query": query})


@mcp.tool()
def web_legal_search(query: str) -> str:
    """联网搜索最新的法律法规、司法解释和法律新闻。
    适合回答用户查询涉及最新动态、时效性强的内容。
    比如"新规"、"2025"、"2026"等时效性关键词的问题。
    优先使用 AnySearch，不可用时自动回退到 ddgs。
    """
    return _web_search.invoke({"query": query})


if __name__ == "__main__":
    # 默认 stdio 传输；需要远程调用时改为 mcp.run(transport="streamable-http")
    mcp.run()
