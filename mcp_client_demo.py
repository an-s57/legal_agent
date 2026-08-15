"""最小 MCP 客户端 demo — 用官方 mcp SDK 连接自己的 server 并调用工具。

用途：
- 演示 MCP 客户端到底做了什么（三步：握手 initialize → 拿工具清单 list_tools → 调用 call_tool）
- 不依赖任何商业客户端（Claude Desktop 等），无账号也能跑
- 以后写 agent 应用时，"agent 去调用 MCP 工具"就是这段代码的模板

用法（WSL 里，Ollama 需在运行）：
    python mcp_client_demo.py
    python mcp_client_demo.py /path/to/venv/bin/python /path/to/mcp_server.py

注意：mcp_server.py 用的是 stdio 传输，客户端在本机启动 server 进程并通过
stdin/stdout 通信——日志走 stderr，协议消息走 stdout，互不干扰。
"""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 默认值按 WSL 环境写；Windows 上跑时用命令行参数覆盖
DEFAULT_COMMAND = "/home/an/legal_agent/venv/bin/python"
DEFAULT_SERVER = "/home/an/legal_agent/mcp_server.py"

QUERY = "消费者权益保护法第五十五条"


async def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COMMAND
    server_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SERVER

    server_params = StdioServerParameters(
        command=command,
        args=[server_path],
    )

    print(f"[1/3] 启动 server: {command} {server_path}")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 第一步：握手（initialize）——互相介绍、确认协议版本
            init = await session.initialize()
            print(f"[2/3] 握手完成: server={init.serverInfo.name} v{init.serverInfo.version}")

            # 第二步：拿工具清单（list_tools）——问"你有什么工具？"
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"[3/3] 可用工具: {names}")

            # 第三步：调用工具（call_tool）——"调用 legal_rag_search，参数是…"
            print(f"\n调用 legal_rag_search，query='{QUERY}' ...")
            result = await session.call_tool("legal_rag_search", {"query": QUERY})

            for block in result.content:
                if block.type == "text":
                    text = block.text
                    print(f"\n===== 检索结果（前 500 字）=====\n{text[:500]}")
                else:
                    print(f"\n(非文本内容块: {block.type})")


if __name__ == "__main__":
    asyncio.run(main())
