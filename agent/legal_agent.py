# legal_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from tools.legal_tools import legal_rag_search, web_legal_search
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.getenv("GLM_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

tools = [legal_rag_search, web_legal_search]

SYSTEM_PROMPT = """你是一个专业的AI法律助手，你有以下工具可以使用：

1. **legal_rag_search** — 在法律文档库中搜索法条原文和案例
2. **web_legal_search** — 联网搜索最新法律法规和司法解释

回答规则：
- 如果用户问的是法律条文定义、概念解释等，优先用 legal_rag_search 查文档库。
- 如果涉及最新动态、司法解释（如"新规"、"2024"、"2025"），用 web_legal_search。
- 把文档库搜索结果和联网搜索结果整合后回答，并标注来源。
- 如果工具没有返回结果，再根据自己的知识回答，但必须标注"请注意核实"。
"""

def create_legal_agent():
    """创建法律助手Agent（LangChain 1.3 新版API，基于LangGraph）"""
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return graph

async def run_legal_agent(user_input: str, chat_history: list, case_summary: str = "") -> dict:
    graph = create_legal_agent()

    # 构建消息列表
    messages = list(chat_history)
    if case_summary:
        messages.append(SystemMessage(content=f"当前案情摘要：{case_summary}"))
    messages.append(HumanMessage(content=user_input))

    # 新版API返回的是 LangGraph 的最终状态
    result = await graph.ainvoke({"messages": messages})

    # 提取最后一条AI消息作为回答
    final_messages = result["messages"]
    ai_messages = [m for m in final_messages if isinstance(m, AIMessage)]
    answer = ai_messages[-1].content if ai_messages else "抱歉，无法生成回答"

    # 提取用过的工具名（兼容main.py的tools_used功能）
    tool_calls_seen = set()
    for m in final_messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls_seen.add(tc["name"])

    return {
        "output": answer,
        "intermediate_steps": [(tc, None) for tc in tool_calls_seen],
    }
