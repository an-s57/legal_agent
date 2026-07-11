"""LangGraph ReAct 法律智能体"""
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.legal_tools import legal_rag_search, web_legal_search

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.getenv("GLM_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

tools = [legal_rag_search, web_legal_search]

SYSTEM_PROMPT = """重要规则：每次回答用户的法律问题前，你都必须先调用 legal_rag_search 进行检索，
确认法条原文后再回答。禁止凭记忆直接回答。 
强制规则：每次回答法律问题前，必须先调用至少一个工具进行检索！禁止凭记忆直接回答法律条文内容。
你是一个专业的AI法律助手，你有以下工具可以使用：

1. **legal_rag_search** — 在法律文档库中搜索法条原文和案例
2. **web_legal_search** — 联网搜索最新法律法规和司法解释

回答规则：
- 如果用户问的是法律条文定义、概念解释等，优先用 legal_rag_search 查文档库。
- 如果涉及最新动态、司法解释（如"新规"、"2026"、"2025"），用 web_legal_search。
- 把文档库搜索结果和联网搜索结果整合后回答，并标注来源。
- 如果工具没有返回结果，再根据自己的知识回答，但必须标注"请注意核实"。
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm_with_tools = llm.bind_tools(tools)


def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"


def create_legal_agent():
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("tools", "llm")
    return graph.compile()


_compiled_graph = create_legal_agent()


async def run_legal_agent(
    user_input: str, chat_history: list, case_summary: str = ""
) -> dict:
    messages = list(chat_history)
    messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    if case_summary:
        messages.append(SystemMessage(content=f"当前案情摘要：{case_summary}"))
    messages.append(HumanMessage(content=user_input))

    result = await _compiled_graph.ainvoke({"messages": messages})

    final_messages = result["messages"]
    ai_messages = [m for m in final_messages if isinstance(m, AIMessage)]
    answer = ai_messages[-1].content if ai_messages else "抱歉，无法生成回答"

    tool_calls_seen = set()
    for m in final_messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls_seen.add(tc["name"])

    return {
        "output": answer,
        "intermediate_steps": [(tc, None) for tc in tool_calls_seen],
    }
