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
    info_complete:bool  #新增planner判断信息是否完整
#TypedDict规定了必须有什么，langgraph需要知道状态有哪些字段
#来管理和传递这些字段

llm_with_tools = llm.bind_tools(tools)


PLANNER_PROMPT="""你是一个法律咨询信息收集员。
你的任务是判断用户提供的信息是否足够给出法律建议。

已知的案情摘要：{case_summary}
用户最新消息：{user_message}

请检查以下维度：
1.event_description--事件描述：发生了什么事？
2.event_time--发生时间：什么时候发生的？
3.damages--后果：造成了什么损失？
4.user_claim --用户诉求：用户想要什么结果？

以上信息如果有确实，以JSON格式返回缺失字段和追问问题：
{{"info_complete":false,"missing_fields":["缺失的字段"],"follow_up":"追问的问题"}}

如果信息足够，返回：
{{"info_complete":true}}
"""

def call_planner(state:AgentState):
    #拿用户最新发的那条消息
    last_message=state["message"][-1]
    user_message=last_message.content

    case_summary="{}"

    prompt=PLANNER_PROMPT.format(
        case_summary=case_summary,
        user_message=user_message
    )
    response=llm.invoke(prompt)
    #调用一次
    import json
    try:
        result=json.loads(response.content.strip().strip("```json").strip("```").strip())
        info_complete=result.get("info_complete",True)
        follow_up=result.get("follow_up","")
    except Exception:
        info_complete=True#解析失败就放行，别卡住用户
        follow_up=""
    
    if not info_complete and follow_up:
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=follow_up)],
            "info_complete": False,
        }#不够，把追问作为AI消息返回
    else:
        # 信息够了，放行，继续react
        return {"info_complete": True}
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
    graph.add_node("planner",call_planner)
    graph.add_node("llm", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "planner")#起点到planner
    graph.add_conditional_edges(
        "planner",
        lambda state:"llm" if state.get("info_complete",True) else "__end__"
    )
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
