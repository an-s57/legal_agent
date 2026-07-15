"""LangGraph ReAct 法律智能体"""
import os
import json
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
    model="glm-4.7",
    openai_api_key=os.getenv("GLM_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

tools = [legal_rag_search, web_legal_search]

SYSTEM_PROMPT = """你是一个专业的AI法律助手。你可以使用以下工具：

1. **legal_rag_search** — 在法律文档库中搜索法条原文和案例
2. **web_legal_search** — 联网搜索最新法律法规和司法解释

回答规则：
- 回答法律问题前必须先调用 legal_rag_search 确认法条原文，禁止凭记忆直接回答。
- 如果涉及最新动态、司法解释（如"新规"、"2026"、"2025"），同时调用 web_legal_search。
- 尽量一次性调用所需的所有工具（并行调用），不要分多次逐一调用。
- 收到工具返回结果后，直接整合回答，不要再反复调用工具。
- 整合检索结果后回答，并标注来源。
- 如果工具没有返回结果，根据自己的知识回答，但必须标注"请注意核实"。
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    info_complete: bool
    case_summary: str


llm_with_tools = llm.bind_tools(tools)

PLANNER_PROMPT = """你是一个法律咨询信息收集员。
你的任务是判断用户的消息属于哪种类型，并决定是否需要追问。

已知的案情摘要：{case_summary}
用户最新消息：{user_message}

第一步：判断消息类型
- 如果是打招呼、闲聊、感谢、简单提问（如"你好""谢谢""你是谁""你能做什么"），直接放行：
{{"info_complete": true}}

第二步：如果是法律咨询，检查以下维度
1. event_description — 事件描述：发生了什么事？
2. event_time — 发生时间：什么时候发生的？
3. damages — 损失/后果：造成了什么损失？
4. user_claim — 用户诉求：用户想要什么结果？

判断原则：只要用户大致提到了某个维度（哪怕不详细），就算该维度已具备。
只有完全没提到某个维度时才算缺失。
例如"上个月在淘宝买了假货花了3000块要求退款"——四个维度都有，应判为完整。

如果信息有缺失，返回：
{{"info_complete": false, "missing_fields": ["缺失的字段"], "follow_up": "追问的问题，语气自然友好，像律师一样"}}

如果信息足够，返回：
{{"info_complete": true}}
"""


def call_planner(state: AgentState):
    last_message = state["messages"][-1]
    user_message = last_message.content

    case_summary = state.get("case_summary", "{}")

    prompt = PLANNER_PROMPT.format(
        case_summary=case_summary,
        user_message=user_message,
    )
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content.strip().strip("```json").strip("```").strip())
        info_complete = result.get("info_complete", True)
        follow_up = result.get("follow_up", "")
    except Exception:
        info_complete = True
        follow_up = ""

    if not info_complete and follow_up:
        return {
            "messages": [AIMessage(content=follow_up)],
            "info_complete": False,
        }
    else:
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
    graph.add_node("planner", call_planner)
    graph.add_node("llm", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: "llm" if state.get("info_complete", True) else "__end__"
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

    result = await _compiled_graph.ainvoke(
        {"messages": messages, "case_summary": case_summary},
        config={"recursion_limit": 12},  # 限制最多 ~5 轮工具调用
    )

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


async def run_legal_agent_stream(
    user_input: str, chat_history: list, case_summary: str = ""
):
    """流式版本 — 逐 token yield，格式 {"type": "token"|"planner_question"|"tool_start"|"tool_end"|"done", ...}"""
    messages = list(chat_history)
    messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    if case_summary:
        messages.append(SystemMessage(content=f"当前案情摘要：{case_summary}"))
    messages.append(HumanMessage(content=user_input))

    state = {"messages": messages, "case_summary": case_summary}

    planner_buf = ""
    tools_used = set()
    llm_has_tool_calls = False

    async for event in _compiled_graph.astream_events(
        state, version="v2",
        config={"recursion_limit": 12},  # 限制最多 ~5 轮工具调用
    ):
        kind = event["event"]
        metadata = event.get("metadata", {})
        node = metadata.get("langgraph_node", "")

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = chunk.content if hasattr(chunk, "content") and chunk.content else ""

            if node == "planner":
                planner_buf += content

            elif node == "llm":
                tc = getattr(chunk, "tool_calls", None)
                if tc:
                    llm_has_tool_calls = True

                if not llm_has_tool_calls and content:
                    yield {"type": "token", "text": content}

        elif kind == "on_chat_model_end":
            if node == "planner":
                try:
                    clean = planner_buf.strip().strip("```json").strip("```").strip()
                    result = json.loads(clean)
                    if not result.get("info_complete", True) and result.get("follow_up"):
                        yield {"type": "planner_question", "text": result["follow_up"]}
                        return
                except Exception:
                    pass
            elif node == "llm":
                llm_has_tool_calls = False

        elif kind == "on_tool_start":
            name = event.get("name", "")
            if name:
                tools_used.add(name)
                yield {"type": "tool_start", "name": name}

        elif kind == "on_tool_end":
            name = event.get("name", "")
            if name:
                yield {"type": "tool_end", "name": name}

    yield {"type": "done", "tools_used": list(tools_used)}
