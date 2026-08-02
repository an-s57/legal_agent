"""LangGraph ReAct 法律智能体"""
import json
import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from llm_client import llm
from tools.legal_tools import legal_rag_search, web_legal_search
from agent.hallucination_guard import check_hallucination, format_hallucination_warning

tools = [legal_rag_search, web_legal_search]


def _extract_json(text: str) -> dict | None:
    """从可能包含 markdown 代码块或多余文本的 LLM 响应中提取 JSON。

    依次尝试：直接解析 → 提取 ```json``` 代码块 → 提取第一个 {...} 块。
    全部失败时返回 None。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

# ── 上下文管理配置 ──────────────────────────────────────────────
# 后续可在此处添加 LLM 摘要压缩层，将超出窗口的旧消息压缩为结构化摘要。
MAX_HISTORY_TURNS = 12       # 传给 LLM 的最大对话轮数（1 轮 = 用户 + AI 各一条）
PLANNER_CONTEXT_TURNS = 3   # 传给 Planner 的最近对话轮数（理解多轮上下文）


def _trim_history(chat_history: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
    """保留最近 max_turns 轮对话，超出部分丢弃。

    原则：数据库存全量（不丢数据），Agent 层决定给 LLM 看多少（这里做截断）。
    丢弃的旧对话中的关键信息已由 case_summary 吸收。
    """
    max_messages = max_turns * 2
    if len(chat_history) <= max_messages:
        return list(chat_history)
    return list(chat_history[-max_messages:])


def _format_planner_context(all_messages: list, recent_turns: int = PLANNER_CONTEXT_TURNS) -> str:
    """从完整消息列表中提取最近 N 轮人机对话，格式化给 Planner。

    只取 HumanMessage 和 AIMessage，跳过 SystemMessage。
    这样 Planner 在追问后能理解用户的简短回复（如"昨天""3000块"）。
    """
    from langchain_core.messages import HumanMessage as HM, AIMessage as AIM

    dialogue = [m for m in all_messages if isinstance(m, (HM, AIM))]
    recent = dialogue[-(recent_turns * 2):]
    lines = []
    for msg in recent:
        role = "用户" if isinstance(msg, HM) else "助手"
        lines.append(f"{role}：{msg.content}")
    return "\n".join(lines) if lines else "（无对话历史）"


SYSTEM_PROMPT = """你是一个专业的AI法律助手。

⚠️ 最高优先级规则：回答法律问题前，你必须先调用工具检索法条。
即使你确定知道答案，也必须先用工具确认。严禁不经检索直接回答。

工具：
1. legal_rag_search — 检索法律文档库中的法条原文和案例
2. web_legal_search — 联网搜索最新法律法规和司法解释

行为守则：
- 看到法律问题 → 立即调用 legal_rag_search
- 问题含"新规/2025/2026/最新/最近"等词 → 同时调用 web_legal_search
- 每个工具只调用一次，不重复
- 收到检索结果后整合回答，标注法条来源
- 工具无结果时用自己的知识回答，末尾加"请注意核实"
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    info_complete: bool
    case_summary: str
    skip_planner: bool


llm_with_tools = llm.bind_tools(tools)

PLANNER_PROMPT = """你是一个法律咨询信息收集员。
你的任务是判断用户的消息属于哪种类型，并决定是否需要追问。

最近对话：
{recent_history}

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
    print(f"[DEBUG] call_planner skip_planner={state.get('skip_planner', False)}", flush=True)
    # 评测模式：跳过 Planner，直接放行进 ReAct
    if state.get("skip_planner", False):
        print("[DEBUG] Planner SKIPPED → info_complete=True", flush=True)
        return {"info_complete": True}

    last_message = state["messages"][-1]
    user_message = last_message.content
    case_summary = state.get("case_summary", "{}")
    recent_history = _format_planner_context(state["messages"])

    prompt = PLANNER_PROMPT.format(
        recent_history=recent_history,
        case_summary=case_summary,
        user_message=user_message,
    )
    _planner_start = time.perf_counter()
    response = llm.invoke(prompt)
    _planner_ms = (time.perf_counter() - _planner_start) * 1000
    print(f"[PERF] stage=planner duration_ms={_planner_ms:.0f}", flush=True)

    result = _extract_json(response.content)
    if result is None:
        print("[WARN] Planner JSON 解析失败，放行进入 ReAct", flush=True)
        return {"info_complete": True}
    info_complete = result.get("info_complete", True)
    follow_up = result.get("follow_up", "")

    if not info_complete and follow_up:
        return {
            "messages": [AIMessage(content=follow_up)],
            "info_complete": False,
        }
    else:
        return {"info_complete": True}


def call_model(state: AgentState):
    _llm_start = time.perf_counter()
    response = llm_with_tools.invoke(state["messages"])
    _llm_ms = (time.perf_counter() - _llm_start) * 1000
    _has_tc = bool(response.tool_calls)
    _tc_names = [t["name"] for t in (response.tool_calls or [])]
    _content_len = len(response.content) if response.content else 0
    print(f"[PERF] stage=llm duration_ms={_llm_ms:.0f} has_tool_calls={_has_tc} tc_names={_tc_names} content_len={_content_len}", flush=True)
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def create_legal_agent():
    graph = StateGraph(AgentState)
    graph.add_node("planner", call_planner)
    graph.add_node("llm", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: "llm" if state.get("info_complete", True) else END
    )
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("tools", "llm")
    return graph.compile()


_compiled_graph = create_legal_agent()


async def run_legal_agent(
    user_input: str,
    chat_history: list,
    case_summary: str = "",
    request_id: str = "",
    skip_planner: bool = False,
) -> dict:
    messages = _trim_history(chat_history)
    messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    if case_summary:
        messages.append(SystemMessage(content=f"当前案情摘要：{case_summary}"))
    # 动态检索提醒：紧贴用户问题，强化"必须先检索"指令
    _retrieval_reminder = "\n\n[系统指令：请先调用 legal_rag_search 检索法条原文，再根据检索结果回答。不要凭记忆直接回答。]"
    messages.append(HumanMessage(content=user_input + _retrieval_reminder))

    result = await _compiled_graph.ainvoke(
        {"messages": messages, "case_summary": case_summary, "skip_planner": skip_planner},
        config={"recursion_limit": 12},  # 限制最多 ~5 轮工具调用
    )

    final_messages = result["messages"]
    # DEBUG ─ 排查 tools_used 为空的问题
    _msg_types = [type(m).__name__ for m in final_messages]
    _ai_tcs = []
    for m in final_messages:
        if isinstance(m, AIMessage):
            _ai_tcs.append({
                "content_len": len(m.content) if m.content else 0,
                "tool_calls": [t["name"] for t in (m.tool_calls or [])],
            })
    print(f"[DEBUG] final_messages types={_msg_types}", flush=True)
    print(f"[DEBUG] AIMessage details={_ai_tcs}", flush=True)
    # END DEBUG
    ai_messages = [m for m in final_messages if isinstance(m, AIMessage)]
    answer = ai_messages[-1].content if ai_messages else "抱歉，无法生成回答"

    # ── 幻觉检测：提取工具返回的检索结果，验证回答中的引用 ──
    tool_results_text = ""
    for m in final_messages:
        if isinstance(m, ToolMessage):
            tool_results_text += m.content + "\n"
    if tool_results_text:
        check = check_hallucination(answer, tool_results_text)
        warning = format_hallucination_warning(check)
        if warning:
            answer = answer + warning

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
    user_input: str,
    chat_history: list,
    case_summary: str = "",
    request_id: str = "",
    skip_planner: bool = False,
):
    """流式版本 — 逐 token yield，格式 {"type": "token"|"planner_question"|"tool_start"|"tool_end"|"done", ...}"""
    messages = _trim_history(chat_history)
    messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    if case_summary:
        messages.append(SystemMessage(content=f"当前案情摘要：{case_summary}"))
    # 动态检索提醒：紧贴用户问题，强化"必须先检索"指令
    _retrieval_reminder = "\n\n[系统指令：请先调用 legal_rag_search 检索法条原文，再根据检索结果回答。不要凭记忆直接回答。]"
    messages.append(HumanMessage(content=user_input + _retrieval_reminder))

    state = {"messages": messages, "case_summary": case_summary, "skip_planner": skip_planner}

    planner_buf = ""
    tools_used = set()
    tool_outputs = []         # 收集工具返回结果，用于幻觉检测
    full_answer = ""          # 累积 LLM 回答文本
    llm_has_tool_calls = False
    llm_text_emitted = False

    async for event in _compiled_graph.astream_events(
        state, version="v2",
        config={"recursion_limit": 12},  # 限制最多 ~5 轮工具调用
    ):
        kind = event["event"]
        metadata = event.get("metadata", {})
        node = metadata.get("langgraph_node", "")

        if kind == "on_chat_model_start" and node == "llm":
            # 每次进入 LLM 节点都重新记录本轮是否已有流式文本。
            # 首轮可能只产生 tool_calls，工具执行后的下一轮才生成最终回答。
            llm_has_tool_calls = False
            llm_text_emitted = False

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = chunk.content if hasattr(chunk, "content") and chunk.content else ""

            if node == "planner":
                planner_buf += content

            elif node == "llm":
                tc = getattr(chunk, "tool_calls", None)
                if tc:
                    llm_has_tool_calls = True

                if not llm_has_tool_calls and content:
                    llm_text_emitted = True
                    full_answer += content
                    yield {"type": "token", "text": content}

        elif kind == "on_chat_model_end":
            if node == "planner":
                # invoke() 不会触发 on_chat_model_stream，planner_buf 可能为空。
                # 兜底：从 on_chat_model_end 的 output 取完整文本。
                if not planner_buf:
                    output = event.get("data", {}).get("output")
                    planner_buf = getattr(output, "content", "") if output else ""
                result = _extract_json(planner_buf)
                if result and not result.get("info_complete", True) and result.get("follow_up"):
                    yield {"type": "planner_question", "text": result["follow_up"]}
                    return
            elif node == "llm":
                # 当前节点用 invoke() 调模型时，部分模型适配器不会触发
                # on_chat_model_stream。此时在结束事件中兜底发送完整回答，
                # 避免前端只收到 done 而没有文本。
                output = event.get("data", {}).get("output")
                content = getattr(output, "content", "") if output else ""
                if not llm_has_tool_calls and not llm_text_emitted and content:
                    full_answer += content
                    yield {"type": "token", "text": content}

        elif kind == "on_tool_start":
            name = event.get("name", "")
            if name:
                tools_used.add(name)
                yield {"type": "tool_start", "name": name}

        elif kind == "on_tool_end":
            name = event.get("name", "")
            if name:
                # 收集工具输出用于幻觉检测
                output_data = event.get("data", {}).get("output")
                if output_data and hasattr(output_data, "content"):
                    tool_outputs.append(output_data.content)
                yield {"type": "tool_end", "name": name}

    # ── 幻觉检测：在流结束后、done 前运行 ──
    if tool_outputs and full_answer:
        tool_results_text = "\n".join(tool_outputs)
        check = check_hallucination(full_answer, tool_results_text)
        warning = format_hallucination_warning(check)
        if warning:
            yield {"type": "token", "text": warning}

    yield {"type": "done", "tools_used": list(tools_used)}
