"""在线合规复核 Agent — 用 GLM 判卷模型复核主 Agent 回答的事实一致性。

设计来源：事实性评测（LLM-as-Judge，GLM 判 DeepSeek）的在线化——
把评测时"离线判卷"的同一套判定逻辑搬进生产链路，在主 Agent 生成回答后、
输出给用户前做一次合规复核（异构模型，避免自评偏好）。

与幻觉守卫（agent/hallucination_guard.py）的分工：
- 幻觉守卫：零 LLM 调用的两层规则校验（引用存在性 + 字符覆盖度），兜底快但粗糙；
- 合规复核：一次 GLM 调用做语义级事实一致性判定，慢但能抓"规则抓不到"的编造。

fail-open：GLM 未配置 key、调用失败或解析失败时返回 verdict=None，不影响主流程。
开关：环境变量 REVIEW_AGENT_ENABLED（默认开启，设 0/false/no 关闭）。
"""
import json
import os
import re

from langchain_core.messages import SystemMessage, HumanMessage

from llm_client import get_judge_llm

REVIEW_ENABLED = os.getenv("REVIEW_AGENT_ENABLED", "1").lower() not in ("0", "false", "no")
MAX_EVIDENCE_CHARS = 6000  # 证据文本上限，防止超出窗口

REVIEW_SYSTEM = (
    "你是法律回答复核员。只依据给定的检索证据（法条原文）判断助手回答的事实正确性，"
    "不做额外引申，不要求措辞一致。每次只输出一个 JSON。"
)

REVIEW_PROMPT = """用户问题：{question}

检索到的法律依据（法条原文摘录）：
{evidence}

助手回答：
{answer}

复核规则：
1. 核心：助手回答的法律结论是否与检索证据规定的规则在事实上一致（允许措辞不同，要求事实要点一致）。
2. 结论与证据相反或明显错误 → 判 0。
3. 编造证据中不存在的法条/引用（如证据没有第 55 条却声称"第 55 条规定"）→ 判 0。
4. 漏掉该问题应回答的关键规则 → 判 0。
5. 结论正确但未引用法条 → 判 1。
6. 与问题无关、只说"信息不足无法回答"、或只是复述题目 → 判 0。

只输出一个 JSON，不要输出任何其他文字：
{{"verdict": 1, "reason": "不超过30字的一句话理由"}}
verdict: 1 = 事实正确，0 = 事实错误"""


def _extract_json(text: str) -> dict | None:
    """从可能包含 markdown 代码块或多余文本的 LLM 响应中提取 JSON。"""
    text = (text or "").strip()
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


def review_answer(
    question: str,
    answer: str,
    evidence: str,
    judge=None,
) -> dict:
    """对主 Agent 的回答做一次 GLM 合规复核。

    返回 {"verdict": 1|0|None, "reason": str}：
    - verdict=1 事实正确；verdict=0 事实错误/编造；verdict=None 未复核
      （开关关闭、缺少证据、缺少 GLM key、调用失败或解析失败，全部 fail-open）。
    judge 参数用于测试注入 stub；生产缺省为 None → 惰性取全局判卷模型。
    """
    if not REVIEW_ENABLED or not evidence or not answer:
        return {"verdict": None, "reason": ""}

    if judge is None:
        judge = get_judge_llm()
        if judge is None:
            return {"verdict": None, "reason": ""}

    prompt = REVIEW_PROMPT.format(
        question=question,
        evidence=evidence[:MAX_EVIDENCE_CHARS],
        answer=answer[:MAX_EVIDENCE_CHARS],
    )
    try:
        resp = judge.invoke(
            [SystemMessage(content=REVIEW_SYSTEM), HumanMessage(content=prompt)]
        )
        parsed = _extract_json(resp.content)
        if parsed and "verdict" in parsed:
            return {
                "verdict": 1 if parsed["verdict"] else 0,
                "reason": str(parsed.get("reason", ""))[:120],
            }
    except Exception:
        pass  # fail-open：复核失败不影响主流程
    return {"verdict": None, "reason": ""}


def format_review_warning(result: dict) -> str:
    """verdict=0 时生成追加到回答末尾的合规提示；其余情况返回空串。"""
    if result.get("verdict") != 0:
        return ""
    reason = result.get("reason", "")
    suffix = f"（复核理由：{reason}）" if reason else ""
    return f"\n\n---\n**合规复核**（仅供参考）：该回答存在与检索依据不一致的风险{suffix}。"
