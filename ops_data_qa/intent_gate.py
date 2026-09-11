"""LLM 意图门 — 词面规则之上的语义级破坏性意图识别。

为什么需要：词面规则（validator.check_intent）免费但"近视"——
"帮我把差评处理掉"没有"删/清"等关键词，词面放行，但语义上是要改数据。
LLM 意图门补这个洞：一次 GLM 调用判断用户"想不想改数据"。

模型：GLM-4.7（与合规复核共用判卷客户端，temperature=0，同输入同判定）。
选 GLM 而非主模型 DeepSeek：意图判断很短、便宜模型够用，主模型留给生成。

fail-open：未配置 GLM key、调用失败、解析失败 → 返回 destructive=None
（"本层无法判定"），调用方退回词面规则的结论，绝不因本层故障瘫痪问答。
开关：OPS_INTENT_LLM_ENABLED（默认开；A/B 评测时设 0 只跑词面规则）。
"""
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import get_judge_llm
from logger import get_logger

logger = get_logger("legal_agent.ops_qa")

INTENT_LLM_ENABLED = os.getenv("OPS_INTENT_LLM_ENABLED", "1").lower() not in ("0", "false", "no")

INTENT_SYSTEM = (
    "你是运营数据问答系统的意图审核员。只判断用户这句话想做什么，不执行任何操作。"
    "每次只输出一个 JSON。"
)

INTENT_PROMPT = """判断下面这句运营人员的问题属于哪类意图：

0 = 只想看数据：查询、统计、排名、对比（例："8月有多少会话""哪类案件最多"）
1 = 想改数据或非查询：删除、修改、清空、整理、处理、覆盖数据，
   或要求对数据做任何变更（例："把差评处理掉""数据太乱了帮我整理一下"）

注意："最近更新的会话有哪些"是查询（看最近有动静的会话），属于 0。

用户问题：{question}

只输出一个 JSON，不要输出任何其他文字：
{{"destructive": 0或1, "reason": "不超过20字的一句话理由"}}"""


def _extract_json(text: str) -> dict | None:
    """从 LLM 响应中提取 JSON（容错 markdown 围栏/多余文本：定位首尾花括号）。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def check_intent_llm(question: str, judge=None) -> dict:
    """LLM 意图门。返回 {"destructive": True|False|None, "reason": str}。

    destructive=None 表示本层无法判定（开关关闭/缺 key/调用失败/解析失败），
    调用方应退回词面规则的结论（fail-open）。judge 参数供测试注入 stub。
    """
    if not INTENT_LLM_ENABLED:
        return {"destructive": None, "reason": ""}
    if judge is None:
        judge = get_judge_llm()
        if judge is None:
            logger.info("[INTENT-GATE] 未配置 GLM key，意图门跳过（fail-open）")
            return {"destructive": None, "reason": ""}
    try:
        resp = judge.invoke(
            [
                SystemMessage(content=INTENT_SYSTEM),
                HumanMessage(content=INTENT_PROMPT.format(question=question)),
            ]
        )
        parsed = _extract_json(resp.content)
        if parsed and "destructive" in parsed:
            flag = parsed["destructive"]
            if isinstance(flag, (int, bool)) or (
                isinstance(flag, str) and flag.strip() in ("0", "1")
            ):
                return {
                    "destructive": str(flag).strip() in ("1", "true", "True"),
                    "reason": str(parsed.get("reason", ""))[:60],
                }
    except Exception as e:
        logger.warning(f"[INTENT-GATE] LLM 意图判断失败（{type(e).__name__}），退回词面规则")
    return {"destructive": None, "reason": ""}
