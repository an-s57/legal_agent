"""LLM 意图门 — 词面规则之上的语义级破坏性意图识别。

为什么需要：词面规则（validator.check_intent）免费但"近视"——
"帮我把差评处理掉"没有"删/清"等关键词，词面放行，但语义上是要改数据。
LLM 意图门补这个洞：一次 GLM 调用判断用户"想不想改数据"。

模型：判卷模型（config.JUDGE_MODEL，默认智谱 GLM-4.5-Air，与合规复核共用判卷客户端，
temperature=0，同输入同判定）。
选 GLM 而非主模型 DeepSeek：意图判断很短、便宜模型够用，主模型留给生成；
顺带保证这层的判卷方与生成方异构（不自己判自己）。

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
1 = 想改数据或非查询：删除、修改、清空、整理、处理、覆盖、合并、补全数据，
   或要求对数据做任何变更

注意这些容易被误判成查询的写法，其实都属于 1（要改数据）：
- "帮我把差评处理掉"            → 1（"处理掉"= 变更数据，不是查询）
- "把重复的会话合并一下"        → 1（合并记录）
- "给所有会话补上案件类型"      → 1（补字段）
- "这些数据太乱了帮我收拾一下"  → 1（整理数据）
- "让评分看起来好看一点"        → 1（篡改数据）

真正的查询是"想看某个数字/列表/分布"，例如：
- "最近更新的会话有哪些"        → 0（看哪些会话有动静）
- "哪类案件占比最高"            → 0

用户问题：{question}

只输出一个 JSON，不要输出任何其他文字：
{{"destructive": 0或1, "reason": "不超过20字的一句话理由"}}"""

# 投票轮数：GLM 单次判定有波动（实测同一批 danger 题两次评测给出不同结论），
# 用多轮多数投票降方差；OPS_INTENT_VOTES=1 可退回单次判定（省成本）。
INTENT_VOTES = max(1, int(os.getenv("OPS_INTENT_VOTES", "3")))


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
    """LLM 意图门。返回 {"destructive": True|False|None, "reason": str, "gate": str}。

    destructive=None 表示本层无法判定（开关关闭/缺 key/调用失败/解析失败），
    调用方应退回词面规则的结论（fail-open）。judge 参数供测试注入 stub。

    gate 字段说清"本层这次到底发生了什么"，供评测报告自证（否则 fail-open
    会让报告看起来像"意图门开了但没拦住"，其实是"意图门根本没工作"）：
      disabled   —— 开关关闭（A 组：只跑词面规则）
      unavailable—— 调用失败/全部投票弃权（fail-open 放行，例：GLM 余额不足 429）
      pass       —— 判定为查询，放行
      block      —— 判定为数据变更请求，拦截
    """
    if not INTENT_LLM_ENABLED:
        return {"destructive": None, "reason": "", "gate": "disabled"}
    if judge is None:
        judge = get_judge_llm()
        if judge is None:
            logger.info("[INTENT-GATE] 未配置 GLM key，意图门跳过（fail-open）")
            return {
                "destructive": None,
                "reason": "",
                "gate": "unavailable",
                "error": "未配置判卷模型 key（GLM_API_KEY）",
            }

    prompt = INTENT_PROMPT.format(question=question)
    votes, reasons = [], []
    last_error = ""
    for i in range(INTENT_VOTES):
        try:
            resp = judge.invoke(
                [SystemMessage(content=INTENT_SYSTEM), HumanMessage(content=prompt)]
            )
            parsed = _extract_json(resp.content)
            if parsed and "destructive" in parsed:
                flag = parsed["destructive"]
                if isinstance(flag, (int, bool)) or (
                    isinstance(flag, str) and flag.strip() in ("0", "1")
                ):
                    votes.append(str(flag).strip() in ("1", "true", "True"))
                    reasons.append(str(parsed.get("reason", ""))[:60])
                else:
                    last_error = f"输出格式异常：{str(resp.content)[:60]}"
            else:
                last_error = f"输出无法解析：{str(resp.content)[:60]}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
            logger.warning(
                f"[INTENT-GATE] LLM 意图判断第 {i + 1}/{INTENT_VOTES} 轮失败"
                f"（{type(e).__name__}），本轮弃权"
            )
        # 提前收敛：剩余票数已不足以翻盘时不再多花钱（注意别用 continue 跳过这段）
        remaining = INTENT_VOTES - i - 1
        yes = sum(votes)
        no = len(votes) - yes
        if yes > no + remaining or no > yes + remaining:
            break

    if not votes:
        logger.warning(
            f"[INTENT-GATE] 全部投票失败，退回词面规则（fail-open）；最后一次错误：{last_error}"
        )
        return {
            "destructive": None,
            "reason": "",
            "gate": "unavailable",
            "error": last_error,
        }

    yes, no = sum(votes), len(votes) - sum(votes)
    # 平票（如 2 票 1:1）判为破坏性：安全场景宁可多拦一次人工复核，也不放行改数据
    destructive = yes >= no
    reason = next((r for v, r in zip(votes, reasons) if v == destructive), reasons[0])
    if INTENT_VOTES > 1:
        reason = f"[{yes}改/{no}查] {reason}"
    logger.info(f"[INTENT-GATE] 投票 {yes}改/{no}查 → destructive={destructive}")
    return {
        "destructive": destructive,
        "reason": reason,
        "gate": "block" if destructive else "pass",
        "votes": f"{yes}改/{no}查",
    }
