"""共享 LLM 客户端配置 — agent 和 memory 模块共用同一个实例。

避免在多个文件中重复创建 ChatOpenAI 和重复调用 load_dotenv()。
"""
import os

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# trust_env=False：忽略 http_proxy/https_proxy 环境变量，直连 DeepSeek。
# 背景：.bashrc 里配了 Clash 代理，ChatOpenAI 默认 trust_env=True 会把
# LLM 请求都塞给代理；Clash 一没开，后端就 APITimeoutError 挂 60s。
# DeepSeek API 直连不需要代理（0.16s），绕开代理反而更稳。
# 参考 rag/retriever.py 里 Ollama 调用同样的 trust_env=False 思路。
def _make_llm(
    temperature: float | None = None,
    model: str | None = None,
    api_base: str | None = None,
) -> ChatOpenAI:
    """按统一参数创建 LLM 实例。

    temperature=None 用默认采样（适合生成回答）；temperature=0 确定性输出
    （适合意图识别这类决策任务，保证同一输入同一判定、可复现）。

    模型可配置（不绑死代码）：model/api_base 参数优先，其次环境变量
    LEGAL_AGENT_MODEL / LEGAL_AGENT_API_BASE / LEGAL_AGENT_API_KEY，
    缺省走 DeepSeek。换模型 = export LEGAL_AGENT_MODEL=新模型，生产与
    评测代码零改动。
    """
    kwargs = dict(
        model=model or os.getenv("LEGAL_AGENT_MODEL", "deepseek-v4-flash"),
        openai_api_key=os.getenv("LEGAL_AGENT_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=api_base or os.getenv("LEGAL_AGENT_API_BASE", "https://api.deepseek.com/v1"),
        timeout=60,        # 单次请求上限 60s，防止 GLM/DeepSeek 慢窗口拖死服务器
        max_retries=1,     # 失败只重试一次，避免无限重试
        http_client=httpx.Client(trust_env=False),  # 忽略环境变量代理，直连
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


llm = _make_llm()                      # 主回答：默认采样，语言更自然
planner_llm = _make_llm(temperature=0) # 意图识别：确定性输出，追问决策可复现


def _make_judge_llm() -> ChatOpenAI:
    """合规复核 Agent 的判卷模型：默认智谱 GLM，OpenAI 兼容接口。

    与主模型（DeepSeek）异构，避免"自己判自己"的自评偏好（同 LLM-as-Judge 评测的设计）。
    配置：LEGAL_AGENT_JUDGE_MODEL / LEGAL_AGENT_JUDGE_API_KEY（缺省 GLM_API_KEY）/
    LEGAL_AGENT_JUDGE_API_BASE（缺省智谱 OpenAI 兼容端点）。
    temperature=0：复核是决策任务，要求同一输入可复现的判定。
    """
    return ChatOpenAI(
        model=os.getenv("LEGAL_AGENT_JUDGE_MODEL", "glm-4.7"),
        openai_api_key=os.getenv("LEGAL_AGENT_JUDGE_API_KEY") or os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("LEGAL_AGENT_JUDGE_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
        timeout=60,
        max_retries=1,
        http_client=httpx.Client(trust_env=False),
        temperature=0,
    )


_judge_llm_cache = None
_judge_llm_resolved = False


def get_judge_llm() -> ChatOpenAI | None:
    """惰性构造合规复核判卷模型；缺少 GLM key 时返回 None（复核层 fail-open 跳过）。

    不在 import 时构造：无 key 的 CI/裸环境不会因缺 key 报错（openai SDK 新版会在
    构造时抛 Missing credentials），只有真正要复核时才建客户端。
    """
    global _judge_llm_cache, _judge_llm_resolved
    if _judge_llm_resolved:
        return _judge_llm_cache
    _judge_llm_resolved = True
    key = os.getenv("LEGAL_AGENT_JUDGE_API_KEY") or os.getenv("GLM_API_KEY")
    if not key:
        return None
    _judge_llm_cache = _make_judge_llm()
    return _judge_llm_cache
