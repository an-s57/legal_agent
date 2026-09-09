"""Redis 缓存客户端 — 意图识别缓存（fail-open，不破坏主流程）。

设计：
- 缓存 Planner 的结构化意图结果（intent + info_complete + missing_fields + follow_up）；
- 缓存键：对用户当前消息做归一化（去空白/标点/小写）后取 sha256，前缀 `intent_cache:`；
- 只缓存"当前消息"的意图判断，不混入案情上下文——意图分类本身不依赖个人案情；
- fail-open：Redis 不可用时自动降级为"每次调 Planner"，绝不影响回答链路。

运行前提：Redis 容器（context-engine-redis-dev，localhost:6379）或任意 Redis 服务。
配置：REDIS_URL（缺省 redis://localhost:6379/0）、INTENT_CACHE_TTL（秒，缺省 86400）、
      CACHE_ENABLED（缺省开，设 0/false 关）。
"""
import hashlib
import json
import os
import re
import threading

from config import VECTORSTORE_VERSION, RETRIEVAL_K_VECTOR, RETRIEVAL_TOP_K

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
INTENT_CACHE_TTL = int(os.getenv("INTENT_CACHE_TTL", str(24 * 3600)))
ANSWER_CACHE_TTL = int(os.getenv("ANSWER_CACHE_TTL", str(24 * 3600)))
# 检索结果只随向量库变、不随时间变 → TTL 只当兜底，真正失效靠版本号。默认 7 天。
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", str(7 * 24 * 3600)))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "1").lower() not in ("0", "false", "no")
PREFIX = "intent_cache:"
# 回答缓存（1.1B）：与意图缓存隔离；版本号取自 config.VECTORSTORE_VERSION，
# 重建/更新向量库后 +1，旧回答缓存自动失效（防止法条更新后仍回放旧答案）。
ANSWER_PREFIX = f"answer_cache:{VECTORSTORE_VERSION}:"
# 检索结果缓存（#3）：key 里再编入 k / top_k，检索配置一变也自动失效。
RETRIEVAL_PREFIX = f"retrieval_cache:{VECTORSTORE_VERSION}_k{RETRIEVAL_K_VECTOR}_tk{RETRIEVAL_TOP_K}:"

_client = None
_client_ok = None  # None=未探测；True/False=探测结果

# ── 命中率统计（纯旁路观测，fail-safe：任何异常都不影响答题）──
# 每层缓存记：hit（命中）/ miss（真未命中）/ unavailable（Redis 不可用，非 miss）/ write（写入）。
# 内存计数，进程重启清零（演示够用）。
_stats_lock = threading.Lock()
_CACHES = ("intent", "answer", "retrieval")
_FIELDS = ("hit", "miss", "unavailable", "write")


def _blank_stats() -> dict:
    return {c: {f: 0 for f in _FIELDS} for c in _CACHES}


_stats = _blank_stats()


def record(cache: str, field: str) -> None:
    """记一次事件。统计是旁路：名字/字段非法或任何异常都静默忽略，绝不抛出。"""
    try:
        if cache in _CACHES and field in _FIELDS:
            with _stats_lock:
                _stats[cache][field] += 1
    except Exception:
        pass


def reset_stats() -> None:
    """清零计数（测试/运维用）。"""
    global _stats
    with _stats_lock:
        _stats = _blank_stats()


def get_cache_stats() -> dict:
    """返回各缓存计数 + 命中率（hit/(hit+miss)）。只读快照。"""
    with _stats_lock:
        snap = {c: dict(_stats[c]) for c in _CACHES}
    for c in _CACHES:
        denom = snap[c]["hit"] + snap[c]["miss"]
        snap[c]["hit_rate"] = round(snap[c]["hit"] / denom, 4) if denom else None
    return snap


def _get_client():
    """惰性创建 Redis 客户端；连接失败返回 None（fail-open）。"""
    global _client, _client_ok
    if not CACHE_ENABLED:
        return None
    if _client_ok is not None:
        return _client if _client_ok else None
    try:
        import redis as _redis
        _client = _redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
        _client.ping()
        _client_ok = True
    except Exception:
        _client_ok = False
        _client = None
    return _client


def normalize_key(message: str) -> str:
    """归一化消息：去首尾空白、去标点、转小写。"""
    if not message:
        return ""
    text = message.strip().lower()
    text = re.sub(r"[\s\u3000]+", "", text)          # 去所有空白（含全角空格）
    text = re.sub(r"[，。！？、；：,.!?;:()（）\"'“”‘’\-—_]", "", text)
    return text


def _key(prefix: str, message: str) -> str:
    """由归一化消息 + 前缀生成 Redis key。"""
    norm = normalize_key(message)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


def build_cache_key(message: str) -> str:
    """由归一化消息生成意图缓存 Redis key。"""
    return _key(PREFIX, message)


def get_intent_decision(message: str) -> dict | None:
    """查缓存。命中返回缓存的意图决策 dict；未命中/不可用返回 None。"""
    client = _get_client()
    if client is None:
        record("intent", "unavailable")
        return None
    if not message:
        return None
    try:
        raw = client.get(build_cache_key(message))
        if raw is None:
            record("intent", "miss")
            return None
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            record("intent", "hit")
            return parsed
        record("intent", "miss")
        return None
    except Exception:
        record("intent", "unavailable")
        return None  # fail-open


def set_intent_decision(message: str, decision: dict) -> bool:
    """写入缓存（带 TTL）。写入失败不报错（fail-open）。"""
    client = _get_client()
    if client is None or not message or not decision:
        return False
    try:
        client.set(build_cache_key(message), json.dumps(decision, ensure_ascii=False), ex=INTENT_CACHE_TTL)
        record("intent", "write")
        return True
    except Exception:
        return False


# ── 回答缓存（1.1B）：只缓存客观知识问题的完整回答 ──
# 读写均 fail-open；调用方保证"仅当 Planner 判定 is_general_knowledge 且无上下文"才写。

def build_answer_key(message: str) -> str:
    """回答缓存 key（带知识库版本号，与意图缓存隔离）。"""
    return _key(ANSWER_PREFIX, message)


def get_answer(message: str) -> str | None:
    """查回答缓存。命中返回缓存的完整回答字符串；未命中/不可用/内容非法返回 None。"""
    client = _get_client()
    if client is None:
        record("answer", "unavailable")
        return None
    if not message:
        return None
    try:
        raw = client.get(build_answer_key(message))
        if raw is None:
            record("answer", "miss")
            return None
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            answer = parsed.get("answer")
            if isinstance(answer, str) and answer:
                record("answer", "hit")
                return answer
        record("answer", "miss")
        return None
    except Exception:
        record("answer", "unavailable")
        return None  # fail-open


def set_answer(message: str, answer: str) -> bool:
    """写入回答缓存（带 TTL）。仅接受非空回答；写失败不报错（fail-open）。"""
    client = _get_client()
    if client is None or not message or not answer:
        return False
    try:
        client.set(
            build_answer_key(message),
            json.dumps({"answer": answer}, ensure_ascii=False),
            ex=ANSWER_CACHE_TTL,
        )
        record("answer", "write")
        return True
    except Exception:
        return False


# ── 检索结果缓存（#3）：缓存 legal_rag_search 的产出（客观法条原文，缓存不会"答错"）──
# 读写均 fail-open；"查不到"的哨兵串也一并缓存（对固定向量库是确定结果）。

def build_retrieval_key(query: str) -> str:
    """检索缓存 key（带版本号 + k + top_k，与向量库/检索配置绑定）。"""
    return _key(RETRIEVAL_PREFIX, query)


def get_retrieval(query: str) -> str | None:
    """查检索缓存。命中返回缓存的检索结果字符串；未命中/不可用/非法返回 None。"""
    client = _get_client()
    if client is None:
        record("retrieval", "unavailable")
        return None
    if not query:
        return None
    try:
        raw = client.get(build_retrieval_key(query))
        if raw is None:
            record("retrieval", "miss")
            return None
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            result = parsed.get("r")
            if isinstance(result, str):
                record("retrieval", "hit")
                return result
        record("retrieval", "miss")
        return None
    except Exception:
        record("retrieval", "unavailable")
        return None  # fail-open


def set_retrieval(query: str, result: str) -> bool:
    """写入检索缓存（带 TTL）。写失败不报错（fail-open）。允许缓存空/未命中哨兵串。"""
    client = _get_client()
    if client is None or not query or result is None:
        return False
    try:
        client.set(
            build_retrieval_key(query),
            json.dumps({"r": result}, ensure_ascii=False),
            ex=RETRIEVAL_CACHE_TTL,
        )
        record("retrieval", "write")
        return True
    except Exception:
        return False
