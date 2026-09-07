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

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
INTENT_CACHE_TTL = int(os.getenv("INTENT_CACHE_TTL", str(24 * 3600)))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "1").lower() not in ("0", "false", "no")
PREFIX = "intent_cache:"

_client = None
_client_ok = None  # None=未探测；True/False=探测结果


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


def build_cache_key(message: str) -> str:
    """由归一化消息生成 Redis key。"""
    norm = normalize_key(message)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"{PREFIX}{digest}"


def get_intent_decision(message: str) -> dict | None:
    """查缓存。命中返回缓存的意图决策 dict；未命中/不可用返回 None。"""
    client = _get_client()
    if client is None or not message:
        return None
    try:
        raw = client.get(build_cache_key(message))
        if raw is None:
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None  # fail-open


def set_intent_decision(message: str, decision: dict) -> bool:
    """写入缓存（带 TTL）。写入失败不报错（fail-open）。"""
    client = _get_client()
    if client is None or not message or not decision:
        return False
    try:
        client.set(build_cache_key(message), json.dumps(decision, ensure_ascii=False), ex=INTENT_CACHE_TTL)
        return True
    except Exception:
        return False
