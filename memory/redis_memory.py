"""Redis 会话缓存 — 替代 SQLite，提供微秒级会话加载速度。"""
import json
import time
from typing import Optional, Dict, List, Any

import redis


class RedisMemory:
    """Redis 会话存储，兼容 case_memory.py 的接口。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, session_ttl: int = 86400):
        """初始化 Redis 连接。
        
        Args:
            host: Redis 服务器地址
            port: Redis 服务器端口
            db: 数据库编号
            password: 密码（如无需留空）
            session_ttl: 会话过期时间（秒），默认 24 小时
        """
        self.client = redis.Redis(
            host=host, port=port, db=db, password=password, decode_responses=True
        )
        self.session_ttl = session_ttl
        # 测试连接
        try:
            self.client.ping()
            print("[OK] Redis 连接成功")
        except Exception as e:
            raise ConnectionError(f"无法连接 Redis: {e}")

    def _key(self, prefix: str, session_id: str) -> str:
        return f"{prefix}:{session_id}"

    # ---- 会话基本信息 ----
    def save_session(self, session_id: str, case_summary: dict = {}) -> None:
        """保存会话基本信息与案情摘要。"""
        key = self._key("session", session_id)
        data = json.dumps(
            {"case_summary": case_summary, "updated_at": int(time.time())},
            ensure_ascii=False,
        )
        self.client.setex(key, self.session_ttl, data)

    def get_session(self, session_id: str) -> Optional[dict]:
        """读取会话信息，返回 {case_summary, updated_at} 或 None。"""
        key = self._key("session", session_id)
        data = self.client.get(key)
        if not data:
            return None
        return json.loads(data)

    def delete_session(self, session_id: str) -> None:
        """删除会话。"""
        key = self._key("session", session_id)
        self.client.delete(key)

    # ---- 消息/对话历史 ----
    def save_messages(self, session_id: str, history: List[Dict[str, str]]) -> None:
        """保存对话历史为列表。每条消息 JSON 存入列表。"""
        key = self._key("messages", session_id)
        # 使用 RPUSH 添加，LTRIM 保留最近 N 条
        pipe = self.client.pipeline()
        for msg in history:
            pipe.rpush(key, json.dumps(msg, ensure_ascii=False))
        pipe.ltrim(key, 0, 99)  # 保留最近 100 条
        pipe.execute()
        self.client.expire(key, self.session_ttl)

    def load_messages(self, session_id: str, max_messages: int = 10) -> List[Dict]:
        """读取最近 max_messages 条对话历史。"""
        key = self._key("messages", session_id)
        raw = self.client.lrange(key, 0, max_messages - 1)
        if not raw:
            return []
        return [json.loads(item) for item in raw]

    # ---- 便捷方法：兼容 case_memory 接口 ----
    def save_exchange(self, session_id: str, human: str, ai: str) -> None:
        """保存一轮对话（兼容 case_memory.save_exchange 接口）。"""
        # 保存会话基本信息
        case_summary = self.get_session(session_id)
        if case_summary is None:
            case_summary = {}
        self.save_session(session_id, case_summary)

        # 追加消息到历史
        key = self._key("messages", session_id)
        msg_entry = {"human": human, "ai": ai}
        self.client.rpush(key, json.dumps(msg_entry, ensure_ascii=False))
        self.client.ltrim(key, -100, -1)  # 保留最近 100 条
        self.client.expire(key, self.session_ttl)

    def load_session_from_db(self, session_id: str) -> dict:
        """从 Redis 加载会话（兼容 case_memory.load_session_from_db 接口）。"""
        session = self.get_session(session_id)
        if session:
            # session 结构: {case_summary: ..., updated_at: ...}
            # messages 需要从 key 中读取
            history = self.load_messages(session_id, MAX_LOAD_MESSAGES=10)
            case_summary = session.get("case_summary", {})
            return {"history": history, "case_summary": case_summary}

        # 空会话
        return {"history": [], "case_summary": {}}


# 全局实例（可选，按需初始化）
redis_memory: Optional[RedisMemory] = None


def init_redis_memory(host: str = "127.0.0.1", port: int = 6379, 
                      db: int = 0, password: Optional[str] = None) -> RedisMemory:
    """初始化全局 RedisMemory 实例。"""
    global redis_memory
    redis_memory = RedisMemory(host=host, port=port, db=db, password=password)
    return redis_memory