"""内存滑动窗口限流器 —— 单进程版。

为什么手写不引第三方库：逻辑只有三十行，时钟可注入、离线可单测，不给项目加依赖。
已知局限（面试可讲）：计数放在进程内存里，多进程 / 多机部署时要换成 Redis 计数
（INCR + EXPIRE 或 ZSET）。当前演示为单进程 uvicorn，够用。
"""
import threading
import time
from collections import deque


def parse_rate_limit(spec: str, default: tuple[int, float] = (10, 60.0)) -> tuple[int, float]:
    """解析 "10/60" → (10, 60.0)。格式非法时退回默认值（fail-open）。"""
    try:
        n, per = str(spec).split("/")
        n_i, per_f = int(n.strip()), float(per.strip())
        if n_i > 0 and per_f > 0:
            return n_i, per_f
    except ValueError:
        pass
    return default


class SlidingWindowLimiter:
    """滑动窗口限流：同一 key 在窗口期内最多放行 max_calls 次。

    用滑动窗口而不是固定窗口：每次判断前先清掉窗口外的旧记录，
    避免固定窗口"边界瞬间双倍放行"的经典毛刺。
    """

    def __init__(self, max_calls: int, period_seconds: float, clock=time.monotonic):
        self.max_calls = max_calls
        self.period = period_seconds
        self._clock = clock          # 时钟可注入，单测用假时钟
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        """放行返回 (True, 0)；拒绝返回 (False, 建议等待的秒数)。"""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.period:
                hits.popleft()
            if len(hits) < self.max_calls:
                hits.append(now)
                return True, 0.0
            return False, max(self.period - (now - hits[0]), 0.0)
