"""限流器单测 — 假时钟离线跑，不睡眠、不依赖网络。

运行：python -m unittest tests.test_ops_ratelimit -v
"""
import unittest

from ops_data_qa.ratelimit import SlidingWindowLimiter, parse_rate_limit


class FakeClock:
    """可手动拨动的时钟：advance(n) 把时间往后拨 n 秒。"""

    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class SlidingWindowLimiterTest(unittest.TestCase):

    def test_窗口内第N次放行第N加1次拒绝(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(3, 60, clock=clock)
        self.assertTrue(lim.allow("u")[0])
        self.assertTrue(lim.allow("u")[0])
        self.assertTrue(lim.allow("u")[0])
        denied, wait = lim.allow("u")
        self.assertFalse(denied)
        self.assertGreater(wait, 0)          # 拒绝时带"建议等待秒数"

    def test_窗口滑动后恢复放行(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(2, 60, clock=clock)
        lim.allow("u")
        lim.allow("u")
        self.assertFalse(lim.allow("u")[0])
        clock.advance(61)                    # 窗口内两条记录全部滑出
        self.assertTrue(lim.allow("u")[0])

    def test_窗口部分滑出_部分容量恢复(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(2, 60, clock=clock)
        lim.allow("u")                       # t=1000
        clock.advance(30)
        lim.allow("u")                       # t=1030
        self.assertFalse(lim.allow("u")[0])
        clock.advance(31)                    # t=1061：t=1000 那条滑出，t=1030 还在
        allowed, _ = lim.allow("u")
        self.assertTrue(allowed)
        self.assertFalse(lim.allow("u")[0])  # 窗口内已有 2 条，再拒

    def test_不同key互不影响(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(1, 60, clock=clock)
        self.assertTrue(lim.allow("alice")[0])
        self.assertTrue(lim.allow("bob")[0])
        self.assertFalse(lim.allow("alice")[0])

    def test_等待秒数随时间递减(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(1, 60, clock=clock)
        lim.allow("u")
        _, w1 = lim.allow("u")
        clock.advance(30)
        _, w2 = lim.allow("u")
        self.assertAlmostEqual(w1, 60.0)
        self.assertAlmostEqual(w2, 30.0)


class ParseRateLimitTest(unittest.TestCase):

    def test_正常解析(self):
        self.assertEqual(parse_rate_limit("10/60"), (10, 60.0))

    def test_带空格与浮点窗口(self):
        self.assertEqual(parse_rate_limit(" 5 / 30.5 "), (5, 30.5))

    def test_非法格式退回默认(self):
        self.assertEqual(parse_rate_limit("abc"), (10, 60.0))
        self.assertEqual(parse_rate_limit("0/60"), (10, 60.0))
        self.assertEqual(parse_rate_limit("10/0"), (10, 60.0))
        self.assertEqual(parse_rate_limit(""), (10, 60.0))


if __name__ == "__main__":
    unittest.main()
