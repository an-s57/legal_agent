"""评测比较器单测 — 全离线，不连库不调模型。

为什么值得单测：比较器是"判卷老师"，它错了会让评测结果整体失真
（第一轮评测就出现过"生成与金标都被坏数据污染 → 27 道假通过"、
"列名不同但值相同被判失败"这类假结论）。所以它的容差规则必须有回归测试兜住。

运行：python -m unittest tests.test_ops_eval_compare -v
"""
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")  # 占位，只保证导入不依赖真实密钥
os.environ.setdefault("CACHE_ENABLED", "0")

import unittest

from ops_data_qa.run_eval import _cell_equal, _compare_result, _multiset_equal


class CellEqualTest(unittest.TestCase):
    """单元格容差：只放过"同一事实的不同写法"，不放过真差异。"""

    def test_字符串完全相同(self):
        self.assertTrue(_cell_equal("劳动纠纷", "劳动纠纷"))

    def test_首尾空格忽略(self):
        self.assertTrue(_cell_equal("12", "12 "))

    def test_浮点尾零(self):
        self.assertTrue(_cell_equal("0.0", "0.0000"))

    def test_整数与带尾零(self):
        self.assertTrue(_cell_equal("78", "78.0"))

    def test_比例与百分比(self):
        self.assertTrue(_cell_equal("0.9444", "94.4"))
        self.assertTrue(_cell_equal("94.4", "0.9444"))

    def test_比例与百分比_再叠加舍入容差(self):
        # #39 实测案例：生成 0.2564（比例），金标 25.6（百分比且 ROUND(...,1)）
        self.assertTrue(_cell_equal("0.2564", "25.6"))
        self.assertTrue(_cell_equal("25.6", "0.2564"))

    def test_舍入精度差(self):
        self.assertTrue(_cell_equal("5.5556", "5.6"))
        self.assertTrue(_cell_equal("5.56", "5.6"))

    def test_整数差1必须判不同(self):
        # 整数不设容差，否则 78 和 79 会被当成"舍入差异"放过 —— 评测就失真了
        self.assertFalse(_cell_equal("78", "79"))

    def test_百分比差0_1超出舍入容差(self):
        self.assertFalse(_cell_equal("25.6", "25.7"))

    def test_单位对了但值不同(self):
        self.assertFalse(_cell_equal("0.2564", "25.7"))

    def test_中文不同(self):
        self.assertFalse(_cell_equal("劳动纠纷", "消费维权"))

    def test_空值与0不等(self):
        self.assertFalse(_cell_equal("", "0"))

    def test_非数字与数字不等(self):
        self.assertFalse(_cell_equal("未知", "3"))


class MultisetEqualTest(unittest.TestCase):
    """行多重集：行序不敏感，行数必须相等。"""

    def test_行序不同视为相等(self):
        self.assertTrue(_multiset_equal([(1, "b"), (2, "a")], [(2, "a"), (1, "b")]))

    def test_行数不同判不等(self):
        self.assertFalse(_multiset_equal([(1,)], [(1,), (2,)]))

    def test_None与空串等价(self):
        self.assertTrue(_multiset_equal([(None,)], [("",)]))


class CompareResultTest(unittest.TestCase):
    """结果形状比较：列数相同按位置比；金标列是子集则投影比。"""

    def test_列名不同但列数相同_按位置比较(self):
        # 典型场景：生成写 AS session_count，金标是 COUNT(*) —— 名字不同、语义相同
        ok, how = _compare_result(["session_count"], [(3,)], ["COUNT(*)"], [(3,)])
        self.assertTrue(ok)
        self.assertIn("列名不同", how)

    def test_生成多给列_按金标列投影(self):
        ok, how = _compare_result(
            ["session_id", "msg_count"], [("s_1", 4)], ["msg_count"], [(4,)]
        )
        self.assertTrue(ok)
        self.assertIn("投影", how)

    def test_列数不同且金标列不存在_判失败(self):
        ok, how = _compare_result(["a", "b"], [(1, 2)], ["c"], [(1,)])
        self.assertFalse(ok)
        self.assertIn("列数不同", how)

    def test_值不同_判失败(self):
        ok, _ = _compare_result(["cnt"], [(3,)], ["cnt"], [(4,)])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
