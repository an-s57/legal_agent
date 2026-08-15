"""幻觉守卫单元测试 — 引用提取 + 引用校验 + 覆盖度检查。

纯函数测试：不调 LLM、不联网。
"""
import unittest

from agent.hallucination_guard import (
    _check_citations_in_docs,
    _extract_law_citations,
    check_hallucination,
)


class ExtractLawCitationsTest(unittest.TestCase):
    """法条引用提取（阿拉伯数字 + 中文数字）。"""

    def test_阿拉伯数字(self) -> None:
        self.assertEqual(
            _extract_law_citations("根据第55条和第23条第1款的规定"),
            ["第55条", "第23条第1款"],
        )

    def test_中文数字(self) -> None:
        self.assertEqual(
            _extract_law_citations("依据第五十五条、第二十三条第一款"),
            ["第五十五条", "第二十三条第一款"],
        )

    def test_无引用返回空(self) -> None:
        self.assertEqual(_extract_law_citations("这里没有任何法律引用"), [])

    def test_混合文本(self) -> None:
        citations = _extract_law_citations("消保法第五十五条适用，另见第3条")
        self.assertIn("第五十五条", citations)
        self.assertIn("第3条", citations)


class CheckCitationsInDocsTest(unittest.TestCase):
    def test_引用在文档中_验证通过(self) -> None:
        docs = "第五十五条 经营者提供商品或者服务有欺诈行为的，应当按照消费者的要求增加赔偿。"
        result = _check_citations_in_docs(["第五十五条"], docs)
        self.assertEqual(result["verified"], ["第五十五条"])
        self.assertEqual(result["unverified"], [])

    def test_引用不在文档中_标记未验证(self) -> None:
        docs = "这里只有消费者权益保护法的内容。"
        result = _check_citations_in_docs(["第五百七十七条"], docs)
        self.assertEqual(result["unverified"], ["第五百七十七条"])
        self.assertEqual(result["verified"], [])


class CheckHallucinationTest(unittest.TestCase):
    def test_引用已验证_无风险(self) -> None:
        answer = "根据消费者权益保护法第五十五条，可以要求三倍赔偿"
        docs = "第五十五条 经营者提供商品或者服务有欺诈行为的，应当按照消费者的要求增加赔偿其受到的损失"
        result = check_hallucination(answer, docs)
        self.assertEqual(result["risk_level"], "none")
        self.assertEqual(result["warnings"], [])

    def test_引用未验证_高风险(self) -> None:
        answer = "根据民法典第五百七十七条，可以要求继续履行"
        docs = "这里只有消费者权益保护法的内容，没有民法典"
        result = check_hallucination(answer, docs)
        self.assertEqual(result["risk_level"], "high")

    def test_覆盖率过低_低风险(self) -> None:
        answer = "xyzabc123"  # 与检索结果字符几乎无重叠
        docs = "第五十五条 经营者提供商品或者服务有欺诈行为的"
        result = check_hallucination(answer, docs)
        self.assertEqual(result["risk_level"], "low")

    def test_答案与文档高度一致_不触发覆盖警告(self) -> None:
        docs = "第五十五条 经营者提供商品或者服务有欺诈行为的，应当按照消费者的要求增加赔偿。"
        answer = "第五十五条 经营者提供商品或者服务有欺诈行为的，应当按照消费者的要求增加赔偿，这是惩罚性赔偿的规定。"
        result = check_hallucination(answer, docs)
        self.assertEqual(result["coverage"]["low_coverage"], False)


if __name__ == "__main__":
    unittest.main()
