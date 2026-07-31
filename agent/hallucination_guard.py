"""幻觉检测 — 两层规则防御，零额外 LLM 调用。

第一层：引用校验 — 正则提取法条引用，验证是否在检索结果中存在。
第二层：检索覆盖度 — 检查回答关键实体是否与检索结果有交集。
"""

import re
from typing import Optional


def _extract_law_citations(text: str) -> list[str]:
    """从文本中提取所有法条引用，如"消保法第55条""第五十五条"。

    支持两种格式：
    - 中文数字：第五十五条、第二十三条
    - 阿拉伯数字：第55条、第23条
    """
    # 阿拉伯数字："第55条""第23条第1款"
    arabic = re.findall(r'第\d+条(?:第\d+款)?', text)
    # 中文数字："第五十五条""第二十三条第一款"
    chinese = re.findall(
        r'第[零一二三四五六七八九十百]+条(?:第[零一二三四五六七八九十百]+款)?',
        text,
    )
    return arabic + chinese


def _check_citations_in_docs(
    citations: list[str],
    retrieved_docs: str,
) -> dict:
    """检查提取到的法条引用是否在检索结果中出现过。

    返回：
    {
        "cited": [...],       # 所有提取到的引用
        "verified": [...],    # 在检索结果中找到的
        "unverified": [...],  # 没找到的（幻觉风险）
    }
    """
    verified = []
    unverified = []

    for citation in citations:
        if citation in retrieved_docs:
            verified.append(citation)
        else:
            unverified.append(citation)

    return {
        "cited": citations,
        "verified": verified,
        "unverified": unverified,
    }


def _check_coverage(
    answer: str,
    retrieved_docs: str,
    threshold: float = 0.3,
) -> dict:
    """检查回答与检索结果的实体重叠度。

    用简单的字符级 n-gram 重叠，而不调 LLM 或 NER 模型。
    threshold 以下视为可能未基于检索结果。
    """
    # 提取检索结果中的关键词（按空格和标点拆）
    doc_chars = set(retrieved_docs)
    answer_chars = set(answer)

    if not answer_chars:
        return {"overlap_ratio": 1.0, "low_coverage": False}

    overlap = answer_chars & doc_chars
    ratio = len(overlap) / len(answer_chars)

    return {
        "overlap_ratio": round(ratio, 3),
        "low_coverage": ratio < threshold,
    }


def check_hallucination(
    answer: str,
    retrieved_docs: str,
) -> dict:
    """对 LLM 生成的回答做两层幻觉检测。

    参数：
        answer: LLM 生成的完整回答
        retrieved_docs: 本轮 RAG 检索返回的文本块（合并为一个字符串）

    返回：
    {
        "risk_level": "none" | "low" | "high",
        "citations": {...},      # 引用校验结果
        "coverage": {...},       # 检索覆盖度结果
        "warnings": [...],       # 人类可读的警告信息
    }
    """
    warnings = []
    risk_level = "none"

    # ── 第一层：引用校验 ──
    citations = _extract_law_citations(answer)
    citation_result = _check_citations_in_docs(citations, retrieved_docs)

    if citation_result["unverified"]:
        unverified_list = "、".join(citation_result["unverified"])
        warnings.append(
            f"⚠️ 以下法条引用未在检索结果中验证：{unverified_list}"
        )

    # ── 第二层：检索覆盖度 ──
    coverage_result = _check_coverage(answer, retrieved_docs)

    if coverage_result["low_coverage"]:
        warnings.append(
            "⚠️ 回答内容与检索结果匹配度较低（{:.0%}），"
            "回答可能未充分依据检索结果".format(coverage_result["overlap_ratio"])
        )

    # ── 综合风险等级 ──
    if citation_result["unverified"]:
        risk_level = "high"
    elif coverage_result["low_coverage"]:
        risk_level = "low"

    return {
        "risk_level": risk_level,
        "citations": citation_result,
        "coverage": coverage_result,
        "warnings": warnings,
    }


def format_hallucination_warning(check_result: dict) -> str:
    """将幻觉检测结果格式化为可追加到回答末尾的文本。

    如果没有警告，返回空字符串。
    """
    warnings = check_result.get("warnings", [])
    if not warnings:
        return ""

    return "\n\n---\n**检索校验**（仅供参考）：\n" + "\n".join(
        f"- {w}" for w in warnings
    )