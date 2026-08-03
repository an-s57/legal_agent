"""诊断 legal_pdfs 下所有 PDF 的文本层质量（v3）。

核心判定：
1. 版式：检测"全国人民代表大会常务委员会公报"页眉 → 该 PDF 是公报两栏版式
2. 拼接行：一行被空格断开成多段，且左段不是条款编号/日期/标题等正常空格
   （公报版式里每行都是"左栏半句 + 右栏半句"）

结论：公报版式 = 向量库 chunk 必然被左右栏拼接污染。

用法: python evaluation/diagnose_pdf_layout.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pdfplumber

PDF_DIR = Path(__file__).parent.parent / "legal_pdfs"

# 正常空格出现的模式：条款编号、日期、标题、落款
_NORMAL_FIRST_SEG = re.compile(r"^(第.+条|第.+款|（.+年|[（(]\d+|全国人民代表大会常务委员会公报|中华人民共和国|主席)")


def is_bulletin_page(text: str) -> bool:
    return "全国人民代表大会常务委员会公报" in text


def is_garbled_line(line: str) -> bool:
    """判定一行是否公报左右栏拼接产生的错乱行。

    特征：行中部被空格断开成 ≥2 段；若左段是编号/日期/标题则不计数
    （那是正常排版空格，如"第二条 ..."）。
    """
    parts = [p for p in line.split(" ") if p.strip()]
    if len(parts) < 2:
        return False
    first = parts[0]
    if _NORMAL_FIRST_SEG.match(first):
        return False
    # 左段应该是一段完整内容（不是孤零零的半个字）
    if len(first) < 6:
        return False
    return True


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"{'PDF 文件':<55}{'页数':>4}{'字数':>8}{'公报页':>6}{'拼接行':>8}  判定")
    print("-" * 92)
    for p in pdfs:
        try:
            with pdfplumber.open(p) as pdf:
                n_pages = len(pdf.pages)
                total_chars = 0
                bulletin_pages = 0
                garbled_lines = 0
                samples = []
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    total_chars += len(text)
                    if is_bulletin_page(text):
                        bulletin_pages += 1
                    for line in text.splitlines():
                        if is_garbled_line(line):
                            garbled_lines += 1
                            if len(samples) < 2:
                                samples.append(f"  第{i}页: {line.strip()[:80]}")
                if total_chars < 100:
                    verdict = "扫描版/无文本层 ⚠"
                elif bulletin_pages >= n_pages / 2:
                    verdict = f"公报两栏版式，拼接污染 ⚠⚠"
                else:
                    verdict = "单栏，正常"
                print(f"{p.name:<55}{n_pages:>4}{total_chars:>8}{bulletin_pages:>6}{garbled_lines:>8}  {verdict}")
                for s in samples:
                    print(s)
        except Exception as e:  # noqa: BLE001
            print(f"{p.name:<55}{'-':>4}{'-':>8}{'-':>6}{'-':>8}  [打开失败] {e}")
    print()


if __name__ == "__main__":
    main()