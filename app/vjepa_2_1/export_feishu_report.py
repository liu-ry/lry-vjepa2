"""Export the tactile technical report for Feishu Docs.

The Markdown report is the source of truth. This script produces:

* an HTML preview with PNG figures embedded as data URIs, suitable for
  copying into a Feishu document;
* a landscape DOCX with the same figures embedded, suitable for direct
  import into Feishu Docs.

DOCX export requires the optional ``python-docx`` package.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/tactile_contextmask_world_model_technical_report.md"
HTML_OUT = (
    ROOT / "docs/tactile_contextmask_world_model_technical_report_feishu.html"
)
DOCX_OUT = (
    ROOT / "docs/tactile_contextmask_world_model_technical_report_feishu.docx"
)
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
INLINE_PATTERN = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
EQUATION_BLOCK_PATTERN = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
EQUATION_FALLBACK = (
    "L_stage-A = L_future + 0.1 L_context + 0.05 L_var + "
    "0.01 L_cov + 1.0 L_temporal + 0.05 L_contact + 0.05 L_slip"
)


def _embed_html_images(markup: str) -> str:
    def replace(match: re.Match[str]) -> str:
        source = match.group(1)
        image_path = (REPORT.parent / source).resolve()
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return (
            f'src="data:image/png;base64,{payload}" '
            'style="display:block;max-width:100%;height:auto;'
            'margin:18px auto;border:1px solid #e2e8f0;border-radius:6px;"'
        )

    return re.sub(r'src="([^"]+\.png)"', replace, markup)


def export_html(source: str):
    source = EQUATION_BLOCK_PATTERN.sub(
        f'<div class="equation">{EQUATION_FALLBACK}</div>',
        source,
    )
    body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = _embed_html_images(body)
    markup = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape("两阶段触觉世界模型技术报告")}</title>
  <style>
    body {{
      margin: 0;
      color: #152238;
      font: 16px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI",
            "Microsoft YaHei", sans-serif;
      background: #f4f7fb;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 14px;
      padding: 10px 20px;
      color: #475569;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid #dbe4ef;
      backdrop-filter: blur(8px);
      font-size: 14px;
    }}
    .toolbar button {{
      border: 0;
      border-radius: 7px;
      padding: 9px 16px;
      color: #fff;
      background: #2563eb;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
    }}
    .toolbar button:hover {{ background: #1d4ed8; }}
    #copyStatus {{ color: #15803d; font-weight: 700; }}
    main {{
      max-width: 1180px;
      margin: 24px auto 56px;
      padding: 36px 42px 72px;
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
    }}
    h1, h2, h3 {{ color: #152238; line-height: 1.3; }}
    h1 {{ font-size: 34px; margin-top: 0; }}
    h2 {{ border-bottom: 1px solid #dde4ee; padding-bottom: 8px; }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      margin: 22px auto;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 18px 0;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d7e0eb;
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{ background: #f1f5f9; text-align: left; }}
    blockquote {{
      margin: 18px 0;
      padding: 12px 18px;
      border-left: 4px solid #2563eb;
      background: #f8fafc;
      color: #475569;
    }}
    .equation {{
      margin: 18px 0;
      padding: 12px 18px;
      border: 1px solid #dbeafe;
      border-radius: 6px;
      background: #eff6ff;
      font-family: "SFMono-Regular", Consolas, monospace;
      text-align: center;
    }}
    pre {{
      overflow-x: auto;
      padding: 14px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
    }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
  </style>
</head>
<body>
<div class="toolbar">
  <button id="copyButton" type="button">复制全文到飞书</button>
  <span>点击按钮后，在飞书文档中直接粘贴；失败时可使用 Ctrl+A / Ctrl+C。</span>
  <span id="copyStatus"></span>
</div>
<main id="report-content">
{body}
</main>
<script>
const copyButton = document.getElementById("copyButton");
const copyStatus = document.getElementById("copyStatus");

async function copyReport() {{
  const article = document.getElementById("report-content");
  const html = '<div class="feishu-document">' + article.innerHTML + "</div>";
  const plainText = article.innerText;
  try {{
    if (navigator.clipboard && window.ClipboardItem) {{
      const item = new ClipboardItem({{
        "text/html": new Blob([html], {{ type: "text/html" }}),
        "text/plain": new Blob([plainText], {{ type: "text/plain" }}),
      }});
      await navigator.clipboard.write([item]);
    }} else {{
      const range = document.createRange();
      range.selectNodeContents(article);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand("copy");
      selection.removeAllRanges();
    }}
    copyStatus.textContent = "已复制，请粘贴到飞书";
    setTimeout(() => {{ copyStatus.textContent = ""; }}, 3500);
  }} catch (error) {{
    copyStatus.textContent = "自动复制失败，请使用 Ctrl+A / Ctrl+C";
    console.error(error);
  }}
}}

copyButton.addEventListener("click", copyReport);
</script>
</body>
</html>
"""
    HTML_OUT.write_text(markup, encoding="utf-8")


def _set_run_font(run, size=None, bold=None, monospace=False):
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run.font.name = "Consolas" if monospace else "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def _add_inline(paragraph, text: str, size=10, color=None):
    from docx.shared import RGBColor

    for part in INLINE_PATTERN.split(text):
        if not part:
            continue
        monospace = part.startswith("`") and part.endswith("`")
        bold = part.startswith("**") and part.endswith("**")
        clean = part[2:-2] if bold else part[1:-1] if monospace else part
        run = paragraph.add_run(clean)
        _set_run_font(run, size=size, bold=bold or None, monospace=monospace)
        if color:
            run.font.color.rgb = RGBColor.from_string(color)


def _shade(paragraph, fill: str):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    properties = paragraph._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def _add_table(document, rows):
    from docx.shared import Pt

    columns = len(rows[0])
    table = document.add_table(rows=len(rows), cols=columns)
    table.style = "Table Grid"
    for row_index, row in enumerate(rows):
        for column_index in range(columns):
            text = row[column_index] if column_index < len(row) else ""
            cell = table.cell(row_index, column_index)
            paragraph = cell.paragraphs[0]
            _add_inline(paragraph, text, size=8)
            if row_index == 0:
                for run in paragraph.runs:
                    run.bold = True
                _shade(paragraph, "F1F5F9")
            paragraph.paragraph_format.space_after = Pt(0)


def export_docx(source: str):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = (
        section.page_height,
        section.page_width,
    )
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)

    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(
        __import__("docx").oxml.ns.qn("w:eastAsia"), "Microsoft YaHei"
    )
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.3

    lines = source.splitlines()
    index = 0
    paragraph_buffer = []

    def flush_paragraphs():
        if not paragraph_buffer:
            return
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _add_inline(paragraph, " ".join(paragraph_buffer))
        paragraph_buffer.clear()

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraphs()
            index += 1
            continue
        if stripped == "---":
            flush_paragraphs()
            index += 1
            continue

        image_match = IMAGE_PATTERN.fullmatch(stripped)
        if image_match:
            flush_paragraphs()
            image_path = (REPORT.parent / image_match.group(2)).resolve()
            picture = document.add_paragraph()
            picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
            picture.add_run().add_picture(str(image_path), width=Inches(9.35))
            caption = document.add_paragraph()
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_inline(
                caption,
                image_match.group(1),
                size=9,
                color="607089",
            )
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraphs()
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            paragraph = document.add_paragraph()
            _shade(paragraph, "F8FAFC")
            _add_inline(paragraph, "\n".join(code_lines), size=9)
            for run in paragraph.runs:
                _set_run_font(run, size=9, monospace=True)
            index += 1
            continue

        if stripped == "$$":
            flush_paragraphs()
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                index += 1
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_inline(paragraph, EQUATION_FALLBACK, size=10, color="152238")
            index += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            flush_paragraphs()
            level = len(heading.group(1))
            paragraph = document.add_heading(level=level - 1)
            _add_inline(
                paragraph,
                heading.group(2),
                size=24 if level == 1 else 16 if level == 2 else 13,
                color="152238",
            )
            index += 1
            continue

        if stripped.startswith("|"):
            flush_paragraphs()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = []
            for table_line in table_lines:
                cells = [cell.strip() for cell in table_line.strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    continue
                rows.append(cells)
            if rows:
                _add_table(document, rows)
            continue

        if stripped.startswith(">"):
            flush_paragraphs()
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip(">").strip())
                index += 1
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.22)
            paragraph.paragraph_format.right_indent = Inches(0.22)
            _shade(paragraph, "F8FAFC")
            _add_inline(paragraph, " ".join(quote_lines), size=9, color="475569")
            continue

        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        numbered = re.match(r"^\d+\.\s+(.*)$", stripped)
        if bullet or numbered:
            flush_paragraphs()
            paragraph = document.add_paragraph(
                style="List Bullet" if bullet else "List Number"
            )
            _add_inline(paragraph, (bullet or numbered).group(1))
            index += 1
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraphs()
    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            if run.font.size is None:
                run.font.size = Pt(10)
            if run.font.color.rgb is None:
                run.font.color.rgb = RGBColor(21, 34, 56)
    document.save(DOCX_OUT)


def main():
    source = REPORT.read_text(encoding="utf-8")
    export_html(source)
    export_docx(source)
    print(f"wrote {HTML_OUT}")
    print(f"wrote {DOCX_OUT}")


if __name__ == "__main__":
    main()
