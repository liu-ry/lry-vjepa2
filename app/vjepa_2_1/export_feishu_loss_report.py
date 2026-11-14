"""Build a Feishu-copyable Stage A/B loss document with LaTeX blocks."""

from __future__ import annotations

import html
from pathlib import Path

import markdown
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/tactile_contextmask_loss_explained.md"
OUTPUT = ROOT / "docs/tactile_contextmask_loss_explained_feishu.html"


def _formula_card(latex: str) -> BeautifulSoup:
    escaped = html.escape(latex)
    escaped_attribute = html.escape(latex, quote=True)
    return BeautifulSoup(
        f"""
        <div class="formula" data-latex="{escaped_attribute}">
          <div class="formula-toolbar">
            <span>LaTeX 公式</span>
            <button class="copy-formula" type="button">复制 LaTeX</button>
          </div>
          <pre><code>{escaped}</code></pre>
        </div>
        """,
        "html.parser",
    )


def _render_article(source: str) -> str:
    body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    soup = BeautifulSoup(body, "html.parser")
    for code in list(soup.select("pre > code.language-latex")):
        pre = code.parent
        pre.replace_with(_formula_card(code.get_text()))
    return str(soup)


def main():
    article = _render_article(SOURCE.read_text(encoding="utf-8"))
    markup = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Stage A / Stage B Loss 详细说明</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: #152238;
      background: #f4f7fb;
      font: 16px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI",
            "Microsoft YaHei", sans-serif;
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
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid #dbe4ef;
      backdrop-filter: blur(8px);
      font-size: 14px;
    }}
    .toolbar button, .copy-formula {{
      border: 0;
      border-radius: 7px;
      padding: 8px 14px;
      color: #fff;
      background: #2563eb;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
    }}
    .toolbar button:hover, .copy-formula:hover {{ background: #1d4ed8; }}
    #copyStatus {{ color: #15803d; font-weight: 700; }}
    main {{
      max-width: 1120px;
      margin: 24px auto 56px;
      padding: 36px 42px 72px;
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
    }}
    h1, h2, h3 {{ color: #152238; line-height: 1.3; }}
    h1 {{ font-size: 32px; margin-top: 0; }}
    h2 {{
      margin-top: 34px;
      padding-bottom: 8px;
      border-bottom: 1px solid #dde4ee;
    }}
    h3 {{ margin-top: 26px; font-size: 20px; }}
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
    .formula {{
      margin: 18px 0;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      background: #eff6ff;
      overflow: hidden;
    }}
    .formula-toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 7px 12px;
      color: #475569;
      background: #dbeafe;
      font-size: 12px;
      font-weight: 700;
    }}
    .copy-formula {{ padding: 4px 10px; font-size: 12px; }}
    .formula pre {{
      margin: 0;
      padding: 14px 16px;
      overflow-x: auto;
      color: #152238;
      background: #eff6ff;
    }}
    .formula code, pre code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 14px;
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
  <button id="copyDocument" type="button">复制全文到飞书</button>
  <span>公式将以可编辑 LaTeX 文本块复制；“复制 LaTeX”可用于飞书原生公式。</span>
  <span id="copyStatus"></span>
</div>
<main id="report-content">
{article}
</main>
<script>
const copyButton = document.getElementById("copyDocument");
const copyStatus = document.getElementById("copyStatus");

async function copyText(text, successMessage) {{
  if (navigator.clipboard && window.isSecureContext) {{
    await navigator.clipboard.writeText(text);
  }} else {{
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }}
  copyStatus.textContent = successMessage;
  setTimeout(() => {{ copyStatus.textContent = ""; }}, 2500);
}}

async function copyDocument() {{
  const article = document.getElementById("report-content");
  const clone = article.cloneNode(true);
  clone.querySelectorAll(".formula-toolbar").forEach((node) => node.remove());
  const richHtml = '<div class="feishu-document">' + clone.innerHTML + "</div>";
  const plainText = article.innerText;
  try {{
    if (navigator.clipboard && window.ClipboardItem) {{
      await navigator.clipboard.write([
        new ClipboardItem({{
          "text/html": new Blob([richHtml], {{ type: "text/html" }}),
          "text/plain": new Blob([plainText], {{ type: "text/plain" }}),
        }}),
      ]);
      copyStatus.textContent = "已复制，请粘贴到飞书";
    }} else {{
      const range = document.createRange();
      range.selectNodeContents(article);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand("copy");
      selection.removeAllRanges();
      copyStatus.textContent = "已复制，请粘贴到飞书";
    }}
  }} catch (error) {{
    copyStatus.textContent = "自动复制失败，请使用 Ctrl+A / Ctrl+C";
    console.error(error);
  }}
}}

copyButton.addEventListener("click", copyDocument);
document.querySelectorAll(".copy-formula").forEach((button) => {{
  button.addEventListener("click", async () => {{
    const latex = button.closest(".formula").dataset.latex;
    try {{
      await copyText(latex, "LaTeX 已复制");
    }} catch (error) {{
      copyStatus.textContent = "LaTeX 复制失败";
      console.error(error);
    }}
  }});
}});
</script>
</body>
</html>
"""
    OUTPUT.write_text(markup, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
