"""Export the chain-autoresearch Markdown report as a Feishu-ready page."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from pathlib import Path

import markdown


IMAGE_SRC = re.compile(r'src="([^"]+)"')


def _embed_images(markup: str, root: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        source = match.group(1)
        if source.startswith(("data:", "http://", "https://")):
            return match.group(0)
        image_path = (root / source).resolve()
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return (
            f'src="data:{mime};base64,{payload}" '
            'style="display:block;max-width:100%;height:auto;'
            'margin:22px auto;border:1px solid #dbe4ef;border-radius:8px;"'
        )

    return IMAGE_SRC.sub(replace, markup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="触觉世界模型技术报告")
    args = parser.parse_args()
    source_path = Path(args.source).resolve()
    source = source_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = _embed_images(body, source_path.parent)
    markup = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(args.title)}</title>
<style>
body {{
  margin: 0;
  color: #172033;
  background: #f3f6fb;
  font: 16px/1.78 -apple-system, BlinkMacSystemFont, "Segoe UI",
        "Microsoft YaHei", sans-serif;
}}
.toolbar {{
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  justify-content: center;
  gap: 14px;
  padding: 10px 20px;
  background: rgba(255,255,255,.96);
  border-bottom: 1px solid #dbe4ef;
  backdrop-filter: blur(8px);
  color: #52627a;
}}
.toolbar button {{
  border: 0;
  border-radius: 7px;
  padding: 9px 18px;
  color: #fff;
  background: #2563eb;
  cursor: pointer;
  font: inherit;
  font-weight: 700;
}}
main {{
  max-width: 1220px;
  margin: 24px auto 60px;
  padding: 38px 46px 72px;
  background: #fff;
  border: 1px solid #e1e7f0;
  border-radius: 10px;
  box-shadow: 0 8px 30px rgba(15,23,42,.06);
}}
h1, h2, h3 {{ color: #172033; line-height: 1.35; }}
h1 {{ margin-top: 0; font-size: 34px; }}
h2 {{ margin-top: 38px; padding-bottom: 8px; border-bottom: 1px solid #dde5ef; }}
h3 {{ margin-top: 26px; }}
img {{ max-width: 100%; }}
table {{ width: 100%; margin: 18px 0; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 8px 10px; border: 1px solid #d7e0eb; vertical-align: top; }}
th {{ background: #f1f5f9; text-align: left; }}
blockquote {{
  margin: 18px 0;
  padding: 12px 18px;
  border-left: 4px solid #2563eb;
  background: #f8fafc;
  color: #52627a;
}}
pre {{
  overflow-x: auto;
  padding: 14px 16px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 7px;
  font-family: "SFMono-Regular", Consolas, monospace;
}}
code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
.copy-status {{ color: #15803d; font-weight: 700; }}
</style>
</head>
<body>
<div class="toolbar">
  <button id="copyButton" type="button">复制全文到飞书</button>
  <span>公式以等宽文本保留，可直接编辑；图表会作为富文本图片复制。</span>
  <span class="copy-status" id="copyStatus"></span>
</div>
<main id="report-content">
{body}
</main>
<script>
const article = document.getElementById("report-content");
const status = document.getElementById("copyStatus");
document.getElementById("copyButton").addEventListener("click", async () => {{
  const rich = '<div class="feishu-document">' + article.innerHTML + "</div>";
  try {{
    if (navigator.clipboard && window.ClipboardItem) {{
      await navigator.clipboard.write([new ClipboardItem({{
        "text/html": new Blob([rich], {{type: "text/html"}}),
        "text/plain": new Blob([article.innerText], {{type: "text/plain"}})
      }})]);
    }} else {{
      const range = document.createRange();
      range.selectNodeContents(article);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand("copy");
    }}
    status.textContent = "已复制，请粘贴到飞书";
  }} catch (error) {{
    status.textContent = "自动复制失败，请使用 Ctrl+A / Ctrl+C";
    console.error(error);
  }}
}});
</script>
</body>
</html>
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markup, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
