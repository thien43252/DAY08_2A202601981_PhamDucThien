"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import shutil
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm"}
NEWS_EXTENSIONS = {".json", ".html", ".htm", ".md", ".txt"}


def _write_markdown(source_path: Path, source_root: Path, output_root: Path, content: str):
    """Write markdown content while preserving the relative folder structure."""
    relative_path = source_path.relative_to(source_root).with_suffix(".md")
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"  ✓ Saved: {output_path}")


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.rglob("*")):
        if filepath.is_file() and filepath.suffix.lower() in LEGAL_EXTENSIONS:
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            _write_markdown(filepath, legal_dir, output_dir, result.text_content)


def convert_news_articles():
    """Convert news files trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(news_dir.rglob("*")):
        if not filepath.is_file() or filepath.suffix.lower() not in NEWS_EXTENSIONS:
            continue

        print(f"Converting: {filepath.name}")

        if filepath.suffix.lower() == ".json":
            data = json.loads(filepath.read_text(encoding="utf-8"))
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            body = data.get("content_markdown") or data.get("content") or ""
            content = header + body
        elif filepath.suffix.lower() in {".html", ".htm"}:
            result = md.convert(str(filepath))
            content = result.text_content
        else:
            content = filepath.read_text(encoding="utf-8")

        _write_markdown(filepath, news_dir, output_dir, content)


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
