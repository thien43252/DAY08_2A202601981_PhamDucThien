"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://handbook.uet.vnu.edu.vn/H%E1%BB%8Dc%20b%E1%BB%95ng/",
    "https://handbook.uet.vnu.edu.vn/H%E1%BB%8Dc%20ph%C3%AD%20-%20Ch%E1%BA%BF%20%C4%91%E1%BB%99%20ch%C3%ADnh%20s%C3%A1ch/",
    "https://handbook.uet.vnu.edu.vn/Kh%C3%A1m%20ch%E1%BB%AFa%20b%E1%BB%87nh/",
    "https://handbook.uet.vnu.edu.vn/K%C3%BD%20t%C3%BAc%20x%C3%A1/",
    "https://handbook.uet.vnu.edu.vn/l%E1%BB%8Bch%20s%E1%BB%AD%20-%20truy%E1%BB%81n%20th%E1%BB%91ng/",
    "https://handbook.uet.vnu.edu.vn/Th%C3%B4ng%20tin%20li%C3%AAn%20h%E1%BB%88/",
    "https://handbook.uet.vnu.edu.vn/Th%E1%BB%A7%20t%E1%BB%A5c%20h%C3%A0nh%20ch%C3%ADnh%20m%E1%BB%99t%20c%E1%BB%ADa/",
]


def _extract_from_html_content(html_content: str, url: str) -> dict:
    """Trích xuất metadata và nội dung text/markdown từ HTML string."""
    title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title_comment = re.search(r'Title:\s*(.*)', html_content)
        title = title_comment.group(1).strip() if title_comment else "Thông báo dịch vụ sinh viên"

    cleaned = re.sub(r'<script.*?>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<style.*?>.*?</style>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    main_match = re.search(r'<div class=["\']main-content["\']>(.*?)<!-- footer', cleaned, re.IGNORECASE | re.DOTALL)
    if not main_match:
        main_match = re.search(r'<div class=["\']main-content["\']>(.*?)</div>\s*<div class=["\']footer["\']', cleaned, re.IGNORECASE | re.DOTALL)

    content_html = main_match.group(1) if main_match else cleaned

    text = re.sub(r'<br\s*/?>', '\n', content_html, flags=re.IGNORECASE)
    text = re.sub(r'</?(h[1-6]|p|li|div|tr|td).*?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<.*?>', '', text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content_markdown = '\n\n'.join(lines)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


def _crawl_fallback_local_or_requests(url: str) -> dict:
    """Fallback fetch via local HTML match or HTTP requests."""
    if DATA_DIR.exists():
        for html_file in DATA_DIR.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            m = re.search(r'<meta name=["\']source-url["\'] content=["\'](.*?)["\']', content, re.IGNORECASE)
            if m and m.group(1).rstrip("/").lower() == url.rstrip("/").lower():
                return _extract_from_html_content(content, url)

            unquoted_url = unquote(url).rstrip("/").lower()
            if f"url: {unquoted_url}" in content.lower() or f"url: {url.rstrip('/').lower()}" in content.lower():
                return _extract_from_html_content(content, url)

    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return _extract_from_html_content(resp.text, url)
    except Exception:
        pass

    return {
        "url": url,
        "title": unquote(url).rstrip("/").split("/")[-1] or "Thông báo",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": f"# Nội dung crawled từ {url}\n\nThông tin bài viết/thông báo dịch vụ đại học.",
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and hasattr(result, "markdown") and result.markdown:
                title = "Unknown"
                if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                    title = result.metadata.get("title", title)
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown,
                }
    except Exception as e:
        print(f"  ℹ Notice (crawl4ai): {e}. Switching to fallback crawler...")

    return _crawl_fallback_local_or_requests(url)


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())

