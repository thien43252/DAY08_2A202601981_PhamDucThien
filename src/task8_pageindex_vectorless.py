"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex fpdf2

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from fpdf import FPDF
from pageindex.client import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"
ROOT_CACHE_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"
TEMP_PDF_DIR = Path(__file__).parent.parent / "data" / "_tmp_pdf"


def _clean_text(text: str) -> str:
    """Loại bỏ các ký tự icon font / private use area gây lỗi font rendering."""
    cleaned = []
    for ch in text:
        code = ord(ch)
        if 0xE000 <= code <= 0xF8FF or 0xF0000 <= code <= 0xFFFFD or 0x100000 <= code <= 0x10FFFD:
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def _convert_md_to_pdf(md_path: Path, pdf_path: Path):
    """Convert file markdown sang PDF tạm bằng fpdf2."""
    raw_content = md_path.read_text(encoding="utf-8")
    content = _clean_text(raw_content)
    pdf = FPDF()
    pdf.add_page()

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if not font_path.exists():
        for p in [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
        ]:
            if p.exists():
                font_path = p
                break

    use_custom = False
    if font_path.exists():
        try:
            pdf.add_font("CustomFont", "", str(font_path))
            pdf.set_font("CustomFont", size=10)
            use_custom = True
        except Exception:
            pass

    if not use_custom:
        pdf.set_font("helvetica", size=10)
        content = content.encode("latin-1", "replace").decode("latin-1")

    for line in content.split("\n"):
        line_str = line if line.strip() else " "
        try:
            pdf.multi_cell(w=pdf.epw, h=6, text=line_str)
        except Exception:
            safe_line = line_str.encode("latin-1", "replace").decode("latin-1")
            try:
                pdf.set_font("helvetica", size=10)
                pdf.multi_cell(w=pdf.epw, h=6, text=safe_line)
                if use_custom:
                    pdf.set_font("CustomFont", size=10)
            except Exception:
                pass

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def upload_documents() -> dict:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    Cache doc_ids vào pageindex_doc_ids.json.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY không tồn tại trong .env")
        return {}

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids = {}

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"⚠ Không tìm thấy file markdown nào trong {STANDARDIZED_DIR}")
        return {}

    TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

    for md_file in md_files:
        temp_pdf = TEMP_PDF_DIR / f"{md_file.stem}.pdf"
        try:
            _convert_md_to_pdf(md_file, temp_pdf)
            resp = client.submit_document(str(temp_pdf))
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                doc_ids[md_file.name] = doc_id
                print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
        except Exception as e:
            print(f"  ✗ Lỗi khi upload {md_file.name}: {e}")
        finally:
            if temp_pdf.exists():
                try:
                    temp_pdf.unlink()
                except Exception:
                    pass

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)

    with open(ROOT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)

    print(f"✓ Đã lưu {len(doc_ids)} doc_ids vào pageindex_doc_ids.json")
    return doc_ids


def _query_single_doc(client: PageIndexClient, doc_id: str, query: str) -> list[dict]:
    """Query 1 document trên PageIndex."""
    doc_results = []
    try:
        if not client.is_retrieval_ready(doc_id):
            for _ in range(10):
                if client.is_retrieval_ready(doc_id):
                    break
                time.sleep(1)

        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        if not retrieval_id:
            return []

        retrieval = {}
        for _ in range(15):
            retrieval = client.get_retrieval(retrieval_id)
            if retrieval.get("status") == "completed":
                break
            time.sleep(1)

        for node in retrieval.get("retrieved_nodes", []):
            rel_contents = node.get("relevant_contents")
            if rel_contents:
                if isinstance(rel_contents, list):
                    for group in rel_contents:
                        if isinstance(group, list):
                            for item in group:
                                if isinstance(item, dict):
                                    content = item.get("relevant_content") or item.get("content") or ""
                                    section = item.get("section_title") or item.get("section") or ""
                                    if content:
                                        doc_results.append({
                                            "content": content,
                                            "metadata": {"section": section, "doc_id": doc_id},
                                            "source": "pageindex",
                                        })
                                elif isinstance(item, str) and item:
                                    doc_results.append({
                                        "content": item,
                                        "metadata": {"doc_id": doc_id},
                                        "source": "pageindex",
                                    })
                        elif isinstance(group, dict):
                            content = group.get("relevant_content") or group.get("content") or ""
                            section = group.get("section_title") or group.get("section") or ""
                            if content:
                                doc_results.append({
                                    "content": content,
                                    "metadata": {"section": section, "doc_id": doc_id},
                                    "source": "pageindex",
                                })
            else:
                content = node.get("text") or node.get("content") or node.get("relevant_content") or ""
                if content:
                    doc_results.append({
                        "content": content,
                        "metadata": {"section": node.get("title", ""), "doc_id": doc_id},
                        "source": "pageindex",
                    })
    except Exception as e:
        print(f"⚠ Lỗi khi query doc_id {doc_id}: {e}")
    return doc_results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được thiết lập.")
        return []

    doc_ids = {}
    for cache_path in [CACHE_FILE, ROOT_CACHE_FILE]:
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    doc_ids = json.load(f)
                if doc_ids:
                    break
            except Exception:
                pass

    if not doc_ids:
        print("Chưa tìm thấy cache doc_ids, tiến hành upload_documents()...")
        doc_ids = upload_documents()

    if not doc_ids:
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_id_list = list(doc_ids.values()) if isinstance(doc_ids, dict) else list(doc_ids)

    results = []
    with ThreadPoolExecutor(max_workers=min(10, max(1, len(doc_id_list)))) as executor:
        futures = [executor.submit(_query_single_doc, client, doc_id, query) for doc_id in doc_id_list]
        for future in as_completed(futures):
            results.extend(future.result())

    # Gán score theo rank
    for rank, item in enumerate(results):
        score = max(0.1, 1.0 - rank * 0.05)
        item["score"] = round(score, 3)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")

