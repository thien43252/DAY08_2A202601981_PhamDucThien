"""
Task 5 — Semantic Search Module & HyDE.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store kết hợp HyDE.

Yêu cầu:
    1. Hoàn thành hàm semantic_search() dùng Cosine similarity.
    2. Viết _generate_hypothetical_doc() cho HyDE (Hypothetical Document Embeddings).
    3. Embed hypothetical doc thay vì query gốc.
    4. Trả về top_k kết quả sorted score descending.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "university_services_docs"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# Singleton cache cho OpenAI Client và Chroma Client
_OPENAI_CLIENT = None
_COLLECTION_CACHE = None


def _get_openai_client():
    """Tạo và cache OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        import os
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _embed_query(text: str) -> list[float] | None:
    """Embed a single query text using OpenAI API."""
    client = _get_openai_client()
    if client is None:
        return None
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL_NAME,
        )
        return response.data[0].embedding
    except Exception:
        return None


def _get_collection():
    """Kết nối và cache ChromaDB Collection."""
    global _COLLECTION_CACHE
    if _COLLECTION_CACHE is None:
        try:
            import chromadb
            if not CHROMA_DIR.exists():
                return None
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _COLLECTION_CACHE = client.get_collection(name=COLLECTION_NAME)
        except BaseException:
            _COLLECTION_CACHE = None
    return _COLLECTION_CACHE


def _generate_hypothetical_doc(query: str) -> str:
    """
    Tạo văn bản giả định (Hypothetical Document) từ câu hỏi gốc (Kỹ thuật HyDE).
    HyDE giúp chuyển đổi câu hỏi ngắn/thiếu ngữ cảnh thành dạng bài viết/thông báo mẫu
    để tăng hiệu quả tìm kiếm vector similarity (cải thiện recall ~10-15%).

    Args:
        query: Câu hỏi hoặc từ khóa tìm kiếm gốc.

    Returns:
        Một văn bản giả định mô phỏng nội dung câu trả lời/thông báo.
    """
    if not query or not query.strip():
        return ""

    hypothetical_doc = (
        f"Thông báo và quy định chi tiết về {query.strip()}.\n"
        f"Hướng dẫn quy trình, điều kiện, đối tượng áp dụng và các mốc thời gian liên quan đến {query.strip()}.\n"
        f"Sinh viên cần nắm rõ các thông tin chính sách, thủ tục hành chính và hỗ trợ từ nhà trường đối với {query.strip()}."
    )
    return hypothetical_doc


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = True) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa (dense retrieval) sử dụng Vector Similarity + HyDE.

    Args:
        query: Câu truy vấn của người dùng.
        top_k: Số lượng kết quả tối đa cần trả về.
        use_hyde: Có sử dụng kỹ thuật HyDE (Hypothetical Document Embeddings) hay không.

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # metadata (source, type, v.v.)
        }
        Kết quả được sắp xếp giảm dần theo score (sorted score descending).
    """
    if not query or not query.strip():
        return []

    # Bước 1 & 2: Viết _generate_hypothetical_doc() cho HyDE & Embed hypothetical doc thay vì query gốc
    if use_hyde:
        hypothetical_doc = _generate_hypothetical_doc(query)
        search_text = hypothetical_doc if hypothetical_doc else query
    else:
        search_text = query

    try:
        query_vector = _embed_query(search_text)
        if query_vector is None:
            return []

        collection = _get_collection()
        if collection is None or collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        output = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(documents)

        # Tính Cosine Similarity Score từ distance
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = max(0.0, 1.0 - float(dist))
            output.append({
                "content": doc,
                "score": round(score, 4),
                "metadata": meta if meta else {}
            })

        # Bước 4: Trả về top_k kết quả sorted score desc
        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]
    except Exception:
        return []



if __name__ == "__main__":
    # Test
    test_query = "what is the tuition fee"
    results = semantic_search(test_query, top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"[{r['score']:.4f}] {r['content'][:100]}...")

