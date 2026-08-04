"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf

try:
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    # PageIndex is optional; hybrid retrieval must remain importable without it.
    def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
        return []


# =============================================================================
# CONFIGURATION
# =============================================================================

def calibrate_score_threshold(
    relevant_queries: list[str], irrelevant_queries: list[str]
) -> float:
    """Return the midpoint between relevant and off-topic dense scores."""
    def best_score(query: str) -> float:
        results = semantic_search(query, top_k=1) or []
        return float(results[0].get("score", 0.0)) if results else 0.0

    relevant_scores = [best_score(query) for query in relevant_queries]
    irrelevant_scores = [best_score(query) for query in irrelevant_queries]
    if not relevant_scores or not irrelevant_scores:
        raise ValueError("Both relevant and irrelevant calibration queries are required")

    return (min(relevant_scores) + max(irrelevant_scores)) / 2


SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if top_k <= 0:
        return []

    search_k = max(top_k * 2, top_k)

    # A missing optional Task 5/6 implementation should not make the complete
    # pipeline crash; an empty branch can still be handled by the other branch
    # or by PageIndex fallback.
    try:
        dense_results = semantic_search(query, top_k=search_k) or []
    except (NotImplementedError, ImportError):
        dense_results = []
    try:
        sparse_results = lexical_search(query, top_k=search_k) or []
    except (NotImplementedError, ImportError):
        sparse_results = []

    # Keep the original dense score before RRF changes the score scale.
    try:
        best_dense_score = float(dense_results[0].get("score", 0.0)) if dense_results else 0.0
    except (TypeError, ValueError):
        best_dense_score = 0.0

    merged = rerank_rrf([dense_results, sparse_results], top_k=search_k)
    hybrid_results = []
    for item in merged:
        result = dict(item)
        result["source"] = "hybrid"
        hybrid_results.append(result)

    if use_reranking and hybrid_results:
        final_results = rerank(
            query, hybrid_results, top_k=top_k, method=RERANK_METHOD
        )
        # ``rerank`` may return copied dictionaries; guarantee the pipeline
        # source marker remains present for downstream citation code.
        for result in final_results:
            result.setdefault("source", "hybrid")
    else:
        final_results = hybrid_results[:top_k]

    # This comparison intentionally uses the original cosine score.  RRF scores
    # are around 0.016 and are not relevance probabilities.
    if best_dense_score < score_threshold:
        try:
            fallback_results = pageindex_search(query, top_k=top_k) or []
        except (NotImplementedError, ImportError):
            fallback_results = []
        if fallback_results:
            normalized = []
            for item in fallback_results[:top_k]:
                result = dict(item)
                result["source"] = "pageindex"
                normalized.append(result)
            return normalized

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
