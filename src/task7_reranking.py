"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import os


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity without requiring NumPy."""
    if len(left) != len(right):
        raise ValueError("Các vector phải có cùng số chiều")
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if top_k <= 0 or not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        return rerank_rrf([candidates], top_k=top_k)

    import requests

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [candidate["content"] for candidate in candidates],
                "top_n": min(top_k, len(candidates)),
            },
            timeout=10,
        )
        response.raise_for_status()
        return [
            {**candidates[result["index"]], "score": result["relevance_score"]}
            for result in response.json()["results"]
        ]
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return rerank_rrf([candidates], top_k=top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if top_k <= 0 or not candidates:
        return []
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param phải nằm trong khoảng [0, 1]")

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    for _ in range(min(top_k, len(candidates))):
        best_idx = remaining[0]
        best_value = float("-inf")
        for index in remaining:
            relevance = _cosine_similarity(
                query_embedding, candidates[index].get("embedding", [])
            )
            diversity_penalty = max(
                (_cosine_similarity(candidates[index]["embedding"], candidates[chosen]["embedding"])
                 for chosen in selected),
                default=0.0,
            )
            value = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if value > best_value:
                best_value = value
                best_idx = index
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [{**candidates[index], "score": _cosine_similarity(query_embedding, candidates[index]["embedding"])}
            for index in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0 or not ranked_lists:
        return []
    if k < 0:
        raise ValueError("k phải >= 0")

    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            if not isinstance(item, dict) or "content" not in item:
                continue
            key = str(item["content"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = dict(item)
                first_seen[key] = order
                order += 1

    ranked = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    results: list[dict] = []
    for key in ranked[:top_k]:
        result = dict(content_map[key])
        result["score"] = scores[key]
        results.append(result)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # Với một danh sách candidates, thứ hạng đầu vào chính là ranker duy nhất.
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
