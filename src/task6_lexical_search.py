"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

from rank_bm25 import BM25Okapi, BM25Plus

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Lazily populated corpus/index so the module works even if conversion runs later.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_TOKENIZED_CORPUS: list[list[str]] = []
_BM25_INDEX: BM25Plus | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenize text with a simple unicode-aware tokenizer."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_corpus() -> list[dict]:
    """Load all markdown documents from data/standardized/."""
    documents: list[dict] = []

    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8")
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if relative_path.parts else "unknown"
        indexed_content = f"{md_file.stem}\n{relative_path.as_posix()}\n\n{content}"
        documents.append(
            {
                "content": indexed_content,
                "metadata": {
                    "source": str(relative_path),
                    "title": md_file.stem,
                    "type": doc_type,
                },
            }
        )

    return documents


def _ensure_index() -> BM25Plus | None:
    """Build the corpus and BM25 index on first use."""
    global CORPUS, _TOKENIZED_CORPUS, _BM25_INDEX

    if _BM25_INDEX is not None:
        return _BM25_INDEX

    CORPUS = _load_corpus()
    if not CORPUS:
        _TOKENIZED_CORPUS = []
        _BM25_INDEX = None
        return None

    _TOKENIZED_CORPUS = [_tokenize(doc["content"]) for doc in CORPUS]
    _BM25_INDEX = BM25Plus(_TOKENIZED_CORPUS)
    return _BM25_INDEX


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus if doc.get("content")]
    return BM25Plus(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25 = _ensure_index()
    if bm25 is None or not CORPUS:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results: list[dict] = []
    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])
        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx]["metadata"],
            }
        )

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
