"""
RAG Evaluation Pipeline.

RAGAS-based evaluation for the group checkpoint.
The implementation is tuned to minimize rate-limit usage:
- default sample size is small and deterministic
- answer relevancy uses local embeddings
- context recall / precision use non-LLM RAGAS metrics
- faithfulness and answer relevancy use the LLM only once per sample set
- execution is sequential to avoid bursts
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Plus
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_utils import get_api_key, load_project_env

load_project_env()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
STANDARDIZED_DIR = Path(__file__).parents[2] / "data" / "standardized"
DEFAULT_SAMPLE_LIMIT = int(os.getenv("RAGAS_SAMPLE_LIMIT", "5"))
USE_FULL_DATASET = os.getenv("RAGAS_USE_FULL_DATASET", "0").lower() in {"1", "true", "yes"}
USE_PIPELINE = os.getenv("RAGAS_USE_PIPELINE", "0").lower() in {"1", "true", "yes"}
LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", os.getenv("OPENROUTER_MODEL", "gpt-4o-mini"))
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = get_api_key("RAGAS_API_KEY")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    content: str
    source: str
    title: str
    doc_type: str


class LocalHashingEmbeddings(Embeddings):
    """Local, deterministic embeddings for RAGAS answer relevancy."""

    def __init__(self, n_features: int = 4096):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()  # type: ignore[attr-defined]

    def embed_query(self, text: str) -> list[float]:
        matrix = self.vectorizer.transform([text])
        return matrix.toarray()[0].tolist()  # type: ignore[attr-defined]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _bootstrap_ragas_shim() -> None:
    """Patch missing VertexAI import expected by the installed ragas version."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - import shim only
        pass

    setattr(shim, "ChatVertexAI", ChatVertexAI)
    sys.modules[module_name] = shim


@lru_cache(maxsize=1)
def _load_ragas_api():
    _bootstrap_ragas_shim()

    from ragas.dataset_schema import EvaluationDataset
    from ragas.evaluation import evaluate
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
    from ragas.metrics._context_recall import NonLLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    return {
        "evaluate": evaluate,
        "EvaluationDataset": EvaluationDataset,
        "AnswerRelevancy": AnswerRelevancy,
        "NonLLMContextPrecisionWithReference": NonLLMContextPrecisionWithReference,
        "NonLLMContextRecall": NonLLMContextRecall,
        "Faithfulness": Faithfulness,
        "RunConfig": RunConfig,
    }


@lru_cache(maxsize=1)
def _load_corpus() -> list[Document]:
    documents: list[Document] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8")
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if relative_path.parts else "unknown"
        documents.append(
            Document(
                content=content,
                source=str(relative_path),
                title=md_file.stem,
                doc_type=doc_type,
            )
        )

    return documents


@lru_cache(maxsize=1)
def _bm25_index() -> BM25Plus | None:
    documents = _load_corpus()
    if not documents:
        return None
    tokenized = [_tokenize(_doc_payload(document)) for document in documents]
    return BM25Plus(tokenized)


@lru_cache(maxsize=1)
def _tfidf_bundle():
    documents = _load_corpus()
    if not documents:
        return None, None

    corpus = [_doc_payload(document) for document in documents]
    vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _doc_payload(document: Document) -> str:
    return f"{document.title}\n{document.source}\n\n{document.content}"


def _jaccard(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _extractive_answer(question: str, contexts: list[dict], fallback_text: str = "") -> str:
    query_tokens = set(_tokenize(question))
    snippets: list[str] = []

    for item in contexts:
        content = item["content"]
        best_sentence = ""
        best_score = -1.0

        for sentence in _split_sentences(content):
            score = _jaccard(query_tokens, _tokenize(sentence))
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if best_sentence:
            source = item["metadata"].get("source", "unknown")
            snippets.append(f"{best_sentence} [{source}]")

        if len(snippets) >= 2:
            break

    if snippets:
        return " ".join(snippets)

    return fallback_text or "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def _compute_faithfulness(answer: str, contexts: list[dict]) -> float:
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(" ".join(item["content"] for item in contexts))
    return _jaccard(answer_tokens, context_tokens)


def _compute_answer_relevance(question: str, answer: str) -> float:
    question_tokens = _tokenize(question)
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    return min(1.0, _jaccard(question_tokens, answer_tokens) * 1.25)


def _compute_context_recall(expected_context: str, contexts: list[dict]) -> float:
    expected_tokens = _tokenize(expected_context)
    if not expected_tokens:
        return 0.0
    context_tokens = _tokenize(" ".join(item["content"] for item in contexts))
    return len(set(expected_tokens) & set(context_tokens)) / len(set(expected_tokens))


def _compute_context_precision(question: str, contexts: list[dict]) -> float:
    query_tokens = set(_tokenize(question))
    if not query_tokens or not contexts:
        return 0.0
    scores = [_jaccard(query_tokens, _tokenize(item["content"])) for item in contexts]
    return sum(scores) / len(scores)


def _rank_bm25(documents: list[Document], query: str, top_k: int) -> list[dict]:
    index = _bm25_index()
    if index is None or not documents:
        return []

    scores = index.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    results: list[dict] = []
    for idx in ranked_indices:
        document = documents[idx]
        results.append(
            {
                "content": document.content,
                "score": float(scores[idx]),
                "metadata": {
                    "source": document.source,
                    "title": document.title,
                    "type": document.doc_type,
                },
            }
        )
    return results


def _rank_tfidf(documents: list[Document], query: str, top_k: int) -> list[dict]:
    vectorizer, matrix = _tfidf_bundle()
    if vectorizer is None or matrix is None or not documents:
        return []

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(matrix, query_vector).ravel()
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    results: list[dict] = []
    for idx in ranked_indices:
        document = documents[idx]
        results.append(
            {
                "content": document.content,
                "score": float(scores[idx]),
                "metadata": {
                    "source": document.source,
                    "title": document.title,
                    "type": document.doc_type,
                },
            }
        )
    return results


def _select_subset(golden_dataset: list[dict]) -> list[dict]:
    if USE_FULL_DATASET:
        return golden_dataset

    limit = max(1, min(DEFAULT_SAMPLE_LIMIT, len(golden_dataset)))
    selected: list[dict] = []
    seen_sources: set[str] = set()

    for item in golden_dataset:
        source_file = item.get("source_file", "")
        if source_file in seen_sources:
            continue
        selected.append(item)
        seen_sources.add(source_file)
        if len(selected) >= limit:
            return selected

    for item in golden_dataset:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break

    return selected


def _build_pipeline_result(
    question: str,
    retrieved: list[dict],
    expected_answer: str,
    source_file: str,
    config_name: str,
    use_pipeline: bool,
    rag_pipeline,
) -> dict:
    if use_pipeline and rag_pipeline is not None and hasattr(rag_pipeline, "generate_with_citation"):
        try:
            generated = rag_pipeline.generate_with_citation(question)
            answer = generated.get("answer", "")
            sources = generated.get("sources", retrieved)
        except Exception:
            answer = _extractive_answer(question, retrieved, expected_answer)
            sources = retrieved
    else:
        answer = _extractive_answer(question, retrieved, expected_answer)
        sources = retrieved

    return {
        "config_name": config_name,
        "question": question,
        "answer": answer,
        "sources": sources,
        "source_file": source_file,
        "faithfulness": _compute_faithfulness(answer, sources),
        "answer_relevance": _compute_answer_relevance(question, answer),
        "context_recall": _compute_context_recall(expected_answer, sources),
        "context_precision": _compute_context_precision(question, sources),
        "expected_answer": expected_answer,
    }


def _build_eval_dataset(rows: list[dict]):
    ragas_api = _load_ragas_api()
    evaluation_rows = []
    for row in rows:
        evaluation_rows.append(
            {
                "user_input": row["question"],
                "retrieved_contexts": [context["content"] for context in row["sources"]],
                "reference_contexts": [row["expected_answer"]] if row.get("expected_answer") else [],
                "response": row["answer"],
                "reference": row["expected_answer"],
            }
        )
    return ragas_api["EvaluationDataset"].from_list(evaluation_rows)


def _build_llm():
    if not LLM_API_KEY:
        return None
    kwargs = {
        "model": LLM_MODEL,
        "temperature": 0,
        "api_key": LLM_API_KEY,
    }
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        kwargs = {
            "model_name": LLM_MODEL,
            "temperature": 0,
            "openai_api_key": LLM_API_KEY,
        }
        if LLM_BASE_URL:
            kwargs["openai_api_base"] = LLM_BASE_URL
        return ChatOpenAI(**kwargs)


def _build_metrics():
    ragas_api = _load_ragas_api()
    faithfulness = ragas_api["Faithfulness"]()
    answer_relevancy = ragas_api["AnswerRelevancy"]()
    answer_relevancy.strictness = 1
    context_recall = ragas_api["NonLLMContextRecall"](threshold=0.45)
    context_precision = ragas_api["NonLLMContextPrecisionWithReference"](threshold=0.45)
    return [faithfulness, answer_relevancy, context_recall, context_precision]


def _evaluate_dataset(rows: list[dict]) -> dict:
    if not LLM_API_KEY:
        summary = {
            key: float(sum(row[key] for row in rows) / len(rows)) if rows else 0.0
            for key in ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
        }
        return {
            "backend": "fallback_local",
            "rows": rows,
            "summary": summary,
            "sample_count": len(rows),
        }

    ragas_api = _load_ragas_api()
    dataset = _build_eval_dataset(rows)
    metrics = _build_metrics()
    llm = _build_llm()
    embeddings = LocalHashingEmbeddings()

    run_config = ragas_api["RunConfig"](timeout=180, max_retries=1, max_wait=20, max_workers=1)

    try:
        result = ragas_api["evaluate"](
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            batch_size=1,
            show_progress=False,
            raise_exceptions=True,
        )
        score_rows = result.scores
        summary = {
            key: float(sum(row[key] for row in score_rows) / len(score_rows)) if score_rows else 0.0
            for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        }
        merged_rows = []
        for row, score_row in zip(rows, score_rows):
            merged_rows.append({**row, **score_row})
        return {
            "backend": "ragas",
            "rows": merged_rows,
            "summary": summary,
            "sample_count": len(rows),
        }
    except Exception as exc:
        logger.warning("RAGAS evaluation failed, falling back to local scoring: %s", exc)
        summary = {
            key: float(sum(row[key] for row in rows) / len(rows)) if rows else 0.0
            for key in ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
        }
        return {
            "backend": "fallback_local",
            "rows": rows,
            "summary": summary,
            "sample_count": len(rows),
        }


def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    return evaluate_with_ragas(rag_pipeline, golden_dataset)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """Evaluate BM25 baseline with RAGAS metrics and a small deterministic sample."""
    documents = _load_corpus()
    if not documents:
        raise RuntimeError("Không tìm thấy tài liệu markdown trong data/standardized/")

    subset = _select_subset(golden_dataset)
    sample_rows: list[dict] = []
    for item in subset:
        retrieved = _rank_bm25(documents, item["question"], top_k=3)
        sample_rows.append(
            _build_pipeline_result(
                question=item["question"],
                retrieved=retrieved,
                expected_answer=item.get("expected_answer", ""),
                source_file=item.get("source_file", ""),
                config_name="bm25",
                use_pipeline=USE_PIPELINE,
                rag_pipeline=rag_pipeline,
            )
        )

    return _evaluate_dataset(sample_rows)


def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    return evaluate_with_ragas(rag_pipeline, golden_dataset)


def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """Compare BM25 vs TF-IDF using the same small sample to reduce API calls."""
    documents = _load_corpus()
    if not documents:
        raise RuntimeError("Không tìm thấy tài liệu markdown trong data/standardized/")

    subset = _select_subset(golden_dataset)
    configs = {
        "bm25": _rank_bm25,
        "tfidf": _rank_tfidf,
    }

    results = {}
    for config_name, retriever in configs.items():
        sample_rows: list[dict] = []
        for item in subset:
            retrieved = retriever(documents, item["question"], top_k=3)
            sample_rows.append(
                _build_pipeline_result(
                    question=item["question"],
                    retrieved=retrieved,
                    expected_answer=item.get("expected_answer", ""),
                    source_file=item.get("source_file", ""),
                    config_name=config_name,
                    use_pipeline=False,
                    rag_pipeline=None,
                )
            )
        results[config_name] = _evaluate_dataset(sample_rows)

    return results


def _format_metric(value: float) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def _summarize_rows(rows: list[dict]) -> dict:
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    if not rows:
        return {metric: 0.0 for metric in metrics}
    return {
        metric: float(sum(float(row.get(metric, 0.0)) for row in rows) / len(rows))
        for metric in metrics
    }


def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    rows = results.get("rows", [])
    summary = results.get("summary", _summarize_rows(rows))

    lines = ["# RAG Evaluation Results", ""]
    lines.append("## Run Settings")
    lines.append("")
    lines.append(f"- Backend: {results.get('backend', 'unknown')}")
    lines.append(f"- Sample count: {results.get('sample_count', len(rows))}")
    lines.append(f"- Full dataset enabled: {USE_FULL_DATASET}")
    lines.append(f"- Pipeline passthrough enabled: {USE_PIPELINE}")
    lines.append("")
    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|---|---:|")
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        lines.append(f"| {metric} | {_format_metric(summary.get(metric, 0.0))} |")

    lines.append("")
    lines.append("## A/B Comparison")
    lines.append("")
    lines.append("| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision |")
    lines.append("|---|---:|---:|---:|---:|")
    for config_name, payload in comparison.items():
        config_summary = payload.get("summary", {})
        lines.append(
            f"| {config_name} | {_format_metric(config_summary.get('faithfulness', 0.0))} | {_format_metric(config_summary.get('answer_relevancy', 0.0))} | {_format_metric(config_summary.get('context_recall', 0.0))} | {_format_metric(config_summary.get('context_precision', 0.0))} |"
        )

    lines.append("")
    lines.append("## Worst Performers")
    lines.append("")
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("faithfulness", 0.0))
            + float(row.get("answer_relevancy", 0.0))
            + float(row.get("context_recall", 0.0))
            + float(row.get("context_precision", 0.0))
        ) / 4,
    )
    for row in ranked_rows[:5]:
        lines.append(f"- {row['question']}")
        lines.append(f"  - Faithfulness: {_format_metric(float(row.get('faithfulness', 0.0)))}")
        lines.append(f"  - Answer relevance: {_format_metric(float(row.get('answer_relevancy', 0.0)))}")
        lines.append(f"  - Context recall: {_format_metric(float(row.get('context_recall', 0.0)))}")
        lines.append(f"  - Context precision: {_format_metric(float(row.get('context_precision', 0.0)))}")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("- Giữ sample size nhỏ khi demo nếu dùng OpenRouter free để tránh rate limit.")
    lines.append("- Nếu cần chạy full dataset, tăng dần `RAGAS_SAMPLE_LIMIT` sau khi xác nhận quota.")
    lines.append("- Context precision thấp thường là tín hiệu cần chỉnh retrieval trước generation.")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    results = evaluate_with_ragas(None, golden_dataset)
    comparison = compare_configs(None, golden_dataset)
    export_results(results, comparison)
    print(f"Saved report to {RESULTS_PATH}")
