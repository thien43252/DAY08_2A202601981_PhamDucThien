"""Task 4: Chunk Markdown documents, embed them, and index them in ChromaDB.

The recursive splitter is used because it preserves paragraphs and sentences where
possible, while its character limit works consistently for both Vietnamese and
English documents.  BGE-M3 is a 1024-dimensional multilingual embedding model,
so it supports the bilingual university-services corpus without an external API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

# 800 characters retains useful context; 100 characters keeps boundary context.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# OpenAI text-embedding-3-small (API-based, 1536 dimensions).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


def load_documents() -> list[dict[str, Any]]:
    """Load non-empty ``.md`` files under :data:`STANDARDIZED_DIR`.

    The relative source path, document type, and filename are retained as
    metadata so later retrieval can cite the original document.
    """
    if not STANDARDIZED_DIR.exists():
        return []

    documents: list[dict[str, Any]] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = path.relative_to(STANDARDIZED_DIR)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": relative_path.as_posix(),
                    "filename": path.name,
                    "type": relative_path.parts[0] if relative_path.parts else "unknown",
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split documents recursively and attach a per-document chunk index."""
    if not documents:
        return []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "。", " ", ""],
        )
        split_text = splitter.split_text
    except ImportError:
        # Fallback keeps the chunking step usable when only the standard
        # library is available; install requirements.txt for production use.
        def split_text(text: str) -> list[str]:
            chunks: list[str] = []
            start = 0
            separators = ("\n\n", "\n", ". ", " ")
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                if end < len(text):
                    boundary = max(text.rfind(separator, start, end) for separator in separators)
                    if boundary > start:
                        end = boundary + 1
                piece = text[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(text):
                    break
                start = max(start + 1, end - CHUNK_OVERLAP)
            return chunks

    chunks: list[dict[str, Any]] = []
    for document in documents:
        metadata = dict(document.get("metadata", {}))
        for chunk_index, content in enumerate(split_text(document["content"])):
            if content.strip():
                chunks.append(
                    {
                        "content": content,
                        "metadata": {**metadata, "chunk_index": chunk_index},
                    }
                )
    return chunks


def _get_openai_client():
    """Create an OpenAI client using the API key from environment."""
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY chưa được thiết lập trong .env")
    return OpenAI(api_key=api_key)


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add OpenAI text-embedding-3-small embeddings to each chunk."""
    if not chunks:
        return []

    client = _get_openai_client()
    texts = [chunk["content"] for chunk in chunks]

    # OpenAI API supports batch embedding, process in batches of 100
    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1} ({len(batch)} chunks)...")
        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
        )
        # Sort by index to preserve order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        batch_embeddings = [item.embedding for item in sorted_data]
        all_embeddings.extend(batch_embeddings)

    for chunk, embedding in zip(chunks, all_embeddings):
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding model trả về {len(embedding)} chiều; cần {EMBEDDING_DIM}."
            )
        chunk["embedding"] = [float(value) for value in embedding]
    return chunks


def get_collection():
    """Open (or create) the persistent ChromaDB collection using cosine distance."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Cần cài chromadb để chạy Task 4.") from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict[str, Any]], reset: bool = True):
    """Persist chunks in ChromaDB.

    ``reset=True`` is the safe default: it removes the named collection before
    re-indexing so deleted or changed source files cannot remain searchable.
    """
    if not chunks:
        raise ValueError("Không có chunk để index.")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("Mọi chunk phải có embedding trước khi index.")

    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Cần cài chromadb để chạy Task 4.") from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            # The collection may not yet exist; creating it below is expected.
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [
        f"{chunk['metadata'].get('source', 'unknown')}:{chunk['metadata'].get('chunk_index', index)}"
        for index, chunk in enumerate(chunks)
    ]
    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def run_pipeline() -> None:
    """Run load → chunk → embed → index and print a compact summary."""
    documents = load_documents()
    if not documents:
        raise FileNotFoundError(
            f"Chưa có Markdown trong {STANDARDIZED_DIR}; hãy chạy Task 3 trước."
        )

    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    collection = index_to_vectorstore(embedded_chunks)
    print(f"Da load {len(documents)} tai lieu, index {collection.count()} chunks vao {CHROMA_DIR}.")


if __name__ == "__main__":
    run_pipeline()
