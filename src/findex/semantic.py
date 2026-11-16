"""Optional dense retrieval using normalized sentence-transformer embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .index_model import Index
from .ranking import BM25, SearchResult, search_numpy
from .tokenize import tokenize

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60


def _load_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Semantic search requires the optional dependency. Install it with "
            "`uv sync --extra semantic`."
        ) from exc
    return SentenceTransformer(model_name)


def _document_text(document: Any) -> str:
    if document.content_path and document.line_offset is not None:
        with Path(document.content_path).open("rb") as stream:
            stream.seek(document.line_offset)
            record = json.loads(stream.readline().decode("utf-8"))
        return str(record.get("text", ""))
    return Path(document.path).read_text(encoding="utf-8", errors="replace")


def _chunks(text: str, words_per_chunk: int = 160, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, words_per_chunk - overlap)
    return [
        " ".join(words[start : start + words_per_chunk])
        for start in range(0, len(words), step)
    ]


def embed_index(
    index: Index,
    output_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> tuple[Path, int]:
    """Encode document chunks and save a normalized float32 .npy matrix.

    A JSON sidecar stores the document/chunk mapping and short source excerpts.
    """
    model = _load_model(model_name)
    texts: list[str] = []
    metadata: list[dict[str, Any]] = []
    for doc_id in sorted(index.documents):
        document = index.documents[doc_id]
        for chunk_no, chunk in enumerate(_chunks(_document_text(document))):
            texts.append(chunk)
            metadata.append(
                {
                    "doc_id": doc_id,
                    "chunk": chunk_no,
                    "title": document.title,
                    "path": document.path,
                    "text": chunk[:600],
                }
            )
    if not texts:
        raise ValueError("No non-empty document chunks were available to embed")

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(metadata):
        raise ValueError("Embedding model returned an unexpected matrix shape")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        matrix = matrix / np.maximum(norms[:, None], 1e-12)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, matrix)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "model": model_name,
                "dimension": int(matrix.shape[1]),
                "chunks": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, len(metadata)


def semantic_search(
    index: Index,
    query: str,
    embeddings_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 10,
) -> list[SearchResult]:
    """Rank documents by their best matching chunk cosine similarity."""
    terms = list(tokenize(query))
    if not terms:
        return []
    model = _load_model(model_name)
    query_vector = np.asarray(
        model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0],
        dtype=np.float32,
    )
    matrix = np.load(Path(embeddings_path), mmap_mode="r")
    sidecar = Path(embeddings_path).with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))["chunks"]
    if matrix.ndim != 2 or matrix.shape[0] != len(metadata):
        raise ValueError("Embedding matrix and metadata do not match")
    if matrix.shape[1] != query_vector.shape[0]:
        raise ValueError("Query embedding dimension does not match stored vectors")

    similarities = np.asarray(matrix @ query_vector, dtype=np.float32)
    best: dict[int, tuple[float, dict[str, Any]]] = {}
    for score, item in zip(similarities, metadata, strict=True):
        doc_id = int(item["doc_id"])
        current = best.get(doc_id)
        if current is None or float(score) > current[0]:
            best[doc_id] = (float(score), item)

    ranked = sorted(best.items(), key=lambda pair: (pair[1][0], pair[0]), reverse=True)
    results: list[SearchResult] = []
    for doc_id, (score, item) in ranked[:top_k]:
        document = index.documents.get(doc_id)
        if document is None:
            continue
        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=document.title,
                path=document.path,
                snippet=str(item.get("text", "")),
                match_type="semantic",
            )
        )
    return results


def reciprocal_rank_fusion(
    lexical: list[SearchResult],
    semantic: list[SearchResult],
    *,
    top_k: int = 10,
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    """Fuse two ranked lists using Reciprocal Rank Fusion."""
    fused: dict[int, float] = {}
    items: dict[int, SearchResult] = {}
    for ranking in (lexical, semantic):
        for rank, item in enumerate(ranking, start=1):
            fused[item.doc_id] = fused.get(item.doc_id, 0.0) + 1.0 / (rrf_k + rank)
            previous = items.get(item.doc_id)
            if previous is None or item.match_type == "semantic":
                items[item.doc_id] = item
    ordered = sorted(fused, key=lambda doc_id: (fused[doc_id], doc_id), reverse=True)
    return [
        SearchResult(
            score=fused[doc_id],
            doc_id=doc_id,
            title=items[doc_id].title,
            path=items[doc_id].path,
            snippet=items[doc_id].snippet,
            match_type="hybrid",
        )
        for doc_id in ordered[:top_k]
    ]


def hybrid_search(
    index: Index,
    query: str,
    embeddings_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 10,
) -> list[SearchResult]:
    lexical = search_numpy(index, list(tokenize(query)), BM25(), top_k=top_k)
    dense = semantic_search(
        index,
        query,
        embeddings_path,
        model_name=model_name,
        top_k=top_k,
    )
    return reciprocal_rank_fusion(lexical, dense, top_k=top_k)
