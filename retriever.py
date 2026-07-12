"""Hybrid retrieval logic for the Engineering Intelligence Hub.

Loads the FAISS dense index, BM25 sparse index, and chunk metadata built by
ingest.py, then exposes hybrid_search(), which combines both retrieval
methods into a single ranked list.

Why hybrid search: dense (embedding) search is good at matching semantic
meaning even when wording differs, but it can miss exact identifiers -- a
specific function name, error code, or config key -- because those don't
carry much "meaning" for an embedding model to latch onto. Sparse (BM25)
keyword search catches those exact matches reliably but misses paraphrases.
Combining both catches more relevant chunks than either alone.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import EMBEDDING_MODEL_NAME, tokenize

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "indexes"

_model = None
_faiss_index = None
_bm25 = None
_metadata = None


class RetrieverNotReadyError(Exception):
    """Raised when hybrid_search is called before indexes have been built.

    Callers (main.py) catch this and return a clear error to the client
    instead of letting the server crash on missing files.
    """


def _load_resources() -> None:
    """Lazily load the embedding model and both indexes into module state.

    Loaded once on first use rather than at import time, so importing this
    module (e.g. from tests) doesn't require the indexes to already exist.
    """
    global _model, _faiss_index, _bm25, _metadata

    if _model is not None:
        return

    dense_path = INDEX_DIR / "dense.index"
    bm25_path = INDEX_DIR / "bm25.pkl"
    metadata_path = INDEX_DIR / "metadata.json"

    if not (dense_path.exists() and bm25_path.exists() and metadata_path.exists()):
        raise RetrieverNotReadyError(
            "Indexes not found. Run 'python ingest.py' before starting the API."
        )

    _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    _faiss_index = faiss.read_index(str(dense_path))
    with open(bm25_path, "rb") as f:
        _bm25 = pickle.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)


def reset_cache() -> None:
    """Clear the cached model/index/metadata so the next call reloads them.

    Called after re-running ingestion (e.g. via the /reindex endpoint) so a
    long-running server process picks up newly rebuilt indexes instead of
    continuing to serve the ones it loaded at startup.
    """
    global _model, _faiss_index, _bm25, _metadata
    _model = None
    _faiss_index = None
    _bm25 = None
    _metadata = None


def _min_max_normalize(scores: dict) -> dict:
    """Min-max normalize a {index: score} dict to the [0, 1] range.

    If every score is identical (zero range), all normalized scores are set
    to 1.0 rather than dividing by zero, so a single matching candidate
    isn't unfairly zeroed out.
    """
    if not scores:
        return {}

    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {idx: 1.0 for idx in scores}

    return {idx: (val - lo) / (hi - lo) for idx, val in scores.items()}


def hybrid_search(query: str, top_k: int = 5) -> list:
    """Return the top_k chunks most relevant to `query`, using hybrid search.

    Runs dense (FAISS) and sparse (BM25) retrieval independently, min-max
    normalizes each score set over the union of retrieved candidates,
    combines them with equal (0.5 / 0.5) weighting, deduplicates by chunk
    index, and returns the top_k merged results.

    Each result is a dict: {"score": float, "metadata": <chunk metadata>}.
    Raises RetrieverNotReadyError if indexes haven't been built yet.
    """
    _load_resources()

    query = (query or "").strip()
    if not query:
        return []

    # --- Dense retrieval ---
    query_embedding = _model.encode([query], normalize_embeddings=True)
    dense_k = min(top_k, _faiss_index.ntotal)
    dense_scores_arr, dense_indices_arr = _faiss_index.search(
        query_embedding.astype("float32"), max(dense_k, 1)
    )
    dense_scores = {}
    for idx, score in zip(dense_indices_arr[0], dense_scores_arr[0]):
        if idx == -1:
            continue
        dense_scores[int(idx)] = float(score)

    # --- Sparse retrieval ---
    query_tokens = tokenize(query)
    all_bm25_scores = _bm25.get_scores(query_tokens)
    sparse_k = min(top_k, len(all_bm25_scores))
    top_sparse_indices = np.argsort(all_bm25_scores)[::-1][:sparse_k]
    sparse_scores = {int(i): float(all_bm25_scores[i]) for i in top_sparse_indices}

    # --- Normalize each score set over the union of candidates ---
    candidate_indices = set(dense_scores) | set(sparse_scores)
    dense_norm = _min_max_normalize(
        {i: dense_scores.get(i, 0.0) for i in candidate_indices}
    )
    sparse_norm = _min_max_normalize(
        {i: sparse_scores.get(i, 0.0) for i in candidate_indices}
    )

    combined = {
        i: 0.5 * dense_norm.get(i, 0.0) + 0.5 * sparse_norm.get(i, 0.0)
        for i in candidate_indices
    }

    ranked_indices = sorted(combined, key=lambda i: combined[i], reverse=True)[:top_k]

    results = []
    for idx in ranked_indices:
        results.append({"score": combined[idx], "metadata": _metadata[idx]})

    return results
