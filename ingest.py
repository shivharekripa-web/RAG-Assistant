"""Ingestion script for the Engineering Intelligence Hub.

Recursively scans DATA_DIR for Markdown (.md) and Python (.py) files, splits
each into source-aware chunks, embeds the chunks, and builds two local
indexes:

  - indexes/dense.index   a FAISS index over sentence-transformer embeddings
  - indexes/bm25.pkl      a BM25Okapi index over tokenized chunk text
  - indexes/metadata.json chunk metadata, position-aligned with both indexes

Run this once after adding files to data/, and re-run any time data/ changes.

    python ingest.py
"""

import ast
import json
import pickle
import re
import sys
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Fallback fixed-size chunking parameters, used only when source-aware
# chunking isn't possible (e.g. a .py file with a syntax error).
FALLBACK_CHUNK_SIZE = 500
FALLBACK_CHUNK_OVERLAP = 50

_HEADING_RE = re.compile(r"^(#{1,6})\s+.*$")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def find_source_files(data_dir: Path) -> list:
    """Recursively find all .md and .py files under data_dir.

    Returns a sorted list of Path objects so ingestion order (and therefore
    chunk indices) is stable across repeated runs.
    """
    files = list(data_dir.rglob("*.md")) + list(data_dir.rglob("*.py"))
    return sorted(files)


def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list:
    """Split text into overlapping fixed-size character chunks.

    Used as a fallback when structure-aware chunking (headings for
    Markdown, ast parsing for Python) doesn't apply. Overlap is included so
    a chunk boundary landing mid-thought doesn't lose context entirely.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)
    step = max(chunk_size - overlap, 1)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start += step

    return chunks


def chunk_markdown_file(path: Path) -> list:
    """Split a Markdown file into one chunk per heading section.

    Each chunk keeps its heading line as part of the chunk text, since the
    heading is often necessary context for understanding the section (e.g.
    "## Known limitations" tells you how to interpret the bullets below it).
    Any content before the first heading becomes its own leading chunk.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    sections = []
    current_lines: list = []

    for line in lines:
        if _HEADING_RE.match(line) and current_lines:
            sections.append("\n".join(current_lines).strip())
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append("\n".join(current_lines).strip())

    sections = [s for s in sections if s]

    if not sections:
        # No headings at all (or empty file) - fall back to fixed-size.
        return [
            {"chunk_text": c, "start_line": None}
            for c in fixed_size_chunks(text, FALLBACK_CHUNK_SIZE, FALLBACK_CHUNK_OVERLAP)
        ]

    return [{"chunk_text": s, "start_line": None} for s in sections]


def chunk_python_file(path: Path) -> list:
    """Split a Python file into one chunk per top-level function or class.

    Uses the ast module to find top-level FunctionDef / AsyncFunctionDef /
    ClassDef nodes and extracts their exact source text (including
    decorators and docstrings) via ast.get_source_segment. If the file has
    top-level code outside any function/class (imports, constants, a
    __main__ block), that's captured as one additional "module-level" chunk
    so it isn't silently dropped. Falls back to fixed-size chunking only if
    the file can't be parsed at all (e.g. a syntax error).
    """
    source = path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            {"chunk_text": c, "start_line": None}
            for c in fixed_size_chunks(source, FALLBACK_CHUNK_SIZE, FALLBACK_CHUNK_OVERLAP)
        ]

    chunks = []
    covered_line_ranges = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(source, node)
            if segment is None or not segment.strip():
                continue
            chunks.append({"chunk_text": segment, "start_line": node.lineno})
            end_line = getattr(node, "end_lineno", node.lineno)
            covered_line_ranges.append((node.lineno, end_line))

    if not chunks:
        # File parsed fine but has no top-level functions/classes at all
        # (e.g. a pure script or a constants-only module).
        return [
            {"chunk_text": c, "start_line": None}
            for c in fixed_size_chunks(source, FALLBACK_CHUNK_SIZE, FALLBACK_CHUNK_OVERLAP)
        ]

    # Capture any top-level lines not covered by a function/class (module
    # docstring, imports, module-level constants) as one extra chunk.
    source_lines = source.splitlines()
    covered = set()
    for start, end in covered_line_ranges:
        covered.update(range(start, end + 1))
    leftover_lines = [
        line for i, line in enumerate(source_lines, start=1) if i not in covered
    ]
    leftover_text = "\n".join(leftover_lines).strip()
    if leftover_text:
        chunks.insert(0, {"chunk_text": leftover_text, "start_line": 1})

    return chunks


def build_chunks(files: list) -> list:
    """Chunk every file in `files` and return a flat list of chunk records.

    Each record has: source_file, chunk_type ("doc" or "code"), chunk_text,
    start_line (None for doc chunks, an integer for code chunks where known).
    """
    all_chunks = []

    for path in files:
        relative_path = str(path)
        if path.suffix == ".md":
            raw_chunks = chunk_markdown_file(path)
            chunk_type = "doc"
        elif path.suffix == ".py":
            raw_chunks = chunk_python_file(path)
            chunk_type = "code"
        else:
            continue

        for raw in raw_chunks:
            all_chunks.append(
                {
                    "source_file": relative_path,
                    "chunk_type": chunk_type,
                    "chunk_text": raw["chunk_text"],
                    "start_line": raw["start_line"],
                }
            )

    return all_chunks


def tokenize(text: str) -> list:
    """Lowercase, alphanumeric-only tokenization used for the BM25 index.

    Kept as a standalone function (rather than inlined) so retriever.py can
    import and reuse the exact same tokenizer for queries, which BM25
    requires for consistent scoring.
    """
    return _WORD_RE.findall(text.lower())


def build_indexes(chunks: list, index_dir: Path) -> None:
    """Embed all chunks, then build and save the FAISS and BM25 indexes.

    The FAISS index uses inner product over L2-normalized embeddings, which
    is equivalent to cosine similarity and lets retriever.py compare scores
    directly with BM25's min-max normalized scores.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [c["chunk_text"] for c in chunks]

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    )

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings.astype("float32"))
    faiss.write_index(faiss_index, str(index_dir / "dense.index"))

    print("Building BM25 index...")
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)


def run_ingestion(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR) -> dict:
    """Run the full ingestion pipeline and return a summary dict.

    This is the programmatic entry point: it does the same work as running
    `python ingest.py` from the command line, but returns a result instead
    of printing-and-exiting, so callers like app.py's "rebuild index"
    button can invoke it in-process and show the outcome in the UI.

    Returns a dict shaped either:
      {"ok": True, "files_processed": int, "chunks_created": int,
       "doc_chunks": int, "code_chunks": int, "index_dir": str}
    or:
      {"ok": False, "error": str}
    """
    if not data_dir.exists():
        return {"ok": False, "error": f"Data directory '{data_dir}' does not exist."}

    files = find_source_files(data_dir)
    if not files:
        return {
            "ok": False,
            "error": f"No .md or .py files found under '{data_dir}'. "
            "Add some files there and try again.",
        }

    chunks = build_chunks(files)
    if not chunks:
        return {
            "ok": False,
            "error": "Files were found but no chunks were produced. Nothing to index.",
        }

    build_indexes(chunks, index_dir)

    doc_chunks = sum(1 for c in chunks if c["chunk_type"] == "doc")
    code_chunks = sum(1 for c in chunks if c["chunk_type"] == "code")

    return {
        "ok": True,
        "files_processed": len(files),
        "chunks_created": len(chunks),
        "doc_chunks": doc_chunks,
        "code_chunks": code_chunks,
        "index_dir": str(index_dir.resolve()),
    }


def main() -> None:
    """CLI entry point: run ingestion and print a human-readable summary."""
    result = run_ingestion()

    if not result["ok"]:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print("\nIngestion complete.")
    print(f"  Files processed : {result['files_processed']}")
    print(
        f"  Chunks created  : {result['chunks_created']}  "
        f"({result['doc_chunks']} doc, {result['code_chunks']} code)"
    )
    print(f"  Indexes saved to: {result['index_dir']}")


if __name__ == "__main__":
    main()
