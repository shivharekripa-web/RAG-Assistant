"""FastAPI backend for the Engineering Intelligence Hub.

Exposes:
  POST /query   accepts a natural-language question, returns a grounded,
                cited answer generated from the retrieved context.
  GET  /health  simple liveness check.

Run with:  uvicorn main:app --reload
"""

import os
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

import ingest
import retriever
from retriever import RetrieverNotReadyError, hybrid_search

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY")
GROQ_API_BASE_URL = os.environ.get("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

UPLOAD_DIR = BASE_DIR / "data" / "uploaded"
ALLOWED_UPLOAD_EXTENSIONS = {".md", ".py"}
GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))

# Below this combined hybrid score, we treat retrieval as "nothing relevant
# found" rather than risk asking the LLM to answer from weak/unrelated
# context. This is a heuristic threshold, not a precise calibration -- tune
# it if the sample data grows to include less related material.
MIN_RELEVANCE_SCORE = 0.15

app = FastAPI(title="Engineering Intelligence Hub")


@app.on_event("startup")
def warmup_retriever() -> None:
    """Preload retrieval resources so the first user query is not blocked by a cold start."""
    try:
        hybrid_search("warmup", top_k=1)
    except Exception:  # noqa: BLE001 - best effort warmup only
        pass


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    sources: List[str]


def _no_answer_response() -> QueryResponse:
    """Build the standard "not enough information" response.

    Shared by the empty-query, no-results, and low-relevance-score paths so
    the client always sees the same message shape for "we can't answer this".
    """
    return QueryResponse(
        answer="I don't have enough information in the indexed documents to answer this.",
        sources=[],
    )


def _format_source_label(chunk_metadata: dict) -> str:
    """Format a chunk's metadata into a human-readable source label.

    Code chunks include the line number (e.g. "src/auth.py:12") since that's
    the most useful pointer back into a source file; doc chunks just show
    the file name.
    """
    source_file = chunk_metadata.get("source_file", "unknown")
    start_line = chunk_metadata.get("start_line")
    if chunk_metadata.get("chunk_type") == "code" and start_line is not None:
        return f"{source_file}:{start_line}"
    return source_file


def _build_prompt(question: str, results: list) -> str:
    """Build the Groq prompt: retrieved context plus strict grounding rules.

    The instructions explicitly forbid answering from outside knowledge and
    require the model to cite which source file(s) support each part of the
    answer, so responses stay traceable back to the actual documents/code.
    """
    context_blocks = []
    for result in results:
        meta = result["metadata"]
        label = _format_source_label(meta)
        context_blocks.append(f"--- Source: {label} ---\n{meta['chunk_text']}")

    context_text = "\n\n".join(context_blocks)

    return (
        "You are an assistant that answers engineering questions using ONLY "
        "the context provided below. Do not use any outside knowledge, and "
        "do not guess or make up information that isn't in the context.\n\n"
        "If the context does not contain enough information to answer the "
        "question, say so explicitly instead of guessing.\n\n"
        "For every part of your answer, cite the source file name(s) it "
        "came from (shown above each context block), in the form "
        "(source: <file>).\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _call_groq(prompt: str) -> str:
    """Send the prompt to the Groq API endpoint."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    url = f"{GROQ_API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer engineering questions using only the provided context "
                    "and cite the relevant sources."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=GROQ_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _fallback_answer(question: str, results: list) -> QueryResponse:
    """Build a simple grounded response from retrieved context when Groq is unavailable."""
    snippets = []
    for result in results[:3]:
        meta = result["metadata"]
        label = _format_source_label(meta)
        text = (meta.get("chunk_text") or "").strip()
        if text:
            snippets.append(f"- {label}: {text}")

    if not snippets:
        return _no_answer_response()

    answer = (
        "I couldn't reach the AI model, so here's the best grounded answer from the indexed source: "
        f"{question}.\n\n"
        + "\n".join(snippets[:3])
    )
    sources = sorted({_format_source_label(r["metadata"]) for r in results})
    return QueryResponse(answer=answer, sources=sources)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question using retrieved, cited context.

    Retrieves relevant chunks via hybrid_search, and returns the standard
    "not enough information" response if nothing sufficiently relevant is
    found. Otherwise builds a grounded prompt, calls Groq, and returns the
    answer alongside the list of source files it was grounded in. Never
    raises -- all failure modes (empty query, no index, Groq errors) are
    caught and turned into a clear response instead of a crash.
    """
    question = (request.question or "").strip()
    if not question:
        return _no_answer_response()

    try:
        results = hybrid_search(question, top_k=5)
    except RetrieverNotReadyError as exc:
        return QueryResponse(answer=f"Error: {exc}", sources=[])
    except Exception as exc:  # noqa: BLE001 - surface any retrieval failure safely
        return QueryResponse(answer=f"Error during retrieval: {exc}", sources=[])

    if not results or results[0]["score"] < MIN_RELEVANCE_SCORE:
        return _no_answer_response()

    if not GROQ_API_KEY:
        return QueryResponse(
            answer=(
                "Error: GROQ_API_KEY is not set. Add it to your .env file "
                "before asking questions."
            ),
            sources=[],
        )

    prompt = _build_prompt(question, results)
    sources = sorted({_format_source_label(r["metadata"]) for r in results})

    try:
        answer_text = _call_groq(prompt).strip()
        if not answer_text:
            return QueryResponse(
                answer="The model returned an empty response. Please try rephrasing your question.",
                sources=sources,
            )
        return QueryResponse(answer=answer_text, sources=sources)
    except Exception:  # noqa: BLE001 - never let a Groq failure crash the API
        return _fallback_answer(question, results)


class UploadResponse(BaseModel):
    """Response body for POST /upload."""

    ok: bool
    message: str
    saved_files: List[str]


class ReindexResponse(BaseModel):
    """Response body for POST /reindex."""

    ok: bool
    message: str


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)) -> UploadResponse:
    """Save uploaded .md/.py files into data/uploaded/ for later ingestion.

    This only saves files to disk -- it does not rebuild the index. Call
    /reindex afterward (the Streamlit UI does this automatically) to make
    the newly uploaded content actually searchable.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    skipped = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            skipped.append(upload.filename)
            continue

        destination = UPLOAD_DIR / Path(upload.filename).name
        contents = await upload.read()
        destination.write_bytes(contents)
        saved.append(str(destination))

    if not saved:
        return UploadResponse(
            ok=False,
            message="No valid .md or .py files were uploaded.",
            saved_files=[],
        )

    message = f"Saved {len(saved)} file(s)."
    if skipped:
        message += f" Skipped {len(skipped)} unsupported file(s): {', '.join(skipped)}."

    return UploadResponse(ok=True, message=message, saved_files=saved)


@app.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    """Re-run ingestion over data/ and reload the retriever's cached index.

    Rebuilds the FAISS/BM25/metadata indexes from everything currently in
    data/ (including anything just uploaded via /upload), then clears the
    retriever's in-memory cache so this running server starts using the new
    indexes immediately, without needing a restart.
    """
    result = ingest.run_ingestion()
    if not result["ok"]:
        return ReindexResponse(ok=False, message=result["error"])

    retriever.reset_cache()

    message = (
        f"Re-indexed {result['files_processed']} file(s) into "
        f"{result['chunks_created']} chunks "
        f"({result['doc_chunks']} doc, {result['code_chunks']} code)."
    )
    return ReindexResponse(ok=True, message=message)


@app.get("/health")
def health() -> dict:
    """Simple liveness check -- returns {"status": "ok"} if the server is up."""
    return {"status": "ok"}
