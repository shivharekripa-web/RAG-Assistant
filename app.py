"""Streamlit chat UI for the Engineering Intelligence Hub.

A minimal chat-style front end over the FastAPI /query endpoint. Run with:

    streamlit run app.py

(with `uvicorn main:app --reload` running separately in another terminal).
"""

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

API_URL = os.getenv("API_URL", "http://localhost:8000/query")
UPLOAD_URL = os.getenv("UPLOAD_URL", "http://localhost:8000/upload")
REINDEX_URL = os.getenv("REINDEX_URL", "http://localhost:8000/reindex")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))
REINDEX_TIMEOUT_SECONDS = int(os.getenv("REINDEX_TIMEOUT_SECONDS", "600"))

st.set_page_config(page_title="Engineering Intelligence Hub", page_icon="🔎")


def ask_question(question: str) -> dict:
    """Send a question to the FastAPI /query endpoint and return the JSON body.

    Network and timeout errors are caught here (rather than left to crash
    the Streamlit app) and turned into a response dict shaped like a normal
    API error, so the rest of the UI can handle both cases identically.
    """
    try:
        response = requests.post(
            API_URL, json={"question": question}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": (
                "Couldn't reach the API server. Make sure it's running with "
                "'uvicorn main:app --reload'."
            ),
            "sources": [],
        }
    except requests.exceptions.Timeout:
        return {
            "answer": "The request timed out. Please try again.",
            "sources": [],
        }
    except requests.exceptions.RequestException as exc:
        return {"answer": f"Request failed: {exc}", "sources": []}


def upload_and_reindex(uploaded_files: list) -> dict:
    """Upload files to the backend, then trigger a reindex, in sequence.

    Returns a dict with "ok" and "message" describing the combined outcome.
    Both steps' errors are caught here so a network hiccup shows up as a
    clear message in the UI instead of an uncaught exception.
    """
    files_payload = [
        (upload.name, upload.getvalue(), "application/octet-stream")
        for upload in uploaded_files
    ]

    try:
        upload_response = requests.post(
            UPLOAD_URL,
            files=[("files", f) for f in files_payload],
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        upload_response.raise_for_status()
        upload_result = upload_response.json()
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "message": "Couldn't reach the API server. Make sure it's running "
            "with 'uvicorn main:app --reload'.",
        }
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "message": f"Upload failed: {exc}"}

    if not upload_result.get("ok"):
        return {"ok": False, "message": upload_result.get("message", "Upload failed.")}

    try:
        reindex_response = requests.post(REINDEX_URL, timeout=REINDEX_TIMEOUT_SECONDS)
        reindex_response.raise_for_status()
        reindex_result = reindex_response.json()
    except requests.exceptions.RequestException as exc:
        return {
            "ok": False,
            "message": f"Files were uploaded, but reindexing failed: {exc}",
        }

    if not reindex_result.get("ok"):
        return {
            "ok": False,
            "message": f"Files were uploaded, but reindexing failed: "
            f"{reindex_result.get('message')}",
        }

    return {
        "ok": True,
        "message": f"{upload_result['message']} {reindex_result['message']}",
    }


def render_sidebar() -> None:
    """Render the sidebar: usage instructions, scope reminder, and uploader."""
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "1. Ask a question about the codebase or docs below.\n"
            "2. Answers are generated only from what's been indexed.\n"
            "3. Expand **📄 Sources** under any answer to see exactly which "
            "files it came from."
        )
        st.caption(
            "Answers are based only on documents and code ingested from the "
            "`data/` folder -- not general knowledge."
        )

        st.divider()
        st.subheader("Add documents")
        uploaded_files = st.file_uploader(
            "Upload .md or .py files",
            type=["md", "py"],
            accept_multiple_files=True,
        )
        if st.button("Add & Rebuild Index", disabled=not uploaded_files):
            with st.spinner("Uploading and rebuilding the index..."):
                result = upload_and_reindex(uploaded_files)
            if result["ok"]:
                st.success(result["message"])
            else:
                st.error(result["message"])


def main() -> None:
    """Render the chat UI: title, question input, and answer with sources."""
    st.title("🔎 Engineering Intelligence Hub")
    st.caption("Ask a question about the codebase or its documentation.")

    render_sidebar()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="e.g. How does task assignment work?")
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Searching the docs and code..."):
            result = ask_question(question.strip())
        st.session_state.history.append(
            {
                "question": question.strip(),
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
            }
        )

    for entry in reversed(st.session_state.history):
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(entry["answer"])
        with st.expander("📄 Sources"):
            if entry["sources"]:
                for source in entry["sources"]:
                    st.markdown(f"- `{source}`")
            else:
                st.markdown("_No sources -- answer wasn't grounded in the indexed documents._")
        st.divider()


if __name__ == "__main__":
    main()
