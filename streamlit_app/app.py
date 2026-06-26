import os

import requests
import streamlit as st
from requests.exceptions import ChunkedEncodingError, ConnectionError, RequestException, Timeout

_base_url = os.getenv("API_URL", "http://localhost:8000/api").rstrip("/")
API_URL = _base_url if _base_url.endswith("/api") else f"{_base_url}/api"
INGEST_TIMEOUT = int(os.getenv("INGEST_TIMEOUT", "300"))
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "120"))


def api_post(path: str, timeout: int, **kwargs) -> requests.Response | None:
    url = f"{API_URL}{path}"
    try:
        response = requests.post(url, timeout=timeout, **kwargs)
        return response
    except Timeout:
        st.error(
            f"Request timed out after {timeout}s. "
            "Try a smaller TXT file, or wait for the backend to finish waking up (free tier cold start)."
        )
        return None
    except (ConnectionError, ChunkedEncodingError):
        st.error(
            "Backend connection dropped during processing. This usually means:\n\n"
            "1. **Render free tier timeout** (~100s) — set `RETRIEVER_MODE=lite` on the backend\n"
            "2. **Backend OOM restart** — use a smaller file or upgrade to Starter ($7/mo)\n"
            "3. **Wrong API_URL** — confirm Streamlit env points to your backend URL\n\n"
            f"Backend URL in use: `{API_URL}`"
        )
        return None
    except RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Document Chat")
st.write("Upload documents and ask questions.")

with st.sidebar.expander("Backend connection"):
    st.code(API_URL, language=None)
    try:
        health = requests.get(API_URL.replace("/api", "") + "/health", timeout=10)
        st.success(f"Backend: {health.json()}" if health.ok else f"Health check failed ({health.status_code})")
    except RequestException:
        st.error("Cannot reach backend /health")

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if st.sidebar.button("Ingest Documents"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file")
    else:
        with st.spinner("Uploading and processing documents..."):
            success = True
            for uploaded_file in uploaded_files:
                response = api_post(
                    "/ingest",
                    timeout=INGEST_TIMEOUT,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    },
                )
                if response is None:
                    success = False
                    break
                if response.status_code != 200:
                    success = False
                    st.sidebar.error(response.text or f"Ingest failed (HTTP {response.status_code})")
                    break

            if success:
                st.sidebar.success("Documents ingested successfully!")

st.header("Ask a Question")

query = st.text_input(
    "Enter your question:",
    placeholder="What is the Transformer architecture?",
)

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            response = api_post("/query", timeout=QUERY_TIMEOUT, json={"query": query})

        if response is None:
            pass
        elif response.status_code == 200:
            data = response.json()
            st.subheader("Answer")
            st.write(data["answer"])
            if data.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(data["sources"], 1):
                        st.markdown(f"**Source {i}**")
                        st.write(src)
        else:
            st.error(response.text or f"Query failed (HTTP {response.status_code})")
