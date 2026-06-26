import os

import requests
import streamlit as st

_base_url = os.getenv("API_URL", "http://localhost:8000/api").rstrip("/")
API_URL = _base_url if _base_url.endswith("/api") else f"{_base_url}/api"
INGEST_TIMEOUT = int(os.getenv("INGEST_TIMEOUT", "600"))
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "120"))


def api_post(path: str, timeout: int, **kwargs) -> requests.Response | None:
    try:
        return requests.post(f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the backend API. It may have crashed or restarted — "
            "this often happens when Docker runs out of memory during PDF processing. "
            "Wait ~30 seconds and try again, use a smaller PDF, or increase Docker memory to 8GB."
        )
        return None
    except requests.exceptions.Timeout:
        st.error(
            f"Request timed out after {timeout}s. Large PDFs can take several minutes — try again or use a smaller file."
        )
        return None


st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Document Chat")
st.write("Upload documents and ask questions.")

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
        with st.spinner("Uploading and processing documents (may take a few minutes)..."):
            success = True
            for uploaded_file in uploaded_files:
                response = api_post(
                    "/ingest",
                    timeout=INGEST_TIMEOUT,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                )
                if response is None:
                    success = False
                    break
                if response.status_code != 200:
                    success = False
                    st.sidebar.error(response.text)
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
            if "sources" in data:
                with st.expander("Sources"):
                    for i, src in enumerate(data["sources"], 1):
                        st.markdown(f"**Source {i}**")
                        st.write(src)
        else:
            st.error(response.text)
