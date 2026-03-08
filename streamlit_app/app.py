import streamlit as st
import requests

API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Chat")
st.write("Upload documents and ask questions.")

# -------------------------
# Upload Documents
# -------------------------

# -------------------------
# Upload Documents
# -------------------------

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("Ingest Documents"):

    if not uploaded_files:
        st.sidebar.warning("Upload at least one file")
    else:

        with st.spinner("Uploading and processing documents..."):

            success = True

            for uploaded_file in uploaded_files:

                response = requests.post(
                    f"{API_URL}/ingest",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }
                )

                if response.status_code != 200:
                    success = False
                    st.sidebar.error(response.text)
                    break

        if success:
            st.sidebar.success("Documents ingested successfully!")

# -------------------------
# Query Section
# -------------------------

st.header("Ask a Question")

query = st.text_input(
    "Enter your question:",
    placeholder="What is the Transformer architecture?"
)

if st.button("Ask"):

    if not query:
        st.warning("Please enter a question")

    else:

        with st.spinner("Thinking..."):

            response = requests.post(
                f"{API_URL}/query",
                json={"query": query}
            )

        if response.status_code == 200:

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