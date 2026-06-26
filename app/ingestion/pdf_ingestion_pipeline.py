import os
from typing import List

from langchain_groq import ChatGroq

from app.retrieval.vector_store import vector_store

PDF_PARSE_STRATEGY = os.getenv("PDF_PARSE_STRATEGY", "fast")
ENRICH_CHUNK_QUESTIONS = os.getenv("ENRICH_CHUNK_QUESTIONS", "false").lower() == "true"


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )


def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    words = [w.lower() for w in text.split() if len(w) > 4]
    return list(set(words))[:max_keywords]


def generate_questions(text: str) -> List[str]:
    prompt = f"""
Generate 3 questions a user might ask about the following text.

Text:
{text[:1500]}

Return only the questions, one per line.
"""
    try:
        response = get_llm().invoke(prompt)
        lines = response.content.split("\n")
        return [q.strip("- ").strip() for q in lines if q.strip()][:3]
    except Exception:
        return []


def _chunk_text(text: str, chunk_size: int = 2400, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def ingest_txt(file_path: str) -> int:
    filename = os.path.basename(file_path)
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    stored_chunks = 0
    for chunk_text in _chunk_text(text):
        metadata = {
            "source": filename,
            "keywords": extract_keywords(chunk_text),
            "type": "text",
        }
        if ENRICH_CHUNK_QUESTIONS:
            metadata["questions"] = generate_questions(chunk_text)

        vector_store.add_text(text=chunk_text, metadata=metadata)
        stored_chunks += 1

    return stored_chunks


def ingest_file(file_path: str) -> int:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    if filename.lower().endswith(".txt"):
        return ingest_txt(file_path)

    from unstructured.chunking.title import chunk_by_title
    from unstructured.partition.pdf import partition_pdf

    print(f"Partitioning document: {file_path} (strategy={PDF_PARSE_STRATEGY})")

    partition_kwargs = {
        "filename": file_path,
        "strategy": PDF_PARSE_STRATEGY,
    }
    if PDF_PARSE_STRATEGY == "hi_res":
        partition_kwargs.update({
            "infer_table_structure": True,
            "extract_image_block_types": ["Image"],
            "extract_image_block_to_payload": True,
        })

    elements = partition_pdf(**partition_kwargs)
    print(f"Extracted {len(elements)} elements")

    images = [el for el in elements if el.category == "Image"]
    tables = [el for el in elements if el.category == "Table"]
    print(f"Found {len(images)} images, {len(tables)} tables")

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )
    print(f"Created {len(chunks)} chunks")

    stored_chunks = 0
    for chunk in chunks:
        chunk_text = chunk.text
        if isinstance(chunk_text, list):
            chunk_text = " ".join(str(t) for t in chunk_text)
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        metadata = chunk.metadata.to_dict() if chunk.metadata else {}
        metadata.update({
            "source": filename,
            "keywords": extract_keywords(chunk_text),
            "type": "text",
        })
        if ENRICH_CHUNK_QUESTIONS:
            metadata["questions"] = generate_questions(chunk_text)

        vector_store.add_text(text=chunk_text, metadata=metadata)
        stored_chunks += 1

    for table in tables:
        html = getattr(table.metadata, "text_as_html", "")
        if isinstance(html, list):
            html = " ".join(str(t) for t in html)
        html = html.strip()
        if html:
            vector_store.add_text(
                text=html,
                metadata={"source": filename, "type": "table"},
            )

    print(f"Ingested {stored_chunks} chunks")
    return stored_chunks
