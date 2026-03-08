import os
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_groq import ChatGroq

from app.retrieval.vector_store import vector_store


# ---------- LLM Setup ----------
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    """Simple keyword extractor."""
    words = [w.lower() for w in text.split() if len(w) > 4]
    unique = list(set(words))
    return unique[:max_keywords]


def generate_questions(text: str) -> List[str]:
    """Use LLM to generate questions about a chunk."""

    prompt = f"""
Generate 3 questions a user might ask about the following text.

Text:
{text[:1500]}

Return only the questions, one per line.
"""
    llm=get_llm()
    try:
        response = llm.invoke(prompt)
        lines = response.content.split("\n")

        questions = [
            q.strip("- ").strip()
            for q in lines
            if q.strip()
        ]

        return questions[:3]

    except Exception:
        return []


def ingest_file(file_path: str) -> int:
    """
    Load a PDF, extract text/tables/images, create smart chunks,
    enrich them with metadata, and add them to the vector store.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)

    print(f"📄 Partitioning document: {file_path}")

    # ---------- 1. Partition PDF ----------
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )

    print(f"✅ Extracted {len(elements)} elements")

    # ---------- 2. Separate images and tables ----------
    images = [el for el in elements if el.category == "Image"]
    tables = [el for el in elements if el.category == "Table"]

    print(f"🖼 Found {len(images)} images")
    print(f"📊 Found {len(tables)} tables")

    # ---------- 3. Create intelligent chunks ----------
    print("🔨 Creating smart chunks...")

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )

    print(f"✅ Created {len(chunks)} chunks")

    stored_chunks = 0

    # ---------- 4. Add chunks to vector store ----------
    for chunk in chunks:
        chunk_text = chunk.text
        if isinstance(chunk_text, list):
         chunk_text = " ".join([str(t) for t in chunk_text])

         
        chunk_text = chunk_text.strip() 
        if not chunk_text:
            continue

        # Generate metadata
        metadata = {}

        if chunk.metadata:
            metadata = chunk.metadata.to_dict()

        metadata.update({
            "source": filename,
            "keywords": extract_keywords(chunk_text),
            "questions": generate_questions(chunk_text),
            "type": "text"
        })

        vector_store.add_text(
            text=chunk_text,
            metadata=metadata
        )

        stored_chunks += 1

    # ---------- 5. Store tables ----------
    for table in tables:

        html = getattr(table.metadata, "text_as_html", "")
        if isinstance(html, list):
            html = " ".join([str(t) for t in html])
        html = html.strip()

        if html:

            vector_store.add_text(
                text=html,
                metadata={
                    "source": filename,
                    "type": "table"
                }
            )

    print(f"✅ Ingested {stored_chunks} chunks")

    return stored_chunks