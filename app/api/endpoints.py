from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import os
import logging

router = APIRouter()
retriever = None
_llm_client = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        from app.llm.llm_client import LLMClient
        _llm_client = LLMClient()
    return _llm_client


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    global retriever

    from langchain_classic.schema import Document
    from app.ingestion.pdf_ingestion_pipeline import ingest_file
    from app.retrieval.retriever import build_retriever
    from app.retrieval.vector_store import get_vector_store

    try:
        file_location = f"temp_uploads/{file.filename}"
        os.makedirs("temp_uploads", exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        ingest_file(file_location)

        store = get_vector_store()
        documents = [
            Document(page_content=doc["text"], metadata=doc["metadata"])
            for doc in store.documents
        ]
        if not documents:
            raise HTTPException(status_code=400, detail="No text chunks extracted")

        retriever = build_retriever(documents)
        logger.info(f"Retriever pipeline built with {len(documents)} chunks")

        return {"message": f"File ingested successfully with {len(documents)} chunks."}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during ingestion")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    global retriever

    if retriever is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    try:
        docs = retriever.invoke(request.query)
        chunks = [doc.page_content for doc in docs]

        answer = get_llm_client().generate_answer(
            query=request.query,
            context_chunks=chunks,
        )

        return QueryResponse(answer=answer, sources=chunks)

    except Exception as e:
        logger.exception("Error during query")
        raise HTTPException(status_code=500, detail=str(e))
