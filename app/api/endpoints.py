import asyncio
import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List

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


def _run_ingest(file_location: str):
    from langchain_core.documents import Document
    from app.ingestion.pdf_ingestion_pipeline import ingest_file
    from app.retrieval.retriever import build_retriever

    logger.info("Starting ingestion for %s", file_location)
    chunks = ingest_file(file_location)
    if not chunks:
        raise ValueError("No text chunks extracted")

    documents = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]
    logger.info("Building retriever for %d chunks", len(documents))
    new_retriever = build_retriever(documents)
    logger.info("Ingestion complete: %d chunks", len(documents))
    return new_retriever, len(documents)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    global retriever

    try:
        file_location = f"temp_uploads/{file.filename}"
        os.makedirs("temp_uploads", exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        new_retriever, chunk_count = await asyncio.to_thread(_run_ingest, file_location)
        retriever = new_retriever

        return {"message": f"File ingested successfully with {chunk_count} chunks."}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during ingestion")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    global retriever

    if retriever is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    try:
        docs = await asyncio.to_thread(retriever.invoke, request.query)
        chunks = [doc.page_content for doc in docs]

        answer = await asyncio.to_thread(
            get_llm_client().generate_answer,
            request.query,
            chunks,
        )

        return QueryResponse(answer=answer, sources=chunks)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Error during query")
        raise HTTPException(status_code=500, detail=str(e))
