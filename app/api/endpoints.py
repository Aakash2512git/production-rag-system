from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import os
import logging

from langchain_classic.schema import Document
from app.ingestion.pdf_ingestion_pipeline import ingest_file
from app.retrieval.retriever import build_retriever
from app.llm.llm_client import LLMClient
from app.retrieval.vector_store import vector_store

router = APIRouter()
retriever = None

llm_client = LLMClient()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Request / Response Models ---------- #
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


# ---------- Ingestion Endpoint ---------- #
@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF, extract text and tables, store in vector store,
    and build the LangChain retriever pipeline.
    """
    global retriever
    try:
        file_location = f"temp_uploads/{file.filename}"
        os.makedirs("temp_uploads", exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Step 1: Ingest into your existing vector_store
        ingest_file(file_location)

        # Step 2: Convert vector_store chunks to LangChain Documents
        documents = [
            Document(page_content=doc["text"], metadata=doc["metadata"])
            for doc in vector_store.documents
        ]
        if not documents:
            raise HTTPException(status_code=400, detail="No text chunks extracted")

        # Step 3: Build retriever pipeline
        retriever = build_retriever(documents)
        logger.info(f"Retriever pipeline built with {len(documents)} chunks")

        return {"message": f"File ingested successfully with {len(documents)} chunks."}

    except Exception as e:
        logger.exception("Error during ingestion")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Query Endpoint ---------- #
@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    global retriever

    if retriever is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    try:
        # Retrieve documents using MultiQuery pipeline
        docs = retriever.invoke(request.query)

        chunks = [doc.page_content for doc in docs]

        answer = llm_client.generate_answer(
            query=request.query,
            context_chunks=chunks
        )

        return QueryResponse(
            answer=answer,
            sources=chunks
        )

    except Exception as e:
        logger.exception("Error during query")
        raise HTTPException(status_code=500, detail=str(e))