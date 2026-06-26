from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.api.endpoints import router

load_dotenv()

app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0",
)

cors_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "service": "RAG Pipeline API",
        "health": "/health",
        "docs": "/docs",
        "ingest": "POST /api/ingest",
        "query": "POST /api/query",
    }


@app.get("/health")
def health():
    groq_configured = bool((os.getenv("GROQ_API_KEY") or "").strip())
    return {
        "status": "ok",
        "groq_api_key_configured": groq_configured,
        "retriever_mode": os.getenv("RETRIEVER_MODE", "full"),
    }
