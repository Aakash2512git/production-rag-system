# app/main.py
from fastapi import FastAPI
from app.api.endpoints import router
from dotenv import load_dotenv  # <-- ADD THIS
import os

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0"
)

# Include endpoints
app.include_router(router, prefix="/api")