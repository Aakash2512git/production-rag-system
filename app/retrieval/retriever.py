import os
import logging
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever

from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq

from app.retrieval.mutiquery import CustomMultiQueryRetriever

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "full").lower()


def build_lite_retriever(documents):
    """BM25-only retriever — fast ingest, fits Render free tier."""
    logger.info("Building lite (BM25) retriever with %d documents", len(documents))
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    logger.info("Lite retriever ready")
    return bm25_retriever


def build_retriever(documents):
    if RETRIEVER_MODE == "lite":
        return build_lite_retriever(documents)

    logger.info(f"Building retriever with {len(documents)} documents")

    # -------------------------------
    # Embeddings
    # -------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    logger.info("Embeddings model loaded")

    # -------------------------------
    # Vector Store
    # -------------------------------

    vectorstore = FAISS.from_documents(documents, embeddings)

    logger.info("FAISS vector store created")

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    # -------------------------------
    # BM25 Retriever
    # -------------------------------

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10

    logger.info("BM25 retriever initialized")

    # -------------------------------
    # Hybrid Retriever
    # -------------------------------

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    logger.info("Hybrid retriever ready")

    # -------------------------------
    # LLM (for MultiQuery)
    # -------------------------------

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    logger.info("LLM initialized")

    # -------------------------------
    # Custom Multi Query Retriever
    # -------------------------------

    multi_query_retriever = CustomMultiQueryRetriever(
        base_retriever=hybrid_retriever,
        llm=llm,
        num_queries=3,
        k=5
    )

    logger.info("MultiQueryRetriever ready")

    # -------------------------------
    # Cohere Reranker
    # -------------------------------

    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY not set in .env")

    compressor = CohereRerank(
        model="rerank-english-v3.0",
        cohere_api_key=cohere_key,
        top_n=5,
    )

    logger.info("Cohere reranker initialized")

    # -------------------------------
    # Contextual Compression
    # -------------------------------

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )

    logger.info("Final retriever pipeline ready")

    return compression_retriever