import logging
import os

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)


def build_full_retriever(documents: list[Document]):
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
    from langchain_cohere import CohereRerank
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq

    from app.retrieval.mutiquery import CustomMultiQueryRetriever

    logger.info("Building full retriever with %d documents", len(documents))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3],
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    multi_query_retriever = CustomMultiQueryRetriever(
        base_retriever=hybrid_retriever,
        llm=llm,
        num_queries=3,
        k=5,
    )

    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY not set")

    compressor = CohereRerank(
        model="rerank-english-v3.0",
        cohere_api_key=cohere_key,
        top_n=5,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever,
    )
