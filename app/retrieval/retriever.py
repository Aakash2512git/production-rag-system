import logging
import os

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "full").lower()


def build_retriever(documents: list[Document]):
    if RETRIEVER_MODE == "lite":
        from app.retrieval.retriever_lite import build_lite_retriever
        return build_lite_retriever(documents)

    from app.retrieval.retriever_full import build_full_retriever
    return build_full_retriever(documents)
