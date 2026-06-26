import logging
import re
from typing import Any, List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class LiteBM25Retriever:
    """In-memory BM25 retriever — no torch, no faiss, fits 512MB RAM."""

    def __init__(self, documents: List[Document], top_k: int = 5):
        self.documents = documents
        self.top_k = top_k
        self._corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)

    def invoke(self, query: str) -> List[Document]:
        tokens = _tokenize(query)
        if not tokens:
            return self.documents[: self.top_k]

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Document] = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            results.append(self.documents[idx])
            if len(results) >= self.top_k:
                break

        return results or self.documents[: self.top_k]


def build_lite_retriever(documents: List[Document]) -> LiteBM25Retriever:
    logger.info("Building lite BM25 retriever with %d documents", len(documents))
    retriever = LiteBM25Retriever(documents)
    logger.info("Lite retriever ready")
    return retriever
