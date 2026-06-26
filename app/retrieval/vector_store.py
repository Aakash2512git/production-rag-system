# app/retrieval/vector_store.py

import faiss
import numpy as np

_store = None


class VectorStore:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

    def add_text(self, text: str, metadata: dict = None):
        embedding = self.model.encode([text])
        self.index.add(np.array(embedding).astype("float32"))
        self.documents.append({
            "text": text,
            "metadata": metadata or {},
        })

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            top_k,
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(distances[0][0]),
                })

        return results


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


class _LazyVectorStore:
    def __getattr__(self, name):
        return getattr(get_vector_store(), name)


vector_store = _LazyVectorStore()
