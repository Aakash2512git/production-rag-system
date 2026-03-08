# app/retrieval/vector_store.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class VectorStore:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Embedding dimension
        self.dimension = 384

        # FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store documents
        self.documents = []

    def add_text(self, text: str, metadata: dict = None):
        """
        Add a text chunk and metadata to the vector store.
        """

        embedding = self.model.encode([text])

        self.index.add(np.array(embedding).astype("float32"))

        self.documents.append({
            "text": text,
            "metadata": metadata or {}
        })

    def search(self, query: str, top_k: int = 5):
        """
        Search similar chunks.
        """

        query_embedding = self.model.encode([query])

        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            top_k
        )

        results = []

        for idx in indices[0]:

            if idx < len(self.documents):

                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(distances[0][0])
                })

        return results


# Singleton instance
vector_store = VectorStore()