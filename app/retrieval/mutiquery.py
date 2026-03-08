from typing import List
from collections import defaultdict
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class QueryVariations(BaseModel):
    queries: List[str]


class CustomMultiQueryRetriever(BaseRetriever):

    base_retriever: any
    llm: any
    num_queries: int = 3
    k: int = 5

    # -----------------------------
    # Generate query variations
    # -----------------------------

    def generate_queries(self, original_query: str) -> List[str]:

        llm_with_tools = self.llm.with_structured_output(QueryVariations)

        prompt = f"""
Generate {self.num_queries} variations of the query.

Query: {original_query}
"""

        try:
            response = llm_with_tools.invoke(prompt)
            queries = response.queries
        except Exception:
            queries = [original_query]

        queries.insert(0, original_query)

        return queries[: self.num_queries]

    # -----------------------------
    # RRF Fusion
    # -----------------------------

    def rrf_fusion(self, results: List[List[Document]], k: int = 60):

        scores = defaultdict(float)
        doc_map = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_id = doc.page_content
                doc_map[doc_id] = doc
                scores[doc_id] += 1 / (k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_map[doc_id] for doc_id, _ in ranked]

    # -----------------------------
    # LangChain Retriever Method
    # -----------------------------

    def _get_relevant_documents(self, query: str) -> List[Document]:

        query_variations = self.generate_queries(query)

        all_results = []

        for q in query_variations:
            docs = self.base_retriever.invoke(q)
            all_results.append(docs)

        fused = self.rrf_fusion(all_results)

        return fused[: self.k]