# app/llm/llm_client.py

import os
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    Wrapper around ChatGroq for RAG answer generation.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant"):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0
        )

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using retrieved context.
        """

        context = "\n\n".join(context_chunks)

        system_prompt = """
        You are a helpful AI assistant answering questions using retrieved context.

        Rules:
        1. Use ONLY the information from the provided context.
        2. If the answer is not in the context, say "I don't know".
        3. Summarize the information in your own words instead of copying sentences.
        4. Keep the answer clear, concise, and factual.
        5. If possible, reference the relevant context.

        Do not add any information that is not in the context.
        """

        user_prompt = f"""
Context:
{context}

Question:
{query}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)

        return response.content