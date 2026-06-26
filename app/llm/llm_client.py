import os
from typing import List

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMClient:
    """Groq client for RAG answer generation."""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.model = model
        self.client = Groq(api_key=api_key, timeout=60.0)

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a helpful AI assistant answering questions using retrieved context.\n\n"
            "Rules:\n"
            "1. Use ONLY the information from the provided context.\n"
            "2. If the answer is not in the context, say \"I don't know\".\n"
            "3. Summarize the information in your own words instead of copying sentences.\n"
            "4. Keep the answer clear, concise, and factual.\n"
            "5. If possible, reference the relevant context.\n\n"
            "Do not add any information that is not in the context."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise _format_groq_error(exc) from exc


def _format_groq_error(exc: Exception) -> RuntimeError:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()

    if "401" in message or "invalid api key" in lowered or "authentication" in lowered:
        return RuntimeError(
            "Groq API key is invalid or missing. Set GROQ_API_KEY in Render environment variables."
        )
    if "connection" in lowered or "connect" in lowered or "timeout" in lowered:
        return RuntimeError(
            "Could not reach Groq API. Verify GROQ_API_KEY on Render and retry in a minute."
        )
    return RuntimeError(f"Groq API error: {message}")
