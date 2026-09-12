"""Query rewriting and grounded answer generation via Groq."""
import os
from typing import List, Optional, Tuple

from groq import Groq
from langchain_core.documents import Document

from .config import GROQ_MODEL_NAME

_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def rewrite_query(
    client: Groq,
    query: str,
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Turn a possibly-ambiguous follow-up question into a standalone search query."""
    history = history or []
    history_text = "\n".join(f"{role}: {message}" for role, message in history)

    prompt = f"""
Rewrite the user's latest question into a clear, standalone
search query for retrieving relevant information from a
technical document.

Use the conversation history to resolve references such as:
"it", "they", "that", "what about scanned ones", etc.

Preserve the user's original meaning.
Do not answer the question.
Return only the rewritten search query.

Conversation history:
{history_text}

Latest user question:
{query}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def generate_answer(client: Groq, query: str, context_chunks: List[Document]) -> str:
    """Generate an answer grounded only in the provided chunks, with citations."""
    context_parts = [
        f"[Source {i + 1}]\n{chunk.page_content}" for i, chunk in enumerate(context_chunks)
    ]
    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the user's question using only the provided context.

Rules:
- Do not use outside knowledge.
- Every factual claim must be supported by the context.
- Cite the source that supports each claim.
- Only use source IDs that appear in the context.
- Use citations in the format [Source 1], [Source 2], etc.
- If the answer is not supported by the context, say:
  "I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
