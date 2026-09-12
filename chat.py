"""Interactive RAG chat over your PDF.

Prerequisite (run once, or whenever the PDF/chunking changes):
    python scripts/build_index.py data/your_file.pdf

Then:
    python chat.py

Each turn: rewrite query with history -> hybrid (dense+BM25) retrieval
-> cross-encoder rerank -> grounded answer generation.
"""
import pickle
from pathlib import Path

from dotenv import load_dotenv

from src.config import (
    CHROMA_PERSIST_DIR,
    CHUNKS_CACHE_PATH,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_TOP_K,
)
from src.generation import generate_answer, get_groq_client, rewrite_query
from src.retrieval import build_bm25_index, hybrid_search, rerank
from src.vector_store import load_vector_store

load_dotenv()


def load_chunks():
    if not CHUNKS_CACHE_PATH.exists():
        raise FileNotFoundError(
            "No chunk cache found. Run `python scripts/build_index.py <pdf>` first."
        )
    with open(CHUNKS_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def main() -> None:
    if not Path(CHROMA_PERSIST_DIR).exists():
        raise FileNotFoundError(
            "No vector store found. Run `python scripts/build_index.py <pdf>` first."
        )

    vector_store = load_vector_store()
    chunks = load_chunks()
    bm25 = build_bm25_index(chunks)
    client = get_groq_client()

    chat_history = []

    print("RAG chat ready. Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        rewritten = rewrite_query(client, query, chat_history)
        print(f"\n(searching for: {rewritten})")

        candidates = hybrid_search(vector_store, bm25, chunks, rewritten, k=DEFAULT_TOP_K)
        top_results = rerank(rewritten, candidates, top_n=DEFAULT_RERANK_TOP_N)

        answer = generate_answer(client, query, top_results)
        print(f"\nAssistant: {answer}\n")

        chat_history.append(("User", query))
        chat_history.append(("Assistant", answer))


if __name__ == "__main__":
    main()
