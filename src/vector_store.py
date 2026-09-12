"""Building and loading the Chroma vector store."""
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_PERSIST_DIR
from .embeddings import get_embedding_model


def build_vector_store(
    chunks: List[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Create a new persisted Chroma store from chunks.

    Run this once (via scripts/build_index.py) whenever the source PDF
    or chunking settings change - not on every chat session.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=persist_directory,
    )


def load_vector_store(persist_directory: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Load an already-built, persisted Chroma store."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embedding_model(),
    )
