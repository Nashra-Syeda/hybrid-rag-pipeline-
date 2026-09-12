"""Build (or rebuild) the vector index and BM25 chunk cache from a PDF.

Run this once whenever you point the pipeline at a new PDF, or change
chunking settings in src/config.py. Re-running it wipes the previous
Chroma collection at the same persist directory.

Usage:
    python scripts/build_index.py data/your_file.pdf
"""
import pickle
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_documents
from src.config import CHROMA_PERSIST_DIR, CHUNKS_CACHE_PATH
from src.ingestion import load_pdf
from src.vector_store import build_vector_store


def main(pdf_path: str) -> None:
    persist_dir = Path(CHROMA_PERSIST_DIR)
    if persist_dir.exists():
        print(f"Removing existing index at {persist_dir} ...")
        shutil.rmtree(persist_dir)

    print(f"Loading {pdf_path} with Docling ...")
    document = load_pdf(pdf_path)

    print("Chunking ...")
    chunks = chunk_documents([document])
    print(f"Created {len(chunks)} chunks.")

    print("Embedding + building Chroma index (this can take a minute) ...")
    build_vector_store(chunks, persist_directory=CHROMA_PERSIST_DIR)

    CHUNKS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nDone. Vector store: {CHROMA_PERSIST_DIR}")
    print(f"Chunk cache (used for BM25 at chat time): {CHUNKS_CACHE_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/build_index.py <path-to-pdf>")
        sys.exit(1)

    main(sys.argv[1])
