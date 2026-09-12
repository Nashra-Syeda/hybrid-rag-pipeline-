# PDF RAG

A small retrieval-augmented generation pipeline over a single PDF:
Docling for parsing, Chroma + HuggingFace embeddings for dense
retrieval, BM25 for sparse retrieval, reciprocal rank fusion to
combine them, a cross-encoder to rerank, and Groq (`openai/gpt-oss-20b`)
for query rewriting and grounded answer generation.

## Structure

```
src/
  config.py       # all tunable settings in one place
  ingestion.py     # PDF -> LangChain Document (Docling)
  chunking.py      # Document -> chunks
  embeddings.py    # shared embedding model instance
  vector_store.py  # build / load the Chroma index
  retrieval.py     # BM25, dense search, RRF hybrid search, reranking
  generation.py    # Groq client, query rewriting, answer generation
scripts/
  build_index.py   # one-off: PDF -> chunks -> embeddings -> Chroma
chat.py            # interactive chat loop, wires everything together
```

Ingestion, retrieval, and generation are separate modules on purpose:
nothing loads Docling or the embedding model more than once per
process, and `chat.py` / `scripts/build_index.py` are just thin
scripts that call into `src/`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your real GROQ_API_KEY
```

Put a PDF in `data/`, then build the index (run once, or again
whenever you change the PDF or chunking settings):

```bash
python scripts/build_index.py data/your_file.pdf
```

Then chat:

```bash
python chat.py
```

Each turn: your question is rewritten into a standalone search query
using conversation history -> hybrid (dense + BM25) retrieval ->
cross-encoder reranking -> answer generated only from the retrieved
chunks, with `[Source N]` citations.

## Notes

- `chroma_db/` and PDFs under `data/` are gitignored — they're
  generated/local artifacts, not source. Regenerate with
  `build_index.py` after cloning.
- Swap the embedding model, reranker, or Groq model in
  `src/config.py`.
