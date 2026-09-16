"""Sparse (BM25), dense (Chroma), hybrid (RRF), and reranked retrieval.
"""

from typing import Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import DEFAULT_RERANK_TOP_N, DEFAULT_TOP_K, RERANKER_MODEL_NAME, RRF_K

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def build_bm25_index(chunks: List[Document]) -> BM25Okapi:
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def bm25_search(
    bm25: BM25Okapi,
    chunks: List[Document],
    query: str,
    n: int = DEFAULT_TOP_K,
) -> List[Document]:
    tokenized_query = query.lower().split()
    return bm25.get_top_n(tokenized_query, chunks, n=n)


def vector_search(vector_store: Chroma, query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
    return vector_store.similarity_search(query, k=k)


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    scores: Dict[str, float] = {}
    documents: Dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content
            documents[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [documents[doc_id] for doc_id in ranked_ids]


def hybrid_search(
    vector_store: Chroma,
    bm25: BM25Okapi,
    chunks: List[Document],
    query: str,
    k: int = DEFAULT_TOP_K,
) -> List[Document]:
    """Dense + sparse retrieval, merged with reciprocal rank fusion."""
    vector_results = vector_search(vector_store, query, k=k)
    bm25_results = bm25_search(bm25, chunks, query, n=k)
    return reciprocal_rank_fusion([vector_results, bm25_results])


def rerank(
    query: str,
    results: List[Document],
    top_n: int = DEFAULT_RERANK_TOP_N,
) -> List[Document]:
    """Re-score candidates with a cross-encoder and keep the top N."""
    if not results:
        return []

    reranker = _get_reranker()
    pairs = [[query, result.page_content] for result in results]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(scores, results), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in reranked[:top_n]]
