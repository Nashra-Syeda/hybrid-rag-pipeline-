"""Embedding model, created once and reused by every caller.

Loading a HuggingFace embedding model takes a couple of seconds and
holds it in memory - the original scripts each created their own copy.
This module hands out a single shared instance instead.
"""
from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL_NAME

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model
