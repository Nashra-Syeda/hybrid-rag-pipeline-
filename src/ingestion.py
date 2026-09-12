"""PDF -> LangChain Document ingestion using Docling.

This is the ONLY place Docling gets instantiated. Every other module
receives already-loaded Documents instead of loading its own copy.
"""
from pathlib import Path
from typing import Union

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Lazily create a single, reusable Docling converter."""
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def load_pdf(pdf_path: Union[str, Path]) -> Document:
    """Parse a PDF with Docling and return it as one LangChain Document.

    The full document is returned as a single Document (Markdown text).
    Use `src.chunking.chunk_documents` to split it before embedding.
    """
    pdf_path = str(pdf_path)
    converter = _get_converter()
    result = converter.convert(pdf_path)
    markdown_text = result.document.export_to_markdown()

    return Document(
        page_content=markdown_text,
        metadata={
            "source": pdf_path,
            "parser": "docling",
        },
    )
