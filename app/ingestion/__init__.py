"""Document ingestion pipeline package for HR Policy RAG Assistant.

Handles validation, multi-format text extraction (PDF, DOCX, PPTX),
text cleaning, section detection, metadata enrichment, and chunking.
"""

from app.ingestion.chunker import DocumentChunker
from app.ingestion.cleaner import clean_extracted_text
from app.ingestion.loader import DocumentLoader, ExtractedPage
from app.ingestion.metadata import DocumentChunk, extract_policy_type, extract_version

__all__ = [
    "DocumentChunk",
    "DocumentChunker",
    "DocumentLoader",
    "ExtractedPage",
    "clean_extracted_text",
    "extract_policy_type",
    "extract_version",
]
