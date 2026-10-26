"""API entry points for XYZ Technologies HR Policy RAG Assistant."""

from __future__ import annotations

from app.api.rag_service import RAGService, answer_question

__all__ = ["RAGService", "answer_question"]
