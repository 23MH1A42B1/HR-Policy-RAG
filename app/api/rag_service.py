"""Shared backend RAG application service for XYZ Technologies HR Policy Assistant.

Acts as the single authoritative service layer consumed by both Streamlit Web UI
and the Slack Bot. Ensures unified retrieval, security, grounding, and LLM synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from app.utils.logging import get_logger

logger = get_logger("app.api.rag_service")


class RAGService:
    """Enterprise RAG Service orchestrating query processing across interfaces."""

    def __init__(self, agent=None) -> None:
        self._agent = agent

    def _get_agent(self):
        if self._agent is not None:
            return self._agent
        from app.agent.agent import get_agent
        return get_agent()

    def answer_question(
        self,
        query: str,
        user_id: str,
        session_id: str,
        channel_id: Optional[str] = None,
        source: str = "web",
        language: str = "English",
    ) -> Dict[str, Any]:
        """Process an employee HR question through the guarded RAG pipeline."""
        clean_q = query.strip() if query else ""
        logger.info(
            f"RAG Service handling query from source='{source}', user='{user_id}', "
            f"session='{session_id}', channel='{channel_id}': '{clean_q[:60]}...'"
        )

        if not clean_q:
            return {
                "answer": "Please ask a question about company HR policies.",
                "sources": [],
                "session_id": session_id,
                "metadata": {
                    "model": "none",
                    "cache_hit": False,
                    "fallback_used": False,
                    "in_scope": False,
                    "retrieval_results": [],
                },
            }

        try:
            agent = self._get_agent()
            response = agent.process_query(
                query=clean_q,
                session_id=session_id,
                language=language,
            )

            # Sanitize citations to guarantee no internal chunk IDs, scores, or vectors leak
            from app.ui.components import format_citation
            formatted_sources: List[Dict[str, Any]] = [
                format_citation(c) for c in (response.sources or [])
            ]

            return {
                "answer": response.answer,
                "sources": formatted_sources,
                "session_id": session_id,
                "metadata": {
                    "model": response.model_used,
                    "cache_hit": response.cache_hit,
                    "fallback_used": response.fallback_used,
                    "reranker_used": response.reranker_used,
                    "in_scope": response.in_scope,
                    "retrieval_results": response.retrieval_results or [],
                },
            }

        except Exception as exc:
            logger.error(f"Error in RAG Service answering question for user='{user_id}': {exc}", exc_info=True)
            # Never leak stack traces, DB errors, or credentials to users
            return {
                "answer": "Sorry, the HR policy service is temporarily unavailable. Please try again later.",
                "sources": [],
                "session_id": session_id,
                "metadata": {
                    "model": "none",
                    "cache_hit": False,
                    "fallback_used": False,
                    "in_scope": False,
                    "error": True,
                    "retrieval_results": [],
                },
            }


# Singleton service instance
_GLOBAL_RAG_SERVICE: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Return singleton RAGService instance."""
    global _GLOBAL_RAG_SERVICE
    if _GLOBAL_RAG_SERVICE is None:
        _GLOBAL_RAG_SERVICE = RAGService()
    return _GLOBAL_RAG_SERVICE


def answer_question(
    query: str,
    user_id: str,
    session_id: str,
    channel_id: Optional[str] = None,
    source: str = "web",
    language: str = "English",
) -> Dict[str, Any]:
    """Public convenience function executing RAGService.answer_question."""
    service = get_rag_service()
    return service.answer_question(
        query=query,
        user_id=user_id,
        session_id=session_id,
        channel_id=channel_id,
        source=source,
        language=language,
    )
