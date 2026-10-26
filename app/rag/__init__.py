"""RAG module for HR Policy Assistant."""

from app.rag.citations import SourceCitation, extract_citations_from_results
from app.rag.generator import LLMConfigurationError, LLMService, get_llm_service
from app.rag.pipeline import answer_question
from app.rag.prompt import build_system_prompt, build_user_prompt

__all__ = [
    "LLMConfigurationError",
    "LLMService",
    "SourceCitation",
    "answer_question",
    "build_system_prompt",
    "build_user_prompt",
    "extract_citations_from_results",
    "get_llm_service",
]
