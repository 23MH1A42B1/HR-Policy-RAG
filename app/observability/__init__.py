"""Observability package for HR Policy RAG Assistant.

Provides LangSmith tracing, latency metrics, step-by-step pipeline profiling,
and evaluation tracking.
"""

from app.observability.langsmith_tracer import (
    LangSmithService,
    RAGPipelineTrace,
    StepTrace,
    get_langsmith_service,
)

__all__ = [
    "LangSmithService",
    "RAGPipelineTrace",
    "StepTrace",
    "get_langsmith_service",
]
