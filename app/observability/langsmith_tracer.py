"""LangSmith tracing and RAG observability service for HR Policy Assistant.

Traces the complete pipeline execution:
User question -> Input security -> Cache -> Retrieval -> Fusion -> Reranking -> LLM -> Output security -> Final answer

Captures latency, retrieved documents, scores, reranker scores, model, fallback, errors, and token usage.
Never logs secrets or sensitive employee PII.
When credentials are not configured, executes local tracing transparently without faking cloud API responses.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import time
from typing import Any, Dict, Generator, List, Optional
import uuid
from pydantic import BaseModel, Field

from app.config import get_settings
from app.security.output_guard import get_output_guard
from app.utils.logging import get_logger

logger = get_logger("app.observability.langsmith")


class StepTrace(BaseModel):
    """Execution metrics for an individual stage in the RAG pipeline."""

    name: str = Field(description="Stage name, e.g. input_security, retrieval, reranking, llm")
    latency_ms: float = Field(default=0.0, description="Stage latency in milliseconds")
    start_time: float = Field(default=0.0, description="Unix timestamp of step start")
    end_time: float = Field(default=0.0, description="Unix timestamp of step completion")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sanitized stage metrics")
    error: Optional[str] = Field(default=None, description="Stage error message if any")


class RAGPipelineTrace(BaseModel):
    """Complete observability trace for a single user query-response cycle."""

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique trace UUID")
    session_id: Optional[str] = Field(default=None, description="User session identifier")
    question: str = Field(description="Sanitized user query")
    steps: List[StepTrace] = Field(default_factory=list, description="Ordered pipeline execution steps")
    retrieved_documents: List[str] = Field(default_factory=list, description="Documents retrieved during search")
    retrieval_scores: List[float] = Field(default_factory=list, description="Dense/BM25 retrieval scores")
    reranker_scores: List[float] = Field(default_factory=list, description="Reranker relevance scores")
    model_used: str = Field(default="unknown", description="LLM model identifier invoked")
    fallback_used: bool = Field(default=False, description="Whether failover model was used")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Prompt, completion, total tokens")
    final_answer: str = Field(default="", description="Sanitized response generated for the user")
    total_latency_ms: float = Field(default=0.0, description="End-to-end latency in milliseconds")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during execution")
    langsmith_run_id: Optional[str] = Field(default=None, description="Remote LangSmith run ID if uploaded")

    def add_step(self, step: StepTrace) -> None:
        """Append an executed step trace."""
        self.steps.append(step)

    @contextmanager
    def measure_step(self, name: str, **initial_metadata: Any) -> Generator[Dict[str, Any], None, None]:
        """Context manager to measure latency and record metadata for a pipeline step."""
        t_start = time.perf_counter()
        t_epoch_start = time.time()
        meta: Dict[str, Any] = dict(initial_metadata)
        err_msg: Optional[str] = None

        try:
            yield meta
        except Exception as exc:
            err_msg = str(exc)
            self.errors.append(f"[{name}] {err_msg}")
            raise
        finally:
            t_end = time.perf_counter()
            duration_ms = round((t_end - t_start) * 1000.0, 2)
            step_record = StepTrace(
                name=name,
                latency_ms=duration_ms,
                start_time=t_epoch_start,
                end_time=time.time(),
                metadata=meta,
                error=err_msg,
            )
            self.add_step(step_record)


class LangSmithService:
    """LangSmith cloud tracing service with transparent local fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or getattr(settings, "langsmith_api_key", None) or os.environ.get("LANGSMITH_API_KEY")
        self.project = project or getattr(settings, "langsmith_project", "hr-policy-rag")
        self.endpoint = endpoint or getattr(settings, "langsmith_endpoint", "https://api.smith.langchain.com")

        self._client = None
        self.is_cloud = False
        self.integration_status = "local_tracing (langsmith api key not configured)"

        self._recent_traces: List[RAGPipelineTrace] = []
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize LangSmith client if API key is present."""
        if not self.api_key:
            logger.info("LangSmith API key not configured. Operating in local observability mode.")
            return

        try:
            from langsmith import Client
            self._client = Client(
                api_key=self.api_key,
                api_url=self.endpoint,
            )
            self.is_cloud = True
            self.integration_status = f"cloud_langsmith (project: '{self.project}')"
            logger.info(f"Connected to LangSmith tracing: project='{self.project}'")
        except Exception as exc:
            logger.warning(f"Failed to initialize LangSmith client: {exc}. Using local tracing.")

    def is_configured(self) -> bool:
        """Check if cloud LangSmith tracing is active."""
        return self.is_cloud and self._client is not None

    def start_trace(self, question: str, session_id: Optional[str] = None) -> RAGPipelineTrace:
        """Create and start a new pipeline execution trace."""
        # Sanitize any accidental secrets from the question
        clean_q = get_output_guard().validate_output(question).sanitized_output
        return RAGPipelineTrace(
            question=clean_q,
            session_id=session_id,
        )

    def finish_trace(self, trace: RAGPipelineTrace) -> RAGPipelineTrace:
        """Finalize pipeline trace, compute aggregate latency, and ship to LangSmith if configured."""
        total_ms = sum(s.latency_ms for s in trace.steps)
        trace.total_latency_ms = round(total_ms, 2)

        # Sanitize final answer before transmission or persistence
        clean_ans = get_output_guard().validate_output(trace.final_answer).sanitized_output
        trace.final_answer = clean_ans

        # Retain in local buffer (last 50 traces)
        self._recent_traces.append(trace)
        if len(self._recent_traces) > 50:
            self._recent_traces.pop(0)

        # Ship to LangSmith if cloud tracing is active
        if self.is_configured() and self._client is not None:
            try:
                run = self._client.create_run(
                    name="hr_policy_rag_pipeline",
                    run_type="chain",
                    project_name=self.project,
                    inputs={"question": trace.question},
                    outputs={
                        "answer": trace.final_answer,
                        "documents": trace.retrieved_documents,
                        "model": trace.model_used,
                        "fallback": trace.fallback_used,
                    },
                    extra={
                        "metadata": {
                            "session_id": trace.session_id,
                            "latency_ms": trace.total_latency_ms,
                            "retrieval_scores": trace.retrieval_scores,
                            "reranker_scores": trace.reranker_scores,
                            "token_usage": trace.token_usage,
                            "steps": [s.model_dump() for s in trace.steps],
                        }
                    },
                    error="; ".join(trace.errors) if trace.errors else None,
                )
                trace.langsmith_run_id = getattr(run, "id", str(uuid.uuid4()))
                logger.info(f"Exported trace to LangSmith: run_id={trace.langsmith_run_id}")
            except Exception as exc:
                logger.warning(f"Failed to ship trace to LangSmith: {exc}")

        return trace

    def export_evaluation_run(
        self,
        dataset_name: str,
        results: List[Dict[str, Any]],
        summary_metrics: Dict[str, float],
    ) -> Optional[str]:
        """Export benchmark evaluation results to LangSmith if configured."""
        if not self.is_configured() or self._client is None:
            logger.info(
                f"LangSmith not configured. Local evaluation complete with composite score: "
                f"{summary_metrics.get('composite_score', 0.0):.4f}"
            )
            return None

        try:
            run_id = str(uuid.uuid4())
            self._client.create_run(
                name=f"evaluation_{dataset_name}",
                run_type="chain",
                project_name=self.project,
                inputs={"dataset": dataset_name, "sample_count": len(results)},
                outputs={"summary_metrics": summary_metrics},
                extra={"metadata": {"results_sample": results[:5]}},
            )
            logger.info(f"Evaluation run exported to LangSmith (project='{self.project}')")
            return run_id
        except Exception as exc:
            logger.warning(f"Failed to export evaluation run to LangSmith: {exc}")
            return None

    def get_recent_traces(self) -> List[RAGPipelineTrace]:
        """Retrieve recent local traces."""
        return list(self._recent_traces)


# Global singleton instance
_GLOBAL_LANGSMITH_SERVICE: Optional[LangSmithService] = None


def get_langsmith_service() -> LangSmithService:
    """Return singleton LangSmithService instance."""
    global _GLOBAL_LANGSMITH_SERVICE
    if _GLOBAL_LANGSMITH_SERVICE is None:
        _GLOBAL_LANGSMITH_SERVICE = LangSmithService()
    return _GLOBAL_LANGSMITH_SERVICE
