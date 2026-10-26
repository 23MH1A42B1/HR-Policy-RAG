"""LiteLLM routing layer with automatic fallback from Vertex AI Gemini to Groq.

Tracks latency, active model, fallback status, and errors without exposing secrets.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional
import litellm
from pydantic import BaseModel, Field

from app.llm.groq import GroqConfig
from app.llm.vertex import VertexConfig
from app.utils.logging import get_logger

logger = get_logger("app.llm.litellm_router")

# Suppress overly noisy internal telemetry from LiteLLM
litellm.telemetry = False
litellm.suppress_debug_info = True


class LLMRoutingError(RuntimeError):
    """Raised when all configured LLM routes fail without exposing secrets."""
    pass


class LLMResponse(BaseModel):
    """Structured response container for routed LLM generations."""

    content: str = Field(description="Generated textual content")
    model_used: str = Field(description="Name of the model that successfully responded")
    provider: str = Field(description="Provider identifier ('vertex_ai' or 'groq')")
    fallback_used: bool = Field(description="Whether a fallback model was invoked")
    latency: float = Field(description="Execution latency in seconds")
    error_message: Optional[str] = Field(default=None, description="Clean error summary if degraded")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Token usage statistics")


def sanitize_error(error: Exception | str) -> str:
    """Sanitize error messages to ensure API keys, tokens, and sensitive headers are stripped."""
    err_str = str(error)
    # Remove bearer tokens, groq/openai/gemini keys, or key= values
    err_str = re.sub(
        r"(?:Bearer\s+|key=|api[_-]?key[:=]\s*)[a-zA-Z0-9_\-\.]{8,}",
        "[REDACTED_SECRET]",
        err_str,
        flags=re.IGNORECASE,
    )
    # Also strip gsk_ or AIza keys directly if present
    err_str = re.sub(r"\b(?:gsk_|AIza)[a-zA-Z0-9_\-]{12,}\b", "[REDACTED_SECRET]", err_str)
    # Remove file system paths that may contain user directories
    err_str = re.sub(r"[A-Za-z]:\\[^:\s\n]+", "[REDACTED_PATH]", err_str)
    return err_str.strip()


class LLMRouter:
    """Orchestrates resilient LLM completions with automatic failover."""

    def __init__(
        self,
        vertex_config: Optional[VertexConfig] = None,
        groq_config: Optional[GroqConfig] = None,
        completion_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.vertex_config = vertex_config or VertexConfig.from_settings()
        self.groq_config = groq_config or GroqConfig.from_settings()
        self._completion_func = completion_func or litellm.completion

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Route prompt to Vertex AI Gemini (primary) with automatic fallback to Groq.

        Args:
            prompt: User / question prompt.
            system_prompt: Optional system grounding instruction.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens in generated completion.

        Returns:
            LLMResponse containing generation content and routing diagnostics.

        Raises:
            LLMRoutingError: If both primary and fallback providers fail.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        routing_errors: List[str] = []
        overall_start = time.perf_counter()

        # ==============================================================
        # 1. Primary Route: Vertex AI Gemini
        # ==============================================================
        if self.vertex_config.is_configured():
            logger.info(
                f"Dispatching to primary model: Vertex AI Gemini ('{self.vertex_config.model}', "
                f"project='{self.vertex_config.project}', location='{self.vertex_config.location}')"
            )
            v_start = time.perf_counter()
            try:
                params = self.vertex_config.get_completion_params(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = self._completion_func(**params)
                latency = round(time.perf_counter() - v_start, 3)

                content = self._extract_content(response)
                usage = self._extract_usage(response)

                logger.info(
                    f"Vertex AI Gemini responded successfully in {latency:.3f}s "
                    f"(model_used='{self.vertex_config.model}', fallback_used=False)"
                )

                return LLMResponse(
                    content=content,
                    model_used=self.vertex_config.model,
                    provider="vertex_ai",
                    fallback_used=False,
                    latency=latency,
                    usage=usage,
                )
            except Exception as exc:
                clean_err = sanitize_error(exc)
                logger.warning(
                    f"Primary Vertex AI Gemini failed ({clean_err}). "
                    "Initiating automatic fallback to Groq..."
                )
                routing_errors.append(f"Vertex AI: {clean_err}")
        else:
            logger.info("Vertex AI Gemini is not configured (project not set). Attempting Groq fallback route.")
            routing_errors.append("Vertex AI: Not configured (missing GOOGLE_CLOUD_PROJECT)")

        # ==============================================================
        # 2. Fallback Route: Groq
        # ==============================================================
        if self.groq_config.is_configured():
            logger.info(f"Dispatching to fallback model: Groq ('{self.groq_config.model}')")
            g_start = time.perf_counter()
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    params = self.groq_config.get_completion_params(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    response = self._completion_func(**params)
                    latency = round(time.perf_counter() - g_start, 3)

                    content = self._extract_content(response)
                    usage = self._extract_usage(response)

                    logger.info(
                        f"Groq fallback responded successfully in {latency:.3f}s "
                        f"(model_used='{self.groq_config.model}', fallback_used=True)"
                    )

                    return LLMResponse(
                        content=content,
                        model_used=self.groq_config.model,
                        provider="groq",
                        fallback_used=True,
                        latency=latency,
                        usage=usage,
                    )
                except Exception as exc:
                    clean_err = sanitize_error(exc)
                    is_rate_limit = "rate limit" in str(exc).lower() or "429" in str(exc)
                    if is_rate_limit and attempt < max_retries:
                        backoff = attempt * 2.5
                        logger.warning(f"Groq rate limit hit (attempt {attempt}/{max_retries}). Backing off for {backoff:.1f}s...")
                        time.sleep(backoff)
                        continue
                    logger.error(f"Fallback Groq call failed ({clean_err}).")
                    routing_errors.append(f"Groq: {clean_err}")
                    break
        else:
            logger.warning("Groq fallback is not configured (missing GROQ_API_KEY).")
            routing_errors.append("Groq: Not configured (missing GROQ_API_KEY)")

        # ==============================================================
        # 3. Double Failure Handling
        # ==============================================================
        total_latency = round(time.perf_counter() - overall_start, 3)
        summary_error = " | ".join(routing_errors)
        logger.error(
            f"All LLM routes failed in {total_latency:.3f}s (fallback_attempted=True, errors='{summary_error}')"
        )

        raise LLMRoutingError(
            "Both primary (Vertex AI Gemini) and fallback (Groq) language model providers were unable to fulfill the request. "
            "Please ensure cloud credentials (GOOGLE_CLOUD_PROJECT or GROQ_API_KEY) are configured."
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract textual content safely from a LiteLLM response."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return str(choice.message.content or "").strip()
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return str(choices[0].get("message", {}).get("content", "")).strip()
        return str(response).strip()

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, Any]:
        """Extract usage metadata safely from a LiteLLM response."""
        if hasattr(response, "usage") and response.usage:
            if hasattr(response.usage, "model_dump"):
                return response.usage.model_dump()
            return dict(response.usage)
        elif isinstance(response, dict) and "usage" in response:
            return dict(response["usage"])
        return {}


def get_router() -> LLMRouter:
    """Factory creating configured LLMRouter instance."""
    return LLMRouter()
