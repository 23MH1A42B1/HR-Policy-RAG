"""Groq fallback LLM configuration and parameters for LiteLLM routing."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.llm.groq")


class GroqConfig(BaseModel):
    """Configuration for Groq fallback LLM provider."""

    api_key: Optional[str] = Field(default=None, description="Groq API Key")
    model: str = Field(default="openai/gpt-oss-20b", description="Groq model identifier")

    @classmethod
    def from_settings(cls) -> GroqConfig:
        """Create GroqConfig from application settings and environment."""
        settings = get_settings()
        api_key = os.environ.get("GROQ_API_KEY") or settings.groq_api_key
        model = os.environ.get("GROQ_MODEL") or getattr(settings, "groq_model", "openai/gpt-oss-20b")
        return cls(api_key=api_key, model=model)

    @property
    def litellm_model(self) -> str:
        """Format model string for LiteLLM Groq provider."""
        clean_model = self.model.strip()
        if clean_model.startswith("groq/"):
            return clean_model
        return f"groq/{clean_model}"

    def is_configured(self) -> bool:
        """Check whether minimum Groq requirements are satisfied."""
        return bool(self.api_key and self.api_key.strip())

    def get_completion_params(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Construct arguments dictionary for litellm.completion."""
        params: Dict[str, Any] = {
            "model": self.litellm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": self.api_key,
        }
        params.update(kwargs)
        return params
