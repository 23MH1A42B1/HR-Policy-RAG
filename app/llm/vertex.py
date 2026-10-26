"""Vertex AI Gemini configuration and parameters for LiteLLM routing."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.llm.vertex")


class VertexConfig(BaseModel):
    """Configuration for Vertex AI Gemini provider."""

    project: Optional[str] = Field(default=None, description="Google Cloud Project ID")
    location: str = Field(default="us-central1", description="Google Cloud region")
    model: str = Field(default="gemini-1.5-pro", description="Gemini model identifier")

    @classmethod
    def from_settings(cls) -> VertexConfig:
        """Create VertexConfig from application settings and environment."""
        settings = get_settings()
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or settings.google_cloud_project
        location = os.environ.get("GOOGLE_CLOUD_LOCATION") or getattr(settings, "google_cloud_location", "us-central1")
        model = os.environ.get("VERTEX_MODEL") or getattr(settings, "vertex_model", "gemini-1.5-pro")
        return cls(project=project, location=location, model=model)

    @property
    def litellm_model(self) -> str:
        """Format model string for LiteLLM Vertex AI provider."""
        clean_model = self.model.strip()
        if clean_model.startswith("vertex_ai/"):
            return clean_model
        return f"vertex_ai/{clean_model}"

    def is_configured(self) -> bool:
        """Check whether minimum Vertex AI requirements are satisfied."""
        return bool(self.project and self.project.strip())

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
            "vertex_project": self.project,
            "vertex_location": self.location,
        }
        params.update(kwargs)
        return params
