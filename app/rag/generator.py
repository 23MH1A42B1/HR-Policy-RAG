"""LLM abstraction and provider implementations for HR Policy RAG.

Provides decoupled LLM generation supporting Gemini, Groq, OpenAI, and a Mock service.
Raises clean configuration errors when credentials are missing without faking answers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import httpx
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.rag.generator")


class LLMConfigurationError(ValueError):
    """Raised when an LLM provider's API key or credentials are missing."""
    pass


class LLMService(ABC):
    """Abstract base class for LLM generation providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from the model given a prompt and optional system prompt."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model identifier."""
        pass


class GeminiLLMService(LLMService):
    """Google Gemini generation service using REST API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.llm_model or "gemini-1.5-pro"

        if not self.api_key:
            raise LLMConfigurationError(
                "Gemini API key is not configured. Please set 'GEMINI_API_KEY' in your .env file or environment."
            )

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Gemini REST API to generate a response."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            },
        }
        if system_prompt:
            body["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        logger.info(f"Dispatching query to Gemini model: '{self.model}'...")
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=body)
                response.raise_for_status()
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("Gemini API returned no candidates.")
                text_parts = candidates[0].get("content", {}).get("parts", [])
                answer = "".join(part.get("text", "") for part in text_parts)
                return answer.strip()
        except httpx.HTTPStatusError as exc:
            logger.error(f"Gemini API error (Status {exc.response.status_code}): {exc.response.text}")
            raise RuntimeError(f"Gemini API returned status {exc.response.status_code}: {exc.response.text}") from exc
        except Exception as exc:
            logger.error(f"Failed to communicate with Gemini API: {exc}")
            raise RuntimeError(f"Gemini API communication failure: {exc}") from exc


class GroqLLMService(LLMService):
    """Groq API generation service."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model = model or "llama-3.3-70b-versatile"

        if not self.api_key:
            raise LLMConfigurationError(
                "Groq API key is not configured. Please set 'GROQ_API_KEY' in your .env file or environment."
            )

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error(f"Groq generation failed: {exc}")
            raise RuntimeError(f"Groq API call failed: {exc}") from exc


class MockLLMService(LLMService):
    """Deterministic mock service for tests without external network or API keys."""

    def __init__(self, model_name: str = "mock-model", custom_response: Optional[str] = None) -> None:
        self._model_name = model_name
        self.custom_response = custom_response

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.custom_response:
            return self.custom_response
        return (
            "Based on the provided Leave Policy (Page 1, Section: 1. Annual Leave Entitlement), "
            "regular full-time employees are entitled to 20 business days of paid annual leave per calendar year."
        )


from app.llm.litellm_router import LLMResponse, LLMRoutingError, LLMRouter, get_router


class LiteLLMService(LLMService):
    """LiteLLM routed service with Vertex AI Gemini primary and Groq fallback."""

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.router = router or get_router()
        settings = get_settings()
        self.temperature = temperature if temperature is not None else getattr(settings, "llm_temperature", 0.2)
        self.max_tokens = max_tokens if max_tokens is not None else getattr(settings, "llm_max_tokens", 1024)
        self.last_response: Optional[LLMResponse] = None

    @property
    def model_name(self) -> str:
        if self.last_response:
            return self.last_response.model_used
        return self.router.vertex_config.model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response via LiteLLM routing with automatic fallback."""
        try:
            self.last_response = self.router.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self.last_response.content
        except LLMRoutingError as exc:
            raise LLMConfigurationError(str(exc)) from exc


def get_llm_service(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLMService:
    """Factory to retrieve the configured LLM service."""
    settings = get_settings()
    active_provider = (provider or settings.llm_provider).lower()
    active_model = model_name or settings.llm_model

    if active_provider in ("router", "litellm"):
        return LiteLLMService()
    elif active_provider == "gemini":
        return GeminiLLMService(model=active_model)
    elif active_provider == "groq":
        return GroqLLMService(model=active_model)
    elif active_provider == "mock":
        return MockLLMService(model_name=active_model)
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: '{active_provider}'. Supported providers: 'router', 'gemini', 'groq', 'mock'."
        )
