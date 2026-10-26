"""Unit tests for LiteLLM routing layer with Vertex AI Gemini and Groq fallback."""

from unittest import mock
import pytest
from app.llm.groq import GroqConfig
from app.llm.litellm_router import (
    LLMRoutingError,
    LLMRouter,
    sanitize_error,
)
from app.llm.vertex import VertexConfig
from app.rag.generator import LiteLLMService


def _create_mock_litellm_response(text: str, model: str = "mock-model") -> mock.MagicMock:
    """Helper creating a LiteLLM-like response mock."""
    resp = mock.MagicMock()
    choice = mock.MagicMock()
    choice.message.content = text
    resp.choices = [choice]
    resp.model = model
    resp.usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
    return resp


def test_primary_vertex_success_without_fallback():
    """Verify primary Vertex AI Gemini call succeeds and fallback is NOT triggered."""
    vertex_cfg = VertexConfig(project="my-gcp-project", location="us-central1", model="gemini-1.5-pro")
    groq_cfg = GroqConfig(api_key="gsk_test_groq_key", model="llama-3.3-70b-versatile")

    mock_completion = mock.MagicMock(
        return_value=_create_mock_litellm_response("Response from Vertex AI Gemini")
    )

    router = LLMRouter(
        vertex_config=vertex_cfg,
        groq_config=groq_cfg,
        completion_func=mock_completion,
    )

    result = router.generate(prompt="What is the leave policy?")

    assert result.content == "Response from Vertex AI Gemini"
    assert result.model_used == "gemini-1.5-pro"
    assert result.provider == "vertex_ai"
    assert result.fallback_used is False
    assert result.latency >= 0.0

    # Ensure only 1 completion call was made (no fallback)
    assert mock_completion.call_count == 1
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "vertex_ai/gemini-1.5-pro"
    assert call_kwargs["vertex_project"] == "my-gcp-project"
    assert call_kwargs["vertex_location"] == "us-central1"


def test_automatic_fallback_to_groq_when_vertex_fails():
    """Verify that when Vertex AI Gemini throws an error, router automatically falls back to Groq."""
    vertex_cfg = VertexConfig(project="my-gcp-project", location="us-central1", model="gemini-1.5-pro")
    groq_cfg = GroqConfig(api_key="gsk_test_groq_key", model="llama-3.3-70b-versatile")

    def completion_side_effect(**kwargs):
        if "vertex_ai" in kwargs.get("model", ""):
            raise RuntimeError("Vertex AI rate limit 429 quota exceeded")
        return _create_mock_litellm_response("Response from Groq Llama 3.3")

    mock_completion = mock.MagicMock(side_effect=completion_side_effect)

    router = LLMRouter(
        vertex_config=vertex_cfg,
        groq_config=groq_cfg,
        completion_func=mock_completion,
    )

    result = router.generate(prompt="What is the sick leave policy?")

    assert result.content == "Response from Groq Llama 3.3"
    assert result.model_used == "llama-3.3-70b-versatile"
    assert result.provider == "groq"
    assert result.fallback_used is True
    assert result.latency >= 0.0

    # 2 calls: first Vertex (failed), second Groq (succeeded)
    assert mock_completion.call_count == 2
    first_call_model = mock_completion.call_args_list[0].kwargs["model"]
    second_call_model = mock_completion.call_args_list[1].kwargs["model"]
    assert "vertex_ai" in first_call_model
    assert "groq" in second_call_model


def test_double_failure_raises_sanitized_routing_error():
    """Verify that when both Vertex AI and Groq fail, a clean LLMRoutingError is raised."""
    vertex_cfg = VertexConfig(project="my-gcp-project", location="us-central1", model="gemini-1.5-pro")
    groq_cfg = GroqConfig(api_key="gsk_secret_groq_key_9999", model="llama-3.3-70b-versatile")

    mock_completion = mock.MagicMock(
        side_effect=RuntimeError("Authorization failed with key=gsk_secret_groq_key_9999")
    )

    router = LLMRouter(
        vertex_config=vertex_cfg,
        groq_config=groq_cfg,
        completion_func=mock_completion,
    )

    with pytest.raises(LLMRoutingError) as excinfo:
        router.generate(prompt="Test prompt")

    err_msg = str(excinfo.value)
    # Must NOT expose raw secrets
    assert "gsk_secret_groq_key_9999" not in err_msg
    assert "Both primary (Vertex AI Gemini) and fallback (Groq)" in err_msg


def test_sanitize_error_strips_keys_and_paths():
    """Verify error sanitizer removes bearer tokens, api keys, and local file paths."""
    raw = "Failed call to https://api.groq.com with Bearer gsk_abcdef123456789 and file C:\\Users\\Alice\\project\\app.py"
    cleaned = sanitize_error(raw)
    assert "gsk_abcdef123456789" not in cleaned
    assert "[REDACTED_SECRET]" in cleaned
    assert "C:\\Users\\Alice\\project\\app.py" not in cleaned
    assert "[REDACTED_PATH]" in cleaned


def test_unconfigured_vertex_bypasses_to_groq():
    """Verify router immediately selects Groq when Vertex project is not configured."""
    vertex_cfg = VertexConfig(project=None, model="gemini-1.5-pro")
    groq_cfg = GroqConfig(api_key="gsk_test_groq_key", model="llama-3.3-70b-versatile")

    mock_completion = mock.MagicMock(
        return_value=_create_mock_litellm_response("Groq response without Vertex configured")
    )

    router = LLMRouter(
        vertex_config=vertex_cfg,
        groq_config=groq_cfg,
        completion_func=mock_completion,
    )

    result = router.generate(prompt="Test prompt")
    assert result.content == "Groq response without Vertex configured"
    assert result.provider == "groq"
    assert result.fallback_used is True
    # Only Groq was called
    assert mock_completion.call_count == 1
    assert "groq" in mock_completion.call_args.kwargs["model"]


def test_environment_configuration_loading():
    """Verify configs resolve correctly from environment variables."""
    with mock.patch.dict(
        "os.environ",
        {
            "GOOGLE_CLOUD_PROJECT": "env-custom-proj",
            "GOOGLE_CLOUD_LOCATION": "europe-west1",
            "VERTEX_MODEL": "gemini-1.5-flash",
            "GROQ_API_KEY": "gsk_env_key",
            "GROQ_MODEL": "llama-3.1-8b-instant",
        },
        clear=True,
    ):
        v_cfg = VertexConfig.from_settings()
        assert v_cfg.project == "env-custom-proj"
        assert v_cfg.location == "europe-west1"
        assert v_cfg.model == "gemini-1.5-flash"
        assert v_cfg.litellm_model == "vertex_ai/gemini-1.5-flash"

        g_cfg = GroqConfig.from_settings()
        assert g_cfg.api_key == "gsk_env_key"
        assert g_cfg.model == "llama-3.1-8b-instant"
        assert g_cfg.litellm_model == "groq/llama-3.1-8b-instant"


def test_litellm_service_wrapper_integration():
    """Verify LiteLLMService wraps LLMRouter and provides standard generate() interface."""
    vertex_cfg = VertexConfig(project="proj", location="us-central1")
    groq_cfg = GroqConfig(api_key="key")

    mock_completion = mock.MagicMock(
        return_value=_create_mock_litellm_response("Synthesized policy answer")
    )
    router = LLMRouter(
        vertex_config=vertex_cfg,
        groq_config=groq_cfg,
        completion_func=mock_completion,
    )

    service = LiteLLMService(router=router)
    answer = service.generate("Question about leave", system_prompt="System prompt")

    assert answer == "Synthesized policy answer"
    assert service.last_response is not None
    assert service.last_response.model_used == "gemini-1.5-pro"
    assert service.last_response.fallback_used is False
