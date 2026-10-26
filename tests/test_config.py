"""Tests for application configuration and settings loading."""

import os
from unittest import mock
from app.config import Settings, get_settings


def test_default_settings():
    """Verify default configuration attributes."""
    settings = Settings()
    assert settings.app_name == "HR Policy RAG Assistant"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"
    assert settings.port == 8501
    assert settings.log_level == "INFO"
    assert settings.debug is False


def test_environment_override():
    """Verify settings can be overridden via environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "APP_NAME": "Test HR Portal",
            "ENVIRONMENT": "test",
            "PORT": "9000",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
        },
    ):
        settings = Settings()
        assert settings.app_name == "Test HR Portal"
        assert settings.environment == "test"
        assert settings.port == 9000
        assert settings.debug is True
        assert settings.log_level == "DEBUG"


def test_get_settings_cached():
    """Verify get_settings returns a cached Settings singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
