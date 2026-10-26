"""Tests for application imports and logging sanity."""

import logging
from app.utils.logging import SafeFormatter, get_logger


def test_imports():
    """Verify all project core modules can be cleanly imported."""
    import app.config
    import app.main
    import app.ui.chat
    import app.ui.components
    import app.ui.sidebar
    import app.utils.logging
    import app.ingestion
    import app.ingestion.loader
    import app.ingestion.pdf_loader
    import app.ingestion.docx_loader
    import app.ingestion.pptx_loader
    import app.ingestion.cleaner
    import app.ingestion.chunker
    import app.ingestion.metadata
    import app.embeddings
    import app.embeddings.embedder
    import app.retrieval
    import app.retrieval.qdrant_store
    import app.rag
    import app.rag.prompt
    import app.rag.citations
    import app.rag.generator
    import app.rag.pipeline

    assert app.config is not None
    assert app.main is not None
    assert app.ui.chat is not None
    assert app.ui.components is not None
    assert app.ui.sidebar is not None
    assert app.utils.logging is not None
    assert app.embeddings is not None
    assert app.retrieval is not None
    assert app.rag is not None


def test_logger_creation():
    """Verify get_logger returns a valid configured logger."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_safe_formatter_redacts_sensitive_keys():
    """Verify that credentials and api keys are redacted by SafeFormatter."""
    formatter = SafeFormatter(fmt="%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Connecting using api_key='sk-abcdef1234567890' and secret: 'super_secret_value'",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "sk-abcdef1234567890" not in formatted
    assert "super_secret_value" not in formatted
    assert "***REDACTED***" in formatted
