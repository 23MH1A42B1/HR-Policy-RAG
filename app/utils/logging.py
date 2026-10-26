"""Logging utility for HR Policy RAG Assistant.

Provides structured and clean logging with safe secret sanitization.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Optional


# Sensitive patterns that must never appear in log records
SENSITIVE_PATTERNS = [
    # Key-value secret assignments
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|auth)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?"), r"\1=***REDACTED***"),
    # Bearer authorization headers
    (re.compile(r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]{10,}\b"), "Bearer ***REDACTED***"),
    # Standalone API keys (Groq, Google Gemini, OpenAI, Jina)
    (re.compile(r"\b(?:gsk_|AIza|jina_|sk-)[a-zA-Z0-9_\-]{12,}\b"), "***REDACTED_KEY***"),
    # Private cryptographic keys
    (re.compile(r"-----BEGIN\s+(?:[A-Z0-9_]+\s+)?PRIVATE\s+KEY-----"), "***REDACTED_PRIVATE_KEY***"),
    # JWT tokens
    (re.compile(r"\beyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\b"), "***REDACTED_JWT***"),
    # Sensitive employee PII (SSN)
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]


class SafeFormatter(logging.Formatter):
    """Formatter that redacts potentially sensitive credentials and PII from log output."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        redacted = original
        for pattern, replacement in SENSITIVE_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted


_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with safe formatter and console handler."""
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = SafeFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = [handler]

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve a configured logger instance for a module."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name or "hr_policy_rag")
