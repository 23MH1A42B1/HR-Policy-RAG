"""Input validation and sanitization guardrail for HR Policy RAG Assistant.

Validates query length, strips control/invisible characters, protects against
bidirectional override attacks, and blocks prompt injection payloads.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.security.prompt_injection import (
    InjectionCheckResult,
    PromptInjectionDetector,
    get_injection_detector,
)
from app.utils.logging import get_logger

logger = get_logger("app.security.input_guard")

# Regex to detect suspicious invisible, zero-width, or bidirectional override characters
SUSPICIOUS_UNICODE_PATTERN = re.compile(
    r"[\u200B-\u200D\uFEFF\u202A-\u202E\u2066-\u2069\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

# Standard safety rejection message when input is deemed malicious
SECURITY_REJECTION_MESSAGE = (
    "Your request could not be processed because it contains prohibited instructions, "
    "unsupported syntax, or security-sensitive directives. Please ask a direct question about company HR policies."
)


class InputValidationResult(BaseModel):
    """Result of comprehensive input guardrail evaluation."""

    is_valid: bool = Field(description="Whether the input is safe to process")
    sanitized_text: str = Field(description="Sanitized and normalized query text")
    rejection_reason: Optional[str] = Field(default=None, description="Explanation if rejected")
    risk_level: str = Field(default="low", description="Assessed risk level: 'low', 'medium', 'high', 'critical'")
    injection_details: Optional[InjectionCheckResult] = Field(default=None, description="Prompt injection evaluation")


class InputGuard:
    """Enterprise input validation and sanitization guardrail."""

    def __init__(
        self,
        injection_detector: Optional[PromptInjectionDetector] = None,
        max_length: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.injection_detector = injection_detector or get_injection_detector()
        self.max_length = (
            max_length if max_length is not None else getattr(settings, "max_input_length", 2000)
        )

    def validate_input(self, raw_query: str) -> InputValidationResult:
        """Validate, sanitize, and screen input query for security threats.

        Evaluation Steps:
            1. Non-empty and length boundary validation.
            2. Removal of null bytes, control codes, and bidirectional override characters.
            3. Character flood / repetition anomaly detection.
            4. Deep prompt injection and jailbreak pattern scanning.
        """
        if not raw_query or not raw_query.strip():
            return InputValidationResult(
                is_valid=False,
                sanitized_text="",
                rejection_reason="Query is empty.",
                risk_level="low",
            )

        # 1. Length constraint check
        if len(raw_query) > self.max_length:
            logger.warning(f"Query length ({len(raw_query)}) exceeds maximum permitted ({self.max_length})")
            return InputValidationResult(
                is_valid=False,
                sanitized_text=raw_query[:self.max_length],
                rejection_reason=f"Query exceeds maximum permitted length of {self.max_length} characters.",
                risk_level="medium",
            )

        # 2. Sanitize invisible characters and normalize Unicode
        # Strip null bytes and control codes
        cleaned = SUSPICIOUS_UNICODE_PATTERN.sub("", raw_query)
        # Normalize Unicode NFC
        cleaned = unicodedata.normalize("NFC", cleaned).strip()

        # 3. Repeated character / token-overflow attack detection
        # e.g., 'a' repeated 150+ times continuously
        if re.search(r"(.)\1{120,}", cleaned):
            logger.warning("Repeated character flood attack detected.")
            return InputValidationResult(
                is_valid=False,
                sanitized_text=cleaned,
                rejection_reason="Repeated character flood detected.",
                risk_level="high",
            )

        # 4. Prompt Injection & Jailbreak Scanning
        injection_res = self.injection_detector.detect(cleaned)
        if injection_res.is_injection:
            logger.warning(
                f"Input rejected by security guardrail: reason='{injection_res.reason}', "
                f"snippet='{injection_res.flagged_snippet}'"
            )
            return InputValidationResult(
                is_valid=False,
                sanitized_text=cleaned,
                rejection_reason=SECURITY_REJECTION_MESSAGE,
                risk_level="critical",
                injection_details=injection_res,
            )

        return InputValidationResult(
            is_valid=True,
            sanitized_text=cleaned,
            risk_level="low",
            injection_details=injection_res,
        )


# Global singleton instance
_GLOBAL_INPUT_GUARD: Optional[InputGuard] = None


def get_input_guard() -> InputGuard:
    """Return singleton InputGuard instance."""
    global _GLOBAL_INPUT_GUARD
    if _GLOBAL_INPUT_GUARD is None:
        _GLOBAL_INPUT_GUARD = InputGuard()
    return _GLOBAL_INPUT_GUARD
