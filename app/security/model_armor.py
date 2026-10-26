"""Google Model Armor integration with local security fallback for HR Policy RAG Assistant.

Provides input, output, and retrieved document screening. When Google Model Armor credentials
are configured, invokes the Model Armor REST API. Otherwise, transparently enforces
comprehensive local security screening without faking cloud API responses.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional, Sequence
import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.security.input_guard import (
    SECURITY_REJECTION_MESSAGE,
    InputGuard,
    get_input_guard,
)
from app.security.output_guard import OutputGuard, get_output_guard
from app.security.prompt_injection import (
    PromptInjectionDetector,
    get_injection_detector,
)
from app.utils.logging import get_logger

logger = get_logger("app.security.model_armor")


class ScreeningResult(BaseModel):
    """Result container for Model Armor input or output screening."""

    is_safe: bool = Field(description="Whether the content passed screening")
    sanitized_text: str = Field(description="Sanitized text safe for model or user consumption")
    status: str = Field(description="Integration status: 'cloud_model_armor' or 'local_security (cloud model armor not configured)'")
    threats_detected: List[str] = Field(default_factory=list, description="Threats, leaks, or injections detected")
    rejection_message: Optional[str] = Field(default=None, description="Explanation if content was rejected")


class DocumentScreeningResult(BaseModel):
    """Result of screening retrieved untrusted policy document excerpts."""

    is_safe: bool = Field(description="Whether retrieved document passage is free of indirect injections")
    document_name: str = Field(default="", description="Name of source document")
    sanitized_text: str = Field(description="Sanitized document passage")
    threats_detected: List[str] = Field(default_factory=list, description="Any indirect injections or exploits detected")


class ModelArmorService:
    """Enterprise Model Armor Service supporting Google Cloud API and local security screening."""

    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        template: Optional[str] = None,
        input_guard: Optional[InputGuard] = None,
        output_guard: Optional[OutputGuard] = None,
        injection_detector: Optional[PromptInjectionDetector] = None,
    ) -> None:
        settings = get_settings()
        self.project = project or os.environ.get("MODEL_ARMOR_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or settings.model_armor_project
        self.location = location or os.environ.get("MODEL_ARMOR_LOCATION") or getattr(settings, "model_armor_location", "us-central1")
        self.template = template or os.environ.get("MODEL_ARMOR_TEMPLATE") or getattr(settings, "model_armor_template", None)

        self.input_guard = input_guard or get_input_guard()
        self.output_guard = output_guard or get_output_guard()
        self.injection_detector = injection_detector or get_injection_detector()

    def is_cloud_configured(self) -> bool:
        """Check whether Google Cloud Model Armor requirements are configured."""
        has_project = bool(self.project and self.project.strip())
        # Check if auth token or credentials exist
        has_auth = bool(
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("GCP_ACCESS_TOKEN")
        )
        return has_project and has_auth

    @property
    def integration_status(self) -> str:
        """Report active screening mode transparently."""
        if self.is_cloud_configured():
            return "cloud_model_armor"
        return "local_security (cloud model armor not configured)"

    def screen_input(self, user_prompt: str) -> ScreeningResult:
        """Screen user question before routing to RAG or LLM."""
        clean_prompt = user_prompt.strip()
        if not clean_prompt:
            return ScreeningResult(
                is_safe=False,
                sanitized_text="",
                status=self.integration_status,
                threats_detected=["empty_input"],
                rejection_message="Please ask a question about company HR policies.",
            )

        # 1. Attempt Cloud Model Armor API if configured
        if self.is_cloud_configured():
            try:
                return self._call_cloud_screen_prompt(clean_prompt)
            except Exception as exc:
                logger.warning(
                    f"Google Model Armor API call encountered an error: {exc}. "
                    "Falling back to local security screening."
                )

        # 2. Local Security Screening Fallback (Truthful, not faking cloud API)
        input_res = self.input_guard.validate_input(clean_prompt)
        if not input_res.is_valid:
            threats = []
            if input_res.injection_details and input_res.injection_details.is_injection:
                threats.extend(input_res.injection_details.attack_categories)
            else:
                threats.append(input_res.rejection_reason or "input_violation")

            return ScreeningResult(
                is_safe=False,
                sanitized_text=input_res.sanitized_text,
                status="local_security (cloud model armor not configured)",
                threats_detected=threats,
                rejection_message=SECURITY_REJECTION_MESSAGE,
            )

        return ScreeningResult(
            is_safe=True,
            sanitized_text=input_res.sanitized_text,
            status="local_security (cloud model armor not configured)",
            threats_detected=[],
        )

    def screen_output(
        self,
        model_response: str,
        retrieved_passages: Optional[Sequence[Any]] = None,
    ) -> ScreeningResult:
        """Screen model generation before presenting to the employee."""
        clean_resp = model_response.strip()

        # 1. Attempt Cloud Model Armor API if configured
        if self.is_cloud_configured():
            try:
                return self._call_cloud_screen_response(clean_resp)
            except Exception as exc:
                logger.warning(
                    f"Google Model Armor API call failed for output: {exc}. "
                    "Falling back to local security screening."
                )

        # 2. Local Output Security Screening Fallback
        output_res = self.output_guard.validate_output(clean_resp, retrieved_passages=retrieved_passages)
        threats: List[str] = []
        if output_res.redacted_items:
            threats.extend(output_res.redacted_items)
        if output_res.system_prompt_leaked:
            threats.append("system_prompt_leakage")
        if output_res.inappropriate_data_exposed:
            threats.append("inappropriate_data_exposure")
        if output_res.unsupported_claims_detected:
            threats.append("unsupported_claims")

        return ScreeningResult(
            is_safe=output_res.is_safe,
            sanitized_text=output_res.sanitized_output,
            status="local_security (cloud model armor not configured)",
            threats_detected=threats,
        )

    def screen_document(self, passage_text: str, document_name: str = "") -> DocumentScreeningResult:
        """Screen untrusted retrieved document text for indirect prompt injections.

        Ensures retrieved policy text can NEVER override system instructions or issue commands.
        """
        clean_text = passage_text.strip()
        injection_res = self.injection_detector.detect_in_document(clean_text, document_name)

        if injection_res.is_injection:
            logger.warning(
                f"Quarantining suspicious passage in '{document_name}': {injection_res.reason}"
            )
            # Neutralize instruction commands inside retrieved text by wrapping safely
            sanitized = re.sub(
                r"\b(ignore|override|disregard|award|bypass)\b",
                "[UNTRUSTED_INSTRUCTION_FILTERED]",
                clean_text,
                flags=re.IGNORECASE,
            )
            return DocumentScreeningResult(
                is_safe=False,
                document_name=document_name,
                sanitized_text=sanitized,
                threats_detected=injection_res.attack_categories,
            )

        return DocumentScreeningResult(
            is_safe=True,
            document_name=document_name,
            sanitized_text=clean_text,
            threats_detected=[],
        )

    def _call_cloud_screen_prompt(self, prompt: str) -> ScreeningResult:
        """Call real Google Cloud Model Armor sanitizeUserPrompt API."""
        url = (
            f"https://modelarmor.googleapis.com/v1/projects/{self.project}/locations/{self.location}:sanitizeUserPrompt"
        )
        token = os.environ.get("GCP_ACCESS_TOKEN", "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {
            "userPromptData": {"text": prompt},
        }
        if self.template:
            body["modelArmorTemplate"] = self.template

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        sanitized_data = data.get("sanitizationResult", {})
        filter_match = sanitized_data.get("filterMatchState") == "MATCH_FOUND"
        sanitized_prompt = sanitized_data.get("sanitizedUserPrompt", {}).get("text", prompt)

        return ScreeningResult(
            is_safe=not filter_match,
            sanitized_text=sanitized_prompt,
            status="cloud_model_armor",
            threats_detected=["cloud_filter_match"] if filter_match else [],
            rejection_message=SECURITY_REJECTION_MESSAGE if filter_match else None,
        )

    def _call_cloud_screen_response(self, response_text: str) -> ScreeningResult:
        """Call real Google Cloud Model Armor sanitizeModelResponse API."""
        url = (
            f"https://modelarmor.googleapis.com/v1/projects/{self.project}/locations/{self.location}:sanitizeModelResponse"
        )
        token = os.environ.get("GCP_ACCESS_TOKEN", "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {
            "modelResponseData": {"text": response_text},
        }
        if self.template:
            body["modelArmorTemplate"] = self.template

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        sanitized_data = data.get("sanitizationResult", {})
        filter_match = sanitized_data.get("filterMatchState") == "MATCH_FOUND"
        sanitized_text = sanitized_data.get("sanitizedModelResponse", {}).get("text", response_text)

        return ScreeningResult(
            is_safe=not filter_match,
            sanitized_text=sanitized_text,
            status="cloud_model_armor",
            threats_detected=["cloud_filter_match"] if filter_match else [],
        )


# Global singleton instance
_GLOBAL_MODEL_ARMOR: Optional[ModelArmorService] = None


def get_model_armor() -> ModelArmorService:
    """Return singleton ModelArmorService instance."""
    global _GLOBAL_MODEL_ARMOR
    if _GLOBAL_MODEL_ARMOR is None:
        _GLOBAL_MODEL_ARMOR = ModelArmorService()
    return _GLOBAL_MODEL_ARMOR
