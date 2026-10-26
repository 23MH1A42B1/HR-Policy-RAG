"""Prompt injection and jailbreak detection for HR Policy RAG Assistant.

Detects direct prompt injection attacks, jailbreak attempts, system prompt extraction,
API key extraction, and indirect injection embedded within untrusted policy documents.
"""

from __future__ import annotations

import re
from typing import List, Optional
from pydantic import BaseModel, Field

from app.utils.logging import get_logger

logger = get_logger("app.security.prompt_injection")

# Injection and jailbreak pattern catalog
INJECTION_PATTERNS = [
    # 1. Instruction override / rule cancellation
    (
        r"\b(ignore|forget|disregard|bypass|override|cancel|dismiss)\s+(all\s+|the\s+|any\s+|prior\s+)*(previous|past|above|system|initial|prior)?\s*(instructions|prompts|rules|commands|guidelines|context|constraints)\b",
        "instruction_override",
        0.95,
    ),
    (
        r"\bdo\s+not\s+follow\s+(the\s+)?(above|previous|system|prior)?\s*(rules|instructions|prompts)\b",
        "instruction_override",
        0.95,
    ),
    (r"\bstart\s+fresh\s+and\s+ignore\b", "instruction_override", 0.90),

    # 2. System prompt extraction
    (
        r"\b(reveal|show(\s+me)?|print|output|display|repeat|tell\s+me|expose|leak|what\s+(is|are))\s+(all\s+|the\s+|your\s+)*(system\s+prompt|initial\s+prompt|hidden\s+prompt|core\s+instructions|system\s+instructions|secret\s+instructions|core\s+rules|initial\s+instructions)\b",
        "system_prompt_extraction",
        0.95,
    ),
    (
        r"\bwhat\s+(are\s+the\s+rules|are\s+your\s+instructions)\s+given\s+to\s+you\s+by\s+(the\s+developer|anthropic|google|openai|admin)\b",
        "system_prompt_extraction",
        0.90,
    ),
    (r"\bverbatim\s+copy\s+of\s+your\s+(instructions|prompt)\b", "system_prompt_extraction", 0.95),
    (r"\bprint\s+the\s+hidden\s+prompt\b", "system_prompt_extraction", 0.95),

    # 3. Secret and credential extraction
    (
        r"\b(show(\s+me)?|reveal|print|give\s+me|tell\s+me|what\s+(is|are))\s+(all\s+|the\s+|your\s+)*(secret\s+)?(api\s*keys?|secret\s*keys?|groq[_-]?api[_-]?key|gemini[_-]?api[_-]?key|openai[_-]?api[_-]?key|qdrant[_-]?api[_-]?key|jina[_-]?api[_-]?key|passwords?|credentials?|tokens?)\b",
        "credential_extraction",
        0.95,
    ),
    (
        r"\b(reveal|print|dump|show(\s+me)?|output|tell\s+me|what\s+(is|are))\s+(all\s+|the\s+|your\s+)*(environment\s*variables?|env\s*vars?|\.env\s*file|system\s*configuration|secret\s*credentials)\b",
        "credential_extraction",
        0.95,
    ),
    (
        r"\b(GROQ_API_KEY|GEMINI_API_KEY|OPENAI_API_KEY|QDRANT_API_KEY|JINA_API_KEY)\b",
        "credential_extraction",
        0.95,
    ),

    # 4. Context / policy document override
    (
        r"\b(ignore|disregard|forget)\s+(the\s+|all\s+)?(retrieved|company|hr|actual)?\s*(policy|policies|documents?|excerpts?|context)\b",
        "policy_override",
        0.95,
    ),
    (
        r"\binstead\s+of\s+(the\s+)?(policy|policies|document|documents),\s*(answer|say|tell|act)\b",
        "policy_override",
        0.90,
    ),

    # 5. Roleplay and jailbreak frameworks (DAN, Developer Mode, Persona hijacking)
    (
        r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(an?\s+)?(DAN|unrestricted|jailbroken|evil\s+ai|ai\s+without\s+rules|developer\s+mode|root|admin)\b",
        "roleplay_jailbreak",
        0.95,
    ),
    (
        r"\b(enable|activate|switch\s+to)\s+(developer\s+mode|god\s+mode|unfiltered\s+mode|unrestricted\s+mode)\b",
        "roleplay_jailbreak",
        0.95,
    ),
    (
        r"\bhypothetical\s+scenario\s+where\s+you\s+have\s+no\s+(rules|restrictions|limits|ethics)\b",
        "roleplay_jailbreak",
        0.90,
    ),

    # 6. Delimiter and structural prompt injection
    (r"(?:###\s*system|---+\s*\[document\s+excerpt|\<\|im_start\|\>|\<\|system\|\>)", "delimiter_injection", 0.85),

    # 7. Indirect injection markers (typically found inside retrieved documents or quotes)
    (
        r"\b(attention\s+(assistant|ai|model|llm|system)|new\s+system\s+instruction|important\s+system\s+update:)\s*[:\-]?\s*(ignore|award|approve|grant|execute)\b",
        "indirect_document_injection",
        0.95,
    ),
]


class InjectionCheckResult(BaseModel):
    """Result of prompt injection evaluation."""

    is_injection: bool = Field(description="Whether text is flagged as a prompt injection attack")
    confidence: float = Field(default=0.0, description="Confidence score between 0.0 and 1.0")
    attack_categories: List[str] = Field(default_factory=list, description="Categorized attack types detected")
    reason: str = Field(default="No injection detected", description="Explanation of findings")
    flagged_snippet: Optional[str] = Field(default=None, description="Excerpt triggering the detection")


class PromptInjectionDetector:
    """Detects direct and indirect prompt injection attempts."""

    def __init__(self, confidence_threshold: float = 0.80) -> None:
        self.confidence_threshold = confidence_threshold
        self._compiled_patterns = [
            (re.compile(pat, re.IGNORECASE), category, score)
            for pat, category, score in INJECTION_PATTERNS
        ]

    def detect(self, text: str) -> InjectionCheckResult:
        """Scan input text against prompt injection and jailbreak signatures."""
        clean_text = text.strip()
        if not clean_text:
            return InjectionCheckResult(is_injection=False)

        max_confidence = 0.0
        categories: List[str] = []
        snippets: List[str] = []

        for regex, category, score in self._compiled_patterns:
            match = regex.search(clean_text)
            if match:
                categories.append(category)
                snippets.append(match.group(0))
                if score > max_confidence:
                    max_confidence = score

        if max_confidence >= self.confidence_threshold:
            unique_categories = sorted(list(set(categories)))
            reason_str = f"Prompt injection detected ({', '.join(unique_categories)})"
            logger.warning(
                f"Prompt injection flagged: confidence={max_confidence:.2f}, "
                f"categories={unique_categories}, snippet='{snippets[0]}'"
            )
            return InjectionCheckResult(
                is_injection=True,
                confidence=max_confidence,
                attack_categories=unique_categories,
                reason=reason_str,
                flagged_snippet=snippets[0],
            )

        return InjectionCheckResult(is_injection=False, confidence=max_confidence)

    def detect_in_document(self, passage_text: str, document_name: str = "") -> InjectionCheckResult:
        """Specifically screen untrusted retrieved document passages for indirect prompt injections."""
        result = self.detect(passage_text)
        if result.is_injection:
            logger.warning(
                f"Indirect prompt injection detected in retrieved document '{document_name}': "
                f"categories={result.attack_categories}"
            )
        return result


# Global singleton detector
_GLOBAL_DETECTOR: Optional[PromptInjectionDetector] = None


def get_injection_detector() -> PromptInjectionDetector:
    """Return singleton PromptInjectionDetector instance."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = PromptInjectionDetector()
    return _GLOBAL_DETECTOR
