"""Scope validation for HR Policy RAG Assistant.

Ensures the assistant remains strictly bounded to enterprise HR policies, benefits,
and workplace guidelines, while politely deflecting out-of-scope inquiries.
"""

from __future__ import annotations

import re
from typing import Optional, Set
from pydantic import BaseModel, Field

from app.utils.logging import get_logger

logger = get_logger("app.agent.scope_checker")

OUT_OF_SCOPE_RESPONSE = "I couldn't find this information in the available XYZ Technologies HR policy documents."

# Comprehensive set of valid HR policy topics, concepts, and workplace terminology
HR_POLICY_KEYWORDS: Set[str] = {
    # Leave & Time Off
    "leave", "vacation", "holiday", "pto", "sick", "parental", "maternity", "paternity",
    "bereavement", "jury", "sabbatical", "carryover", "carry forward", "accrual", "accrue", "day off", "time off",
    "absence", "fmla", "caregiver", "unpaid leave", "casual leave", "annual leave",
    # Working Hours & Attendance
    "attendance", "working hours", "hours", "overtime", "timesheet", "shift", "tardy", "core hours",
    "core collaboration hours", "clock in", "clock out", "break", "rest period", "weekend", "on-call", "late",
    # Compensation, Travel & Expenses
    "expense", "allowance", "per diem", "reimbursement", "travel", "flight", "hotel", "lodging",
    "mileage", "receipt", "meal", "breakfast", "lunch", "dinner", "stipend", "equipment stipend",
    "home office allowance", "budget", "approval limit", "l1-l3", "l1", "l2", "l3", "taxes",
    # Remote Work & Workplace
    "remote", "wfh", "work from home", "hybrid", "telecommute", "office", "ergonomics",
    "internet allowance", "desk", "monitor", "laptop", "equipment", "relocation",
    # Health, Benefits & Retirement
    "benefit", "insurance", "health", "medical", "dental", "vision", "401k", "401(k)", "pension",
    "wellness", "disability", "life insurance", "gym", "fsa", "hsa", "mental health", "learning support",
    "learning benefit", "accident cover",
    # Conduct, Governance & Employment
    "conduct", "harassment", "discrimination", "diversity", "inclusion", "policy", "guideline",
    "rule", "disciplinary", "termination", "resignation", "notice period", "probation", "performance",
    "review", "evaluation", "promotion", "grievance", "whistleblower", "dress code", "ethics", "rating scale",
    "assets", "exit", "separation", "mfa", "passwords", "security", "data classification", "accounts",
    # Identifiers & General Corporate HR
    "hr", "human resources", "people operations", "employee", "manager", "contractor", "intern",
    "full-time", "part-time", "worker", "employment", "company", "xyz", "xyz technologies", "effective date",
}

# Patterns characteristic of completely out-of-scope general assistant requests
OUT_OF_SCOPE_PATTERNS = [
    # Programming / coding requests
    r"\b(write|create|debug|generate|fix)\s+(a\s+)?(python|javascript|typescript|java|c\+\+|html|css|sql|rust|code|script|function|program|algorithm|class)\b",
    # Culinary / recipes
    r"\b(recipe|bake|cook|ingredients|how\s+to\s+(make|cook|bake)\s+(cake|pasta|cookies|pizza|bread))\b",
    # Creative fiction / poetry
    r"\b(write\s+(a\s+)?(poem|story|song|essay|novel|haiku|joke|fairy\s*tale))\b",
    # Financial speculation / trading
    r"\b(buy|sell|predict|invest\s+in)\s+(crypto|bitcoin|stock|stocks|shares|ethereum|market|forex)\b",
    # World trivia / weather / general knowledge unrelated to workplace
    r"\b(weather|temperature|forecast|rain|sunny|climate)\b",
    r"\b(capital\s+of\s+[a-z]+|who\s+is\s+the\s+president\s+of|distance\s+to\s+the\s+moon|speed\s+of\s+light)\b",
    # Math homework / general tutoring
    r"\b(solve\s+for\s+x|integrate\s+the\s+equation|derivative\s+of)\b",
]


class ScopeCheckResult(BaseModel):
    """Result of HR policy scope assessment."""

    is_in_scope: bool = Field(description="Whether the query relates to HR policies")
    reason: str = Field(description="Explanation of classification decision")
    decline_message: str = Field(default=OUT_OF_SCOPE_RESPONSE, description="Polite deflection message")


class HRScopeChecker:
    """Evaluates whether an incoming user query falls within enterprise HR domain."""

    def __init__(self) -> None:
        self._compiled_out_patterns = [
            re.compile(pat, re.IGNORECASE) for pat in OUT_OF_SCOPE_PATTERNS
        ]

    def check_conversational_intent(self, query: str, language: str = "English") -> Optional[str]:
        """Identify greetings, self-introductions, or thanks that should receive immediate conversational answers."""
        clean = query.strip().lower()
        from app.rag.multilingual import detect_language

        detected = detect_language(query)
        target_lang = (detected or language or "English").lower()

        # Telugu greetings
        if clean in ("నమస్కారం", "హలో", "ధన్యవాదాలు") or (target_lang == "telugu" and clean in ("hi", "hello", "hii", "hey")):
            return (
                "నమస్కారం! నేను XYZ Technologies HR పాలసీ అసిస్టెంట్‌ని. "
                "సెలవు విధానాలు (Leave Policy), వర్క్ ఫ్రమ్ హోమ్ నిబంధనలు (WFH), పని వేళలు, "
                "మరియు ఉద్యోగుల ప్రయోజనాల గురించి మీకు ఎలా సహాయపడగలను?"
            )

        # Hindi greetings
        if clean in ("नमस्ते", "नमस्कार", "धन्यवाद", "प्रणाम") or (target_lang == "hindi" and clean in ("hi", "hello", "hii", "hey")):
            return (
                "नमस्ते! मैं XYZ Technologies का HR पॉलिसी असिस्टेंट हूँ। "
                "कंपनी की छुट्टियों की नीति (Leave Policy), वर्क फ्रॉम होम (WFH), कार्य समय, "
                "और कर्मचारी लाभों के बारे में मैं आपकी क्या सहायता कर सकता हूँ?"
            )

        # Tamil greetings
        if clean in ("வணக்கம்", "ஹலோ", "நன்றி") or (target_lang == "tamil" and clean in ("hi", "hello", "hii", "hey")):
            return (
                "வணக்கம்! நான் XYZ Technologies HR கொள்கை உதவியாளர். "
                "விடுப்பு கொள்கை (Leave Policy), வீட்டில் இருந்து வேலை செய்யும் விதிமுறைகள் (WFH), வேலை நேரம் "
                "மற்றும் பணியாளர் நன்மைகள் குறித்து உங்களுக்கு எவ்வாறு உதவ முடியும்?"
            )

        # Direct word matches (English default)
        if clean in (
            "hi", "hello", "hii", "hiii", "hey", "heyy", "howdy", "greetings",
            "good morning", "good afternoon", "good evening", "good day",
            "hey there", "hello there",
        ):
            return (
                "Hello! I am your XYZ Technologies HR Policy Assistant. "
                "I'm here to help you understand our company policies, leave entitlements, remote and hybrid work rules, "
                "working hours, travel expenses, and employee benefits. How can I assist you today?"
            )
        if clean in (
            "who are you", "who are you?", "what are you", "what are you?",
            "what can you do", "what can you do?", "help", "help me", "how can you help me?",
        ):
            return (
                "I am the official HR Policy Assistant for XYZ Technologies Private Limited. "
                "You can ask me questions about our company policies—including Annual Leave, Sick Leave, "
                "hybrid work arrangements, expense reimbursement limits, performance appraisals, and IT security guidelines."
            )
        if clean in (
            "thanks", "thank you", "thank you!", "thanks!", "thanks a lot", "thank you so much",
        ):
            return (
                "You're very welcome! Please feel free to ask if you have any other questions about XYZ Technologies HR policies."
            )
        return None

    def check_scope(self, query: str) -> ScopeCheckResult:
        """Classify query as in-scope HR policy question or out-of-scope inquiry."""
        clean_q = query.strip()
        from app.rag.multilingual import translate_query_to_english_for_search

        # Map regional queries to English HR terms so they match keyword rules
        search_mapped_q = translate_query_to_english_for_search(clean_q)
        lower_q = search_mapped_q.lower()

        if not clean_q:
            return ScopeCheckResult(
                is_in_scope=False,
                reason="Empty query",
                decline_message="Please ask a question about company HR policies.",
            )

        # 1. Check for blatant out-of-scope patterns (coding, creative writing, recipes, trading)
        for pattern in self._compiled_out_patterns:
            if pattern.search(lower_q):
                logger.info(f"Query flagged as out-of-scope (matched pattern: {pattern.pattern}): '{clean_q[:50]}...'")
                return ScopeCheckResult(
                    is_in_scope=False,
                    reason=f"Matched out-of-scope pattern: {pattern.pattern}",
                    decline_message=OUT_OF_SCOPE_RESPONSE,
                )

        # 2. Check for explicit policy codes or identifiers (e.g. HR-POL-2026-01, EMP-1029)
        if re.search(r"\b(hr-[a-z0-9\-]+|emp-[0-9]+)\b", lower_q):
            return ScopeCheckResult(
                is_in_scope=True,
                reason="Contains explicit policy or employee identifier",
            )

        # 3. Check for HR keywords or workplace terminology
        # Tokenize query words
        words = set(re.findall(r"\b[a-zA-Z0-9_\-\(\)]+\b", lower_q))
        matching_keywords = words & HR_POLICY_KEYWORDS

        if matching_keywords:
            return ScopeCheckResult(
                is_in_scope=True,
                reason=f"Matched HR policy concepts: {', '.join(sorted(matching_keywords)[:3])}",
            )

        # Multi-word phrase checks (e.g., "work from home", "carry over", "time off")
        for phrase in ("work from home", "carry over", "time off", "core hours", "paid leave", "sick day"):
            if phrase in lower_q:
                return ScopeCheckResult(
                    is_in_scope=True,
                    reason=f"Matched HR policy phrase: '{phrase}'",
                )

        # 4. Check conversational / general organizational inquiry (e.g. "what policies do you have?", "hello", "can you help me?")
        conversational_starters = ("what", "how", "can i", "do we", "is there", "tell me", "explain", "where", "who", "hello", "hi")
        if any(lower_q.startswith(starter) for starter in conversational_starters):
            # If it's a general question that didn't trigger out-of-scope patterns,
            # allow it so we don't incorrectly reject edge-case employee questions
            return ScopeCheckResult(
                is_in_scope=True,
                reason="General inquiry without out-of-scope indicators; allowed for policy search",
            )

        # If it doesn't match HR keywords and doesn't look like a standard question, decline politely
        logger.info(f"Query deemed out of HR scope: '{clean_q[:60]}...'")
        return ScopeCheckResult(
            is_in_scope=False,
            reason="Query does not relate to company HR policies or workplace guidelines",
            decline_message=OUT_OF_SCOPE_RESPONSE,
        )

    def detect_query_filters(self, query: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Detect target (status, policy_id, policy_type) from user query.

        Rules:
            1. Status:
               - If query explicitly references historical context ("before april 2026", "old policy", "archived", "v1.0", "in 2025", "previous"): status = "archived"
               - For all other standard/current policy inquiries: status = "active"
            2. Policy ID:
               - If query explicitly mentions policy codes (e.g. "HR-LEAVE-001", "HR-WFH-001", "FIN-EXP-001", "IT-SEC-001"): extract policy_id
            3. Policy Type:
               - Inferred category from keywords if clear.
        """
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # 1. Historical vs Current Status Filter
        historical_patterns = [
            r"\bbefore\s+(april\s+)?2026\b",
            r"\bprior\s+to\s+(april\s+)?2026\b",
            r"\bold\s+(policy|leave|entitlement|rules?)\b",
            r"\bprevious\s+(policy|leave|entitlement|rules?|version)\b",
            r"\barchived\b",
            r"\bv1(?:\.0)?\b",
            r"\bversion\s+1(?:\.0)?\b",
            r"\bin\s+2025\b",
            r"\bwhat\s+was\s+the\s+.*before\b",
        ]
        is_historical = any(re.search(pat, lower_q) for pat in historical_patterns)
        status = "archived" if is_historical else "active"

        # 2. Specific Policy ID Extraction
        policy_id_m = re.search(r"\b(HR-[A-Z0-9\-]+|FIN-[A-Z0-9\-]+|IT-[A-Z0-9\-]+)\b", clean_q, re.IGNORECASE)
        policy_id = policy_id_m.group(1).upper() if policy_id_m else None

        # 3. Policy Type is kept None here to prevent over-filtering across related policies
        # (e.g. core hours spans remote work vs attendance)
        policy_type = None

        return status, policy_id, policy_type


# Global singleton instance
_GLOBAL_SCOPE_CHECKER: Optional[HRScopeChecker] = None


def get_scope_checker() -> HRScopeChecker:
    """Return singleton HRScopeChecker instance."""
    global _GLOBAL_SCOPE_CHECKER
    if _GLOBAL_SCOPE_CHECKER is None:
        _GLOBAL_SCOPE_CHECKER = HRScopeChecker()
    return _GLOBAL_SCOPE_CHECKER
