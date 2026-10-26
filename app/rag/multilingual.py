"""Multilingual support engine for HR Policy RAG Assistant.

Supports English, Hindi (हिंदी), Telugu (తెలుగు), and Tamil (தமிழ்).
Provides script detection, cross-lingual search translation, language-specific
prompts, starter suggestions, and greetings.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from app.utils.logging import get_logger

logger = get_logger("app.rag.multilingual")

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "English": "English",
    "Hindi": "Hindi (हिंदी)",
    "Telugu": "Telugu (తెలుగు)",
    "Tamil": "Tamil (தமிழ்)",
}

STARTER_QUESTIONS: Dict[str, List[str]] = {
    "English": [
        "How many annual leave days do employees get?",
        "Can I work from home under the remote work policy?",
        "What are the core collaboration hours?",
        "When should expense claims normally be submitted?",
    ],
    "Hindi": [
        "कर्मचारियों को वर्ष में कितनी वार्षिक छुट्टियां (Annual Leave) मिलती हैं?",
        "क्या मैं रिमोट वर्क पॉलिसी के तहत घर से काम (WFH) कर सकता हूँ?",
        "कंपनी के मुख्य कार्य घंटे (Core Hours) क्या हैं?",
        "यात्रा और अन्य खर्चों की प्रतिपूर्ति (Expense Claims) के क्या नियम हैं?",
    ],
    "Telugu": [
        "ఉద్యోగులకు సంవత్సరానికి ఎన్ని వార్షిక సెలవులు (Annual Leave) లభిస్తాయి?",
        "నేను రిమోట్ వర్క్ పాలసీ ప్రకారం ఇంటి నుండి పని (WFH) చేయవచ్చా?",
        "కంపెనీ ముఖ్య పని వేళలు (Core Hours) ఏమిటి?",
        "ఖర్చుల రీయింబర్స్‌మెంట్ క్లెయిమ్‌లను ఎప్పుడు సమర్పించాలి?",
    ],
    "Tamil": [
        "ஊழியர்களுக்கு வருடத்திற்கு எத்தனை வருடாந்திர விடுப்புகள் (Annual Leave) கிடைக்கும்?",
        "தொலைதூரப் பணி கொள்கையின் கீழ் நான் வீட்டிலிருந்து பணியாற்றலாமா?",
        "நிறுவனத்தின் முக்கிய வேலை நேரம் (Core Hours) என்ன?",
        "பயண மற்றும் செலவு திருப்பிச் செலுத்தும் கோரிக்கைகளை எப்போது சமர்ப்பிக்க வேண்டும்?",
    ],
}

WELCOME_MESSAGES: Dict[str, str] = {
    "English": "Hello! I am your HR Policy Assistant. Ask me anything about company HR policies.",
    "Hindi": "नमस्ते! मैं आपका HR पॉलिसी असिस्टेंट हूँ। कंपनी की नीतियों और नियमों के बारे में मुझसे कुछ भी पूछें।",
    "Telugu": "నమస్కారం! నేను మీ HR పాలసీ అసిస్టెంట్‌ని. కంపెనీ పాలసీల గురించి నన్ను ఏమైనా అడగండి.",
    "Tamil": "வணக்கம்! நான் உங்கள் HR கொள்கை உதவியாளர். நிறுவனத்தின் கொள்கைகள் குறித்து என்னிடம் கேளுங்கள்.",
}

# Cross-lingual keyword dictionary for offline/quick mapping to English policy terms
KEYWORD_MAPPINGS = {
    # Hindi
    "छुट्टी": "leave",
    "छुट्टियां": "leave",
    "वार्षिक छुट्टी": "annual leave",
    "बीमारी की छुट्टी": "sick leave",
    "घर से काम": "work from home remote hybrid",
    "रिमोट": "remote work",
    "कार्यालय": "office work",
    "प्रतिपूर्ति": "expense reimbursement",
    "भत्ता": "allowance",
    "यात्रा": "travel policy",
    "बीमा": "health insurance",
    "वेतन": "salary payroll",
    "घंटे": "working hours core hours",
    "इस्तीफा": "resignation notice period",
    "आचार संहिता": "code of conduct",
    # Telugu
    "సెలవు": "leave",
    "సెలవులు": "annual casual sick leave",
    "వార్షిక సెలవు": "annual leave",
    "ఇంటి నుండి పని": "work from home remote hybrid",
    "ఆఫీస్": "office working days",
    "రీయింబర్స్‌మెంట్": "expense reimbursement",
    "భత్యం": "allowance stipend",
    "ప్రయాణం": "travel expenses",
    "బీమా": "health insurance medical",
    "జీతం": "salary compensation",
    "పని వేళలు": "working hours core hours",
    "రాజీనామా": "resignation notice period",
    "ప్రవర్తనా నియమావళి": "code of conduct",
    # Tamil
    "விடுப்பு": "leave",
    "விடுப்புகள்": "annual casual sick leave",
    "வருடாந்திர விடுப்பு": "annual leave",
    "வீட்டிலிருந்து வேலை": "work from home remote hybrid",
    "அலுவலகம்": "office working days",
    "திருப்பிச் செலுத்துதல்": "expense reimbursement",
    "படி": "allowance",
    "பயணம்": "travel expenses",
    "காப்பீடு": "health insurance medical",
    "சம்பளம்": "salary compensation",
    "வேலை நேரம்": "working hours core hours",
    "ராஜினாமா": "resignation notice period",
    "நடத்தை விதிமுறை": "code of conduct",
}


def detect_language(text: str) -> Optional[str]:
    """Detect if the text contains Devanagari (Hindi), Telugu, or Tamil scripts."""
    if not text:
        return None
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "Hindi"
        if 0x0C00 <= cp <= 0x0C7F:
            return "Telugu"
        if 0x0B80 <= cp <= 0x0BFF:
            return "Tamil"
    return None


def get_language_directive(language: str) -> str:
    """Generate system instruction to enforce fluent output in the selected regional language."""
    lang = (language or "English").lower()
    if "hindi" in lang:
        return (
            "\n- LANGUAGE REQUIREMENT (HINDI):\n"
            "  * You MUST formulate your entire response in clear, natural, and polite Hindi (हिंदी).\n"
            "  * State policy entitlements, rules, and procedures accurately in Hindi.\n"
            "  * Keep formal policy document codes (e.g., HR-LEAVE-001, HR-WFH-001) and numbers in standard Latin format for reference."
        )
    elif "telugu" in lang:
        return (
            "\n- LANGUAGE REQUIREMENT (TELUGU):\n"
            "  * You MUST formulate your entire response in clear, natural, and polite Telugu (తెలుగు).\n"
            "  * State policy entitlements, rules, and procedures accurately in Telugu.\n"
            "  * Keep formal policy document codes (e.g., HR-LEAVE-001, HR-WFH-001) and numbers in standard Latin format for reference."
        )
    elif "tamil" in lang:
        return (
            "\n- LANGUAGE REQUIREMENT (TAMIL):\n"
            "  * You MUST formulate your entire response in clear, natural, and polite Tamil (தமிழ்).\n"
            "  * State policy entitlements, rules, and procedures accurately in Tamil.\n"
            "  * Keep formal policy document codes (e.g., HR-LEAVE-001, HR-WFH-001) and numbers in standard Latin format for reference."
        )
    return ""


def translate_query_to_english_for_search(query: str, active_language: str = "English") -> str:
    """Translate regional language query into English keywords for effective vector/BM25 retrieval."""
    detected = detect_language(query)
    target_lang = detected or active_language

    if target_lang == "English" and not detected:
        return query

    # Check if keyword dictionary contains direct matches
    matched_terms: List[str] = []
    for term, eng_equiv in KEYWORD_MAPPINGS.items():
        if term in query:
            matched_terms.append(eng_equiv)

    if matched_terms:
        enhanced = f"{query} {' '.join(matched_terms)}"
        logger.info(f"Enhanced regional query with English keywords: '{enhanced}'")
        return enhanced

    # If no dictionary keywords matched, return original query (FastEmbed can handle partial multilingual or fall back)
    return query
