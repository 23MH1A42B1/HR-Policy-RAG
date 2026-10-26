"""User feedback tracking and intelligent follow-up suggestions for HR Policy RAG Assistant."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from app.utils.logging import get_logger

logger = get_logger("app.ui.feedback")

FEEDBACK_FILE = Path("data/feedback_log.json")


def generate_followup_suggestions(
    query: str,
    answer: str,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Generate 2-3 relevant follow-up question chips based on the topic discussed."""
    combined = (query + " " + answer).lower()
    citations = citations or []
    policy_ids = {c.get("policy_id", "").upper() for c in citations if c.get("policy_id")}

    # 1. Leave & Time Off Policy
    if "leave" in combined or "pto" in combined or "vacation" in combined or "HR-LEAVE-001" in policy_ids:
        if "carry" not in combined:
            return [
                "Can I carry forward unused annual leave days?",
                "How does casual and sick leave work?",
                "What is the notice period for taking annual leave?",
            ]
        else:
            return [
                "How many total leave days do employees get per year?",
                "Are unused casual leave days encashed or expired?",
                "What happens to my leave balance if I resign?",
            ]

    # 2. Remote & Hybrid Work Policy
    if "remote" in combined or "hybrid" in combined or "wfh" in combined or "home" in combined or "HR-WFH-001" in policy_ids:
        return [
            "How many days per week must employees work from the office?",
            "What are the mandatory core collaboration hours?",
            "Can I claim an internet or home office allowance?",
        ]

    # 3. Expenses & Travel Reimbursement
    if "expense" in combined or "travel" in combined or "reimbursement" in combined or "flight" in combined or "FIN-EXP-001" in policy_ids:
        return [
            "What are the daily meal per diem limits during domestic travel?",
            "What is the maximum deadline to submit expense reimbursement claims?",
            "Can I claim professional learning and certification costs?",
        ]

    # 4. Benefits & Learning
    if "benefit" in combined or "learning" in combined or "course" in combined or "certification" in combined or "insurance" in combined or "HR-BEN-001" in policy_ids:
        return [
            "What is the maximum annual learning and certification reimbursement?",
            "What health and medical insurance coverage does XYZ offer?",
            "What are the parental leave entitlements for new parents?",
        ]

    # 5. Working Hours & Attendance
    if "attendance" in combined or "hours" in combined or "shift" in combined or "late" in combined or "HR-ATTEND-001" in policy_ids:
        return [
            "What are the standard core working hours?",
            "Is there a grace period for logging in to work?",
            "How does overtime compensation work for eligible roles?",
        ]

    # 6. Conduct, Security & General Policies
    if "conduct" in combined or "security" in combined or "password" in combined or "device" in combined or "IT-SEC-001" in policy_ids:
        return [
            "What is the company policy on gifts and conflicts of interest?",
            "What are the password and MFA security requirements?",
            "What is the disciplinary procedure for policy violations?",
        ]

    # Default general follow-ups
    return [
        "How many annual leave days do employees get?",
        "Can I work from home under the remote work policy?",
        "What learning benefits are available for employees?",
    ]


def record_feedback(
    message_id: str,
    query: str,
    answer: str,
    rating: str,  # 'helpful' or 'unhelpful'
    user_email: Optional[str] = None,
    comment: Optional[str] = None,
) -> bool:
    """Record an employee rating for an assistant response."""
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        entries: List[Dict[str, Any]] = []

        if FEEDBACK_FILE.exists():
            try:
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading feedback file: {e}")
                entries = []

        # Update existing entry if user changes their vote for the same message
        for item in entries:
            if item.get("message_id") == message_id:
                item["rating"] = rating
                item["timestamp"] = time.time()
                if comment:
                    item["comment"] = comment
                break
        else:
            entries.append({
                "message_id": message_id,
                "query": query[:200],
                "answer_preview": answer[:250],
                "rating": rating,
                "user_email": user_email or "anonymous",
                "timestamp": time.time(),
                "comment": comment or "",
            })

        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

        logger.info(f"Recorded feedback '{rating}' for query: '{query[:40]}...' by user '{user_email}'")
        return True
    except Exception as exc:
        logger.error(f"Failed to record feedback: {exc}")
        return False


def get_feedback_metrics() -> Dict[str, Any]:
    """Retrieve feedback statistics for the Admin Analytics Dashboard."""
    if not FEEDBACK_FILE.exists():
        return {
            "total_feedback": 0,
            "helpful_count": 0,
            "unhelpful_count": 0,
            "satisfaction_rate": 100.0,
            "recent_entries": [],
        }

    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            entries: List[Dict[str, Any]] = json.load(f)

        total = len(entries)
        helpful = sum(1 for e in entries if e.get("rating") == "helpful")
        unhelpful = sum(1 for e in entries if e.get("rating") == "unhelpful")
        rate = round((helpful / total * 100), 1) if total > 0 else 100.0

        return {
            "total_feedback": total,
            "helpful_count": helpful,
            "unhelpful_count": unhelpful,
            "satisfaction_rate": rate,
            "recent_entries": list(reversed(entries[-15:])),
        }
    except Exception as exc:
        logger.warning(f"Error computing feedback metrics: {exc}")
        return {
            "total_feedback": 0,
            "helpful_count": 0,
            "unhelpful_count": 0,
            "satisfaction_rate": 100.0,
            "recent_entries": [],
        }


def get_topic_analytics() -> Dict[str, int]:
    """Categorize logged employee queries into HR policy topic clusters."""
    counts = {
        "Leave & Time Off": 0,
        "Remote & Hybrid Work": 0,
        "Expenses & Travel": 0,
        "Benefits & Insurance": 0,
        "Working Hours & Attendance": 0,
        "General HR Policies": 0,
    }
    if not FEEDBACK_FILE.exists():
        return counts

    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            entries: List[Dict[str, Any]] = json.load(f)

        for e in entries:
            q = (e.get("query", "") + " " + e.get("answer_preview", "")).lower()
            if any(w in q for w in ["leave", "pto", "vacation", "sick", "casual"]):
                counts["Leave & Time Off"] += 1
            elif any(w in q for w in ["remote", "hybrid", "wfh", "home"]):
                counts["Remote & Hybrid Work"] += 1
            elif any(w in q for w in ["expense", "travel", "flight", "claim", "reimbursement", "hotel"]):
                counts["Expenses & Travel"] += 1
            elif any(w in q for w in ["benefit", "insurance", "health", "learning", "course", "medical"]):
                counts["Benefits & Insurance"] += 1
            elif any(w in q for w in ["hours", "shift", "attendance", "overtime", "late"]):
                counts["Working Hours & Attendance"] += 1
            else:
                counts["General HR Policies"] += 1

        return counts
    except Exception as exc:
        logger.warning(f"Error computing topic analytics: {exc}")
        return counts
