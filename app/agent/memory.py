"""Conversation memory management for HR Policy RAG Assistant.

Maintains sliding-window per-session history and contextualizes follow-up queries
without overwhelming the LLM prompt context.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.agent.memory")


class ConversationTurn(BaseModel):
    """Represents a single exchange between user and assistant."""

    user_query: str = Field(description="Employee's question")
    assistant_response: str = Field(description="Assistant's answer")
    timestamp: float = Field(default_factory=time.time, description="Exchange timestamp")


class SessionMemory:
    """Manages conversational history for a single user session."""

    def __init__(self, session_id: str, max_turns: int = 5) -> None:
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []

    def add_turn(self, user_query: str, assistant_response: str) -> None:
        """Add an exchange and enforce sliding window turn limit."""
        turn = ConversationTurn(
            user_query=user_query.strip(),
            assistant_response=assistant_response.strip(),
            timestamp=time.time(),
        )
        self.turns.append(turn)

        # Enforce sliding window turn limit (never send unlimited history)
        if len(self.turns) > self.max_turns:
            trimmed = len(self.turns) - self.max_turns
            self.turns = self.turns[-self.max_turns:]
            logger.debug(f"Trimmed {trimmed} oldest conversation turn(s) for session {self.session_id}")

    def get_history(self) -> List[ConversationTurn]:
        """Return all active turns within the sliding window."""
        return list(self.turns)

    def get_recent_context_str(self, max_turns: Optional[int] = None) -> str:
        """Format recent dialogue turns into a compact context string for prompting."""
        limit = max_turns or self.max_turns
        active_turns = self.turns[-limit:]
        if not active_turns:
            return ""

        formatted: List[str] = []
        for idx, turn in enumerate(active_turns, start=1):
            formatted.append(f"Employee: {turn.user_query}")
            # Truncate assistant response snippet to prevent context explosion
            resp_snippet = turn.assistant_response[:300].strip()
            if len(turn.assistant_response) > 300:
                resp_snippet += "..."
            formatted.append(f"Assistant: {resp_snippet}")

        return "\n".join(formatted)

    def contextualize_query(self, query: str) -> str:
        """Heuristically resolve follow-up queries and pronoun references using recent history.

        Example:
            Prior turn: "How many annual leave days do full-time employees get?"
            Follow-up: "What about carryover?"
            Resolved: "What about carryover for annual leave?"
        """
        clean_q = query.strip()
        if not self.turns:
            return clean_q

        last_turn = self.turns[-1]
        lower_q = clean_q.lower()

        # Follow-up trigger patterns (anaphoric references, conjunctions, short elliptical questions)
        followup_triggers = [
            "what about",
            "how about",
            "and what",
            "and how",
            "can i also",
            "does it apply",
            "what is the limit",
            "is that allowed",
            "for part time",
            "for contractors",
            "carryover",
            "rollover",
        ]

        is_followup = (
            len(clean_q.split()) <= 6
            or any(t in lower_q for t in followup_triggers)
            or any(pronoun in lower_q.split() for pronoun in ("it", "they", "this", "that", "these", "those"))
        )

        if is_followup:
            # Extract key topic words from prior question (e.g. 'annual leave', 'remote work', 'meal expense')
            prior_q = last_turn.user_query.strip()
            logger.info(
                f"Detected follow-up question: '{clean_q}'. Contextualizing with prior turn: '{prior_q[:50]}...'"
            )
            return f"{clean_q} (in context of: {prior_q})"

        return clean_q

    def clear(self) -> None:
        """Clear session memory."""
        self.turns.clear()


class MemoryManager:
    """Registry maintaining active SessionMemory instances."""

    def __init__(self, default_max_turns: Optional[int] = None, max_sessions: int = 5000) -> None:
        settings = get_settings()
        self.default_max_turns = (
            default_max_turns
            if default_max_turns is not None
            else getattr(settings, "memory_max_turns", 5)
        )
        self.max_sessions = max_sessions
        self.sessions: Dict[str, SessionMemory] = {}

    def get_session(self, session_id: Optional[str]) -> SessionMemory:
        """Retrieve or create a SessionMemory instance."""
        sid = session_id or "default_session"
        if sid not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                # Evict oldest session
                first_key = next(iter(self.sessions))
                del self.sessions[first_key]
            self.sessions[sid] = SessionMemory(session_id=sid, max_turns=self.default_max_turns)
        return self.sessions[sid]

    def clear_session(self, session_id: str) -> None:
        """Remove a session from memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Global memory manager singleton
_GLOBAL_MEMORY: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Return singleton MemoryManager instance."""
    global _GLOBAL_MEMORY
    if _GLOBAL_MEMORY is None:
        _GLOBAL_MEMORY = MemoryManager()
    return _GLOBAL_MEMORY
