"""Slack conversation session and thread isolation manager.

Maps Slack DM channels, threads, and user identifiers to deterministic RAG session IDs.
Prevents cross-contamination of conversation memory between different employees.
"""

from __future__ import annotations

from typing import Optional


def get_slack_session_id(
    user_id: str,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
) -> str:
    """Generate a clean, isolated session ID for a Slack interaction.

    Rules:
    1. If part of a thread in a channel, isolate by channel and thread_ts.
    2. If a direct message (DM), isolate by user_id and channel_id.
    3. If a channel top-level message, isolate by channel_id and user_id.
    """
    clean_user = (user_id or "unknown_user").strip()
    clean_channel = (channel_id or "direct").strip()

    if thread_ts:
        clean_ts = str(thread_ts).replace(".", "_")
        return f"slack_thread_{clean_channel}_{clean_ts}"

    if clean_channel.startswith("D"):  # Slack Direct Message channels start with 'D'
        return f"slack_dm_{clean_user}_{clean_channel}"

    return f"slack_channel_{clean_channel}_{clean_user}"
