"""Unit tests for Slack session identification and conversation memory isolation."""

import pytest
from app.slack.memory import get_slack_session_id


def test_dm_session_generation():
    """Verify Slack DM generates deterministic user-channel isolated session ID."""
    session_id = get_slack_session_id(user_id="U12345", channel_id="D98765")
    assert session_id == "slack_dm_U12345_D98765"


def test_thread_session_generation():
    """Verify thread session ID is tied strictly to channel and thread timestamp."""
    session_id = get_slack_session_id(
        user_id="U12345",
        channel_id="C99999",
        thread_ts="1712345678.123456",
    )
    assert session_id == "slack_thread_C99999_1712345678_123456"


def test_thread_isolation_between_threads():
    """Verify different threads in the same channel do not share session IDs."""
    sess1 = get_slack_session_id(user_id="U1", channel_id="C1", thread_ts="100.1")
    sess2 = get_slack_session_id(user_id="U2", channel_id="C1", thread_ts="200.2")
    assert sess1 != sess2


def test_employee_isolation_in_dms():
    """Verify Employee A and Employee B never share session IDs in DMs."""
    sess_a = get_slack_session_id(user_id="U_ALICE", channel_id="D_ALICE")
    sess_b = get_slack_session_id(user_id="U_BOB", channel_id="D_BOB")
    assert sess_a != sess_b
    assert "U_ALICE" in sess_a
    assert "U_BOB" in sess_b


def test_safe_fallbacks_for_empty_values():
    """Verify safe fallback strings if user or channel is empty."""
    sess = get_slack_session_id(user_id="", channel_id="")
    assert "unknown_user" in sess
