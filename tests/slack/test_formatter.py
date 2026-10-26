"""Unit tests for Slack Message and Citation Formatter."""

import pytest
from app.slack.formatter import format_slack_response, format_slack_sources_text


def test_format_single_source():
    """Verify single source formatting matches requirements."""
    sources = [
        {
            "document_name": "Leave and Time-Off Policy",
            "policy_id": "HR-LEAVE-001",
            "version": "2.0",
            "section": "Active Entitlements",
            "page": 2,
            # Leaked technical fields that must NOT appear
            "chunk_id": "secret_chunk_99",
            "score": 0.95,
            "vector": [0.1, 0.2],
        }
    ]

    text = format_slack_sources_text(sources)
    assert "*Sources (1)*" in text
    assert "📄 *Leave and Time-Off Policy*" in text
    assert "`HR-LEAVE-001` • Version 2.0" in text
    assert "Section: Active Entitlements" in text
    assert "Page: 2" in text

    # Strict privacy check
    assert "secret_chunk" not in text
    assert "score" not in text
    assert "vector" not in text


def test_format_multiple_sources():
    """Verify formatting when multiple sources support an answer."""
    sources = [
        {
            "document_name": "Employee Handbook",
            "policy_id": "HR-HANDBOOK-001",
            "version": "2.0",
        },
        {
            "document_name": "Leave and Time-Off Policy",
            "policy_id": "HR-LEAVE-001",
            "version": "2.0",
        },
    ]

    text = format_slack_sources_text(sources)
    assert "*Sources (2)*" in text
    assert "📄 *Employee Handbook*" in text
    assert "📄 *Leave and Time-Off Policy*" in text


def test_format_slack_response_block_kit():
    """Verify Block Kit structure contains section, divider, and sources."""
    answer = "You are entitled to 18 working days of Annual Leave per year."
    sources = [
        {
            "document_name": "Leave and Time-Off Policy",
            "policy_id": "HR-LEAVE-001",
            "version": "2.0",
            "page": 1,
        }
    ]

    resp = format_slack_response(answer=answer, sources=sources, is_admin=False)
    assert resp["text"].startswith("You are entitled to 18 working days")
    assert len(resp["blocks"]) >= 3  # answer section, divider, sources section

    block_types = [b["type"] for b in resp["blocks"]]
    assert "section" in block_types
    assert "divider" in block_types


def test_admin_diagnostics_omitted_for_normal_users():
    """Verify developer retrieval metadata is excluded for normal employees."""
    answer = "Normal answer."
    meta = {"model": "openai/gpt-oss-20b", "cache_hit": True}

    # Normal user
    normal_resp = format_slack_response(answer, sources=[], is_admin=False, metadata=meta)
    assert "Admin Diagnostics" not in normal_resp["text"]
    assert not any("Admin Diagnostics" in str(b) for b in normal_resp["blocks"])

    # Admin user
    admin_resp = format_slack_response(answer, sources=[], is_admin=True, metadata=meta)
    assert any("Admin Diagnostics" in str(b) for b in admin_resp["blocks"])
