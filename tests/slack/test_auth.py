"""Unit tests for Slack Authentication, Authorization, and Rate Limiting."""

from unittest.mock import MagicMock
import pytest

from app.slack.auth import (
    UNAUTHORIZED_SLACK_MESSAGE,
    SlackAuthorizer,
    SlackRateLimiter,
)
from app.slack.config import SlackSettings


def test_authorized_user_via_allowed_list():
    """Verify Slack user in SLACK_ALLOWED_USERS is permitted."""
    settings = SlackSettings(
        allowed_users=["U12345", "U67890"],
        admin_users=["U12345"],
        allowed_email_domains=["xyz.com"],
    )
    authorizer = SlackAuthorizer(settings=settings)

    # Standard allowed user
    res1 = authorizer.authorize_user("U67890")
    assert res1.is_authorized is True
    assert res1.is_admin is False

    # Admin allowed user
    res2 = authorizer.authorize_user("U12345")
    assert res2.is_authorized is True
    assert res2.is_admin is True


def test_unauthorized_user_rejected_with_exact_message():
    """Verify unauthorized Slack user is rejected with clear message."""
    settings = SlackSettings(
        allowed_users=["U12345"],
        admin_users=[],
        allowed_email_domains=["xyz.com"],
    )
    authorizer = SlackAuthorizer(settings=settings)

    # Unknown user without matching client email
    res = authorizer.authorize_user("U_UNKNOWN")
    assert res.is_authorized is False
    assert res.rejection_message == UNAUTHORIZED_SLACK_MESSAGE
    assert "Only authorized company employees" in res.rejection_message


def test_authorized_user_via_slack_client_email_domain():
    """Verify employee is authorized if Slack profile email ends with allowed domain."""
    settings = SlackSettings(
        allowed_users=[],
        admin_users=[],
        allowed_email_domains=["xyz.com"],
    )
    authorizer = SlackAuthorizer(settings=settings)

    mock_client = MagicMock()
    mock_client.users_info.return_value = {
        "ok": True,
        "user": {
            "profile": {
                "email": "employee@xyz.com",
            }
        },
    }

    res = authorizer.authorize_user("U_NEW_EMPLOYEE", client=mock_client)
    assert res.is_authorized is True
    assert res.user_email == "employee@xyz.com"


def test_unauthorized_user_external_domain_rejected():
    """Verify employee with external domain (e.g. gmail.com) is rejected."""
    settings = SlackSettings(
        allowed_users=["U_SPECIFIC"],
        admin_users=[],
        allowed_email_domains=["xyz.com"],
    )
    authorizer = SlackAuthorizer(settings=settings)

    mock_client = MagicMock()
    mock_client.users_info.return_value = {
        "ok": True,
        "user": {
            "profile": {
                "email": "intruder@gmail.com",
            }
        },
    }

    res = authorizer.authorize_user("U_EXTERNAL", client=mock_client)
    assert res.is_authorized is False
    assert res.rejection_message == UNAUTHORIZED_SLACK_MESSAGE


def test_empty_user_id_rejected():
    """Verify empty or None user ID is rejected safely."""
    authorizer = SlackAuthorizer()
    res = authorizer.authorize_user("")
    assert res.is_authorized is False


def test_slack_rate_limiter_allows_and_blocks():
    """Verify rate limiter blocks after reaching limit and returns retry_after."""
    limiter = SlackRateLimiter(max_requests=3, window_seconds=60)
    user = "U_SPAMMER"

    # First 3 requests permitted
    for _ in range(3):
        is_limited, retry = limiter.check_rate_limit(user)
        assert is_limited is False
        assert retry is None

    # 4th request must be rate limited
    is_limited, retry = limiter.check_rate_limit(user)
    assert is_limited is True
    assert retry is not None
    assert retry > 0

    # Another user is not affected
    is_limited_other, _ = limiter.check_rate_limit("U_ANOTHER")
    assert is_limited_other is False
