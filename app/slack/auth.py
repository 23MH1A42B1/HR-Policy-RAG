"""Slack User Authentication, Authorization, and Rate Limiting.

Ensures that incoming Slack messages are securely verified against authorized
XYZ Technologies employee accounts and prevents brute-force / spamming via rate limits.
"""

from __future__ import annotations

import collections
import time
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.slack.config import SlackSettings, get_slack_settings
from app.utils.logging import get_logger

logger = get_logger("app.slack.auth")

UNAUTHORIZED_SLACK_MESSAGE = (
    "🔒 You are not authorized to access the XYZ Technologies HR Policy Assistant. "
    "Only authorized company employees may access internal HR policies. "
    "Please contact People Operations if you believe this is in error."
)


class SlackAuthResult(BaseModel):
    """Result of Slack user authorization evaluation."""

    is_authorized: bool = Field(description="Whether the user is permitted to query the HR bot")
    is_admin: bool = Field(default=False, description="Whether the user has administrator privileges")
    user_email: Optional[str] = Field(default=None, description="Resolved employee email from Slack API")
    rejection_message: Optional[str] = Field(default=None, description="Message to return if unauthorized")


class SlackAuthorizer:
    """Evaluates whether a Slack user ID is an authorized employee."""

    def __init__(self, settings: Optional[SlackSettings] = None) -> None:
        self.settings = settings or get_slack_settings()

    def authorize_user(
        self,
        slack_user_id: str,
        client: Optional[Any] = None,
    ) -> SlackAuthResult:
        """Validate Slack user against allowed users or corporate domain."""
        if not slack_user_id:
            return SlackAuthResult(
                is_authorized=False,
                rejection_message=UNAUTHORIZED_SLACK_MESSAGE,
            )

        clean_id = slack_user_id.strip()
        is_admin = clean_id in self.settings.admin_users

        # 1. Direct allowed users ID check
        if clean_id in self.settings.allowed_users:
            logger.info(f"Slack user '{clean_id}' authorized via SLACK_ALLOWED_USERS.")
            return SlackAuthResult(
                is_authorized=True,
                is_admin=is_admin,
            )

        # 2. Email domain verification via Slack WebClient (if available)
        resolved_email: Optional[str] = None
        if client is not None:
            try:
                user_info_resp = client.users_info(user=clean_id)
                if user_info_resp.get("ok"):
                    profile = user_info_resp.get("user", {}).get("profile", {})
                    resolved_email = profile.get("email", "").lower().strip()
            except Exception as exc:
                logger.warning(f"Could not fetch Slack user profile for '{clean_id}': {exc}")

        if resolved_email:
            for domain in self.settings.allowed_email_domains:
                if resolved_email.endswith(f"@{domain}"):
                    logger.info(f"Slack user '{clean_id}' ({resolved_email}) authorized via domain '@{domain}'.")
                    return SlackAuthResult(
                        is_authorized=True,
                        is_admin=is_admin,
                        user_email=resolved_email,
                    )

        # If allowed_users and allowed_email_domains are both unconfigured (development default open for tests)
        if not self.settings.allowed_users and not self.settings.allowed_email_domains:
            return SlackAuthResult(
                is_authorized=True,
                is_admin=is_admin,
                user_email=resolved_email,
            )

        logger.warning(f"Slack user '{clean_id}' rejected: Not in allowed users or authorized domain.")
        return SlackAuthResult(
            is_authorized=False,
            is_admin=False,
            user_email=resolved_email,
            rejection_message=UNAUTHORIZED_SLACK_MESSAGE,
        )


class SlackRateLimiter:
    """In-memory sliding window rate limiter per Slack user."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Mapping: user_id -> deque of request timestamps
        self._requests: Dict[str, collections.deque] = {}

    def check_rate_limit(self, user_id: str) -> Tuple[bool, Optional[int]]:
        """Check if user has exceeded rate limit.

        Returns (is_limited, retry_after_seconds).
        """
        now = time.time()
        if user_id not in self._requests:
            self._requests[user_id] = collections.deque()

        user_history = self._requests[user_id]

        # Purge timestamps outside the window
        while user_history and user_history[0] <= now - self.window_seconds:
            user_history.popleft()

        if len(user_history) >= self.max_requests:
            oldest = user_history[0]
            retry_after = max(1, int(oldest + self.window_seconds - now))
            logger.warning(f"Slack rate limit triggered for user '{user_id}'. Retry after {retry_after}s.")
            return True, retry_after

        # Record this request
        user_history.append(now)
        return False, None

    def reset(self, user_id: Optional[str] = None) -> None:
        """Reset rate limiter state for tests."""
        if user_id:
            self._requests.pop(user_id, None)
        else:
            self._requests.clear()
