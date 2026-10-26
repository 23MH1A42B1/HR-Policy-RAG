"""Event and command handlers for XYZ Technologies Slack Bot.

Handles:
- Direct Messages (DMs)
- Channel @app_mention events
- /ask-hr Slash commands
Enforces authorization, rate-limiting, thread-level session isolation, and input security.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Optional

from app.api.rag_service import answer_question
from app.slack.auth import SlackAuthorizer, SlackRateLimiter
from app.slack.formatter import format_slack_response
from app.slack.memory import get_slack_session_id
from app.utils.logging import get_logger

logger = get_logger("app.slack.handlers")


class SlackEventHandler:
    """Orchestrates Slack interactions with RAG backend and security layers."""

    def __init__(
        self,
        authorizer: Optional[SlackAuthorizer] = None,
        rate_limiter: Optional[SlackRateLimiter] = None,
        rag_service_fn: Optional[Callable] = None,
    ) -> None:
        self.authorizer = authorizer or SlackAuthorizer()
        self.rate_limiter = rate_limiter or SlackRateLimiter()
        self.rag_service_fn = rag_service_fn or answer_question

    def handle_message(self, event: Dict[str, Any], say: Callable, client: Optional[Any] = None) -> None:
        """Handle Direct Message (DM) sent to the bot."""
        start_time = time.time()
        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        text = event.get("text", "").strip()
        thread_ts = event.get("thread_ts")

        # Ignore bot messages, message edits, or system subtypes
        if event.get("bot_id") or event.get("subtype"):
            return

        session_id = get_slack_session_id(user_id=user_id, channel_id=channel_id, thread_ts=thread_ts)
        logger.info(f"Slack DM event: user='{user_id}', session='{session_id}'")

        # 1. Authorization Check
        auth_result = self.authorizer.authorize_user(slack_user_id=user_id, client=client)
        if not auth_result.is_authorized:
            say(
                text=auth_result.rejection_message or "You are not authorized to use this bot.",
                thread_ts=thread_ts,
            )
            return

        # 2. Rate Limiting Check
        is_limited, retry_after = self.rate_limiter.check_rate_limit(user_id)
        if is_limited:
            say(
                text=f"⚠️ You have sent too many queries. Please wait {retry_after}s before asking again.",
                thread_ts=thread_ts,
            )
            return

        if not text:
            say(text="Please ask a question about company HR policies.", thread_ts=thread_ts)
            return

        # 3. Process query through unified RAG service
        try:
            rag_output = self.rag_service_fn(
                query=text,
                user_id=user_id,
                session_id=session_id,
                channel_id=channel_id,
                source="slack_dm",
            )
            payload = format_slack_response(
                answer=rag_output.get("answer", ""),
                sources=rag_output.get("sources", []),
                is_admin=auth_result.is_admin,
                metadata=rag_output.get("metadata", {}),
            )
            say(text=payload["text"], blocks=payload["blocks"], thread_ts=thread_ts)
            latency = time.time() - start_time
            logger.info(f"Slack DM completed: user='{user_id}', latency={latency:.2f}s, success=True")
        except Exception as exc:
            latency = time.time() - start_time
            logger.error(f"Slack DM error: user='{user_id}', latency={latency:.2f}s, err={exc}")
            say(
                text="Sorry, the HR policy service is temporarily unavailable. Please try again later.",
                thread_ts=thread_ts,
            )

    def handle_app_mention(self, event: Dict[str, Any], say: Callable, client: Optional[Any] = None) -> None:
        """Handle @mentions in channels."""
        start_time = time.time()
        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        raw_text = event.get("text", "")
        ts = event.get("ts")
        thread_ts = event.get("thread_ts") or ts  # Keep reply in thread to maintain clean channels

        # Remove bot mention: e.g. <@U12345> or <@U_BOT_ID>
        clean_text = re.sub(r"<@[^>]+>", "", raw_text).strip()
        session_id = get_slack_session_id(user_id=user_id, channel_id=channel_id, thread_ts=thread_ts)
        logger.info(f"Slack Mention event: user='{user_id}', session='{session_id}'")

        # 1. Authorization Check
        auth_result = self.authorizer.authorize_user(slack_user_id=user_id, client=client)
        if not auth_result.is_authorized:
            say(
                text=auth_result.rejection_message or "You are not authorized to use this bot.",
                thread_ts=thread_ts,
            )
            return

        # 2. Rate Limiting Check
        is_limited, retry_after = self.rate_limiter.check_rate_limit(user_id)
        if is_limited:
            say(
                text=f"⚠️ You have sent too many queries. Please wait {retry_after}s before asking again.",
                thread_ts=thread_ts,
            )
            return

        if not clean_text:
            say(
                text="Hello! How can I assist you with XYZ Technologies HR policies today?",
                thread_ts=thread_ts,
            )
            return

        # 3. Process query through unified RAG service
        try:
            rag_output = self.rag_service_fn(
                query=clean_text,
                user_id=user_id,
                session_id=session_id,
                channel_id=channel_id,
                source="slack_mention",
            )
            payload = format_slack_response(
                answer=rag_output.get("answer", ""),
                sources=rag_output.get("sources", []),
                is_admin=auth_result.is_admin,
                metadata=rag_output.get("metadata", {}),
            )
            say(text=payload["text"], blocks=payload["blocks"], thread_ts=thread_ts)
            latency = time.time() - start_time
            logger.info(f"Slack Mention completed: user='{user_id}', latency={latency:.2f}s, success=True")
        except Exception as exc:
            latency = time.time() - start_time
            logger.error(f"Slack Mention error: user='{user_id}', latency={latency:.2f}s, err={exc}")
            say(
                text="Sorry, the HR policy service is temporarily unavailable. Please try again later.",
                thread_ts=thread_ts,
            )

    def handle_slash_command(
        self,
        ack: Callable,
        command: Dict[str, Any],
        respond: Callable,
        client: Optional[Any] = None,
    ) -> None:
        """Handle /ask-hr <question> command with immediate ack."""
        # MUST acknowledge immediately to satisfy Slack's 3-second timeout SLA
        ack()
        start_time = time.time()

        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")
        query_text = command.get("text", "").strip()

        session_id = get_slack_session_id(user_id=user_id, channel_id=channel_id)
        logger.info(f"Slack Slash command: user='{user_id}', session='{session_id}'")

        # 1. Authorization Check
        auth_result = self.authorizer.authorize_user(slack_user_id=user_id, client=client)
        if not auth_result.is_authorized:
            respond(
                text=auth_result.rejection_message or "You are not authorized to use this bot.",
                response_type="ephemeral",
            )
            return

        # 2. Rate Limiting Check
        is_limited, retry_after = self.rate_limiter.check_rate_limit(user_id)
        if is_limited:
            respond(
                text=f"⚠️ You have sent too many queries. Please wait {retry_after}s before asking again.",
                response_type="ephemeral",
            )
            return

        if not query_text:
            respond(
                text="Please provide an HR policy question, e.g. `/ask-hr How many annual leave days do I get?`",
                response_type="ephemeral",
            )
            return

        # 3. Process query through unified RAG service
        try:
            rag_output = self.rag_service_fn(
                query=query_text,
                user_id=user_id,
                session_id=session_id,
                channel_id=channel_id,
                source="slack_slash_command",
            )
            payload = format_slack_response(
                answer=rag_output.get("answer", ""),
                sources=rag_output.get("sources", []),
                is_admin=auth_result.is_admin,
                metadata=rag_output.get("metadata", {}),
            )
            respond(
                text=payload["text"],
                blocks=payload["blocks"],
                response_type="in_channel",
            )
            latency = time.time() - start_time
            logger.info(f"Slack Slash command completed: user='{user_id}', latency={latency:.2f}s, success=True")
        except Exception as exc:
            latency = time.time() - start_time
            logger.error(f"Slack Slash command error: user='{user_id}', latency={latency:.2f}s, err={exc}")
            respond(
                text="Sorry, the HR policy service is temporarily unavailable. Please try again later.",
                response_type="ephemeral",
            )
