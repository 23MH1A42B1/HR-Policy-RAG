"""Runner script for XYZ Technologies HR Policy Slack Bot using Socket Mode.

Usage:
    uv run python scripts/run_slack.py
"""

from __future__ import annotations

import sys
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.slack.app import create_slack_app
from app.slack.config import get_slack_settings
from app.utils.logging import get_logger

logger = get_logger("scripts.run_slack")


def main() -> None:
    # Ensure UTF-8 stdout encoding on Windows
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    settings = get_slack_settings()

    if not settings.is_configured:
        print("\n" + "=" * 60)
        print("[!] Slack Bot Configuration Missing")
        print("=" * 60)
        print("Please configure the following environment variables in your .env file:")
        print("  * SLACK_BOT_TOKEN=xoxb-... (Bot User OAuth Token)")
        print("  * SLACK_APP_TOKEN=xapp-... (App-Level Token for Socket Mode)")
        print("  * SLACK_ALLOWED_USERS=U12345,U67890 (Optional allowed user IDs)")
        print("  * SLACK_ADMIN_USERS=U_ADMIN (Optional admin user IDs)")
        print("\nRefer to README.md for complete Slack App creation instructions.")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("⚡ XYZ HR Assistant Slack Bot is starting...")
    print("=" * 60)

    app = create_slack_app(settings=settings)
    handler = SocketModeHandler(app, settings.app_token)

    print("⚡ XYZ HR Assistant Slack Bot is running")
    print("Connected in Socket Mode. Listening for DMs, @mentions, and /ask-hr commands...\n")

    try:
        handler.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down XYZ HR Assistant Slack Bot...")
    except Exception as exc:
        logger.error(f"Slack Bot terminated with error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
