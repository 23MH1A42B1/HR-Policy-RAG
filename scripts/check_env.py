"""Environment validation script for HR Policy RAG Assistant.

Checks the local Python environment and verifies configuration load.
"""

from __future__ import annotations

import sys
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("scripts.check_env")


def main() -> None:
    logger.info("Validating HR Policy RAG Assistant environment...")
    try:
        settings = get_settings()
        logger.info(
            "Configuration successfully loaded",
            extra={
                "app_name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
            },
        )
        print(f"OK: {settings.app_name} v{settings.app_version} [{settings.environment}]")
    except Exception as exc:
        logger.error(f"Configuration check failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
