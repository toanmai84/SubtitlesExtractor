"""Logging utilities — Loguru-based (v2.29).

Plug & Play logging với Loguru:
    * ``setup_loguru()`` — API mới (preferred, dùng level string).
    * ``setup_logging()`` — backward-compat wrapper (level integer).
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.logging.loguru_config import (
    InterceptHandler,
    setup_logging,
    setup_loguru,
)

__all__ = ["InterceptHandler", "setup_logging", "setup_loguru"]
