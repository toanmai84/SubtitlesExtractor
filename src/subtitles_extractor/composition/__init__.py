"""Composition root — điểm duy nhất của app được phép tạo adapter cụ thể."""

from __future__ import annotations

from subtitles_extractor.composition.bootstrap import (
    bootstrap_for_cli,
    bootstrap_for_gui,
)
from subtitles_extractor.composition.container import ApplicationContainer

__all__ = [
    "ApplicationContainer",
    "bootstrap_for_cli",
    "bootstrap_for_gui",
]
