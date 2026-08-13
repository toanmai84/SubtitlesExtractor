"""Hạ tầng dịch phụ đề bằng AI (adapter cho các nhà cung cấp như Gemini)."""

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

__all__ = ["GeminiSubtitleTranslator"]
