"""Test [v3.23.36] lọc model free tier (loại pro/đời cũ/non-text)."""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


class TestFreeTierModelFilter:
    def test_keeps_free_flash_models(self) -> None:
        f = GeminiSubtitleTranslator._is_free_tier_text_model
        for m in ("gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3-flash",
                  "gemini-2.5-flash", "gemini-2.5-flash-lite"):
            assert f(m) is True, m

    def test_rejects_pro(self) -> None:
        f = GeminiSubtitleTranslator._is_free_tier_text_model
        for m in ("gemini-2.5-pro", "gemini-3.1-pro"):
            assert f(m) is False, m

    def test_rejects_old_gen(self) -> None:
        f = GeminiSubtitleTranslator._is_free_tier_text_model
        for m in ("gemini-2.0-flash", "gemini-2-flash", "gemini-1.5-flash"):
            assert f(m) is False, m

    def test_rejects_non_text(self) -> None:
        f = GeminiSubtitleTranslator._is_free_tier_text_model
        for m in ("text-embedding-004", "imagen-3", "veo-2",
                  "gemini-2.5-flash-tts", "gemma-2"):
            assert f(m) is False, m
