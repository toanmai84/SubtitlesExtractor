"""Test [v3.23.52] thinking_level cho Gemini 3.x vs thinking_budget cho 2.5."""

from __future__ import annotations

from unittest.mock import MagicMock

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


class _FakeThinkingConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _make_translator_with_fake_types():
    tr = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    fake_types = MagicMock()
    fake_types.ThinkingConfig = _FakeThinkingConfig
    fake_types.GenerateContentConfig = _FakeConfig
    tr._types_module = fake_types
    return tr


class TestFamilyDetection:
    def test_gemini_3_family(self) -> None:
        for m in ("gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3-flash"):
            assert GeminiSubtitleTranslator._is_gemini_3_family(m) is True

    def test_not_gemini_3(self) -> None:
        for m in ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-pro", ""):
            assert GeminiSubtitleTranslator._is_gemini_3_family(m) is False


class TestThinkingConfig:
    def test_gemini_3_uses_thinking_level(self) -> None:
        tr = _make_translator_with_fake_types()
        cfg = tr._build_config(
            0.2, {}, "sys", enable_thinking=True, thinking_budget=-1,
            model_name="gemini-3.1-flash-lite", thinking_level="low",
        )
        tc = cfg.kwargs["thinking_config"]
        assert tc.kwargs == {"thinking_level": "low"}
        # KHÔNG kèm thinking_budget (tránh lỗi 400).
        assert "thinking_budget" not in tc.kwargs

    def test_gemini_25_uses_budget(self) -> None:
        tr = _make_translator_with_fake_types()
        cfg = tr._build_config(
            0.2, {}, "sys", enable_thinking=True, thinking_budget=2048,
            model_name="gemini-2.5-flash",
        )
        tc = cfg.kwargs["thinking_config"]
        assert tc.kwargs == {"thinking_budget": 2048}
        assert "thinking_level" not in tc.kwargs

    def test_invalid_level_falls_back_to_low(self) -> None:
        tr = _make_translator_with_fake_types()
        cfg = tr._build_config(
            0.2, {}, "sys", enable_thinking=True,
            model_name="gemini-3.5-flash", thinking_level="ultra",
        )
        assert cfg.kwargs["thinking_config"].kwargs == {"thinking_level": "low"}

    def test_thinking_disabled_no_config(self) -> None:
        tr = _make_translator_with_fake_types()
        cfg = tr._build_config(
            0.2, {}, "sys", enable_thinking=False, model_name="gemini-3.1-flash-lite",
        )
        assert "thinking_config" not in cfg.kwargs

    def test_budget_zero_disables_thinking(self) -> None:
        tr = _make_translator_with_fake_types()
        cfg = tr._build_config(
            0.2, {}, "sys", enable_thinking=True, thinking_budget=0,
            model_name="gemini-3.1-flash-lite",
        )
        assert "thinking_config" not in cfg.kwargs
