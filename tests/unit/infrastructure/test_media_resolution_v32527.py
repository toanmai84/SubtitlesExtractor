"""[v3.23.127] Test: media_resolution (Gemini 3.x) đưa đúng vào GenerateContentConfig.

Đặt độ phân giải video THẤP khi dịch (tiết kiệm token) / VỪA khi phân tích cues (nhìn rõ
mặt). Test xác nhận tham số được map đúng enum và chỉ thêm khi được yêu cầu.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


class _FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeMediaResolution:
    MEDIA_RESOLUTION_LOW = "LOW"
    MEDIA_RESOLUTION_MEDIUM = "MEDIUM"
    MEDIA_RESOLUTION_HIGH = "HIGH"


def _translator() -> GeminiSubtitleTranslator:
    tr = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    fake_types = MagicMock()
    fake_types.GenerateContentConfig = _FakeConfig
    fake_types.MediaResolution = _FakeMediaResolution
    tr._types_module = fake_types
    return tr


def test_resolve_levels() -> None:
    tr = _translator()
    assert tr._resolve_media_resolution("low") == "LOW"
    assert tr._resolve_media_resolution("medium") == "MEDIUM"
    assert tr._resolve_media_resolution("high") == "HIGH"
    assert tr._resolve_media_resolution("bogus") is None
    assert tr._resolve_media_resolution("") is None


def test_build_config_sets_media_resolution_when_requested() -> None:
    tr = _translator()
    cfg = tr._build_config(
        0.2, {}, "sys", model_name="gemini-3.5-flash", media_resolution="low"
    )
    assert cfg.kwargs["media_resolution"] == "LOW"


def test_build_config_omits_media_resolution_by_default() -> None:
    tr = _translator()
    cfg = tr._build_config(0.2, {}, "sys", model_name="gemini-3.5-flash")
    assert "media_resolution" not in cfg.kwargs


def test_build_config_omits_when_sdk_lacks_enum() -> None:
    tr = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    fake_types = MagicMock()
    fake_types.GenerateContentConfig = _FakeConfig
    fake_types.MediaResolution = None  # SDK cũ
    tr._types_module = fake_types
    cfg = tr._build_config(
        0.2, {}, "sys", model_name="gemini-3.5-flash", media_resolution="low"
    )
    assert "media_resolution" not in cfg.kwargs


def test_resolve_with_real_sdk_enum() -> None:
    # Với SDK thật, low phải map ra enum LOW của google-genai.
    from google.genai import types

    tr = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    tr._types_module = types
    expected = types.MediaResolution.MEDIA_RESOLUTION_LOW
    assert tr._resolve_media_resolution("low") == expected
