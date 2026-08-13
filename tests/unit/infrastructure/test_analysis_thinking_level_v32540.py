"""[v3.23.140] Test mức Thinking cho PHÂN TÍCH ngữ cảnh: setting + adapter chuẩn hoá.

Trước đây thinking_level của phân tích bị HARDCODE "low" và không cho người dùng chọn. Nay
thành setting ``analysis_thinking_level`` (mặc định "medium"), truyền vào adapter và dùng
cho cả phân tích chính lẫn Visual Cues.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.settings.application_settings import (
    TranslationSettings,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def test_setting_default_is_medium() -> None:
    assert TranslationSettings().analysis_thinking_level == "medium"


def test_setting_rejects_invalid_value() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TranslationSettings(analysis_thinking_level="ultra")


def test_adapter_stores_and_normalizes_level() -> None:
    a_high = GeminiSubtitleTranslator(api_key="K", analysis_thinking_level="high")
    assert a_high._analysis_thinking_level == "high"

    a_bad = GeminiSubtitleTranslator(api_key="K", analysis_thinking_level="nonsense")
    assert a_bad._analysis_thinking_level == "medium"  # giá trị lạ -> mặc định an toàn

    a_default = GeminiSubtitleTranslator(api_key="K")
    assert a_default._analysis_thinking_level == "medium"


def test_adapter_case_insensitive() -> None:
    a = GeminiSubtitleTranslator(api_key="K", analysis_thinking_level="HIGH")
    assert a._analysis_thinking_level == "high"
