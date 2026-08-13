"""[v3.23.125] Test: few-shot examples xuất hiện ở các giai đoạn DỊCH, không ở preprocess.

Dùng mẫu gọi UNBOUND với stub nhẹ để tránh khởi tạo client thật (segfault headless).
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

_MARKER = "VÍ DỤ MINH HOẠ"


def _ctx() -> TranslationContext:
    return TranslationContext(
        target_lang="Tiếng Việt", source_lang="Tiếng Trung", overview="phim thử"
    )


def _stub() -> SimpleNamespace:
    # _system_instruction chỉ cần self._canonical_names_directive (staticmethod).
    return SimpleNamespace(
        _canonical_names_directive=GeminiSubtitleTranslator._canonical_names_directive
    )


def _instruction(kind: TranslationStageKind) -> str:
    stage = TranslationStageConfig(kind=kind, model_name="gemini-3.5-flash")
    return GeminiSubtitleTranslator._system_instruction(_stub(), stage, _ctx(), False)


def test_literal_stage_has_examples() -> None:
    assert _MARKER in _instruction(TranslationStageKind.LITERAL)


def test_style_stage_has_examples() -> None:
    assert _MARKER in _instruction(TranslationStageKind.STYLE)


def test_localize_stage_has_examples() -> None:
    assert _MARKER in _instruction(TranslationStageKind.LOCALIZE)


def test_preprocess_stage_has_no_examples() -> None:
    # Preprocess chỉ sửa lỗi gốc, không dịch -> không cần ví dụ phong cách câu đích.
    assert _MARKER not in _instruction(TranslationStageKind.PREPROCESS)


def test_examples_cover_key_principles() -> None:
    text = _instruction(TranslationStageKind.STYLE)
    # Có minh hoạ: gọn, không chú thích, xưng hô nhất quán.
    assert "chú thích" in text
    assert "xưng hô" in text.lower() or "Xưng hô" in text
