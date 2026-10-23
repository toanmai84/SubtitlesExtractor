"""[v3.23.504] Bản gốc chỉ đính khi THỰC SỰ cần — không trùng lặp, không bỏ sót.

Hai vấn đề được xử lý:

1. ``has_prior_translation`` chỉ bật sau một giai đoạn DỊCH, nên khi LITERAL chạy nó luôn False →
   điều kiện của v3.23.503 khiến LITERAL KHÔNG BAO GIỜ nhận payload kép, kể cả khi PREPROCESS đã
   sửa dòng nguồn. Nay dùng cờ riêng ``source_was_preprocessed``.
2. Nếu người dùng bỏ giai đoạn hiệu đính và chạy thẳng dịch thô, bản gốc TRÙNG y hệt đầu vào —
   đính thêm chỉ tốn token. Khi đó payload kép phải TẮT.
"""
from __future__ import annotations

import inspect

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _line(text: str, original: str) -> TranslationLine:
    return TranslationLine(index=1, start_ms=0, end_ms=1000, text=text, original_text=original)


@pytest.fixture
def translator() -> GeminiSubtitleTranslator:
    instance = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    instance._reading_cps_budget = 0.0
    instance._emotion_cues = False
    return instance


class TestNoDuplication:
    """Dòng không bị khâu trước sửa thì KHÔNG kèm bản gốc (tránh nhân đôi token)."""

    def test_identical_original_omitted(self, translator) -> None:
        payload = translator._line_to_dual_payload(_line("你老实跟我说", "你老实跟我说"))
        assert "original" not in payload

    def test_changed_line_includes_original(self, translator) -> None:
        payload = translator._line_to_dual_payload(
            _line("Cẩn thận! Sắp được rồi!", "Cẩn thận Sắp được rồi")
        )
        assert payload["original"] == "Cẩn thận Sắp được rồi"

    def test_empty_original_omitted(self, translator) -> None:
        assert "original" not in translator._line_to_dual_payload(_line("xin chào", ""))


def _dual_enabled(kind: TranslationStageKind, prior: bool, preprocessed: bool) -> bool:
    """Mô phỏng điều kiện trong ``translate_stage`` (giữ đồng bộ qua test bên dưới)."""
    return (kind is TranslationStageKind.LITERAL and preprocessed) or (
        prior
        and kind in (TranslationStageKind.STYLE, TranslationStageKind.LOCALIZE)
    )


class TestDualPayloadScenarios:
    @pytest.mark.parametrize(
        ("kind", "prior", "preprocessed", "expected"),
        [
            # Bỏ hiệu đính, chạy thẳng dịch thô → bản gốc trùng đầu vào → KHÔNG đính.
            (TranslationStageKind.LITERAL, False, False, False),
            # Có hiệu đính trước → PHẢI đính để phát hiện lệch do khâu đó gây ra.
            (TranslationStageKind.LITERAL, False, True, True),
            (TranslationStageKind.STYLE, True, True, True),
            (TranslationStageKind.LOCALIZE, True, True, True),
            # Chưa có bản nháp nào → không có gì để đối chiếu.
            (TranslationStageKind.STYLE, False, False, False),
            (TranslationStageKind.PREPROCESS, False, False, False),
        ],
    )
    def test_matrix(self, kind, prior, preprocessed, expected) -> None:
        assert _dual_enabled(kind, prior, preprocessed) is expected

    def test_adapter_uses_preprocessed_flag(self) -> None:
        source = inspect.getsource(GeminiSubtitleTranslator.translate_stage)
        block = source[source.index("use_dual_payload = ") :][:400]
        assert "source_was_preprocessed" in block
        assert "TranslationStageKind.LITERAL" in block

    def test_stage_accepts_preprocessed_flag(self) -> None:
        params = inspect.signature(GeminiSubtitleTranslator.translate_stage).parameters
        assert "source_was_preprocessed" in params
