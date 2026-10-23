"""[v3.23.513] Model đánh số lại line_no từ 1 → phải ghép theo VỊ TRÍ, không bỏ qua.

Các model "lite" có thói quen đánh số lại ``line_no`` từ 1 cho mỗi lô. Khi đó map theo line_no
trượt sạch, mà nhánh dự phòng theo vị trí lại không chạy (vì item vẫn "có" line_no hợp lệ) → MỌI
dòng của lô giữ nguyên văn bản gốc, tức còn nguyên chữ Hán, kèm log "AI trả rỗng cho dòng N".

Nghịch lý của lỗi cũ: line_no RÁC (âm hoặc chữ) lại ghép đúng, chỉ số HỢP LỆ-nhưng-lệch mới hỏng.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _line(index: int, text: str) -> TranslationLine:
    return TranslationLine(index=index, start_ms=0, end_ms=100, text=text)


@pytest.fixture
def translator() -> GeminiSubtitleTranslator:
    return GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)


BATCH = [_line(100, "你好"), _line(101, "再见")]


class TestRenumberedLineNumbers:
    def test_renumbered_from_one_merges_by_position(self, translator) -> None:
        merged = translator._merge_ai_items(
            BATCH,
            [{"line_no": 1, "text": "Xin chào"}, {"line_no": 2, "text": "Tạm biệt"}],
            False,
        )
        assert [line.text for line in merged] == ["Xin chào", "Tạm biệt"]

    def test_offset_numbering_merges_by_position(self, translator) -> None:
        merged = translator._merge_ai_items(
            BATCH,
            [{"line_no": 5000, "text": "Xin chào"}, {"line_no": 5001, "text": "Tạm biệt"}],
            False,
        )
        assert [line.text for line in merged] == ["Xin chào", "Tạm biệt"]

    def test_garbage_line_no_still_works(self, translator) -> None:
        merged = translator._merge_ai_items(
            BATCH,
            [{"line_no": -5, "text": "Xin chào"}, {"line_no": "x", "text": "Tạm biệt"}],
            False,
        )
        assert [line.text for line in merged] == ["Xin chào", "Tạm biệt"]


class TestNoRegression:
    def test_correct_line_no_still_matched_by_id(self, translator) -> None:
        merged = translator._merge_ai_items(
            BATCH,
            [{"line_no": 100, "text": "Xin chào"}, {"line_no": 101, "text": "Tạm biệt"}],
            False,
        )
        assert [line.text for line in merged] == ["Xin chào", "Tạm biệt"]

    def test_out_of_order_matched_by_id_not_position(self, translator) -> None:
        """Model xáo trộn thứ tự nhưng line_no ĐÚNG → vẫn ghép vào đúng câu gốc."""
        merged = translator._merge_ai_items(
            BATCH,
            [{"line_no": 101, "text": "Tạm biệt"}, {"line_no": 100, "text": "Xin chào"}],
            False,
        )
        assert [line.text for line in merged] == ["Xin chào", "Tạm biệt"]

    def test_partial_overlap_not_treated_as_renumbered(self, translator) -> None:
        """Có giao nhau line_no → KHÔNG chuyển sang ghép vị trí (tránh xê dịch nhầm)."""
        merged = translator._merge_ai_items(
            BATCH,
            [{"line_no": 100, "text": "Xin chào"}, {"line_no": 999, "text": "Lạc đề"}],
            False,
        )
        assert merged[0].text == "Xin chào"

    def test_count_mismatch_does_not_shift(self, translator) -> None:
        """Số lượng KHÔNG khớp → không ghép vị trí (không biết thiếu ở đâu)."""
        merged = translator._merge_ai_items(BATCH, [{"line_no": 1, "text": "Xin chào"}], False)
        assert merged[0].text == "你好"  # giữ nguyên gốc thay vì đoán bừa
