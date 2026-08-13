"""Test [v3.23.18] _validate_batch chấp nhận line_no lệch (model đánh số lại từ 1)."""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _BatchPartialError,
    _BatchValidationError,
)


def _adapter() -> GeminiSubtitleTranslator:
    return GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)


class TestRenumberByPosition:
    def test_static_renumber(self) -> None:
        items = [{"line_no": 1, "text": "a"}, {"line_no": 2, "text": "b"}]
        GeminiSubtitleTranslator._renumber_items_by_position(items, 400)
        assert [it["line_no"] for it in items] == [401, 402]


class TestValidateFlexible:
    def test_offset_line_no_accepted(self) -> None:
        # Model trả line_no 1-100 cho batch bắt đầu ở 401 → chấp nhận + renumber.
        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 101)]}
        _adapter()._validate_batch(payload, 100, 400)
        assert payload["subtitles"][0]["line_no"] == 401
        assert payload["subtitles"][-1]["line_no"] == 500

    def test_exact_match_accepted(self) -> None:
        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 101)]}
        _adapter()._validate_batch(payload, 100, 0)  # không raise

    def test_missing_tail_patches(self) -> None:
        # Thiếu 1-2 dòng cuối → renumber + vá đuôi.
        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 100)]}
        try:
            _adapter()._validate_batch(payload, 100, 0)
            raise AssertionError("không raise")
        except _BatchPartialError as exc:
            assert exc.missing_line_nos == [100]

    def test_overflow_raises(self) -> None:
        # Dư nhiều → ValidationError (halving).
        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 96)]}
        try:
            _adapter()._validate_batch(payload, 25, 0)
            raise AssertionError("không raise")
        except _BatchValidationError:
            pass

    def test_severe_shortage_raises(self) -> None:
        # Nhận 20/100 (thiếu nhiều thật) → ValidationError.
        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 21)]}
        try:
            _adapter()._validate_batch(payload, 100, 0)
            raise AssertionError("không raise")
        except _BatchValidationError:
            pass
