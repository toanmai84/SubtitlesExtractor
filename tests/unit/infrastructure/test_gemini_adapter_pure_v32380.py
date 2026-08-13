"""[v3.23.80] Test hàm thuần lõi adapter dịch Gemini (chống lỗi dữ liệu dịch).

Phủ ba hàm không cần gọi mạng:
- ``_merge_ai_items``: ghép kết quả AI vào câu gốc theo ``line_no`` (chống xáo trộn) và
  fallback giữ nguyên gốc khi AI trả rỗng (chống "nuốt chữ").
- ``_compute_patch_windows``: gộp cửa sổ ngữ cảnh để vá dòng thiếu.
- ``_sanitize_json_text``: bóc JSON khỏi văn bản "lắm lời" của model.
"""

from __future__ import annotations

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _compute_patch_windows,
    _sanitize_json_text,
)

_merge_ai_items = GeminiSubtitleTranslator._merge_ai_items


def _line(index: int, text: str) -> TranslationLine:
    return TranslationLine(index=index, start_ms=0, end_ms=1000, text=text)


class TestMergeAiItems:
    def test_maps_by_line_no_even_when_shuffled(self) -> None:
        batch = [_line(1, "one"), _line(2, "two"), _line(3, "three")]
        ai_items = [
            {"line_no": 3, "text": "ba"},
            {"line_no": 1, "text": "một"},
            {"line_no": 2, "text": "hai"},
        ]
        merged = _merge_ai_items(batch, ai_items, is_preprocess=False)
        assert [m.text for m in merged] == ["một", "hai", "ba"]

    def test_empty_text_falls_back_to_source(self) -> None:
        batch = [_line(1, "keep me")]
        merged = _merge_ai_items(
            batch, [{"line_no": 1, "text": "   "}], is_preprocess=False
        )
        assert merged[0].text == "keep me"

    def test_mixed_unmapped_item_assigned_in_order(self) -> None:
        # Item GIỮA thiếu line_no → phải gán cho câu gốc GIỮA (theo thứ tự còn lại),
        # không để câu giữa rơi về văn bản gốc.
        batch = [_line(1, "one"), _line(2, "two"), _line(3, "three")]
        ai_items = [
            {"line_no": 1, "text": "một"},
            {"text": "hai"},  # thiếu line_no
            {"line_no": 3, "text": "ba"},
        ]
        merged = _merge_ai_items(batch, ai_items, is_preprocess=False)
        assert [m.text for m in merged] == ["một", "hai", "ba"]

    def test_all_unmapped_positional_fallback(self) -> None:
        batch = [_line(1, "one"), _line(2, "two")]
        ai_items = [{"text": "một"}, {"text": "hai"}]  # model cũ không trả line_no
        merged = _merge_ai_items(batch, ai_items, is_preprocess=False)
        assert [m.text for m in merged] == ["một", "hai"]


class TestComputePatchWindows:
    def test_empty_inputs(self) -> None:
        assert _compute_patch_windows([], set(), 2) == []
        assert _compute_patch_windows([1, 2, 3], set(), 2) == []

    def test_single_missing_window_clamped(self) -> None:
        # batch line_no [10..14], thiếu 10 (vị trí 0), padding 2 → [0, 3).
        assert _compute_patch_windows([10, 11, 12, 13, 14], {10}, 2) == [(0, 3)]

    def test_adjacent_windows_merge(self) -> None:
        # thiếu vị trí 1 và 3, padding 1 → (0,3) và (2,5) chồng nhau → gộp (0,5).
        assert _compute_patch_windows([1, 2, 3, 4, 5], {2, 4}, 1) == [(0, 5)]

    def test_disjoint_windows_stay_separate(self) -> None:
        # thiếu vị trí 0 và 6, padding 1 → (0,2) và (5,7) tách rời.
        idx = [1, 2, 3, 4, 5, 6, 7]
        assert _compute_patch_windows(idx, {1, 7}, 1) == [(0, 2), (5, 7)]


class TestSanitizeJsonText:
    def test_strips_markdown_fence(self) -> None:
        assert _sanitize_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_leading_chatter(self) -> None:
        assert _sanitize_json_text('Dạ, kết quả đây: {"a": 1}') == '{"a": 1}'

    def test_strips_trailing_garbage(self) -> None:
        assert _sanitize_json_text('[{"x": 1}] xong nhé!') == '[{"x": 1}]'

    def test_plain_json_unchanged(self) -> None:
        assert _sanitize_json_text('{"k": "v"}') == '{"k": "v"}'


class TestValidateBatchBlankDetection:
    """[v3.23.80] Dòng có line_no nhưng text RỖNG phải được coi là thiếu → vá lại."""

    @staticmethod
    def _translator() -> GeminiSubtitleTranslator:
        return GeminiSubtitleTranslator(api_key="test-key")

    @staticmethod
    def _payload(texts: list[str], start_idx: int = 0) -> dict:
        return {
            "subtitles": [
                {"line_no": start_idx + i + 1, "text": t}
                for i, t in enumerate(texts)
            ]
        }

    def test_clean_batch_passes_without_exception(self) -> None:
        tr = self._translator()
        payload = self._payload([f"dòng {i}" for i in range(10)])
        # Không ném ngoại lệ = hợp lệ.
        tr._validate_batch(payload, expected_count=10, start_idx=0)

    def test_blank_text_line_triggers_partial_patch(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
            _BatchPartialError,
        )

        tr = self._translator()
        texts = [f"dòng {i}" for i in range(10)]
        texts[4] = "   "  # line_no 5 rỗng
        payload = self._payload(texts)
        try:
            tr._validate_batch(payload, expected_count=10, start_idx=0)
            raise AssertionError("Phải ném _BatchPartialError cho dòng rỗng")
        except _BatchPartialError as exc:
            assert 5 in exc.missing_line_nos

    def test_excess_items_triggers_validation_error(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
            _BatchValidationError,
        )

        tr = self._translator()
        payload = self._payload([f"d {i}" for i in range(15)])  # dư so với expected 10
        try:
            tr._validate_batch(payload, expected_count=10, start_idx=0)
            raise AssertionError("Phải ném _BatchValidationError khi trả dư dòng")
        except _BatchValidationError:
            pass
