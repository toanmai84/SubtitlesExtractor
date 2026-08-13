"""Test [v3.23.9] tăng độ bền dịch: retryDelay server, ghép theo ID, validate tập."""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _BatchValidationError,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


class TestServerRetryDelay:
    def test_parses_retry_delay_seconds(self) -> None:
        err = Exception("429 ... 'retryDelay': '36s'")
        delay = GeminiSubtitleTranslator._server_retry_delay(err)
        assert delay is not None and 37.0 <= delay <= 38.0

    def test_parses_please_retry_in(self) -> None:
        err = Exception("Please retry in 28.96s")
        delay = GeminiSubtitleTranslator._server_retry_delay(err)
        assert delay is not None and 30.0 <= delay <= 31.0

    def test_no_delay_when_absent(self) -> None:
        assert GeminiSubtitleTranslator._server_retry_delay(Exception("503 UNAVAILABLE")) is None


class TestMergeByLineNo:
    def test_merges_by_id_when_shuffled(self) -> None:
        batch = [
            TranslationLine(index=1, start_ms=0, end_ms=1, text="Hello"),
            TranslationLine(index=2, start_ms=1, end_ms=2, text="World"),
        ]
        # AI trả xáo trộn thứ tự.
        ai = [{"line_no": 2, "text": "Thế giới"}, {"line_no": 1, "text": "Xin chào"}]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, ai, is_preprocess=True)
        assert merged[0].index == 1 and merged[0].text == "Xin chào"
        assert merged[1].index == 2 and merged[1].text == "Thế giới"

    def test_fallback_to_position_without_line_no(self) -> None:
        batch = [
            TranslationLine(index=1, start_ms=0, end_ms=1, text="A"),
            TranslationLine(index=2, start_ms=1, end_ms=2, text="B"),
        ]
        ai = [{"text": "X"}, {"text": "Y"}]  # không có line_no
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, ai, is_preprocess=True)
        assert merged[0].text == "X" and merged[1].text == "Y"

    def test_empty_keeps_original(self) -> None:
        batch = [TranslationLine(index=1, start_ms=0, end_ms=1, text="Keep")]
        ai = [{"line_no": 1, "text": "   "}]  # rỗng
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, ai, is_preprocess=True)
        assert merged[0].text == "Keep"


class TestPartialBatchPatch:
    """[v3.23.11] Thiếu ít dòng → _BatchPartialError (vá riêng); thiếu nhiều → ValidationError."""

    def _adapter(self) -> GeminiSubtitleTranslator:
        return GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)

    def test_few_missing_raises_partial(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _BatchPartialError,
        )

        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 96)]}
        try:
            self._adapter()._validate_batch(payload, 100, 0)
            raise AssertionError("không raise")
        except _BatchPartialError as exc:
            assert exc.missing_line_nos == [96, 97, 98, 99, 100]

    def test_many_missing_raises_validation(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _BatchValidationError,
            _BatchPartialError,
        )

        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 51)]}
        try:
            self._adapter()._validate_batch(payload, 100, 0)
            raise AssertionError("không raise")
        except _BatchPartialError:
            raise AssertionError("phải là ValidationError, không phải Partial")
        except _BatchValidationError:
            pass

    def test_complete_no_raise(self) -> None:
        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 101)]}
        self._adapter()._validate_batch(payload, 100, 0)  # không raise

    def test_small_batch_missing_few_patches_tail(self) -> None:
        # [v3.23.18] Batch nhỏ, model trả đúng thứ tự nhưng thiếu 1-2 dòng cuối →
        # renumber theo vị trí + vá đuôi (PartialError) thay vì halving.
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _BatchPartialError,
        )

        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 4)]}
        try:
            self._adapter()._validate_batch(payload, 5, 0)
            raise AssertionError("không raise")
        except _BatchPartialError as exc:
            assert exc.missing_line_nos == [4, 5]

    def test_large_overflow_raises_validation(self) -> None:
        # Model trả DƯ nhiều (95 cho batch 25) → ValidationError (halving).
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _BatchValidationError,
        )

        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 96)]}
        try:
            self._adapter()._validate_batch(payload, 25, 0)
            raise AssertionError("không raise")
        except _BatchValidationError:
            pass
