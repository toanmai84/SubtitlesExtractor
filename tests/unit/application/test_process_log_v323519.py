"""[v3.23.519] File chuẩn đoán phải ghi DIỄN BIẾN quá trình dịch, không chỉ kết quả cuối.

Trước đây file xuất chỉ có kết quả cuối của 4 giai đoạn. Lô nào bị model trả sai, thử lại mấy lần,
lô nào bị phát hiện lệch dòng, lô nào đành giữ nguyên văn bản gốc — chỉ nằm trong nhật ký ứng dụng
và mất khi đóng chương trình, nên mỗi lần chuẩn đoán lại phải xin thêm app_log.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.application.services.translation_diagnostics import (
    build_diagnostics_bundle,
)
from subtitles_extractor.application.services.translation_process_log import (
    TranslationProcessLog,
    get_process_log,
    record_event,
)


@pytest.fixture(autouse=True)
def _clean_log():
    get_process_log().clear()
    yield
    get_process_log().clear()


class TestRecorder:
    def test_records_and_counts_by_type(self) -> None:
        record_event("batch_retry", attempt=1)
        record_event("batch_retry", attempt=2)
        record_event("batch_gave_up", depth=5)
        snapshot = get_process_log().snapshot()
        assert snapshot["counts"] == {"batch_retry": 2, "batch_gave_up": 1}

    def test_keeps_fields(self) -> None:
        record_event("batch_misaligned", lines=[257, 258], shift=3)
        event = get_process_log().snapshot()["events"][0]
        assert event["lines"] == [257, 258]
        assert event["shift"] == 3

    def test_clear_resets(self) -> None:
        record_event("batch_retry")
        get_process_log().clear()
        assert get_process_log().snapshot()["events"] == []

    def test_cap_prevents_unbounded_growth(self) -> None:
        log = TranslationProcessLog(max_events=5)
        for index in range(20):
            log.record("batch_retry", attempt=index)
        snapshot = log.snapshot()
        assert len(snapshot["events"]) == 5
        assert snapshot["dropped"] == 15

    def test_never_raises_on_bad_input(self) -> None:
        """Ghi nhật ký là việc phụ — không được phép làm hỏng khâu dịch."""
        record_event("odd", value=object())  # không JSON-serializable
        assert get_process_log().snapshot()["counts"]["odd"] == 1


class TestBundleIntegration:
    def test_bundle_contains_process_log(self) -> None:
        record_event("batch_retry", attempt=1, reason="Model trả DƯ 26 dòng")
        record_event("batch_misaligned", lines=[257], shift=3)
        bundle = build_diagnostics_bundle(
            app_version="3.23.519",
            exported_at="2026-09-10T00:00:00Z",
            video_path="/x/phim.mp4",
            source_events=[],
            translated_events=[],
            session=None,
        )
        assert bundle["process_log"]["counts"] == {"batch_retry": 1, "batch_misaligned": 1}

    def test_bundle_without_events_still_valid(self) -> None:
        bundle = build_diagnostics_bundle(
            app_version="3.23.519",
            exported_at="2026-09-10T00:00:00Z",
            video_path="/x/phim.mp4",
            source_events=[],
            translated_events=[],
            session=None,
        )
        assert bundle["process_log"] == {"events": [], "counts": {}, "dropped": 0}


class TestAdapterHooks:
    """Adapter phải ghi ở đủ các điểm quan trọng của vòng thử lại."""

    def test_all_hook_points_present(self) -> None:
        import inspect

        from subtitles_extractor.infrastructure.translation import (
            gemini_translation_adapter as module,
        )

        source = inspect.getsource(module)
        for event in ("batch_retry", "batch_misaligned", "batch_gave_up"):
            assert f'"{event}"' in source, event


class TestBatchContentSample:
    """[v3.23.520] Lô có vấn đề phải ghi cả câu GỐC lẫn bản model TRẢ VỀ (kể cả bản sai).

    Biết lô nào lỗi mà không có nội dung thì không suy ra được model hiểu sai ở đâu. Chỉ trích cho
    lô CÓ VẤN ĐỀ và giới hạn kích thước — ghi trọn 5540 dòng × 4 giai đoạn là vô ích.
    """

    @staticmethod
    def _line(index: int, text: str):
        from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

        return TranslationLine(index=index, start_ms=0, end_ms=100, text=text)

    def _sample(self, batch, payload):
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _batch_content_sample,
        )

        return _batch_content_sample(batch, payload)

    def test_pairs_source_with_returned_text(self) -> None:
        batch = [self._line(1, "王梅人好"), self._line(2, "支付宝到账")]
        payload = {"subtitles": [
            {"line_no": 1, "text": "Alipay tiền vào tài khoản."},
            {"line_no": 2, "text": "Vương Mai tốt bụng."},
        ]}
        rows = self._sample(batch, payload)
        assert rows[0]["source"] == "王梅人好"
        assert rows[0]["returned"] == "Alipay tiền vào tài khoản."

    def test_notes_overflow_from_model(self) -> None:
        """Ca thật: lô 1 dòng nhưng model trả 26 mục — phải ghi lại ghi chú đó."""
        batch = [self._line(100, "支付宝到账一千元")]
        payload = {"subtitles": [{"line_no": n, "text": f"ngữ cảnh {n}"} for n in range(75, 101)]}
        rows = self._sample(batch, payload)
        assert any("26 mục" in str(row.get("note", "")) for row in rows)

    def test_limits_lines_and_length(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            MAX_SAMPLE_CHARS,
            MAX_SAMPLE_LINES,
        )

        batch = [self._line(i, "x" * 500) for i in range(1, 60)]
        payload = {"subtitles": [{"line_no": i, "text": "y" * 500} for i in range(1, 60)]}
        rows = self._sample(batch, payload)
        assert len(rows) <= MAX_SAMPLE_LINES + 1  # +1 cho dòng ghi chú
        assert len(rows[0]["source"]) <= MAX_SAMPLE_CHARS

    def test_missing_or_malformed_payload_is_safe(self) -> None:
        batch = [self._line(1, "王梅人好")]
        assert self._sample(batch, None)[0]["returned"] == ""
        assert self._sample(batch, {"subtitles": ["rác"]})[0]["returned"] == ""

    def test_renumbered_payload_matched_by_position(self) -> None:
        batch = [self._line(100, "王梅人好"), self._line(101, "支付宝到账")]
        payload = {"subtitles": [{"text": "Vương Mai tốt bụng."}, {"text": "Alipay nhận tiền."}]}
        rows = self._sample(batch, payload)
        assert rows[0]["returned"] == "Vương Mai tốt bụng."

    def test_adapter_records_content_with_errors(self) -> None:
        import inspect

        from subtitles_extractor.infrastructure.translation import (
            gemini_translation_adapter as module,
        )

        source = inspect.getsource(module)
        assert '"batch_invalid"' in source
        assert "content=_batch_content_sample(batch, data)" in source


class TestFocusedSampling:
    """[v3.23.521] Mẫu nội dung phải bám DÒNG BỊ CỜ, không phải 12 dòng đầu lô.

    Dữ liệu thật: lô 541-600 bị cờ ở dòng 575, 587, 591 — mẫu cũ chỉ ghi dòng 541-552 nên chỗ cần
    xem không có trong tệp, không đối chiếu được gì.
    """

    @staticmethod
    def _batch_and_payload():
        from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

        batch = [
            TranslationLine(index=i, start_ms=0, end_ms=100, text=f"gốc{i}")
            for i in range(541, 601)
        ]
        payload = {"subtitles": [{"line_no": i, "text": f"dịch{i}"} for i in range(541, 601)]}
        return batch, payload

    def _sample(self, focus=None):
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _batch_content_sample,
        )

        batch, payload = self._batch_and_payload()
        return _batch_content_sample(batch, payload, focus)

    def test_sample_centers_on_flagged_lines(self) -> None:
        rows = [row["line_no"] for row in self._sample([575, 587, 591])]
        assert 575 in rows and 587 in rows and 591 in rows

    def test_sample_includes_neighbours_for_context(self) -> None:
        rows = [row["line_no"] for row in self._sample([575])]
        assert {574, 575, 576}.issubset(set(rows))

    def test_falls_back_to_batch_start_without_focus(self) -> None:
        rows = [row["line_no"] for row in self._sample()]
        assert rows[0] == 541

    def test_focus_outside_batch_falls_back(self) -> None:
        rows = [row["line_no"] for row in self._sample([9999])]
        assert rows[0] == 541


class TestAlignmentRetryCap:
    """[v3.23.521] Không ép dịch lại vô hạn vì lệch dòng.

    Đo thật: một lô bị bắt 12 lần liên tiếp — mỗi lần là một lượt gọi API mà kết quả vẫn lệch,
    tốn quota và còn nguy cơ bỏ cuộc giữ nguyên chữ Hán (tệ hơn bản dịch lệch).
    """

    def test_threshold_is_small(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            MAX_ALIGNMENT_RETRIES,
        )

        assert 1 <= MAX_ALIGNMENT_RETRIES <= 3

    def test_validator_counts_and_stops(self) -> None:
        import inspect

        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        source = inspect.getsource(GeminiSubtitleTranslator._translate_single_batch)
        assert "alignment_retries = 0" in source
        assert "nonlocal alignment_retries" in source
        assert "alignment_retries < MAX_ALIGNMENT_RETRIES" in source
