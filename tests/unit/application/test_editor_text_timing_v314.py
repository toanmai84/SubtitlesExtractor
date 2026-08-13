"""Unit test cho Phần 3 — thuật toán văn bản & thời gian (v3.14.4).

Phủ: No-Domino insert, marker thẻ nguyên khối khi split, chống rách ký tự có dấu.
"""

from __future__ import annotations

import unicodedata

from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _event(idx: int, text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(
        index=idx, text=text, interval=TimeInterval(start, end), confidence=Confidence(1.0)
    )


def _service(events: list[SubtitleEvent]) -> SubtitleEditorService:
    svc = SubtitleEditorService()
    svc.load(events)
    return svc


class TestNoDominoInsert:
    def test_insert_does_not_shift_following_events(self) -> None:
        events = [
            _event(1, "A", 0.0, 1.0),
            _event(2, "B", 1.2, 2.2),   # khe hẹp sau A (chỉ 0.2s)
            _event(3, "C", 5.0, 6.0),
        ]
        svc = _service(events)
        state = svc.insert_after(0, "Chèn")
        # Câu B và C phải GIỮ NGUYÊN mốc thời gian (không bị đẩy).
        assert state.events[2].interval.start_sec == 1.2
        assert state.events[3].interval.start_sec == 5.0

    def test_inserted_event_min_duration(self) -> None:
        events = [_event(1, "A", 0.0, 1.0), _event(2, "B", 1.1, 2.0)]
        svc = _service(events)
        state = svc.insert_after(0, "X")
        inserted = state.events[1]
        assert (inserted.interval.end_sec - inserted.interval.start_sec) >= 0.149

    def test_insert_with_wide_gap_uses_desired(self) -> None:
        events = [_event(1, "A", 0.0, 1.0), _event(2, "B", 10.0, 11.0)]
        svc = _service(events)
        state = svc.insert_after(0, "X")
        inserted = state.events[1]
        # Khe rộng → dùng thời lượng mong muốn (~2s), vẫn không đụng B.
        assert inserted.interval.end_sec <= 10.0
        assert (inserted.interval.end_sec - inserted.interval.start_sec) > 1.0


class TestSplitTagProtection:
    def test_html_tag_not_corrupted(self) -> None:
        svc = _service([_event(1, "<b>Xin chào các bạn hiền</b>", 0.0, 4.0)])
        state = svc.split(0, 2.0)
        joined = state.events[0].text + state.events[1].text
        # Không còn ký tự marker điều khiển rò rỉ.
        assert "\x00" not in joined
        assert "\x0b" not in joined and "\x0c" not in joined
        assert "_INTERNAL_TAG_" not in joined
        # Thẻ <b> hoặc </b> còn nguyên vẹn (không bị cắt đôi).
        assert "<b>" in joined or "</b>" in joined

    def test_no_marker_leftover(self) -> None:
        svc = _service([_event(1, "{\\an8}Dòng phụ đề trên cao đây nhé", 0.0, 4.0)])
        state = svc.split(0, 2.0)
        for ev in state.events[:2]:
            assert "_INTERNAL_TAG_" not in ev.text


class TestDiacriticSplit:
    def test_no_orphan_combining_mark(self) -> None:
        # Chuỗi dạng NFD: mỗi 'ê' = 'e' + combining circumflex (khó tách an toàn).
        decomposed = unicodedata.normalize("NFD", "êêêêêêêêêê")
        svc = _service([_event(1, decomposed, 0.0, 4.0)])
        state = svc.split(0, 2.0)
        # Nửa sau KHÔNG được bắt đầu bằng combining mark mồ côi.
        text2 = state.events[1].text
        if text2:
            assert unicodedata.combining(text2[0]) == 0


class TestMultiRoiDedup:
    """[#13] Khử trùng câu trùng do 2 ROI giao nhau."""

    def test_duplicate_high_overlap_merged_keeps_higher_conf(self) -> None:
        from subtitles_extractor.application.services.event_deduplication import (
            merge_events_from_rois as _merge_events,
        )
        roi_a = [_event_conf(1, "Xin chào thế giới", 1.0, 3.0, 0.80)]
        roi_b = [_event_conf(1, "Xin chào thế giới", 1.0, 3.0, 0.95)]
        merged = _merge_events([roi_a, roi_b])
        assert len(merged) == 1
        assert merged[0].confidence.value == 0.95  # giữ câu tin cậy hơn

    def test_distinct_events_not_merged(self) -> None:
        from subtitles_extractor.application.services.event_deduplication import (
            merge_events_from_rois as _merge_events,
        )
        roi_a = [_event_conf(1, "Câu một", 1.0, 3.0, 0.9)]
        roi_b = [_event_conf(1, "Câu hai khác hẳn", 5.0, 7.0, 0.9)]
        assert len(_merge_events([roi_a, roi_b])) == 2

    def test_partial_overlap_below_threshold_kept(self) -> None:
        from subtitles_extractor.application.services.event_deduplication import (
            merge_events_from_rois as _merge_events,
        )
        # Chồng thời gian ít (<80% câu ngắn) → giữ cả hai dù text giống.
        roi_a = [_event_conf(1, "Giống nhau", 1.0, 3.0, 0.9)]
        roi_b = [_event_conf(1, "Giống nhau", 2.8, 4.8, 0.9)]
        assert len(_merge_events([roi_a, roi_b])) == 2


def _event_conf(idx, text, start, end, conf):
    return SubtitleEvent(
        index=idx, text=text, interval=TimeInterval(start, end), confidence=Confidence(conf)
    )
