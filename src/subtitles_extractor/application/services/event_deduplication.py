"""Khử trùng & gộp sự kiện phụ đề từ nhiều vùng ROI (logic thuần, không Qt).

Tách khỏi worker (tầng presentation) để dễ kiểm thử và tái sử dụng. Giải quyết
"Ghost Duplication": hai vùng ROI giao nhau bắt cùng một câu hai lần.
"""

from __future__ import annotations

import itertools

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent


def _time_overlap_ratio(first: SubtitleEvent, second: SubtitleEvent) -> float:
    """Tỷ lệ chồng lấn thời gian so với câu NGẮN hơn (∈ [0, 1])."""
    overlap = min(first.end_sec, second.end_sec) - max(first.start_sec, second.start_sec)
    if overlap <= 0:
        return 0.0
    shorter = min(first.end_sec - first.start_sec, second.end_sec - second.start_sec)
    return overlap / shorter if shorter > 0 else 0.0


def _text_similarity(first_text: str, second_text: str) -> float:
    """Độ giống nội dung ∈ [0, 1] (rapidfuzz nếu có, fallback difflib)."""
    a, b = first_text.strip(), second_text.strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz.fuzz import ratio

        return ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio()


def deduplicate_overlapping_events(
    events: list[SubtitleEvent],
    *,
    time_overlap_threshold: float = 0.80,
    text_similarity_threshold: float = 0.80,
) -> list[SubtitleEvent]:
    """Khử các câu trùng do ROI giao nhau, giữ câu có Confidence cao hơn.

    Args:
        events: Danh sách sự kiện (nên đã sắp theo thời gian).
        time_overlap_threshold: Ngưỡng chồng lấn thời gian (so câu ngắn hơn).
        text_similarity_threshold: Ngưỡng giống nội dung.

    Returns:
        Danh sách đã khử trùng, sắp theo thời gian.
    """
    if len(events) < 2:
        return list(events)
    ordered = sorted(events, key=lambda e: (e.start_sec, e.end_sec))
    kept: list[SubtitleEvent] = []
    for event in ordered:
        duplicate_index = None
        for index, existing in enumerate(kept):
            if existing.end_sec < event.start_sec:
                continue
            if (
                _time_overlap_ratio(existing, event) >= time_overlap_threshold
                and _text_similarity(existing.text, event.text) >= text_similarity_threshold
            ):
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(event)
        elif event.confidence.value > kept[duplicate_index].confidence.value:
            kept[duplicate_index] = event
    kept.sort(key=lambda e: (e.start_sec, e.end_sec))
    return kept


def merge_events_from_rois(
    events_per_roi: list[list[SubtitleEvent]],
) -> list[SubtitleEvent]:
    """Gom sự kiện từ nhiều ROI, khử trùng giao thoa, đánh lại số thứ tự."""
    all_events = list(itertools.chain.from_iterable(events_per_roi))
    deduped = deduplicate_overlapping_events(all_events)
    for new_index, event in enumerate(deduped, start=1):
        event.index = new_index
    return deduped
