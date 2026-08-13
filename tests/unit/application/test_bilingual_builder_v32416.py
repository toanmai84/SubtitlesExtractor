"""[v3.23.116] Test hàm thuần ghép phụ đề song ngữ (gốc + dịch)."""

from __future__ import annotations

from subtitles_extractor.application.services.bilingual_builder import (
    build_bilingual_events,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _ev(index: int, text: str, start: float = 0.0, end: float = 1.0) -> SubtitleEvent:
    return SubtitleEvent(
        index=index, text=text, interval=TimeInterval(start_sec=start, end_sec=end)
    )


def test_merges_original_above_translation_by_default() -> None:
    src = [_ev(1, "你好"), _ev(2, "再见")]
    trans = [_ev(1, "Xin chào"), _ev(2, "Tạm biệt")]
    out = build_bilingual_events(src, trans)
    assert out[0].text == "你好\nXin chào"
    assert out[1].text == "再见\nTạm biệt"
    # Giữ thời gian/uid của bản dịch.
    assert out[0].interval == trans[0].interval


def test_translation_on_top() -> None:
    src = [_ev(1, "你好")]
    trans = [_ev(1, "Xin chào")]
    out = build_bilingual_events(src, trans, translation_on_top=True)
    assert out[0].text == "Xin chào\n你好"


def test_missing_source_keeps_translation_only() -> None:
    src = [_ev(1, "你好")]
    trans = [_ev(2, "Câu không có gốc")]  # index 2 không khớp gốc
    out = build_bilingual_events(src, trans)
    assert out[0].text == "Câu không có gốc"


def test_identical_text_not_duplicated() -> None:
    # Câu STYLE/LOCALIZE có thể trả về y hệt gốc -> không nhân đôi.
    src = [_ev(1, "OK")]
    trans = [_ev(1, "OK")]
    out = build_bilingual_events(src, trans)
    assert out[0].text == "OK"


def test_does_not_mutate_inputs() -> None:
    src = [_ev(1, "你好")]
    trans = [_ev(1, "Xin chào")]
    build_bilingual_events(src, trans)
    assert src[0].text == "你好"
    assert trans[0].text == "Xin chào"


def test_count_follows_translation() -> None:
    src = [_ev(1, "a"), _ev(2, "b"), _ev(3, "c")]
    trans = [_ev(1, "A"), _ev(2, "B")]  # dịch gộp còn 2 câu
    out = build_bilingual_events(src, trans)
    assert len(out) == 2
