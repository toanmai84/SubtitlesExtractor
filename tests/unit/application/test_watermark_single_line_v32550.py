"""[v3.23.150] Test strip_persistent_watermark_lines: drop event MỘT DÒNG thuần watermark.

Bug: watermark Latin dai dẳng được XÁC ĐỊNH bằng thống kê multi-line (>= 5 lần và >= 8%
multi-line events) nhưng khi xuất hiện MỘT MÌNH (single-line event — lúc màn hình chỉ còn
logo, không thoại) thì nhánh ``len(raw_lines) < 2`` giữ nguyên -> phụ đề đầu ra lẫn hàng
loạt event "ARBOR"/"HOME" đơn độc. Fix: single-line có nội dung ∈ watermark_lines -> drop.
"""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    strip_persistent_watermark_lines,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _event(index: int, text: str) -> SubtitleEvent:
    return SubtitleEvent(
        index=index, text=text,
        interval=TimeInterval(float(index), float(index) + 1.0),
    )


def _movie_with_watermark() -> list[SubtitleEvent]:
    """10 event multi-line dính 'ARBOR' + 3 event single-line thuần 'ARBOR'."""
    events: list[SubtitleEvent] = []
    idx = 1
    for _ in range(10):
        events.append(_event(idx, f"ARBOR\n你好世界 {idx}"))
        idx += 1
    for _ in range(3):
        events.append(_event(idx, "ARBOR"))
        idx += 1
    events.append(_event(idx, "这是真正的对话"))
    return events


def test_single_line_pure_watermark_dropped() -> None:
    events = _movie_with_watermark()
    result = strip_persistent_watermark_lines(events)
    texts = [e.text for e in result]
    # Single-line thuần watermark bị DROP hoàn toàn.
    assert "ARBOR" not in texts
    # Multi-line được strip dòng watermark, giữ phần thoại.
    assert all("ARBOR" not in t for t in texts)
    assert any(t.startswith("你好世界") for t in texts)
    # Thoại thường không bị đụng.
    assert "这是真正的对话" in texts


def test_single_line_real_dialogue_kept() -> None:
    # Câu Latin thật chỉ xuất hiện 1 lần single-line -> KHÔNG đủ bằng chứng -> giữ.
    events = _movie_with_watermark()
    events.append(_event(99, "Let me explain everything"))
    result = strip_persistent_watermark_lines(events)
    assert "Let me explain everything" in [e.text for e in result]


def test_no_watermark_no_change() -> None:
    events = [
        _event(1, "你好\n世界"),
        _event(2, "第二句"),
        _event(3, "第三句\n继续"),
    ]
    result = strip_persistent_watermark_lines(events)
    assert [e.text for e in result] == [e.text for e in events]


def test_cjk_frequent_line_untouched() -> None:
    # Dòng CJK lặp nhiều (vd tên nhân vật lặp) KHÔNG bị coi là watermark
    # (chỉ Latin-dominant mới đủ điều kiện) -> single-line CJK giữ nguyên.
    events: list[SubtitleEvent] = []
    idx = 1
    for _ in range(10):
        events.append(_event(idx, "小明\n你要去哪里"))
        idx += 1
    events.append(_event(idx, "小明"))
    result = strip_persistent_watermark_lines(events)
    assert "小明" in [e.text for e in result]
