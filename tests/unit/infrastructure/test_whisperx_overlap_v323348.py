"""Test làm sạch câu phụ đề từ WhisperX — v3.23.348.

HAI LỖI ĐO ĐƯỢC, không có bước làm sạch nào trước đây:

1. **Chồng lấn.** Hai đoạn ``10.00–13.50`` và ``12.80–15.00`` (rất hay gặp khi bật phân
   tách người nói) chồng nhau 0,7 giây. Trình phát hiện HAI dòng cùng lúc.
2. **Sai thứ tự.** Câu ở giây 20 được đánh số 1, câu ở giây 10 đánh số 2 — chỉ số gán
   theo thứ tự ĐẦU VÀO chứ không theo thời gian. Tệp SRT có mốc giảm dần.

Nguyên tắc sửa: **cắt bớt, không bỏ câu**. Nội dung vẫn còn, chỉ rút ngắn thời gian
hiển thị. Nếu cắt xong không còn thời lượng dương thì giữ nguyên — thà chồng nhẹ còn
hơn mất chữ.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.stt.whisperx_adapter import WhisperXAdapter


def _segments(*items: tuple[float, float, str]) -> list[dict]:
    return [{"start": s, "end": e, "text": t} for s, e, t in items]


def _bounds(events) -> list[tuple[float, float]]:
    return [(e.interval.start_sec, e.interval.end_sec) for e in events]


# ── Chồng lấn ────────────────────────────────────────────────────────────────
def test_overlapping_cues_are_trimmed() -> None:
    events = WhisperXAdapter._segments_to_events(
        _segments((10.0, 13.5, "câu một"), (12.8, 15.0, "câu hai")), None
    )
    assert events[0].interval.end_sec <= events[1].interval.start_sec


def test_trim_keeps_a_small_gap() -> None:
    """Khe hở nhỏ để trình phát không hiển thị chồng khung hình."""
    events = WhisperXAdapter._segments_to_events(
        _segments((10.0, 13.5, "a"), (12.8, 15.0, "b")), None
    )
    gap = events[1].interval.start_sec - events[0].interval.end_sec
    assert 0.0 < gap <= 0.05


def test_trim_never_drops_a_cue() -> None:
    """Cắt bớt chứ KHÔNG bỏ câu — mất chữ tệ hơn chồng nhẹ."""
    events = WhisperXAdapter._segments_to_events(
        _segments((5.0, 9.0, "dài"), (5.0, 6.0, "ngắn")), None
    )
    assert len(events) == 2


def test_fully_nested_cue_is_kept_intact() -> None:
    """Không cắt được (sẽ mất thời lượng) thì giữ nguyên."""
    events = WhisperXAdapter._segments_to_events(
        _segments((5.0, 9.0, "dài"), (5.0, 6.0, "ngắn")), None
    )
    assert all(e.interval.end_sec > e.interval.start_sec for e in events)


def test_three_way_overlap_resolved_pairwise() -> None:
    events = WhisperXAdapter._segments_to_events(
        _segments((0.0, 5.0, "a"), (3.0, 8.0, "b"), (6.0, 10.0, "c")), None
    )
    for current, following in zip(events, events[1:]):
        assert current.interval.end_sec <= following.interval.start_sec


# ── Thứ tự ───────────────────────────────────────────────────────────────────
def test_out_of_order_segments_are_sorted() -> None:
    events = WhisperXAdapter._segments_to_events(
        _segments((20.0, 22.0, "câu sau"), (10.0, 12.0, "câu trước")), None
    )
    assert events[0].text == "câu trước"
    assert _bounds(events) == [(10.0, 12.0), (20.0, 22.0)]


def test_indexes_renumbered_after_sorting() -> None:
    """Chỉ số phải khớp thứ tự thời gian, không phải thứ tự đầu vào."""
    events = WhisperXAdapter._segments_to_events(
        _segments((20.0, 22.0, "sau"), (10.0, 12.0, "trước")), None
    )
    assert [e.index for e in events] == [1, 2]
    assert events[0].text == "trước"


def test_timestamps_are_monotonic() -> None:
    """Bất biến chính của một tệp phụ đề hợp lệ."""
    events = WhisperXAdapter._segments_to_events(
        _segments((30.0, 32.0, "c"), (10.0, 12.0, "a"), (20.0, 25.0, "b")), None
    )
    starts = [e.interval.start_sec for e in events]
    assert starts == sorted(starts)


# ── Không hồi quy ────────────────────────────────────────────────────────────
def test_clean_input_is_unchanged() -> None:
    events = WhisperXAdapter._segments_to_events(
        _segments((1.0, 2.0, "a"), (3.0, 4.0, "b")), None
    )
    assert _bounds(events) == [(1.0, 2.0), (3.0, 4.0)]


def test_adjacent_cues_are_not_trimmed() -> None:
    """Kề nhau nhưng KHÔNG chồng — không được đụng vào."""
    events = WhisperXAdapter._segments_to_events(
        _segments((1.0, 2.0, "a"), (2.0, 3.0, "b")), None
    )
    assert _bounds(events) == [(1.0, 2.0), (2.0, 3.0)]


def test_single_cue_passes_through() -> None:
    events = WhisperXAdapter._segments_to_events(_segments((1.0, 2.0, "a")), None)
    assert len(events) == 1


def test_empty_input() -> None:
    assert WhisperXAdapter._segments_to_events([], None) == []


# ── Hằng số ──────────────────────────────────────────────────────────────────
def test_gap_constant_is_small() -> None:
    """Khe hở phải đủ nhỏ để người xem không thấy hụt."""
    from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
        _MIN_CUE_GAP_SEC,
    )

    assert 0.0 < _MIN_CUE_GAP_SEC <= 0.05


@pytest.mark.parametrize("count", [0, 1])
def test_sanitise_handles_short_lists(count: int) -> None:
    events = WhisperXAdapter._segments_to_events(
        _segments(*[(float(i), float(i) + 1.0, "x") for i in range(count)]), None
    )
    assert len(events) == count
