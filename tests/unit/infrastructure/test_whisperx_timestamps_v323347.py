"""Test mốc thời gian phụ đề từ WhisperX — v3.23.347.

TRIỆU CHỨNG NGƯỜI DÙNG BÁO: *"phụ đề trích xuất ra thỉnh thoảng có câu bị lệch mốc thời
gian bắt đầu so với giọng nói."*

NGUYÊN NHÂN: ``_normalize_words`` khởi tạo ``last_end = 0.0``. WhisperX thường trả
``start=None`` cho những từ nó không căn chỉnh được (số, ký hiệu, từ ngoài từ điển).
Khi từ đó là từ **ĐẦU TIÊN** của đoạn, nó nhận mốc ``0.0`` — tức đầu phim.

Đo trên ví dụ thật: đoạn thoại ở giây 125,4 bị đẩy về giây 0 — **lệch 125 giây**.

Điều này giải thích chữ "thỉnh thoảng": từ ở GIỮA đoạn kế thừa mốc của từ trước (hợp
lý), chỉ từ ĐẦU đoạn mới rơi về 0.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.stt.whisperx_adapter import WhisperXAdapter


def _words(*items: tuple[str, float | None, float | None]) -> list[dict]:
    return [{"word": text, "start": start, "end": end} for text, start, end in items]


# ── Lỗi chính: từ đầu đoạn mất mốc bắt đầu ───────────────────────────────────
def test_first_word_without_start_anchors_to_segment() -> None:
    """LỖI ĐÃ SỬA: trước đây rơi về 0.0 — lệch tới cả trăm giây."""
    result = WhisperXAdapter._normalize_words(
        _words(("Bát", None, None), ("đệ", 125.4, 125.7)),
        segment_start=125.3,
    )
    assert result[0][1] == pytest.approx(125.3)


def test_first_word_never_lands_at_zero_when_segment_is_late() -> None:
    """Bất biến quan trọng: câu ở phút thứ 20 không được bắt đầu ở giây 0."""
    result = WhisperXAdapter._normalize_words(
        _words(("xxx", None, None), ("yyy", 1200.5, 1200.9)),
        segment_start=1200.0,
    )
    assert result[0][1] > 1000.0


def test_middle_word_still_inherits_previous_end() -> None:
    """KHÔNG hồi quy: từ giữa kế thừa mốc của từ trước là hành vi đúng."""
    result = WhisperXAdapter._normalize_words(
        _words(("Thẩm", 200.0, 200.3), ("tướng", None, None), ("quân", 200.9, 201.2)),
        segment_start=199.9,
    )
    assert result[1][1] == pytest.approx(200.3)


def test_cue_start_comes_from_first_word() -> None:
    """Mốc câu lấy từ từ đầu tiên — nên từ đầu sai là cả câu sai."""
    cues = WhisperXAdapter._split_words_into_cues(
        _words(("Xin", None, None), ("chào", 50.2, 50.5)),
        None,
        50.1,
        50.6,
    )
    assert cues
    assert cues[0][1] == pytest.approx(50.1)


# ── Lỗi phụ: từ cuối mất mốc kết thúc ────────────────────────────────────────
def test_last_word_without_end_uses_segment_end() -> None:
    """Không có neo thì câu kết thúc ngay tại mốc bắt đầu -> cắt cụt tiếng nói."""
    result = WhisperXAdapter._normalize_words(
        _words(("Ta", 10.0, 10.3), ("thấy", 10.4, None)),
        segment_start=9.9,
        segment_end=10.9,
    )
    assert result[-1][2] == pytest.approx(10.9)


def test_middle_word_without_end_uses_next_word_start() -> None:
    """Chặt hơn nhiều so với ``end = start`` (thời lượng 0)."""
    result = WhisperXAdapter._normalize_words(
        _words(("A", 1.0, None), ("B", 1.8, 2.0)),
        segment_start=0.9,
        segment_end=2.0,
    )
    assert result[0][2] == pytest.approx(1.8)


def test_no_zero_duration_when_bounds_available() -> None:
    """Từ nào cũng phải có thời lượng dương nếu có neo để suy ra."""
    result = WhisperXAdapter._normalize_words(
        _words(("A", 1.0, None), ("B", 1.8, None)),
        segment_start=0.9,
        segment_end=2.5,
    )
    assert all(end > start for _text, start, end in result)


# ── Không hồi quy ────────────────────────────────────────────────────────────
def test_complete_words_are_untouched() -> None:
    result = WhisperXAdapter._normalize_words(
        _words(("X", 5.0, 5.5)), segment_start=4.9, segment_end=6.0
    )
    assert result == [("X", 5.0, 5.5)]


def test_defaults_preserve_old_behaviour() -> None:
    """Gọi không truyền neo vẫn chạy — mã cũ không vỡ."""
    result = WhisperXAdapter._normalize_words(_words(("A", None, None)))
    assert result[0][1] == 0.0


def test_invalid_values_do_not_crash() -> None:
    """Giá trị rác từ SDK không được làm sập cả phiên phiên âm."""
    result = WhisperXAdapter._normalize_words(
        [{"word": "A", "start": "xyz", "end": None}], segment_start=3.0
    )
    assert result[0][1] == pytest.approx(3.0)


def test_non_dict_entries_are_skipped() -> None:
    result = WhisperXAdapter._normalize_words(
        ["rác", {"word": "A", "start": 1.0, "end": 1.5}], segment_start=0.0
    )
    assert len(result) == 1


def test_empty_word_text_is_skipped() -> None:
    result = WhisperXAdapter._normalize_words(
        [{"word": "", "start": 1.0, "end": 1.5}], segment_start=0.0
    )
    assert result == []


# ── Đọc mốc đoạn an toàn ─────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("segment", "key", "expected"),
    [
        ({"start": 12.5}, "start", 12.5),
        ({"end": 20.0}, "end", 20.0),
        ({"start": None}, "start", 0.0),
        ({"start": "rác"}, "start", 0.0),
        ({}, "start", 0.0),
        ("không phải dict", "start", 0.0),
    ],
)
def test_segment_bound_reader(segment: object, key: str, expected: float) -> None:
    from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
        _safe_segment_bound,
    )

    assert _safe_segment_bound(segment, key) == pytest.approx(expected)
