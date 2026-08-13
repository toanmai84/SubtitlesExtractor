"""Unit tests bảo vệ v3.6 bugfixes trong SubtitleEditorService.

Bug A: shift_all không reset _starts_cache → cursor video sai
Bug B: auto_fix_timeline push undo vô nghĩa khi applied_fixes=0
Bug C: undo/redo trả stale event.index sau _reindex mutation
Bug D: split() để lại null byte khi split_idx rơi vào giữa marker
"""
from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _make_event(index: int, start: float, end: float, text: str = "test") -> SubtitleEvent:
    return SubtitleEvent(
        index=index,
        text=text,
        interval=TimeInterval(start, end),
        confidence=Confidence(0.9),
        frame_count=5,
    )


def _make_service(*events: SubtitleEvent) -> SubtitleEditorService:
    svc = SubtitleEditorService()
    svc.load(list(events))
    return svc


# ---------------------------------------------------------------------------
# Bug A: shift_all — _starts_cache stale
# ---------------------------------------------------------------------------


class TestShiftAllStartsCacheInvalidated:
    """Bug A: shift_all phải reset _starts_cache."""

    def test_find_event_correct_after_shift(self) -> None:
        """find_event_index_at_time phải trả đúng kết quả sau shift_all."""
        svc = _make_service(
            _make_event(1, 1.0, 2.0, "Hello"),
            _make_event(2, 3.0, 4.0, "World"),
        )
        # Warm up cache
        assert svc.find_event_index_at_time(1.5) == 0
        assert svc.find_event_index_at_time(3.5) == 1

        # Shift +10s
        svc.shift_all(10.0)

        # Sau shift: event 0 ở [11, 12], event 1 ở [13, 14]
        # Cache cũ [1.0, 3.0] phải bị xóa, cache mới [11.0, 13.0] phải được tạo
        assert svc.find_event_index_at_time(11.5) == 0, (
            "Bug A: shift_all không reset _starts_cache → "
            "find_event_index_at_time trả sai index"
        )
        assert svc.find_event_index_at_time(13.5) == 1

    def test_starts_cache_is_none_after_shift(self) -> None:
        """_starts_cache phải là None ngay sau shift_all (sẽ được rebuild lazily)."""
        svc = _make_service(_make_event(1, 1.0, 2.0))
        _ = svc.find_event_index_at_time(1.5)  # Build cache
        assert svc._starts_cache is not None

        svc.shift_all(5.0)
        assert svc._starts_cache is None, (
            "Bug A: _starts_cache phải được reset sau shift_all"
        )

    def test_negative_shift_also_invalidates_cache(self) -> None:
        """Shift âm cũng phải reset cache."""
        svc = _make_service(
            _make_event(1, 10.0, 11.0),
            _make_event(2, 12.0, 13.0),
        )
        _ = svc.find_event_index_at_time(10.5)  # Build cache
        svc.shift_all(-5.0)
        assert svc._starts_cache is None

        # Cache phải rebuild đúng: events giờ ở [5,6] và [7,8]
        assert svc.find_event_index_at_time(5.5) == 0
        assert svc.find_event_index_at_time(7.5) == 1


# ---------------------------------------------------------------------------
# Bug B: auto_fix_timeline — spurious undo entry
# ---------------------------------------------------------------------------


class TestAutoFixTimelineUndoNotPushedWhenNoFix:
    """Bug B: auto_fix_timeline không push undo khi applied_fixes=0."""

    def test_undo_stack_empty_when_no_fix_applied(self) -> None:
        """Nếu tất cả cặp đều bị skip bởi guard conditions → undo stack không thay đổi."""
        # Tạo 2 events hoàn toàn đảo ngược nhau (start >= end của cái kia)
        # → guard `if curr.start_sec >= nxt.end_sec: continue` sẽ skip
        svc = _make_service(
            _make_event(1, 5.0, 10.0),   # curr: [5, 10]
            _make_event(2, 0.0, 0.5),    # nxt: [0, 0.5] — hoàn toàn trước curr
        )
        # Trường hợp này curr.end > nxt.start (overlap) nhưng curr.start > nxt.end
        # → bị guard skip, applied_fixes = 0

        initial_undo_size = len(svc._undo_stack)
        result = svc.auto_fix_timeline()

        # Không có fix nào thực sự được áp dụng
        # (hoặc = 0 vì pre-check cho phép tiếp tục nhưng guard block skip)
        if result == 0:
            assert len(svc._undo_stack) == initial_undo_size, (
                "Bug B: auto_fix_timeline không được push undo khi applied_fixes=0"
            )

    def test_undo_stack_grows_only_when_fix_applied(self) -> None:
        """Undo stack CHỈ tăng khi thực sự có fix được áp dụng."""
        svc = _make_service(
            _make_event(1, 1.0, 3.0),   # Overlap với event 2
            _make_event(2, 2.5, 4.0),
        )
        initial_undo_size = len(svc._undo_stack)
        result = svc.auto_fix_timeline()

        if result > 0:
            assert len(svc._undo_stack) == initial_undo_size + 1, (
                "Khi có fix thực sự, undo stack phải tăng đúng 1"
            )
        else:
            assert len(svc._undo_stack) == initial_undo_size, (
                "Không có fix → undo stack không thay đổi"
            )

    def test_undo_after_real_fix_restores_original(self) -> None:
        """Sau khi auto_fix rồi undo, sự kiện trở về trạng thái gốc."""
        svc = _make_service(
            _make_event(1, 1.0, 3.5),   # Overlap
            _make_event(2, 3.0, 5.0),
        )
        original_end_0 = svc._events[0].end_sec
        original_start_1 = svc._events[1].start_sec

        result = svc.auto_fix_timeline()
        assert result > 0, "Phải có fix trong test case này"

        # Undo
        state = svc.undo()
        assert state.events[0].end_sec == original_end_0, "Undo phải khôi phục end_sec của event 0"
        assert state.events[1].start_sec == original_start_1, "Undo phải khôi phục start_sec của event 1"


# ---------------------------------------------------------------------------
# Bug C: undo/redo — stale event.index
# ---------------------------------------------------------------------------


class TestUndoRedoEventIndexConsistency:
    """Bug C: undo/redo phải trả về event.index nhất quán."""

    def test_undo_delete_restores_correct_indices(self) -> None:
        """Sau delete rồi undo, tất cả event.index phải đúng thứ tự 1,2,3."""
        svc = _make_service(
            _make_event(1, 0.0, 1.0, "A"),
            _make_event(2, 1.5, 2.5, "B"),
            _make_event(3, 3.0, 4.0, "C"),
        )
        svc.delete(0)  # Xóa A → B.index=1, C.index=2 (after reindex)

        state = svc.undo()  # Phục hồi A
        indices = [ev.index for ev in state.events]
        assert indices == [1, 2, 3], (
            f"Bug C: Sau undo delete, indices phải là [1,2,3] nhưng nhận {indices}. "
            "Nguyên nhân: _reindex() mutate shared objects trong snapshot."
        )

    def test_undo_insert_restores_correct_indices(self) -> None:
        """Sau insert rồi undo, indices trở về đúng."""
        svc = _make_service(
            _make_event(1, 0.0, 1.0, "A"),
            _make_event(2, 5.0, 6.0, "B"),
        )
        svc.insert_after(0)  # Insert giữa A và B

        state = svc.undo()
        indices = [ev.index for ev in state.events]
        assert indices == [1, 2], (
            f"Bug C: Sau undo insert, indices phải là [1,2] nhưng nhận {indices}."
        )

    def test_redo_preserves_correct_indices(self) -> None:
        """Redo sau undo cũng phải có indices đúng."""
        svc = _make_service(
            _make_event(1, 0.0, 1.0, "A"),
            _make_event(2, 1.5, 2.5, "B"),
            _make_event(3, 3.0, 4.0, "C"),
        )
        svc.delete(1)  # Xóa B
        svc.undo()     # Phục hồi B
        state = svc.redo()  # Redo: xóa B lại

        indices = [ev.index for ev in state.events]
        assert indices == [1, 2], (
            f"Bug C: Sau redo delete, indices phải là [1,2] nhưng nhận {indices}."
        )

    def test_multiple_undos_keep_indices_consistent(self) -> None:
        """Nhiều lần undo liên tiếp vẫn duy trì indices nhất quán."""
        svc = _make_service(
            _make_event(1, 0.0, 1.0, "A"),
            _make_event(2, 1.5, 2.5, "B"),
            _make_event(3, 3.0, 4.0, "C"),
        )
        svc.delete(2)  # Xóa C → [A(1), B(2)]
        svc.delete(0)  # Xóa A → [B(1)]

        # Undo 1: phục hồi A → [A(?), B(?)]
        state = svc.undo()
        indices = [ev.index for ev in state.events]
        assert indices == [1, 2], f"Undo 1: expected [1,2] got {indices}"

        # Undo 2: phục hồi C → [A(?), B(?), C(?)]
        state = svc.undo()
        indices = [ev.index for ev in state.events]
        assert indices == [1, 2, 3], f"Undo 2: expected [1,2,3] got {indices}"

    def test_srt_export_has_correct_sequence_numbers_after_undo(self) -> None:
        """SRT export sau undo phải có số thứ tự 1, 2, 3... liên tiếp."""
        svc = _make_service(
            _make_event(1, 0.0, 1.0, "First"),
            _make_event(2, 2.0, 3.0, "Second"),
            _make_event(3, 4.0, 5.0, "Third"),
        )
        svc.delete(0)   # Xóa First
        state = svc.undo()  # Phục hồi First

        event_indices = [ev.index for ev in state.events]
        # Đảm bảo không có duplicate và đúng thứ tự
        assert event_indices == sorted(set(event_indices)), (
            "Indices phải là unique và tăng dần"
        )
        assert event_indices[0] == 1, "Index đầu tiên phải là 1"
        for i in range(len(event_indices) - 1):
            assert event_indices[i+1] == event_indices[i] + 1, (
                f"Indices phải liên tiếp, nhưng nhận {event_indices}"
            )


# ---------------------------------------------------------------------------
# Bug D: split() — null byte trong tag marker bị cắt đôi
# ---------------------------------------------------------------------------


class TestSplitTagMarkerNotBroken:
    """Bug D: split() không được để lại null byte khi marker bị cắt."""

    def test_split_with_html_tag_no_null_bytes(self) -> None:
        """Split subtitle có HTML tag không được có ký tự null trong kết quả."""
        svc = _make_service(
            _make_event(1, 0.0, 5.0, "<b>Hello World This Is Test</b>")
        )
        svc.split(0, 2.5)
        for ev in svc._events:
            assert "\x00" not in ev.text, (
                f"Bug D: null byte tìm thấy trong '{ev.text!r}' sau khi split. "
                "Marker bị cắt đôi khi split_idx rơi vào giữa ký tự \\x00...\\x00."
            )

    def test_split_with_ass_tag_no_null_bytes(self) -> None:
        """Split subtitle có ASS tag không được có null byte."""
        svc = _make_service(
            _make_event(1, 0.0, 5.0, "{\\an8}Hello World Test Content")
        )
        svc.split(0, 2.5)
        for ev in svc._events:
            assert "\x00" not in ev.text, (
                f"Null byte tìm thấy: {ev.text!r}"
            )

    def test_split_multiple_tags_produces_valid_text(self) -> None:
        """Split với nhiều tag phải tạo text hợp lệ (không null, không marker thừa)."""
        svc = _make_service(
            _make_event(1, 0.0, 5.0, "<b>Hello</b> <i>World</i> <b>Test</b>")
        )
        svc.split(0, 2.5)
        for ev in svc._events:
            assert "\x00" not in ev.text, f"Null byte in {ev.text!r}"
            # Không có marker dư (dạng số giữa null byte)
            import re
            assert not re.search(r'\x00\d+\x00', ev.text), (
                f"Marker thừa trong {ev.text!r}"
            )

    def test_split_plain_text_still_works(self) -> None:
        """Split text thông thường (không có tag) vẫn hoạt động đúng."""
        svc = _make_service(
            _make_event(1, 0.0, 5.0, "Hello World How Are You")
        )
        state = svc.split(0, 2.5)
        assert len(state.events) == 2
        combined = state.events[0].text + state.events[1].text
        # Text gốc phải được bảo toàn (whitespace có thể thay đổi)
        assert "Hello" in combined
        assert "World" in combined
