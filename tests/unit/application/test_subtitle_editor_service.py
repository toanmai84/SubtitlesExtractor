"""Test :class:`SubtitleEditorService` — undo/redo + edit operations."""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _make_event(idx: int, text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(
        index=idx,
        text=text,
        interval=TimeInterval(start, end),
        confidence=Confidence(0.9),
    )


@pytest.fixture
def initial_events() -> list[SubtitleEvent]:
    return [
        _make_event(1, "Câu thứ nhất", 0.0, 2.0),
        _make_event(2, "Câu thứ hai", 3.0, 5.0),
        _make_event(3, "Câu thứ ba", 6.0, 8.0),
    ]


@pytest.fixture
def service(initial_events: list[SubtitleEvent]) -> SubtitleEditorService:
    s = SubtitleEditorService()
    s.load(initial_events)
    return s


class TestLoadAndState:
    def test_load_clears_history(self, service: SubtitleEditorService) -> None:
        state = service.snapshot_state()
        assert len(state.events) == 3
        assert not state.can_undo
        assert not state.can_redo
        assert not state.is_dirty

    def test_load_reindexes(self) -> None:
        s = SubtitleEditorService()
        events = [_make_event(99, "x", 0, 1), _make_event(50, "y", 2, 3)]
        state = s.load(events)
        assert [e.index for e in state.events] == [1, 2]


class TestUpdateText:
    def test_update_text(self, service: SubtitleEditorService) -> None:
        state = service.update_text(0, "Đã sửa")
        assert state.events[0].text == "Đã sửa"
        assert state.is_dirty
        assert state.can_undo

    def test_update_text_invalid_index(self, service: SubtitleEditorService) -> None:
        with pytest.raises(IndexError):
            service.update_text(999, "x")


class TestUpdateTiming:
    def test_update_timing_valid(self, service: SubtitleEditorService) -> None:
        state = service.update_timing(0, 1.0, 3.0)
        assert state.events[0].start_sec == 1.0
        assert state.events[0].end_sec == 3.0

    def test_update_timing_invalid_rolls_back(
        self, service: SubtitleEditorService
    ) -> None:
        with pytest.raises(ConfigurationError):
            service.update_timing(0, 5.0, 1.0)  # end < start
        # Sau lỗi, không có undo entry mới được tạo.
        assert not service.snapshot_state().can_undo


class TestInsertDelete:
    def test_insert_after(self, service: SubtitleEditorService) -> None:
        state = service.insert_after(0, "Mới chèn")
        assert len(state.events) == 4
        assert state.events[1].text == "Mới chèn"
        # Index re-numbered.
        assert [e.index for e in state.events] == [1, 2, 3, 4]

    def test_insert_at_beginning(self, service: SubtitleEditorService) -> None:
        state = service.insert_after(-1, "Đầu tiên")
        assert state.events[0].text == "Đầu tiên"

    def test_delete(self, service: SubtitleEditorService) -> None:
        state = service.delete(1)
        assert len(state.events) == 2
        assert state.events[0].text == "Câu thứ nhất"
        assert state.events[1].text == "Câu thứ ba"
        assert [e.index for e in state.events] == [1, 2]


class TestSplitMerge:
    def test_split(self, service: SubtitleEditorService) -> None:
        state = service.split(0, 1.0)  # event[0] = (0, 2), tách tại 1.0
        assert len(state.events) == 4
        assert state.events[0].interval.end_sec == 1.0
        assert state.events[1].interval.start_sec == 1.0

    def test_split_outside_interval_fails(
        self, service: SubtitleEditorService
    ) -> None:
        with pytest.raises(ConfigurationError):
            service.split(0, 3.0)  # 3.0 > end_sec=2.0

    def test_merge_with_next(self, service: SubtitleEditorService) -> None:
        state = service.merge_with_next(0)
        assert len(state.events) == 2
        # Câu gộp có start = câu 1, end = câu 2, text nối bằng \n.
        assert state.events[0].interval.start_sec == 0.0
        assert state.events[0].interval.end_sec == 5.0
        assert "Câu thứ nhất" in state.events[0].text
        assert "Câu thứ hai" in state.events[0].text

    def test_merge_last_fails(self, service: SubtitleEditorService) -> None:
        with pytest.raises(IndexError):
            service.merge_with_next(2)  # index cuối


class TestShiftAll:
    def test_shift_positive(self, service: SubtitleEditorService) -> None:
        state = service.shift_all(2.5)
        assert state.events[0].start_sec == 2.5
        assert state.events[0].end_sec == 4.5

    def test_shift_negative(self, service: SubtitleEditorService) -> None:
        # Với initial events [(0,2), (3,5), (6,8)], shift -0.5 sẽ khiến
        # event đầu start=-0.5 → fail. Cần tạo service với event start ≥ 0.5.
        s = SubtitleEditorService()
        s.load([_make_event(1, "x", 1.0, 2.0), _make_event(2, "y", 3.0, 4.0)])
        state = s.shift_all(-0.5)
        assert state.events[0].start_sec == 0.5
        assert state.events[0].end_sec == 1.5

    def test_shift_too_negative_fails(
        self, service: SubtitleEditorService
    ) -> None:
        with pytest.raises(ConfigurationError):
            service.shift_all(-100.0)


class TestUndoRedo:
    def test_undo_after_edit(self, service: SubtitleEditorService) -> None:
        original_text = service.snapshot_state().events[0].text
        service.update_text(0, "Đã đổi")
        state_after_undo = service.undo()
        assert state_after_undo.events[0].text == original_text
        assert state_after_undo.can_redo

    def test_redo_after_undo(self, service: SubtitleEditorService) -> None:
        service.update_text(0, "Đã đổi")
        service.undo()
        state_after_redo = service.redo()
        assert state_after_redo.events[0].text == "Đã đổi"

    def test_new_action_clears_redo_stack(
        self, service: SubtitleEditorService
    ) -> None:
        service.update_text(0, "A")
        service.undo()
        # Sau undo, redo stack có phần tử.
        assert service.snapshot_state().can_redo
        # Action mới ⇒ clear redo stack.
        service.update_text(1, "B")
        assert not service.snapshot_state().can_redo

    def test_undo_on_empty_history(self) -> None:
        s = SubtitleEditorService()
        # Chưa load gì — undo không gây lỗi.
        state = s.undo()
        assert state.events == []


class TestMarkClean:
    def test_mark_clean_resets_dirty(self, service: SubtitleEditorService) -> None:
        service.update_text(0, "A")
        assert service.snapshot_state().is_dirty
        clean = service.mark_clean()
        assert not clean.is_dirty
