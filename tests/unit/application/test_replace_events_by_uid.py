"""Test :meth:`SubtitleEditorService.replace_events_by_uid`.

Bao phủ logic merge với overlap detection — sửa bug v2.15: index-based
replace gây trùng lặp khi events mới overlap với events cũ không thuộc
selection.
"""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _make_event(
    index: int, text: str, start: float, end: float, uid: str | None = None
) -> SubtitleEvent:
    """Tạo SubtitleEvent với UID có thể chỉ định (giúp test).

    Khi ``uid=None``, UUID4 ngẫu nhiên sẽ được tạo (giống production).
    """
    kwargs: dict = {
        "index": index,
        "text": text,
        "interval": TimeInterval(start, end),
        "confidence": Confidence(0.9),
        "frame_count": 5,
    }
    if uid is not None:
        kwargs["uid"] = uid
    return SubtitleEvent(**kwargs)


class TestReplaceEventsByUid:
    def test_replaces_single_event(self) -> None:
        service = SubtitleEditorService()
        e1 = _make_event(1, "Câu 1", 0.0, 1.0, uid="uid-1")
        e2 = _make_event(2, "Câu 2", 1.5, 2.5, uid="uid-2")
        service.load([e1, e2])

        new_event = _make_event(99, "Câu 1 sửa", 0.0, 1.0)
        state = service.replace_events_by_uid(["uid-1"], [new_event])

        assert len(state.events) == 2
        assert state.events[0].text == "Câu 1 sửa"
        assert state.events[1].text == "Câu 2"
        # Index được tự reindex từ 1.
        assert state.events[0].index == 1
        assert state.events[1].index == 2

    def test_remove_only_when_new_events_empty(self) -> None:
        service = SubtitleEditorService()
        e1 = _make_event(1, "Giữ", 0.0, 1.0, uid="keep")
        e2 = _make_event(2, "Xoá", 1.5, 2.5, uid="remove")
        service.load([e1, e2])

        state = service.replace_events_by_uid(["remove"], [])
        assert len(state.events) == 1
        assert state.events[0].text == "Giữ"

    def test_no_op_when_both_empty(self) -> None:
        service = SubtitleEditorService()
        e1 = _make_event(1, "x", 0.0, 1.0, uid="u")
        service.load([e1])
        state = service.replace_events_by_uid([], [])
        assert len(state.events) == 1

    def test_replaces_multiple_uids(self) -> None:
        service = SubtitleEditorService()
        events = [
            _make_event(1, "A", 0.0, 1.0, uid="a"),
            _make_event(2, "B", 1.5, 2.5, uid="b"),
            _make_event(3, "C", 3.0, 4.0, uid="c"),
        ]
        service.load(events)

        new = [
            _make_event(99, "B-mới", 1.5, 2.5),
            _make_event(99, "C-mới", 3.0, 4.0),
        ]
        state = service.replace_events_by_uid(["b", "c"], new)
        assert len(state.events) == 3
        assert state.events[0].text == "A"
        assert state.events[1].text == "B-mới"
        assert state.events[2].text == "C-mới"

    def test_overlap_protection_clips_keep_event(self) -> None:
        """Khi new_event overlap với một keep_event không nằm trong
        ``uids_to_remove``, keep_event phải bị cắt ngắn để tránh chồng."""
        service = SubtitleEditorService()
        events = [
            _make_event(1, "Giữ", 0.0, 5.0, uid="keep"),
            _make_event(2, "Xoá", 6.0, 8.0, uid="remove"),
        ]
        service.load(events)

        # New event overlap với "Giữ" (kéo dài 0-5) ở vùng 4-5.
        new = [_make_event(99, "Mới", 4.0, 7.0)]
        state = service.replace_events_by_uid(["remove"], new)

        # "Giữ" bị cắt: end_sec đẩy về 4.0; "Mới" giữ nguyên 4.0-7.0.
        assert len(state.events) == 2
        keep_event = next(e for e in state.events if e.text == "Giữ")
        new_event = next(e for e in state.events if e.text == "Mới")
        assert keep_event.end_sec == 4.0
        assert new_event.start_sec == 4.0
        assert new_event.end_sec == 7.0

    def test_overlap_drops_too_short_keep_event(self) -> None:
        """Nếu sau cắt keep_event ngắn hơn 0.05s, drop nó hẳn."""
        service = SubtitleEditorService()
        events = [
            _make_event(1, "Quá ngắn", 0.0, 1.0, uid="short"),
            _make_event(2, "Xoá", 5.0, 6.0, uid="remove"),
        ]
        service.load(events)

        # New event nuốt gần hết "Quá ngắn".
        new = [_make_event(99, "Mới", 0.02, 6.0)]
        state = service.replace_events_by_uid(["remove"], new)

        # "Quá ngắn" bị drop (chỉ còn 0.02s sau khi cắt).
        assert len(state.events) == 1
        assert state.events[0].text == "Mới"

    def test_unknown_uid_is_silently_ignored(self) -> None:
        """UID không tồn tại không gây crash — cứ thêm new_events bình thường."""
        service = SubtitleEditorService()
        e1 = _make_event(1, "A", 0.0, 1.0, uid="a")
        service.load([e1])

        new = [_make_event(99, "B", 5.0, 6.0)]
        state = service.replace_events_by_uid(["does-not-exist"], new)
        assert len(state.events) == 2

    def test_undo_after_replace(self) -> None:
        """Sau replace, undo phải khôi phục state cũ."""
        service = SubtitleEditorService()
        events = [_make_event(1, "Cũ", 0.0, 1.0, uid="u1")]
        service.load(events)

        new = [_make_event(99, "Mới", 0.0, 1.0)]
        service.replace_events_by_uid(["u1"], new)

        undone_state = service.undo()
        assert len(undone_state.events) == 1
        assert undone_state.events[0].text == "Cũ"

    def test_uids_remain_stable_across_replace(self) -> None:
        """Các events không bị thay thế phải giữ UID cũ — quan trọng để
        các thao tác Re-OCR liên tiếp không mất ánh xạ."""
        service = SubtitleEditorService()
        events = [
            _make_event(1, "A", 0.0, 1.0, uid="aaa"),
            _make_event(2, "B", 1.5, 2.5, uid="bbb"),
        ]
        service.load(events)

        new = [_make_event(99, "B-mới", 1.5, 2.5)]
        state = service.replace_events_by_uid(["bbb"], new)

        a_event = next(e for e in state.events if e.text == "A")
        assert a_event.uid == "aaa"
