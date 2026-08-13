"""[v3.23.151] Test trang chỉnh sửa: bảo toàn uid/bounding_box + chống chồng phụ đề.

Ba bug được vá:
1. ``merge_with_next`` không truyền uid/bounding_box -> uid sinh MỚI (Re-OCR theo uid
   mất khớp), box thành None (overlay không vẽ được) — trong khi ``apply_merge_groups``
   cùng file giữ đủ.
2. ``split`` mất bounding_box ở cả hai nửa + nửa đầu mất liên kết uid gốc.
3. ``_clip_against_new_events``: event cũ BAO TRÙM vùng Re-OCR (không chạm biên nào)
   -> hai nhánh cắt biên đều bỏ qua -> giữ nguyên -> phụ đề CHỒNG lên vùng mới.
"""

from __future__ import annotations

from itertools import pairwise

from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
    _union_bounding_boxes,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _event(
    index: int,
    text: str,
    start: float,
    end: float,
    box: tuple[int, int, int, int] | None = None,
) -> SubtitleEvent:
    return SubtitleEvent(
        index=index, text=text, interval=TimeInterval(start, end), bounding_box=box
    )


# ── Fix 1: merge_with_next giữ uid + hợp nhất bounding_box ───────────────


def test_merge_keeps_first_uid_and_unions_boxes() -> None:
    service = SubtitleEditorService()
    first = _event(1, "你好", 0.0, 1.0, box=(10, 100, 50, 120))
    second = _event(2, "世界", 1.2, 2.0, box=(60, 98, 120, 122))
    service.load([first, second])
    original_uid = first.uid

    state = service.merge_with_next(0)

    assert len(state.events) == 1
    merged = state.events[0]
    assert merged.uid == original_uid
    assert merged.bounding_box == (10, 98, 120, 122)
    assert merged.text == "你好世界"  # CJK nối liền không space


def test_merge_box_none_falls_back_to_other() -> None:
    service = SubtitleEditorService()
    service.load([
        _event(1, "Hello", 0.0, 1.0, box=None),
        _event(2, "world", 1.2, 2.0, box=(5, 6, 7, 8)),
    ])
    state = service.merge_with_next(0)
    assert state.events[0].bounding_box == (5, 6, 7, 8)
    assert state.events[0].text == "Hello world"  # Latin nối bằng space


def test_union_bounding_boxes_both_none() -> None:
    assert _union_bounding_boxes(None, None) is None


# ── Fix 2: split giữ bounding_box + nửa đầu kế thừa uid ──────────────────


def test_split_preserves_box_and_first_half_uid() -> None:
    service = SubtitleEditorService()
    original = _event(1, "Xin chào mọi người nhé", 0.0, 4.0, box=(1, 2, 3, 4))
    service.load([original])
    original_uid = original.uid

    state = service.split(0, 2.0)

    assert len(state.events) == 2
    first_half, second_half = state.events
    assert first_half.bounding_box == (1, 2, 3, 4)
    assert second_half.bounding_box == (1, 2, 3, 4)
    assert first_half.uid == original_uid
    assert second_half.uid != original_uid  # nửa sau là event mới


# ── Fix 3: event cũ bao trùm vùng Re-OCR phải bị cắt (không chồng) ───────


def test_containing_event_clipped_keeps_longer_side() -> None:
    service = SubtitleEditorService()
    service.load([_event(1, "câu dài bao trùm", 0.0, 10.0)])
    new_events = [_event(0, "vùng mới", 2.0, 4.0)]

    state = service.replace_events(indices_to_remove=[], new_events=new_events)

    events = sorted(state.events, key=lambda e: e.start_sec)
    assert len(events) == 2
    # Đuôi (4->10 = 6s) dài hơn đầu (0->2 = 2s) -> giữ đuôi.
    kept = next(e for e in events if e.text == "câu dài bao trùm")
    assert kept.start_sec == 4.0 and kept.end_sec == 10.0
    # Không còn cặp nào chồng nhau.
    for left, right in pairwise(events):
        assert left.end_sec <= right.start_sec


def test_containing_event_equal_sides_keeps_head() -> None:
    service = SubtitleEditorService()
    service.load([_event(1, "bao trùm cân", 0.0, 10.0)])
    state = service.replace_events(
        indices_to_remove=[], new_events=[_event(0, "giữa", 4.0, 6.0)]
    )
    kept = next(e for e in state.events if e.text == "bao trùm cân")
    assert kept.start_sec == 0.0 and kept.end_sec == 4.0


def test_clip_edge_behaviors_unchanged() -> None:
    # Hành vi cũ giữ nguyên: tách biệt -> không đổi; bị đè hoàn toàn -> drop.
    service = SubtitleEditorService()
    service.load([
        _event(1, "tách biệt", 0.0, 1.0),
        _event(2, "bị đè", 2.0, 3.0),
    ])
    state = service.replace_events(
        indices_to_remove=[], new_events=[_event(0, "đè lên", 1.9, 3.1)]
    )
    texts = [e.text for e in state.events]
    assert "tách biệt" in texts
    assert "bị đè" not in texts
    assert "đè lên" in texts
