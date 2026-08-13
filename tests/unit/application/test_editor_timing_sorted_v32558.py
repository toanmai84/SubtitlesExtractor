"""[v3.23.158] Test update_timing GIỮ bất biến "danh sách sorted theo start".

Bug: gõ start tuỳ ý qua spinbox/nudge khiến event vượt qua hàng xóm nhưng danh sách
GIỮ NGUYÊN vị trí -> mọi bisect (``find_event_index_at_time``, chèn sub trên waveform)
và duyệt-cặp-kề (``auto_fix_timeline``) chạy trên danh sách KHÔNG sorted -> kết quả
sai tuỳ tiện (highlight sai dòng khi phát video). Fix: sort lại CHỈ khi thứ tự thật
sự bị phá (hot-path kéo waveform đã được widget clamp nên không bao giờ kích hoạt).
"""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _event(index: int, text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(index=index, text=text, interval=TimeInterval(start, end))


def _service() -> SubtitleEditorService:
    service = SubtitleEditorService()
    service.load([
        _event(1, "câu một", 10.0, 12.0),
        _event(2, "câu hai", 20.0, 22.0),
        _event(3, "câu ba", 30.0, 32.0),
    ])
    return service


def test_moving_event_before_neighbor_resorts_list() -> None:
    service = _service()
    moved_uid = service.snapshot_state().events[2].uid  # "câu ba"
    # Kéo "câu ba" (30s) về 5s — vượt qua CẢ hai event trước.
    state = service.update_timing(2, 5.0, 7.0)
    starts = [e.start_sec for e in state.events]
    assert starts == sorted(starts)  # bất biến sorted được khôi phục
    assert state.events[0].text == "câu ba"
    assert state.events[0].uid == moved_uid  # cùng event, chỉ đổi vị trí
    assert [e.index for e in state.events] == [1, 2, 3]  # reindex lại 1..N


def test_find_event_index_correct_after_resort() -> None:
    service = _service()
    service.update_timing(2, 5.0, 7.0)  # "câu ba" về đầu
    # bisect trên _starts_cache phải trả đúng event tại 6s (là "câu ba").
    idx = service.find_event_index_at_time(6.0)
    assert idx == 0
    assert service.snapshot_state().events[idx].text == "câu ba"


def test_normal_timing_change_keeps_position() -> None:
    service = _service()
    state = service.update_timing(1, 19.5, 22.5)  # vẫn nằm giữa 10s và 30s
    assert [e.text for e in state.events] == ["câu một", "câu hai", "câu ba"]


def test_undo_restores_original_order() -> None:
    service = _service()
    service.update_timing(2, 5.0, 7.0)
    state = service.undo()
    assert [e.text for e in state.events] == ["câu một", "câu hai", "câu ba"]
    starts = [e.start_sec for e in state.events]
    assert starts == [10.0, 20.0, 30.0]
