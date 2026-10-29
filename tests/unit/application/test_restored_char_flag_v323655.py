"""[v3.23.655] Câu được khôi phục ký tự OCR làm rơi (``一`` đầu câu…) → bộ lọc "Nghi ngờ OCR" đưa ra soát."""

from __future__ import annotations

from subtitles_extractor.application.services.ocr_suspicion import SUSPECT_RESTORED_CHAR, ocr_suspicions
from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
    OCR_NOTE_RESTORED,
    mark_restored_events,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    collect_restored_texts,
    restore_dropped_yi_prefix,
)
from subtitles_extractor.application.services.subtitle_editor_service import SubtitleEditorService
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _event(i: int, text: str) -> SubtitleEvent:
    return SubtitleEvent(index=i, text=text, interval=TimeInterval(float(i), float(i) + 0.9), frame_count=20)


def test_restorer_reports_restored_text() -> None:
    candidates = [("直盯着他看", 0.99)] * 6 + [("一直盯着他看", 0.98)] * 4
    with collect_restored_texts() as found:
        out = restore_dropped_yi_prefix("直盯着他看", candidates)
    assert out == "一直盯着他看" and found == {"一直盯着他看"}
    assert restore_dropped_yi_prefix("直盯着他看", candidates) == out   # ngoài khối with: không thu, không lỗi


def test_marked_event_is_suspect_and_text_edit_clears_it() -> None:
    events = [_event(1, "一直盯着他看"), _event(2, "走吧")]
    assert mark_restored_events(events, {"一直盯着他看"}) == 1
    assert events[0].ocr_notes == (OCR_NOTE_RESTORED,)
    assert ocr_suspicions(events).get(0) == [SUSPECT_RESTORED_CHAR]
    service = SubtitleEditorService()
    service.load(events)
    service.update_timing(0, 1.0, 1.5)
    assert service.fast_snapshot().events[0].ocr_notes == (OCR_NOTE_RESTORED,)   # đổi giờ: giữ cờ
    service.update_text(0, "一直盯着她看")
    assert service.fast_snapshot().events[0].ocr_notes == ()                     # đã soát/sửa chữ: bỏ cờ


def test_repository_roundtrip_keeps_notes(tmp_path) -> None:
    from subtitles_extractor.infrastructure.database.sqlite_subtitle_repository import SqliteSubtitleRepository

    repo = SqliteSubtitleRepository(tmp_path / "s.db")
    event = _event(1, "一直盯着他看")
    event.ocr_notes = (OCR_NOTE_RESTORED,)
    repo.save_events("v.mp4", [event, _event(2, "走吧")])
    loaded = repo.load_events("v.mp4")
    assert loaded[0].ocr_notes == (OCR_NOTE_RESTORED,) and loaded[1].ocr_notes == ()


def test_short_restored_text_does_not_flag_other_lines_v323655() -> None:
    """[v3.23.655] Rà soát: khớp chuỗi con làm câu khôi phục ngắn "一个" đánh dấu oan mọi câu chứa nó."""
    events = [_event(1, "我要一个包子"), _event(2, "一个"), _event(3, "第一行\n一直盯着他看")]
    assert mark_restored_events(events, {"一个", "一直盯着他看"}) == 2
    assert [bool(e.ocr_notes) for e in events] == [False, True, True]
