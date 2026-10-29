"""[v3.23.659] Dịch vụ Chỉnh sửa: Auto-Fix/Gộp giữ ghi chú OCR, sửa giờ sai không xoá Làm lại, SRT đánh số liên tục."""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_editor_service import SubtitleEditorService
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import SrtExporter


def _ev(i, start, end, text="câu", notes=("restored_char",)):
    return SubtitleEvent(index=i, text=text, interval=TimeInterval(start, end), ocr_notes=notes)


def test_auto_fix_overlap_keeps_ocr_notes():
    svc = SubtitleEditorService()
    svc.load([_ev(1, 0.0, 2.0), _ev(2, 1.5, 3.0)])
    assert svc.auto_fix_timeline() == 1
    assert all(e.ocr_notes == ("restored_char",) for e in svc.snapshot_state().events)


def test_auto_fix_small_gap_keeps_ocr_notes():
    svc = SubtitleEditorService()
    svc.load([_ev(1, 0.0, 2.0), _ev(2, 2.05, 3.0)])
    assert svc.auto_fix_timeline() == 1
    assert all(e.ocr_notes == ("restored_char",) for e in svc.snapshot_state().events)


def test_merge_keeps_notes_of_chosen_text():
    svc = SubtitleEditorService()
    svc.load([_ev(1, 0.0, 1.0, "ab", ()), _ev(2, 1.0, 2.0, "abcd", ("restored_char",))])
    assert svc.apply_merge_groups([[0, 1]]) == 1
    (only,) = svc.snapshot_state().events
    assert only.text == "abcd" and only.ocr_notes == ("restored_char",)


def test_invalid_timing_keeps_redo_history():
    svc = SubtitleEditorService()
    svc.load([_ev(1, 0.0, 1.0), _ev(2, 2.0, 3.0)])
    svc.update_text(0, "mới")
    svc.undo()
    assert svc.snapshot_state().can_redo
    with pytest.raises(Exception):
        svc.update_timing(0, 5.0, 1.0)  # kết thúc trước bắt đầu
    state = svc.snapshot_state()
    assert state.can_redo, "Giờ sai bị từ chối nhưng ngăn Làm lại đã bị xoá."
    assert svc.redo().events[0].text == "mới"


def test_srt_numbering_is_sequential():
    events = [_ev(5, 0.0, 1.0, "a"), _ev(5, 1.0, 2.0, "b"), _ev(0, 2.0, 3.0, "c")]
    content = SrtExporter._build_content(events)
    numbers = [block.split("\n", 1)[0] for block in content.strip().split("\n\n")]
    assert numbers == ["1", "2", "3"]
