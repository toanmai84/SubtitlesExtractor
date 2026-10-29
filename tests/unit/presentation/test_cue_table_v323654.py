"""[v3.23.654] Bảng sửa Visual Cues ở trang Dịch: sửa bằng tên Việt, gán/đảo hàng loạt, dòng sửa thành cue người dùng."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.visual_cue_serializer import deserialize_visual_cues
from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import CUE_ORIGIN_USER
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.presentation.widgets.cue_table_dialog import (
    COL_ADDRESSEE,
    COL_ISSUE,
    COL_SPEAKER,
    CueTableDialog,
    display_names,
)

ROSTER = "林逸 (Lâm Dật) [nam, ~20] — chính\n苏瑶 (Tô Dao) [nữ, ~18] — sư muội"
CUES = '[{"id":1,"spk":"林逸","to":"苏瑶","cue":""},{"id":2,"spk":"苏瑶","to":"苏瑶","cue":"giận"}]'
ROWS = [(0.0, "师妹你来了"), (2.0, "师兄"), (4.0, "走吧")]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _dialog() -> CueTableDialog:
    return CueTableDialog(None, CUES, ROWS, ["Sư muội tới rồi", "Sư huynh", "Đi thôi"], ROSTER, lambda key: key)


def test_display_names() -> None:
    assert display_names(ROSTER) == {"林逸": "Lâm Dật (林逸)", "苏瑶": "Tô Dao (苏瑶)"}


def test_issues_shown_and_vietnamese_name_edit(app) -> None:
    dialog = _dialog()
    assert dialog._table.item(1, COL_ISSUE).text()                 # người nói = người nghe
    assert dialog._table.item(0, COL_SPEAKER).text() == "Lâm Dật (林逸)"
    dialog._table.item(1, COL_ADDRESSEE).setText("Lâm Dật")       # gõ tên Việt
    cues = {c.line_no: c for c in deserialize_visual_cues(dialog.result_json())}
    assert cues[2].addressee == "林逸" and cues[2].origin == CUE_ORIGIN_USER
    assert cues[1].origin == ""                                     # dòng không sửa giữ nguồn AI
    assert dialog.changed_lines == [2]
    assert dialog.recheck() == 1                                     # còn dòng 3 thiếu người nói


def test_bulk_assign_and_swap(app) -> None:
    dialog = _dialog()
    dialog._table.selectRow(0)
    assert dialog.swap_selected() == 1
    dialog._table.clearSelection()
    dialog._table.selectRow(2)
    assert dialog.assign(COL_SPEAKER, "苏瑶") == 1
    cues = {c.line_no: c for c in deserialize_visual_cues(dialog.result_json())}
    assert (cues[1].speaker, cues[1].addressee) == ("苏瑶", "林逸")
    assert cues[3].speaker == "苏瑶" and cues[3].origin == CUE_ORIGIN_USER


def test_translate_page_applies_table_result(app, monkeypatch) -> None:
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    page = TranslatePage(bootstrap_for_gui())
    saved: list = []
    page._view_model.save_context_edits = lambda path, **k: saved.append(k)
    page._source_video_path = "/tmp/ep1.mp4"
    page._view_model._source_events = [
        SubtitleEvent(index=i + 1, text=text, interval=TimeInterval(start, start + 1.5)) for i, (start, text) in enumerate(ROWS)
    ]
    page._characters_edit.setPlainText(ROSTER)
    page._visual_cues_edit.setPlainText(CUES)
    assert page.refresh_cue_issues() == 2

    def accept(dialog) -> bool:
        dialog._table.selectRow(2)
        dialog.assign(COL_SPEAKER, "林逸")
        return True

    monkeypatch.setattr(TranslatePage, "_exec_cue_dialog", staticmethod(accept))
    assert page._open_cue_table()
    cues = {c.line_no: c for c in deserialize_visual_cues(page._visual_cues_edit.toPlainText())}
    assert cues[3].speaker == "林逸"
    assert saved
    assert page.refresh_cue_issues() == 1


def test_ok_does_not_wipe_cues_synced_while_open(app) -> None:
    """Rà soát độc lập: đồng bộ người nói từ TTS chạy trong lúc bảng mở; OK từng ghi đè bằng bản lúc mở bảng."""
    from subtitles_extractor.application.services.visual_cue_serializer import apply_user_speakers

    dialog = _dialog()
    dialog._table.selectRow(2)
    dialog.assign(COL_SPEAKER, "苏瑶")
    synced, _ = apply_user_speakers(CUES, {1: "苏瑶"})          # trang TTS ghi vào ô cues khi bảng đang mở
    cues = {c.line_no: c for c in deserialize_visual_cues(dialog.merged_into(synced))}
    assert cues[1].speaker == "苏瑶" and cues[1].origin == CUE_ORIGIN_USER
    assert cues[3].speaker == "苏瑶"


def test_recheck_button_follows_suspicious_count_v323655(app, monkeypatch) -> None:
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    page = TranslatePage(bootstrap_for_gui())
    page._view_model._source_events = [
        SubtitleEvent(index=i + 1, text=text, interval=TimeInterval(start, start + 1.5)) for i, (start, text) in enumerate(ROWS)
    ]
    page._characters_edit.setPlainText(ROSTER)
    page._visual_cues_edit.setPlainText(CUES)
    assert page.refresh_cue_issues() == 2
    assert page._btn_cue_recheck.isEnabled()
    calls: list = []
    monkeypatch.setattr(page, "_combined_keys_text", lambda: "KEY")
    monkeypatch.setattr(page, "_selected_video_path", lambda: "/tmp/ep1.mp4")
    monkeypatch.setattr(page._view_model, "start_cue_recheck", lambda **kw: calls.append(kw) or True)
    assert page._on_cue_recheck_clicked()
    assert calls and calls[0]["visual_cues"] == CUES


def test_table_edit_keeps_ai_label_for_auto_release_v323655(app) -> None:
    """[v3.23.655] Rà soát: sửa ở bảng cũng phải cất nhãn AI gốc để "Tự động" ở trang TTS trả về được."""
    from subtitles_extractor.application.services.visual_cue_serializer import CUE_AUTO, apply_user_cues

    dialog = _dialog()
    dialog._table.item(0, COL_SPEAKER).setText("Tô Dao")
    raw, _ = apply_user_cues(dialog.result_json(), {1: (CUE_AUTO, None)})
    cue = {c.line_no: c for c in deserialize_visual_cues(raw)}[1]
    assert (cue.speaker, cue.addressee, cue.origin) == ("林逸", "苏瑶", "")
