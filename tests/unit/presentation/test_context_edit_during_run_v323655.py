"""[v3.23.655] Bug #9: sửa ô Visual Cues TRONG LÚC dịch → dịch xong bị kết quả quét bổ sung ghi đè."""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    merge_cues_three_way,
)

BASE = '[{"id":1,"spk":"A","to":"B","cue":""},{"id":2,"spk":"B","to":"A","cue":""}]'
# Lượt dịch quét bổ sung dòng 3 + đổi dòng 2 (người dùng không đụng dòng 2).
THEIRS = '[{"id":1,"spk":"A","to":"B","cue":""},{"id":2,"spk":"C","to":"A","cue":""},{"id":3,"spk":"A","to":"C","cue":""}]'
# Trong lúc dịch người dùng sửa dòng 1.
MINE = '[{"id":1,"spk":"Z","to":"B","cue":""},{"id":2,"spk":"B","to":"A","cue":""}]'


def _spk(raw: str) -> dict[int, str]:
    return {c.line_no: c.speaker for c in deserialize_visual_cues(raw)}


def test_merge_keeps_user_edit_and_new_scan() -> None:
    assert _spk(merge_cues_three_way(BASE, MINE, THEIRS)) == {1: "Z", 2: "C", 3: "A"}


def test_merge_untouched_box_takes_result() -> None:
    assert merge_cues_three_way(BASE, BASE, THEIRS) == THEIRS


def test_merge_user_delete_and_broken_json() -> None:
    mine = '[{"id":2,"spk":"B","to":"A","cue":""}]'          # xoá dòng 1
    assert 1 not in _spk(merge_cues_three_way(BASE, mine, THEIRS))
    assert merge_cues_three_way(BASE, "[{", THEIRS) == "[{"   # sửa dở: không phá


def test_translate_page_merges_topped_up_cues(monkeypatch) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    QApplication.instance() or QApplication([])
    page = TranslatePage(bootstrap_for_gui())
    saved: list[str] = []
    page._view_model.save_context_edits = lambda _path, **kw: saved.append(kw["visual_cues"])
    page._session_video_path = lambda: "/tmp/ep1.mp4"
    page._visual_cues_edit.setPlainText(BASE)
    page._on_busy_changed(True)                 # bắt đầu lượt dịch: chụp nội dung ô
    page._visual_cues_edit.setPlainText(MINE)   # người dùng sửa trong lúc dịch
    page._on_visual_cues_topped_up(THEIRS)      # dịch xong
    page._on_busy_changed(False)
    assert _spk(page._visual_cues_edit.toPlainText()) == {1: "Z", 2: "C", 3: "A"}
    assert saved and _spk(saved[-1]) == {1: "Z", 2: "C", 3: "A"}     # bản ĐÃ GỘP được lưu vào phiên
