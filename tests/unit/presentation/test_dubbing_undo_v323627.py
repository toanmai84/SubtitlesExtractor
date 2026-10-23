"""[v3.23.627] Lồng tiếng: hoàn tác / làm lại chỉnh sửa dòng (trước khi TTS)."""  # noqa: RUF002

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from tests.unit.presentation.test_dubbing_add_delete_lines_v323625 import (  # noqa: F401
    _view_model,
)
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (  # noqa: F401
    HUSBAND,
    LINES,
    P495,
    P496,
    _events,
    app,
    page,
)


def _fresh(app_):  # noqa: ANN001, ANN202
    from types import SimpleNamespace

    from subtitles_extractor.presentation.view_models.tts_page_view_model import (
        TTSPageViewModel,
    )

    vm = TTSPageViewModel(SimpleNamespace())
    vm.set_source_events(_events())
    return vm


class TestUndoRedo:
    def test_each_operation_is_one_undo_step(self, app: QApplication) -> None:  # noqa: F811
        vm = _fresh(app)
        baseline = [(e.text, e.start_sec, e.end_sec) for e in vm.source_events]
        vm.set_speaker_override([P495], HUSBAND)
        vm.update_event_text(P496, "Được, em giúp anh.")
        vm.delete_lines([0])
        pos = vm.insert_line(LINES[3][2] + 0.1, LINES[3][2] + 1.0, "Dòng thêm")
        vm.merge_lines([pos - 1, pos])                       # thao tác lồng nhau = MỘT bước
        vm.split_line(2, "a", "b", (vm.source_events[2].start_sec + vm.source_events[2].end_sec) / 2)
        assert len(vm._undo_stack) == 6
        while vm.can_undo:
            assert vm.undo()
        assert [(e.text, e.start_sec, e.end_sec) for e in vm.source_events] == baseline
        assert not vm.has_line_edits() and vm.edited_positions == frozenset()
        assert not vm.line_structure_changed
        while vm.can_redo:
            assert vm.redo()
        assert len(vm.source_events) == len(LINES) - 1 + 1 - 1 + 1  # xoá, thêm, gộp(-1), tách(+1)
        assert vm.speaker_overrides == {P495: HUSBAND}   # -1 (xoá) +1 (thêm) -1 (gộp) +1 (tách)

    def test_new_edit_clears_redo_and_noops_are_not_recorded(self, app: QApplication) -> None:  # noqa: F811
        vm = _fresh(app)
        vm.update_event_text(P496, LINES[P496][7] + "  ")    # chỉ khoảng trắng: không đổi gì
        assert not vm.can_undo
        vm.set_speaker_override([P495], HUSBAND)
        vm.undo()
        assert vm.can_redo
        vm.set_speaker_override([P496], HUSBAND)
        assert not vm.can_redo and vm.can_undo
        assert vm.set_line_timing(99, 1.0, 2.0) is False and len(vm._undo_stack) == 1

    def test_undo_restores_text_and_speaker_edits(self, app: QApplication) -> None:  # noqa: F811
        vm = _fresh(app)
        vm.update_event_text(P496, "Được, em giúp anh.")
        vm.undo()
        assert vm.source_events[P496].text == LINES[P496][7]
        assert not vm.has_line_edits()
        vm.redo()
        assert vm.source_events[P496].text == "Được, em giúp anh."
        assert vm.line_edits().texts[P496].original == LINES[P496][7]

    def test_no_undo_after_tts_results_exist(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        vm.set_speaker_override([P495], HUSBAND)
        assert not vm.can_undo and not vm._undo_stack         # audio từng dòng không khôi phục được
        vm2 = _fresh(app)
        vm2.set_speaker_override([P495], HUSBAND)
        assert vm2.can_undo
        vm2._on_worker_finished([])                           # kết quả TTS tới: bỏ lịch sử
        assert not vm2._undo_stack and not vm2._redo_stack

    def test_new_subtitle_resets_history(self, app: QApplication) -> None:  # noqa: F811
        vm = _fresh(app)
        vm.set_speaker_override([P495], HUSBAND)
        vm.set_source_events(_events())
        assert not vm.can_undo and not vm.can_redo


class TestUndoOnThePage:
    def test_buttons_and_shortcuts(self, page) -> None:  # noqa: F811
        controller = page._line_edits
        page._view_model.set_source_events(_events())
        assert not controller._btn_undo.isEnabled()
        controller.apply_speaker([P495], HUSBAND)
        controller._update_buttons()
        assert controller._btn_undo.isEnabled() and not controller._btn_redo.isEnabled()
        assert controller.undo()
        assert page._view_model.speaker_overrides == {}
        assert "Đã hoàn tác" in page._stage_lbl.text()
        controller._update_buttons()
        assert controller._btn_redo.isEnabled()
        assert controller.redo()
        assert page._view_model.speaker_overrides == {P495: HUSBAND}

    def test_undo_a_delete_brings_the_row_back(self, page, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
        from subtitles_extractor.presentation.pages import tts_line_edits

        page._view_model.set_source_events(_events())
        monkeypatch.setattr(tts_line_edits, "confirm_delete", lambda *_a: True)
        assert page._line_edits.delete_lines([P495]) == 1
        assert page._table.rowCount() == len(LINES) - 1
        assert page._line_edits.undo()
        assert page._table.rowCount() == len(LINES)
        assert page._table.item(P495, 2).text() == LINES[P495][7]
