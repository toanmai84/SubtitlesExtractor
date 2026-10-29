"""[v3.23.654] Trang TTS sửa được "nói với ai"; giá trị được lưu theo phụ đề, hoàn tác được và gửi về Visual Cues."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.dubbing_line_edits import decode_line_edits, encode_line_edits
from subtitles_extractor.application.services.visual_cue_serializer import (
    apply_user_cues,
    deserialize_visual_cues,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import CUE_ORIGIN_USER
from subtitles_extractor.presentation.pages.tts_line_edits import COLUMN_ADDRESSEE
from subtitles_extractor.presentation.view_models.tts_page_view_model import TTSPageViewModel
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (  # noqa: F401
    HUSBAND,
    LINES,
    P495,
    P496,
    WIFE,
    _events,
    _row_of,
    _source,
    app,
    page,
)


def _vm() -> TTSPageViewModel:
    vm = TTSPageViewModel(SimpleNamespace())
    vm.set_source_events(_events())
    vm.prepare_dubbing(_source())
    return vm


def test_override_wins_over_cues_and_is_undoable(app: QApplication) -> None:  # noqa: F811
    vm = _vm()
    assert vm.line_addressee(P495) == LINES[P495][5]          # theo Visual Cues
    assert vm.set_addressee_override([P495], WIFE) == 1
    assert vm.line_addressee(P495) == WIFE
    assert vm.set_addressee_override([P496], "") == 1         # không nói với ai
    assert vm.line_addressee(P496) == ""
    assert P495 not in vm.edited_positions                    # không cần TTS lại (giọng theo người nói)
    assert vm.undo()
    assert vm.line_addressee(P496) == LINES[P496][5]
    assert vm.revert_line_edits([P495]) == []
    assert vm.line_addressee(P495) == LINES[P495][5]


def test_addressee_persists_and_follows_line_moves(app: QApplication) -> None:  # noqa: F811
    vm = _vm()
    vm.set_addressee_override([P496], WIFE)
    vm.delete_lines([0])                                       # dòng dời lên 1 vị trí
    assert vm.addressee_overrides == {P496 - 1: WIFE}
    saved = decode_line_edits(encode_line_edits(vm.line_edits()))
    fresh = TTSPageViewModel(SimpleNamespace())
    fresh.set_source_events(_events())
    fresh.restore_line_edits(saved)
    assert fresh.addressee_overrides == {P496 - 1: WIFE}


def test_cue_corrections_carry_speaker_and_addressee(app: QApplication) -> None:  # noqa: F811
    vm = _vm()
    vm.set_speaker_override([P495], HUSBAND)
    vm.set_addressee_override([P495], WIFE)
    vm.set_addressee_override([P496], "")
    assert vm.cue_corrections() == {P495 + 1: (HUSBAND, WIFE), P496 + 1: (None, "")}
    raw, changed = apply_user_cues(_source().visual_cues, vm.cue_corrections())
    cues = {c.line_no: c for c in deserialize_visual_cues(raw)}
    assert changed == [P495 + 1, P496 + 1]
    assert (cues[P495 + 1].speaker, cues[P495 + 1].addressee) == (HUSBAND, WIFE)
    assert cues[P496 + 1].speaker == LINES[P496][4] and cues[P496 + 1].addressee == ""
    assert cues[P496 + 1].origin == CUE_ORIGIN_USER


def test_page_column_and_menu(page) -> None:  # noqa: F811
    page._view_model.set_source_events(_events())
    page._dubbing_panel.set_dubbing_enabled(True)
    page._line_edits.refresh()
    assert not page._table.isColumnHidden(COLUMN_ADDRESSEE)
    menu = page._line_edits.build_addressee_menu([P495])
    labels = [action.text() for action in menu.actions()]
    assert any(label.startswith("Tự động") for label in labels)
    assert page._line_edits.apply_addressee([P495], WIFE) == 1
    page._line_edits.refresh()
    row = _row_of(page, P495)
    assert page._table.item(row, COLUMN_ADDRESSEE).text()
    context = [a.text() for a in page._line_edits.build_context_menu([P495]).actions()]
    assert any("Nói với" in text for text in context)
