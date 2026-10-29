"""[v3.23.652] TTS: xoá nhầm nhiều dòng phải lấy lại được; ảnh chụp danh sách dòng không đè bản dịch mới."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.dubbing_line_edits import decode_line_edits, encode_line_edits
from subtitles_extractor.presentation.view_models.tts_page_view_model import TTSPageViewModel
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (  # noqa: F401
    HUSBAND,
    LINES,
    _events,
    app,
    page,
)


def _vm() -> TTSPageViewModel:
    vm = TTSPageViewModel(SimpleNamespace())
    vm.set_source_events(_events())
    return vm


def test_restore_deleted_lines_in_place(app: QApplication) -> None:
    vm = _vm()
    original = [e.text for e in vm.source_events]
    vm.delete_lines([1, 2, 5])
    assert [o for o, *_ in vm.missing_original_lines()] == [1, 2, 5]
    assert vm.restore_original_lines([2, 5]) == 2
    texts = [e.text for e in vm.source_events]
    assert texts == [t for i, t in enumerate(original) if i != 1]
    assert [e.index for e in vm.source_events] == list(range(1, len(texts) + 1))
    assert vm.undo()  # một bước hoàn tác cho cả thao tác
    assert len(vm.source_events) == len(LINES) - 3


def test_reset_to_original_after_mass_edits(app: QApplication) -> None:
    vm = _vm()
    original = [(e.text, e.start_sec, e.end_sec) for e in vm.source_events]
    vm.delete_lines(list(range(0, len(LINES) - 2)))
    vm.insert_line(9999.0, 10000.0, "Dòng thêm tay")
    vm.set_speaker_override([0], HUSBAND)
    vm.update_event_text(0, "lời sửa")
    assert vm.reset_to_original(keep_speakers=True)
    assert [(e.text, e.start_sec, e.end_sec) for e in vm.source_events] == original
    assert not vm.line_structure_changed
    assert vm.speaker_overrides == {len(LINES) - 2: HUSBAND}  # người nói của dòng gốc được giữ
    assert vm.undo()
    assert len(vm.source_events) == 3


def test_new_translation_wins_over_saved_snapshot(app: QApplication) -> None:
    vm = _vm()
    vm.delete_lines([0])
    vm.update_event_text(2, "Lời tôi tự sửa")      # dòng gốc 3, sửa tay trên bản dịch cũ
    saved = decode_line_edits(encode_line_edits(vm.line_edits()))

    retranslated = [replace(e, text=f"Bản dịch mới {i}") for i, e in enumerate(_events())]
    fresh = TTSPageViewModel(SimpleNamespace())
    fresh.set_source_events(retranslated)          # dịch lại: cùng mốc giờ → cùng dấu phụ đề
    assert fresh.subtitle_fingerprint == vm.subtitle_fingerprint
    fresh.restore_line_edits(saved)
    assert fresh.restored_structure == (1, 0)
    texts = [e.text for e in fresh.source_events]
    # Trước đây: chữ của bản dịch CŨ trong ảnh chụp đè bản dịch mới, kể cả dòng không ai sửa.
    assert texts == [f"Bản dịch mới {i}" for i in range(1, len(LINES))]


def test_user_text_edit_survives_reload_of_same_translation(app: QApplication) -> None:
    vm = _vm()
    vm.delete_lines([0])
    vm.update_event_text(2, "Lời tôi tự sửa")
    saved = decode_line_edits(encode_line_edits(vm.line_edits()))
    fresh = _vm()
    fresh.restore_line_edits(saved)
    assert fresh.source_events[2].text == "Lời tôi tự sửa"
    assert fresh.has_line_edits([2])


def test_page_restore_buttons_and_bulk_delete_text(page, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    from subtitles_extractor.presentation.pages import tts_line_edits

    controller = page._line_edits
    page._view_model.set_source_events(_events())
    assert not controller._btn_restore_deleted.isEnabled()
    questions: list[str] = []
    monkeypatch.setattr(tts_line_edits, "confirm_delete", lambda _p, _t, text: questions.append(text) or True)
    monkeypatch.setattr(tts_line_edits, "BULK_DELETE_MIN_LINES", 3)
    assert controller.delete_lines([0, 1, 2]) == 3
    assert "3/" in questions[-1] and "Khôi phục" in questions[-1]
    assert controller._btn_restore_deleted.isEnabled()
    assert controller._btn_restore_original.isEnabled()

    monkeypatch.setattr(tts_line_edits, "ask_restore_deleted", lambda _p, _t, _i, lines: [lines[0][0]])
    assert controller.restore_deleted() == 1
    assert page._table.rowCount() == len(LINES) - 2

    monkeypatch.setattr(tts_line_edits, "confirm_restore_original", lambda *_a: True)
    assert controller.restore_original()
    assert page._table.rowCount() == len(LINES)
    assert not controller._btn_restore_deleted.isEnabled()


def test_restore_after_tts_keeps_results_consistent(app: QApplication, tmp_path, monkeypatch) -> None:
    from tests.unit.presentation.test_dubbing_add_delete_lines_v323625 import _view_model

    vm, _started = _view_model(tmp_path, monkeypatch)
    vm.delete_lines([1, 3])
    assert vm.restore_original_lines(None) == 2
    indices = sorted(int(r.event_index) for r in vm._last_results)
    assert indices == list(range(len(LINES)))
    restored = {int(r.event_index): r for r in vm._last_results}
    # [v3.23.654] Audio dòng đã xoá nằm trong thùng rác → dòng chèn lại dùng lại audio, không cần TTS lại.
    assert not restored[1].was_skipped and not restored[3].was_skipped
    assert not ({1, 3} & set(vm.edited_positions))
    assert vm.reset_to_original() is False or len(vm.source_events) == len(LINES)


def test_restored_line_respects_retimed_neighbours(app: QApplication) -> None:
    from subtitles_extractor.application.services.dubbing_line_structure import valid_snapshot

    vm = _vm()
    vm.delete_lines([2])
    vm.set_line_timing(1, 2026.0, 2026.5)          # hợp lệ: dòng kế giờ bắt đầu 2026.733
    assert vm.restore_original_lines([2]) == 1
    starts = [e.start_sec for e in vm.source_events]
    assert starts == sorted(starts)
    assert valid_snapshot(vm.line_edits().lines, len(LINES))  # lưu được, không bị bỏ lặng lẽ lần nạp sau


def test_mass_restore_is_one_refresh(page, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    import time

    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.presentation.pages import tts_line_edits

    events = [SubtitleEvent(index=i + 1, text=f"Câu {i}", interval=TimeInterval(i * 2.0, i * 2.0 + 1.5))
              for i in range(1500)]
    page._view_model.set_source_events(events)
    monkeypatch.setattr(tts_line_edits, "confirm_delete", lambda *_a: True)
    page._line_edits.delete_lines(list(range(100, 1300)))
    emitted: list[int] = []
    page._view_model.structure_changed.connect(lambda: emitted.append(1))
    monkeypatch.setattr(tts_line_edits, "ask_restore_deleted", lambda _p, _t, _i, lines: [o for o, *_ in lines])
    started = time.monotonic()
    assert page._line_edits.restore_deleted() == 1200
    elapsed = time.monotonic() - started
    assert len(emitted) == 1
    assert page._table.rowCount() == 1500
    assert elapsed < 10, elapsed  # trước đây ~25 s cho 600 dòng
