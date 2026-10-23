"""[v3.23.626] Lồng tiếng: sửa MỐC thời gian, GỘP dòng, TÁCH dòng (bổ sung cho thêm/xoá dòng)."""  # noqa: RUF002

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.dubbing_line_edits import (
    decode_line_edits,
    encode_line_edits,
)
from subtitles_extractor.application.services.dubbing_line_structure import suggest_split
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


class TestSuggestSplit:
    def test_cuts_at_the_punctuation_nearest_the_middle(self) -> None:
        first, second, split = suggest_split("Anh đến rồi, em đợi mãi, sao giờ mới tới", 10.0, 14.0)
        assert (first, second) == ("Anh đến rồi, em đợi mãi,", "sao giờ mới tới")
        assert 10.0 < split < 14.0
        assert split == pytest.approx(10.0 + 4.0 * len(first) / (len(first) + len(second)))

    def test_falls_back_to_space_then_to_the_middle(self) -> None:
        assert suggest_split("một hai ba bốn", 0.0, 4.0)[:2] == ("một hai", "ba bốn")
        first, second, _ = suggest_split("abcdefgh", 0.0, 2.0)
        assert (first, second) == ("abcd", "efgh")
        assert suggest_split("a", 0.0, 2.0) == ("a", "", 1.0)


class TestViewModelEdits:
    def test_timing_edit_updates_event_and_result_and_is_persisted(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        old = vm.source_events[2]
        assert vm.set_line_timing(2, old.start_sec + 0.2, old.end_sec + 0.5)
        assert vm.source_events[2].start_sec == pytest.approx(old.start_sec + 0.2)
        assert vm.last_results[2].start_sec == pytest.approx(old.start_sec + 0.2)
        assert vm.last_results[2].adjusted_start_sec == -1.0
        assert vm.edited_positions == {2}
        assert vm.line_structure_changed                       # mốc đổi cũng được lưu
        saved = decode_line_edits(encode_line_edits(vm.line_edits()))
        assert saved.lines is not None and saved.lines[2].start_sec == pytest.approx(
            old.start_sec + 0.2
        )

    def test_timing_edit_rejects_bad_or_reordering_values(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        events = vm.source_events
        assert not vm.set_line_timing(2, 5.0, 5.0)                       # kết thúc ≤ bắt đầu
        assert not vm.set_line_timing(2, events[1].start_sec - 1, 9999)  # vượt dòng trước
        assert not vm.set_line_timing(2, events[3].start_sec + 1, 9999)  # vượt dòng sau
        assert not vm.set_line_timing(99, 1.0, 2.0)
        assert not vm.line_structure_changed

    def test_merge_adjacent_lines(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        vm.set_speaker_override([P495], HUSBAND)
        assert vm.merge_lines([P495, P496]) == P495
        assert len(vm.source_events) == len(LINES) - 1
        merged = vm.source_events[P495]
        assert merged.text == f"{LINES[P495][7]} {LINES[P496][7]}"
        assert merged.start_sec == pytest.approx(LINES[P495][1])
        assert merged.end_sec == pytest.approx(max(LINES[P495][3], LINES[P496][3]))
        assert vm.speaker_overrides == {P495: HUSBAND}         # dòng đầu giữ người nói
        assert vm.last_results[P495].text == merged.text
        assert [r.event_index for r in vm.last_results] == list(range(len(LINES) - 1))
        assert P495 in vm.edited_positions

    def test_merge_needs_adjacent_lines(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        assert vm.merge_lines([1, 3]) is None
        assert vm.merge_lines([2]) is None
        assert len(vm.source_events) == len(LINES)

    def test_split_line_keeps_the_speaker_on_both_halves(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        vm.set_speaker_override([P496], HUSBAND)
        event = vm.source_events[P496]
        mid = (event.start_sec + event.end_sec) / 2
        assert vm.split_line(P496, "Được,", "em giúp anh.", mid) == P496 + 1
        assert len(vm.source_events) == len(LINES) + 1
        first, second = vm.source_events[P496], vm.source_events[P496 + 1]
        assert (first.text, second.text) == ("Được,", "em giúp anh.")
        assert first.end_sec == pytest.approx(mid) and second.start_sec == pytest.approx(mid)
        assert second.end_sec == pytest.approx(event.end_sec)
        assert vm.speaker_overrides[P496] == vm.speaker_overrides[P496 + 1] == HUSBAND
        assert vm.last_results[P496].text == "Được,"
        assert vm.last_results[P496 + 1].was_skipped          # chờ TTS
        assert {P496, P496 + 1} <= vm.edited_positions

    def test_split_rejects_bad_input(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        event = vm.source_events[2]
        assert vm.split_line(2, "a", "", (event.start_sec + event.end_sec) / 2) is None
        assert vm.split_line(2, "a", "b", event.end_sec + 1) is None
        assert len(vm.source_events) == len(LINES)


class TestPageEdits:
    def test_timing_merge_split_from_the_controller(
        self, page, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        from subtitles_extractor.presentation.pages import tts_line_edits

        controller = page._line_edits
        page._view_model.set_source_events(_events())
        event = page._view_model.source_events[P495]

        monkeypatch.setattr(
            tts_line_edits, "ask_line_timing",
            lambda *_a: (event.start_sec + 0.1, event.end_sec),
        )
        assert controller.edit_timing(P495)
        assert page._view_model.source_events[P495].start_sec == pytest.approx(
            event.start_sec + 0.1
        )
        assert "Đã đổi mốc" in page._stage_lbl.text()

        assert controller.merge_lines([P495, P496]) == P495
        assert page._table.rowCount() == len(LINES) - 1

        monkeypatch.setattr(
            tts_line_edits, "ask_split",
            lambda _p, _t, _l, a, b, start, end, split: (a, b, split),
        )
        assert controller.split_line(P495) == P495 + 1
        assert page._table.rowCount() == len(LINES)

        monkeypatch.setattr(tts_line_edits, "ask_split", lambda *_a: None)
        assert controller.split_line(P495) is None

    def test_button_states_and_menu(self, page) -> None:  # noqa: F811
        controller = page._line_edits
        page._view_model.set_source_events(_events())
        page._table.selectRow(P495)
        assert controller._btn_timing.isEnabled() and controller._btn_split.isEnabled()
        assert not controller._btn_merge.isEnabled()           # chỉ một dòng
        page._table.selectRow(P496)
        page._table.selectRow(P495)
        controller._update_buttons()
        texts = [a.text() for a in controller.build_context_menu([P495, P496]).actions()]
        for prefix in ("Sửa mốc", "Gộp các dòng", "Tách dòng", "Thêm dòng", "Xoá dòng"):
            assert any(t.startswith(prefix) for t in texts), prefix
        assert controller._can_merge([4, 5, 6]) and not controller._can_merge([4, 6])
