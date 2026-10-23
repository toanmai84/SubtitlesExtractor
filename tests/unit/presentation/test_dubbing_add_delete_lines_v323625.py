"""[v3.23.625] Lồng tiếng: THÊM / XOÁ dòng (trước đây danh sách dòng cố định theo phụ đề)."""  # noqa: RUF002

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.dubbing_line_edits import (
    LineEdits,
    LineSnapshot,
    TextEdit,
    decode_line_edits,
    encode_line_edits,
    remember_line_edits,
)
from subtitles_extractor.application.services.dubbing_line_structure import (
    default_line_timing,
    delete_mapping,
    insert_mapping,
    insertion_position,
    remap_keys,
    valid_snapshot,
)
from subtitles_extractor.application.use_cases.resynthesize_lines import (
    ResynthesizeLinesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.infrastructure.tts.segment_store import (
    renumber_results,
    segment_path,
)
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (  # noqa: F401
    HUSBAND,
    LINES,
    P495,
    P496,
    _events,
    _row_of,
    app,
    page,
)


class TestPureStructure:
    def test_mappings(self) -> None:
        assert delete_mapping(5, [1, 3]) == {0: 0, 2: 1, 4: 2}
        assert insert_mapping(3, 1) == {0: 0, 1: 2, 2: 3}
        assert remap_keys({0: "a", 1: "b", 2: "c"}, delete_mapping(3, [1])) == {0: "a", 1: "c"}

    def test_insertion_keeps_time_order(self) -> None:
        assert insertion_position([1.0, 2.0, 2.0, 5.0], 2.0) == 3
        assert insertion_position([1.0, 2.0], 0.0) == 0

    def test_default_timing_fits_the_gap(self) -> None:
        starts, ends = [0.0, 5.0, 5.2], [1.0, 4.0, 6.0]
        assert default_line_timing(starts, ends, 0) == (1.0, 3.0)           # khoảng trống rộng
        assert default_line_timing(starts, ends, 1) == (4.0, 5.2)           # không lấn dòng kế
        assert default_line_timing([0.0, 1.1], [1.0, 2.0], 0) == (1.0, 2.5)  # quá hẹp: 1.5s
        assert default_line_timing(starts, ends, 2) == (6.0, 8.0)           # sau dòng cuối
        assert default_line_timing(starts, ends, None) == (0.0, 1.5)   # dòng đầu đã bắt đầu lúc 0
        assert default_line_timing(starts, ends, 2, media_duration_s=6.5) == (6.0, 6.5)

    def test_snapshot_roundtrip_and_validation(self) -> None:
        lines = (LineSnapshot(0, 0.0, 1.0, "a"), LineSnapshot(-1, 1.0, 2.0, "mới"),
                 LineSnapshot(2, 3.0, 4.0, "c"))
        edits = LineEdits(speakers={1: "X"}, texts={0: TextEdit("o", "a")}, lines=lines)
        assert decode_line_edits(encode_line_edits(edits)) == edits
        assert not edits.is_empty
        assert LineEdits(lines=()).is_empty is False  # xoá hết dòng gốc vẫn là một chỉnh sửa
        assert valid_snapshot(lines, 3)
        assert not valid_snapshot(lines, 2)                       # origin 2 vượt phụ đề gốc
        assert not valid_snapshot((lines[0], lines[0]), 3)        # lặp dòng gốc
        bad = encode_line_edits(edits)
        bad["lines"] = [[0, 0.0, 1.0, "a"], ["x"]]
        assert decode_line_edits(bad).lines is None               # hỏng thì bỏ cả danh sách
        store = remember_line_edits({}, "fp", edits)
        assert "fp" in store and remember_line_edits(store, "fp", LineEdits()) == {}


def _segment(tmp: Path, position: int, level: float) -> str:
    path = segment_path(tmp, position)
    sf.write(path, np.full(2400, level, dtype=np.float32), 24000)
    return str(path)


def _results(tmp: Path) -> list[TTSSegmentResult]:
    return [
        TTSSegmentResult(
            event_index=i, start_sec=float(i), end_sec=i + 0.5, text=f"t{i}",
            audio_duration_s=0.1, adjusted_start_sec=float(i),
            segment_audio_path=_segment(tmp, i, 0.1 * (i + 1)),
        )
        for i in range(4)
    ]


class TestRenumberSegments:
    def test_delete_moves_files_and_removes_the_deleted_one(self, tmp_path: Path) -> None:
        results = _results(tmp_path)
        out = renumber_results(results, delete_mapping(4, [1]))
        assert [r.event_index for r in out] == [0, 1, 2]
        assert [r.text for r in out] == ["t0", "t2", "t3"]
        for r in out:                                     # tệp nằm đúng tên theo vị trí mới
            assert r.segment_audio_path == str(segment_path(tmp_path, r.event_index))
        levels = [round(float(sf.read(r.segment_audio_path)[0][0]), 2) for r in out]
        assert levels == [0.1, 0.3, 0.4]                  # audio đi theo ĐÚNG dòng của nó
        assert not segment_path(tmp_path, 3).exists()

    def test_insert_shifts_files_up_without_overwriting(self, tmp_path: Path) -> None:
        out = renumber_results(_results(tmp_path), insert_mapping(4, 1))
        assert [r.event_index for r in out] == [0, 2, 3, 4]
        levels = [round(float(sf.read(r.segment_audio_path)[0][0]), 2) for r in out]
        assert levels == [0.1, 0.2, 0.3, 0.4]
        assert not segment_path(tmp_path, 1).exists()     # chỗ trống cho dòng mới


class TestResynthesizeRemerge:
    def test_remerge_without_tts_drops_deleted_audio(self, tmp_path: Path) -> None:
        results = renumber_results(_results(tmp_path), delete_mapping(4, [3]))
        out = tmp_path / "master.wav"
        sf.write(out, np.ones(24000 * 5, dtype=np.float32), 24000)
        merged = ResynthesizeLinesUseCase(adapter=object()).execute(  # type: ignore[arg-type]
            full_request=TTSRequest(events=[]), selected_indices=set(),
            prior_results=results, output_path=out, sample_rate=24000,
            total_length_samples=24000 * 5, remerge_only=True,
        )
        assert len(merged) == 3
        master, _ = sf.read(out)
        assert abs(master[int(3.05 * 24000)]) < 1e-3     # chỗ dòng đã xoá im lặng
        assert master[int(1.0 * 24000) + 10] > 0.15      # dòng còn lại vẫn có tiếng

    def test_remerge_with_no_audible_line_is_silence(self, tmp_path: Path) -> None:
        out = tmp_path / "master.wav"
        sf.write(out, np.ones(24000 * 5, dtype=np.float32), 24000)
        ResynthesizeLinesUseCase(adapter=object()).execute(  # type: ignore[arg-type]
            full_request=TTSRequest(events=[]), selected_indices=set(), prior_results=[],
            output_path=out, sample_rate=24000, total_length_samples=24000 * 5,
            remerge_only=True,
        )
        assert float(np.abs(sf.read(out)[0]).max()) == 0.0


def _view_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from subtitles_extractor.presentation.view_models.tts_page_view_model import (
        TTSPageViewModel,
    )
    from subtitles_extractor.presentation.workers import resynthesize_worker

    started: list[dict] = []

    class _Worker:
        def __init__(self, _use_case: object, **kwargs: object) -> None:
            started.append(kwargs)
            signal = SimpleNamespace(connect=lambda *_a: None)
            self.progress_changed = self.finished_ok = signal
            self.failed = self.finished = self.cancelled = signal

        def start(self) -> None:
            return None

        def isRunning(self) -> bool:  # noqa: N802
            return False

    monkeypatch.setattr(resynthesize_worker, "ResynthesizeWorker", _Worker)
    output = tmp_path / "out.wav"
    sf.write(output, np.zeros(2400, dtype=np.float32), 24000)
    view_model = TTSPageViewModel(
        SimpleNamespace(make_resynthesize_use_case=lambda _adapter: object())
    )
    view_model.set_source_events(_events())
    count = len(LINES)
    view_model._last_adapter = object()
    view_model._last_output_path = output
    view_model._last_results = [
        TTSSegmentResult(
            event_index=i, start_sec=LINES[i][1], end_sec=LINES[i][2], text=LINES[i][7],
            audio_duration_s=0.1, segment_audio_path=_segment(tmp_path, i, 0.01 * (i + 1)),
        )
        for i in range(count)
    ]
    view_model._last_request = TTSRequest(
        events=_events(), line_voices=tuple(f"v{i}" for i in range(count))
    )
    return view_model, started


class TestViewModel:
    def test_delete_shifts_every_per_position_state(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        vm.set_speaker_override([P496], HUSBAND)
        vm.update_event_text(P496, "Được, em giúp anh.")
        assert vm.delete_lines([0, 1]) == 2
        assert len(vm.source_events) == len(LINES) - 2
        assert vm.speaker_overrides == {P496 - 2: HUSBAND}
        assert set(vm.line_edits().texts) == {P496 - 2}
        assert vm.edited_positions == {P496 - 2}
        assert [e.index for e in vm.source_events] == list(range(1, len(LINES) - 1))
        assert [r.event_index for r in vm.last_results] == list(range(len(LINES) - 2))
        assert vm.last_results[P496 - 2].text == LINES[P496][7]
        assert vm.last_request.line_voices[P496 - 2] == f"v{P496}"
        assert vm.line_structure_changed

    def test_cannot_delete_everything_or_while_busy(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        assert vm.delete_lines(range(len(LINES))) == 0
        vm._is_busy = True
        assert vm.delete_lines([0]) == 0
        assert vm.insert_line(1.0, 2.0, "x") is None

    def test_insert_goes_in_time_order_with_a_placeholder_result(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vm, started = _view_model(tmp_path, monkeypatch)
        vm.set_speaker_override([P496], HUSBAND)
        start = LINES[2][2] + 0.1
        position = vm.insert_line(start, start + 1.0, "  Lời mới  ")
        assert position == 3
        assert vm.source_events[position].text == "Lời mới"
        assert vm.source_events[position].start_sec == start
        assert vm.speaker_overrides == {P496 + 1: HUSBAND}
        assert vm.edited_positions == {position, P496 + 1}
        placeholder = vm.last_results[position]
        assert placeholder.was_skipped and placeholder.text == "Lời mới"
        assert vm.last_results[position + 1].text == LINES[3][7]
        assert vm.last_request.line_voices[position] == ""
        assert vm.last_request.line_voices[position + 1] == "v3"
        # TTS lại dòng mới: yêu cầu mang đúng danh sách dòng mới.
        assert vm.resynthesize_lines({position})
        request = started[-1]["full_request"]
        assert request.events[position].text == "Lời mới"
        assert len(request.events) == len(LINES) + 1

    def test_insert_validates_input(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vm, _ = _view_model(tmp_path, monkeypatch)
        assert vm.insert_line(5.0, 5.0, "x") is None
        assert vm.insert_line(5.0, 6.0, "   ") is None
        assert vm.insert_line(-1.0, 2.0, "x") is None
        assert len(vm.source_events) == len(LINES)

    def test_remerge_starts_a_worker_without_selected_lines(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vm, started = _view_model(tmp_path, monkeypatch)
        vm.delete_lines([1])
        assert vm.remerge_master()
        assert started[-1]["remerge_only"] is True
        assert started[-1]["selected_indices"] == set()
        assert len(started[-1]["prior_results"]) == len(LINES) - 1

    def test_structure_persists_and_returns_with_the_same_subtitle(
        self, app: QApplication
    ) -> None:
        from subtitles_extractor.presentation.view_models.tts_page_view_model import (
            TTSPageViewModel,
        )

        vm = TTSPageViewModel(SimpleNamespace())
        vm.set_source_events(_events())
        vm.delete_lines([0])
        position = vm.insert_line(LINES[3][2] + 0.1, LINES[3][2] + 1.0, "Dòng thêm")
        vm.set_speaker_override([position], HUSBAND)
        saved = decode_line_edits(encode_line_edits(vm.line_edits()))

        fresh = TTSPageViewModel(SimpleNamespace())
        fresh.set_source_events(_events())               # cùng phụ đề gốc nạp lại
        assert fresh.subtitle_fingerprint == vm.subtitle_fingerprint
        fresh.restore_line_edits(saved)
        assert [e.text for e in fresh.source_events] == [e.text for e in vm.source_events]
        assert [e.start_sec for e in fresh.source_events] == [
            e.start_sec for e in vm.source_events
        ]
        assert fresh.speaker_overrides == {position: HUSBAND}
        assert fresh.line_structure_changed
        assert [e.index for e in fresh.source_events] == list(range(1, len(LINES) + 1))


class TestPage:
    def test_buttons_and_delete_with_confirmation(
        self, page, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        from subtitles_extractor.presentation.pages import tts_line_edits

        controller = page._line_edits
        page._view_model.set_source_events(_events())
        assert controller._btn_add_line.isEnabled()
        assert not controller._btn_delete_lines.isEnabled()
        page._table.selectRow(_row_of(page, P495))
        assert controller._btn_delete_lines.isEnabled()

        monkeypatch.setattr(tts_line_edits, "confirm_delete", lambda *_a: False)
        assert controller.delete_lines([P495]) == 0
        assert page._table.rowCount() == len(LINES)

        monkeypatch.setattr(tts_line_edits, "confirm_delete", lambda *_a: True)
        assert controller.delete_lines([P495, P496]) == 2
        assert page._table.rowCount() == len(LINES) - 2
        assert page._table.item(P495, 2).text() == LINES[P496 + 1][7]

    def test_add_line_dialog_flow(self, page, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
        from subtitles_extractor.presentation.pages import tts_line_edits
        from subtitles_extractor.presentation.widgets.add_line_dialog import NewLineRequest

        controller = page._line_edits
        page._dubbing_panel.set_dubbing_enabled(True)
        page._view_model.set_source_events(_events())
        seen: dict = {}

        def fake_dialog(_parent, _title, _labels, start, end, speakers):
            seen.update(start=start, end=end, speakers=[s[0] for s in speakers])
            return NewLineRequest(start, end, "Dòng thêm tay", HUSBAND)

        monkeypatch.setattr(tts_line_edits, "ask_new_line", fake_dialog)
        position = controller.add_line(P495)
        assert position == P495 + 1
        assert seen["start"] == pytest.approx(LINES[P495][3] if False else _events()[P495].end_sec)
        assert "Chân Kiếm Thanh" in " ".join(seen["speakers"])
        assert page._table.rowCount() == len(LINES) + 1
        assert page._table.item(position, 2).text() == "Dòng thêm tay"
        assert page._view_model.speaker_overrides[position] == HUSBAND
        assert "Đã thêm dòng" in page._stage_lbl.text()

        monkeypatch.setattr(tts_line_edits, "ask_new_line", lambda *_a: None)
        assert controller.add_line(0) is None
        assert page._table.rowCount() == len(LINES) + 1

    def test_context_menu_has_add_and_delete(self, page) -> None:  # noqa: F811
        page._view_model.set_source_events(_events())
        menu = page._line_edits.build_context_menu([P495])
        texts = [a.text() for a in menu.actions()]
        assert any(t.startswith("Thêm dòng") for t in texts)
        assert any(t.startswith("Xoá dòng") for t in texts)


class TestPageWithResults:
    def test_delete_after_tts_rebuilds_results_table(
        self, page, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        from subtitles_extractor.presentation.pages import tts_line_edits

        page._view_model.set_source_events(_events())
        results = [
            TTSSegmentResult(
                event_index=i, start_sec=LINES[i][1], end_sec=LINES[i][2], text=LINES[i][7],
                audio_duration_s=0.5,
            )
            for i in range(len(LINES))
        ]
        page._view_model._last_results = results
        page._on_result_ready(results)
        monkeypatch.setattr(tts_line_edits, "confirm_delete", lambda *_a: True)
        assert page._line_edits.delete_lines([0]) == 1
        # Lần TTS không giữ audio từng dòng → không ghép lại được, người dùng được báo rõ.
        assert "TTS lại" in page._stage_lbl.text()
        assert page._table.rowCount() == len(LINES) - 1
        assert page._table.item(0, 2).text() == LINES[1][7]
        assert [r.event_index for r in page._last_results] == list(range(len(LINES) - 1))
