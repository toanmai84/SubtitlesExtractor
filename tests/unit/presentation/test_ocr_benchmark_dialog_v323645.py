"""[v3.23.645] Hộp thoại đo tốc độ OCR trong trang Trích xuất."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.services.ocr_benchmark import (
    DEFAULT_SCENARIOS,
    ScenarioResult,
    recommend,
)
from subtitles_extractor.infrastructure.i18n.json_translator import JsonTranslator
from subtitles_extractor.presentation.widgets.ocr_benchmark_dialog import (
    OcrBenchmarkDialog,
    format_result_row,
)

SC = {s.key: s for s in DEFAULT_SCENARIOS}
EVENTS = ((1.0, 2.0, "một"), (3.0, 4.0, "hai"))


def _result(key: str, wall: float, events=EVENTS, error: str = "") -> ScenarioResult:
    return ScenarioResult(scenario=SC[key], wall_s=wall, ocr_s=wall * 0.9, ocr_frames=200, batches=10,
                          events=tuple(events), error=error)


class _FakeVm(QObject):
    benchmark_progress = Signal(int, int, str)
    benchmark_finished = Signal(object, object, bool)
    benchmark_failed = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, can_start: bool = True) -> None:
        super().__init__()
        self.is_busy = False
        self.can_start = can_start
        self.started_with: dict | None = None
        self.applied = None
        self.cancelled = False

    def start_ocr_benchmark(self, keys, probe_seconds=60.0, center_ratio=0.5) -> bool:
        self.started_with = {"keys": keys, "seconds": probe_seconds, "ratio": center_ratio}
        return self.can_start

    def apply_benchmark_recommendation(self, recommendation) -> bool:
        self.applied = recommendation
        return True

    def cancel_extraction(self) -> None:
        self.cancelled = True


@pytest.fixture
def translator() -> JsonTranslator:
    from subtitles_extractor.composition.bootstrap import _resolve_data_dir

    return JsonTranslator(_resolve_data_dir())


def test_row_formatting_marks_identical_lossy_and_failed(qtbot) -> None:
    reference = _result("current", 100)
    identical = _result("det8", 50)
    lossy = _result("gate200", 20, events=EVENTS[:1])
    broken = _result("det4", 0, error="OOM")
    recommend([reference, identical, lossy, broken])                       # điền .comparison
    assert format_result_row(reference, reference)[7] == "(mốc so sánh)"
    row = format_result_row(identical, reference)
    assert row[7] == "giống hệt" and row[8] == "2.00×"
    assert "thiếu 1" in format_result_row(lossy, reference)[7]
    assert format_result_row(broken, reference)[7].startswith("LỖI")


def test_dialog_start_passes_the_selection_and_shows_results(qtbot, translator) -> None:
    vm = _FakeVm()
    dialog = OcrBenchmarkDialog(vm, translator)
    qtbot.addWidget(dialog)
    dialog._boxes["det4"].setChecked(True)
    dialog._spin_seconds.setValue(45)
    dialog._on_start()
    assert vm.started_with is not None
    assert vm.started_with["seconds"] == 45.0 and "current" in vm.started_with["keys"]
    assert "det4" in vm.started_with["keys"] and not dialog._btn_start.isEnabled() and dialog._btn_cancel.isEnabled()

    results = [_result("current", 100), _result("det8", 50), _result("gate200", 30)]
    rec = recommend(results)
    vm.benchmark_progress.emit(50, 300, "Kịch bản 2/3: det8")
    assert "Kịch bản 2/3" in dialog._status.text()
    vm.benchmark_finished.emit(results, rec, False)
    assert dialog._table.rowCount() == 3 and dialog._btn_start.isEnabled()
    assert dialog._btn_apply.isEnabled() and "Đề xuất" in dialog._summary.text()
    dialog._on_apply()
    assert vm.applied is rec and not dialog._btn_apply.isEnabled()


def test_dialog_cannot_start_message_and_no_apply_without_changes(qtbot, translator) -> None:
    vm = _FakeVm(can_start=False)
    dialog = OcrBenchmarkDialog(vm, translator)
    qtbot.addWidget(dialog)
    dialog._on_start()
    assert dialog._btn_start.isEnabled() and "Chưa thể đo" in dialog._status.text()
    vm.benchmark_finished.emit([_result("current", 100), _result("det4", 98)],
                               recommend([_result("current", 100), _result("det4", 98)]), False)
    assert not dialog._btn_apply.isEnabled() and "không cần đổi" in dialog._summary.text() or "Không có thay đổi" in dialog._summary.text()


def test_dialog_handles_cancel_failure_and_empty_stretch(qtbot, translator) -> None:
    vm = _FakeVm()
    dialog = OcrBenchmarkDialog(vm, translator)
    qtbot.addWidget(dialog)
    dialog._on_start()
    dialog.reject()                                                       # đang đo: không đóng được
    assert dialog.isVisible() is False or dialog._running
    vm.benchmark_failed.emit("lỗi gì đó")
    assert not dialog._running and "lỗi gì đó" in dialog._status.text()
    vm.benchmark_finished.emit([_result("current", 100, events=())], None, False)
    assert "không có phụ đề" in dialog._summary.text()
    vm.benchmark_finished.emit([_result("current", 100)], None, True)
    assert "huỷ" in dialog._status.text().lower()
    assert not dialog._btn_apply.isEnabled()


def test_current_scenario_is_always_measured(qtbot, translator) -> None:
    dialog = OcrBenchmarkDialog(_FakeVm(), translator)
    qtbot.addWidget(dialog)
    assert dialog._boxes["current"].isChecked() and not dialog._boxes["current"].isEnabled()


def test_start_stays_disabled_until_the_view_model_is_idle(qtbot, translator) -> None:
    """[v3.23.645] ``finished`` đến trước khi luồng kết thúc: bấm lại ngay sẽ gặp "không bắt đầu được"."""
    vm = _FakeVm()
    dialog = OcrBenchmarkDialog(vm, translator)
    qtbot.addWidget(dialog)
    dialog._on_start()
    vm.is_busy = True
    vm.benchmark_finished.emit([_result("current", 10)], None, False)
    assert not dialog._btn_start.isEnabled()
    vm.is_busy = False
    vm.busy_changed.emit(False)
    assert dialog._btn_start.isEnabled()


def test_applying_a_recommendation_tells_the_settings_page_to_reload(qtbot) -> None:
    """[v3.23.645] Trang Cài đặt chỉ đổ form một lần: không nạp lại thì lần Lưu sau hoàn tác đề xuất."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from subtitles_extractor.application.services.ocr_benchmark import Recommendation
    from subtitles_extractor.presentation.view_models.extract_page_view_model import (
        ExtractPageViewModel,
    )

    container = MagicMock()
    vm = ExtractPageViewModel(container)
    seen: list[bool] = []
    vm.benchmark_settings_applied.connect(lambda: seen.append(True))
    rec = Recommendation(key="det8", label_vi="x", reason_vi="y", settings_patch={"hardware": {"det_batch_size": 8}})
    assert vm.apply_benchmark_recommendation(rec) is True
    container.settings_service.update.assert_called_once_with(hardware={"det_batch_size": 8})
    assert seen == [True]
    container.settings_service.update.side_effect = OSError("đĩa đầy")
    assert vm.apply_benchmark_recommendation(rec) is False and seen == [True]
    assert vm.apply_benchmark_recommendation(SimpleNamespace(settings_patch={})) is False
