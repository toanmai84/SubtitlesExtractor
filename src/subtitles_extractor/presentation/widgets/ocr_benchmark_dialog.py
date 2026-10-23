"""[v3.23.645] Hộp thoại "Đo tốc độ OCR": chọn đoạn + kịch bản → đo → bảng kết quả → áp dụng đề xuất.

Chạy ngay trong trang Trích xuất (không cần mở cmd): mỗi kịch bản là một lượt trích xuất thật trên đoạn phim ngắn bằng video, ROI
và cài đặt HIỆN TẠI; kết quả được so với "cấu hình hiện tại" để biết cấu hình nhanh hơn có làm đổi phụ đề hay không.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from subtitles_extractor.application.services.ocr_benchmark import (
    DEFAULT_KEYS,
    DEFAULT_SCENARIOS,
    Recommendation,
    ScenarioResult,
)
from subtitles_extractor.presentation.fluent_compat import PrimaryPushButton, PushButton

if TYPE_CHECKING:
    from subtitles_extractor.infrastructure.i18n.json_translator import JsonTranslator
    from subtitles_extractor.presentation.view_models.extract_page_view_model import ExtractPageViewModel

logger = logging.getLogger(__name__)

__all__ = ["OcrBenchmarkDialog", "format_result_row"]

_COLUMNS = (
    "bench.col_scenario", "bench.col_total", "bench.col_ocr", "bench.col_other", "bench.col_frames",
    "bench.col_batches", "bench.col_events", "bench.col_diff", "bench.col_speed",
)


def format_result_row(
    result: ScenarioResult, reference: ScenarioResult | None,
) -> list[str]:
    """Các ô của một dòng bảng kết quả (hàm thuần, kiểm thử được)."""
    label = result.scenario.label_vi
    if result.error:
        return [label, "—", "—", "—", "—", "—", "—", f"LỖI: {result.error[:60]}", "—"]
    comparison = result.comparison
    if reference is None or result is reference or comparison is None:
        diff = "(mốc so sánh)" if result is reference else "—"
    elif comparison.identical:
        diff = "giống hệt"
    else:
        diff = f"thiếu {comparison.n_missing} · thừa {comparison.n_extra} · lệch ≤{max(comparison.max_start_diff_s, comparison.max_end_diff_s):.2f}s"
    speed = "1.0×"
    if reference is not None and result.wall_s > 0 and reference.wall_s > 0 and not reference.error:
        speed = f"{reference.wall_s / result.wall_s:.2f}×"
    return [
        label, f"{result.wall_s:.1f}", f"{result.ocr_s:.1f}", f"{result.other_s:.1f}", str(result.ocr_frames),
        f"{result.batches} (TB {result.avg_batch:.0f})", str(len(result.events)), diff, speed,
    ]


class OcrBenchmarkDialog(QDialog):
    """Hộp thoại đo A/B OCR; điều khiển :class:`ExtractPageViewModel` và hiện kết quả."""

    def __init__(
        self, view_model: ExtractPageViewModel, translator: JsonTranslator, parent: QDialog | None = None,
    ) -> None:
        super().__init__(parent)
        self._vm = view_model
        self._t = translator
        self._recommendation: Recommendation | None = None
        self._running = False
        self._boxes: dict[str, QCheckBox] = {}
        self.setWindowTitle(self._t.translate("bench.title"))
        self.setMinimumSize(880, 560)
        self._build_ui()
        self._vm.benchmark_progress.connect(self._on_progress)
        self._vm.benchmark_finished.connect(self._on_finished)
        self._vm.benchmark_failed.connect(self._on_failed)
        self._vm.busy_changed.connect(self._refresh_start_enabled)

    # ── Dựng giao diện ──────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        intro = QLabel(self._t.translate("bench.intro"))
        intro.setWordWrap(True)
        root.addWidget(intro)

        options = QHBoxLayout()
        options.addWidget(QLabel(self._t.translate("bench.length")))
        self._spin_seconds = QSpinBox()
        self._spin_seconds.setRange(20, 300)
        self._spin_seconds.setValue(60)
        self._spin_seconds.setSuffix(" s")
        options.addWidget(self._spin_seconds)
        options.addWidget(QLabel(self._t.translate("bench.position")))
        self._cb_position = QComboBox()
        for key, ratio in (("ex_pos_mid", 0.5), ("ex_pos_q1", 0.25), ("ex_pos_q3", 0.75), ("ex_pos_start", 0.05), ("ex_pos_end", 0.95)):
            self._cb_position.addItem(self._t.translate(f"extract.{key}"), ratio)
        options.addWidget(self._cb_position)
        options.addStretch(1)
        root.addLayout(options)

        root.addWidget(QLabel(self._t.translate("bench.scenarios")))
        for scenario in DEFAULT_SCENARIOS:
            box = QCheckBox(scenario.label_vi)
            box.setChecked(scenario.key in DEFAULT_KEYS)
            if scenario.key == "current":
                box.setChecked(True)
                box.setEnabled(False)               # luôn đo: là mốc so sánh
            if scenario.gate:
                box.setToolTip(self._t.translate("bench.gate_warning"))
            self._boxes[scenario.key] = box
            root.addWidget(box)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([self._t.translate(c) for c in _COLUMNS])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self._summary)

        buttons = QHBoxLayout()
        self._btn_start = PrimaryPushButton(self._t.translate("bench.start"))
        self._btn_start.clicked.connect(self._on_start)
        self._btn_cancel = PushButton(self._t.translate("bench.cancel"))
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._vm.cancel_extraction)
        self._btn_apply = PushButton(self._t.translate("bench.apply"))
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_close = PushButton(self._t.translate("bench.close"))
        self._btn_close.clicked.connect(self.reject)
        for button in (self._btn_start, self._btn_cancel, self._btn_apply):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self._btn_close)
        root.addLayout(buttons)

    # ── Hành động ───────────────────────────────────────────────────────────
    def _refresh_start_enabled(self, *_: object) -> None:
        """Bắt đầu chỉ bấm được khi không đo VÀ view model đã hết bận (``finished`` đến trước khi luồng kết thúc)."""
        self._btn_start.setEnabled(not self._running and not self._vm.is_busy)

    def _selected_keys(self) -> list[str]:
        return [key for key, box in self._boxes.items() if box.isChecked()]

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._refresh_start_enabled()
        self._btn_cancel.setEnabled(running)
        self._btn_close.setEnabled(not running)
        self._spin_seconds.setEnabled(not running)
        self._cb_position.setEnabled(not running)
        for key, box in self._boxes.items():
            box.setEnabled(not running and key != "current")
        self._progress.setVisible(running)

    def _on_start(self) -> None:
        self._table.setRowCount(0)
        self._summary.setText("")
        self._recommendation = None
        self._btn_apply.setEnabled(False)
        ratio = self._cb_position.currentData()
        started = self._vm.start_ocr_benchmark(
            self._selected_keys(), probe_seconds=float(self._spin_seconds.value()),
            center_ratio=float(ratio) if ratio is not None else 0.5,
        )
        if not started:
            self._status.setText(self._t.translate("bench.cannot_start"))
            return
        self._set_running(True)
        self._progress.setRange(0, 0)
        self._status.setText(self._t.translate("bench.running"))

    def _on_progress(self, current: int, total: int, message: str) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(min(current, total))
        self._status.setText(message)

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._status.setText(self._t.translate("bench.failed").replace("{error}", message))

    def _on_finished(
        self, results: Sequence[ScenarioResult], recommendation: Recommendation | None, cancelled: bool,
    ) -> None:
        self._set_running(False)
        self._fill_table(results)
        self._recommendation = recommendation
        if cancelled:
            self._status.setText(self._t.translate("bench.cancelled"))
        else:
            self._status.setText(self._t.translate("bench.done"))
        self._summary.setText(self._summary_html(results, recommendation, cancelled))
        self._btn_apply.setEnabled(bool(recommendation and recommendation.changes_anything))

    def _fill_table(self, results: Sequence[ScenarioResult]) -> None:
        reference = next((r for r in results if r.scenario.key == "current" and not r.error), None)
        self._table.setRowCount(len(results))
        for row, result in enumerate(results):
            for col, text in enumerate(format_result_row(result, reference)):
                item = QTableWidgetItem(text)
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, col, item)

    def _summary_html(
        self, results: Sequence[ScenarioResult], recommendation: Recommendation | None, cancelled: bool,
    ) -> str:
        reference = next((r for r in results if r.scenario.key == "current" and not r.error), None)
        if cancelled and recommendation is None:
            return self._t.translate("bench.sum_cancelled")
        if reference is None:
            return self._t.translate("bench.sum_no_reference")
        if not reference.events:
            return self._t.translate("bench.sum_no_subtitles")
        if recommendation is None:
            return self._t.translate("bench.sum_none")
        parts = [f"<b>{self._t.translate('bench.sum_recommend')}</b> {recommendation.label_vi}<br>{recommendation.reason_vi}"]
        for warning in recommendation.warnings_vi:
            parts.append(f"<span style='color:#c27c00'>⚠ {warning}</span>")
        if not recommendation.changes_anything:
            parts.append(self._t.translate("bench.sum_keep"))
        return "<br>".join(parts)

    def _on_apply(self) -> None:
        if self._recommendation is None or not self._recommendation.changes_anything:
            return
        if self._vm.apply_benchmark_recommendation(self._recommendation):
            self._status.setText(self._t.translate("bench.applied"))
            self._btn_apply.setEnabled(False)
        else:
            self._status.setText(self._t.translate("bench.apply_failed"))

    def reject(self) -> None:
        if self._running:                       # đang đo: không cho đóng (Esc / X) — dùng nút Huỷ
            return
        super().reject()
