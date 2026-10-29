"""[v3.23.658] Trang "Phim bộ": bảng trạng thái từng tập theo khâu + "Chạy tiếp tất cả" (mặc định dừng sau khâu Dịch)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.application.services.series_pipeline import (
    STAGE_ORDER,
    EpisodeStatus,
    SeriesStage,
    StageState,
    remaining_work,
)
from subtitles_extractor.presentation.view_models.series_pipeline_controller import SeriesPipelineController

logger = logging.getLogger(__name__)

_SETTINGS_KEY_FOLDER = "series/last_folder"
_STATE_ICONS = {
    StageState.DONE: "✓",
    StageState.READY: "▶",
    StageState.BLOCKED: "·",
    StageState.FAILED: "⚠",
}
_STATE_COLORS = {
    StageState.DONE: "#2e7d32",
    StageState.READY: "#1565c0",
    StageState.BLOCKED: "#9e9e9e",
    StageState.FAILED: "#c62828",
}


class SeriesPage(QWidget):
    """Dây chuyền phim bộ: chọn thư mục, xem trạng thái, chạy tiếp các khâu còn thiếu."""

    def __init__(
        self, controller: SeriesPipelineController, translator: Any, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SeriesPage")
        self._controller = controller
        self._translator = translator
        self._tr = translator.translate
        self._build_ui()
        controller.statuses_changed.connect(self._populate)
        controller.stage_started.connect(self._on_stage_started)
        controller.run_finished.connect(self._on_run_finished)
        controller.running_changed.connect(self._on_running_changed)
        last = str(QSettings().value(_SETTINGS_KEY_FOLDER, "", type=str) or "")
        if last and Path(last).is_dir():
            self._open(Path(last))

    # ── giao diện ──────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(self._tr("series.intro"))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        top = QHBoxLayout()
        self._btn_folder = QPushButton(self._tr("series.btn_folder"))
        self._btn_folder.clicked.connect(self._on_pick_folder)
        self._lbl_folder = QLabel(self._tr("series.no_folder"))
        self._btn_refresh = QPushButton(self._tr("series.btn_refresh"))
        self._btn_refresh.clicked.connect(self._controller.refresh)
        top.addWidget(self._btn_folder)
        top.addWidget(self._lbl_folder, 1)
        top.addWidget(self._btn_refresh)
        layout.addLayout(top)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel(self._tr("series.lbl_stop_after")))
        self._cb_stop = QComboBox()
        self._cb_stop.addItem(self._tr("series.stop_translate"), SeriesStage.TRANSLATE.value)
        self._cb_stop.addItem(self._tr("series.stop_tts"), SeriesStage.TTS.value)
        self._cb_stop.addItem(self._tr("series.stop_publish"), SeriesStage.PUBLISH.value)
        self._cb_stop.setToolTip(self._tr("series.tip_stop_after"))
        self._cb_stop.currentIndexChanged.connect(self._on_stop_changed)
        run_row.addWidget(self._cb_stop)
        run_row.addStretch()
        self._btn_run = QPushButton(self._tr("series.btn_run"))
        self._btn_run.setToolTip(self._tr("series.tip_run"))
        self._btn_run.clicked.connect(self._on_run)
        self._btn_stop = QPushButton(self._tr("series.btn_stop"))
        self._btn_stop.clicked.connect(self._controller.stop)
        run_row.addWidget(self._btn_run)
        run_row.addWidget(self._btn_stop)
        layout.addLayout(run_row)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        layout.addWidget(self._lbl_status)

        self._table = QTableWidget(0, 1 + len(STAGE_ORDER) + 1)
        self._table.setHorizontalHeaderLabels([
            self._tr("series.col_episode"), self._tr("series.col_extract"), self._tr("series.col_translate"),
            self._tr("series.col_tts"), self._tr("series.col_publish"), self._tr("series.col_note"),
        ])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(len(STAGE_ORDER) + 1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, 1)
        self._on_running_changed(False)

    # ── sự kiện ────────────────────────────────────────────────────────────
    def _on_pick_folder(self) -> None:
        start = str(self._controller.folder or "")
        folder = QFileDialog.getExistingDirectory(self, self._tr("series.btn_folder"), start)
        if folder:
            self._open(Path(folder))

    def _open(self, folder: Path) -> None:
        if self._controller.is_running:
            return
        self._controller.open_folder(folder)
        QSettings().setValue(_SETTINGS_KEY_FOLDER, str(folder))
        self._lbl_folder.setText(str(folder))
        index = self._cb_stop.findData(self._controller.state.stop_after.value)
        self._cb_stop.blockSignals(True)
        self._cb_stop.setCurrentIndex(max(0, index))
        self._cb_stop.blockSignals(False)
        if self._controller.state.running:
            # Lượt trước chưa xong (tắt app giữa chừng): nhắc chạy tiếp.
            self._lbl_status.setText(self._tr("series.resume_hint"))

    def _on_stop_changed(self, _index: int) -> None:
        value = self._cb_stop.currentData()
        if value:
            self._controller.set_stop_after(SeriesStage(value))
            self._populate(self._controller.statuses)

    def _on_run(self) -> None:
        error = self._controller.run()
        if error:
            QMessageBox.information(self, self._tr("series.title"), error)

    def _on_stage_started(self, stage: str, count: int) -> None:
        label = self._tr(f"series.col_{stage}")
        self._lbl_status.setText(self._tr("series.running_stage", stage=label, n=count))
        self._populate(self._controller.statuses)

    def _on_run_finished(self, message: str) -> None:
        self._lbl_status.setText(message.splitlines()[0] if message else "")
        QMessageBox.information(self, self._tr("series.title"), message)

    def _on_running_changed(self, running: bool) -> None:
        self._btn_run.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._btn_folder.setEnabled(not running)
        self._btn_refresh.setEnabled(not running)
        self._cb_stop.setEnabled(not running)

    # ── bảng ───────────────────────────────────────────────────────────────
    def _populate(self, statuses: list[EpisodeStatus]) -> None:
        running = self._controller.current_stage
        failures = self._controller.state.failures
        self._table.setRowCount(len(statuses))
        for row, status in enumerate(statuses):
            self._table.setItem(row, 0, QTableWidgetItem(status.name))
            for col, stage in enumerate(STAGE_ORDER, start=1):
                state = status.states[stage]
                text = _STATE_ICONS[state]
                if running is stage and state in (StageState.READY, StageState.FAILED):
                    text = "⏳"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setForeground(QColor(_STATE_COLORS[state]))
                output = status.outputs.get(stage)
                item.setToolTip(str(output) if output else self._tr(f"series.state_{state.value}"))
                self._table.setItem(row, col, item)
            note = "; ".join(f"{self._tr(f'series.col_{k}')}: {v}" for k, v in failures.get(status.name, {}).items())
            self._table.setItem(row, len(STAGE_ORDER) + 1, QTableWidgetItem(note))
        if not self._controller.is_running and statuses:
            todo = remaining_work(statuses, self._controller.state.stop_after)
            parts = [f"{self._tr(f'series.col_{s.value}')} {n}" for s, n in todo.items() if n]
            done = sum(1 for s in statuses if s.next_stage() is None)
            if not self._controller.state.running:
                self._lbl_status.setText(self._tr(
                    "series.summary", n=len(statuses), done=done,
                    todo=", ".join(parts) if parts else self._tr("series.nothing_todo"),
                ))


__all__ = ["SeriesPage"]
