"""Trang Log — hiển thị nhật ký chạy của ứng dụng theo thời gian thực.

Nhận log qua :class:`QtLogBridge`, cho phép lọc theo cấp độ, tạm dừng cuộn,
xoá và xuất ra file để dễ debug/kiểm tra.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.infrastructure.logging.qt_log_handler import (
    LogEntry,
    QtLogBridge,
)
from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.theme import feedback as _feedback
from subtitles_extractor.presentation.theme import metrics as _m

logger = logging.getLogger(__name__)

_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "SUCCESS": 25, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_LEVEL_COLOR = {
    "DEBUG": "#888888", "INFO": "#e0e0e0", "SUCCESS": "#4caf50",
    "WARNING": "#ffb300", "ERROR": "#ff5252", "CRITICAL": "#ff1744",
}


class LogPage(QWidget):
    """Trang hiển thị log thời gian thực với lọc cấp độ và xuất file."""

    def __init__(self, log_bridge: QtLogBridge, translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LogPage")
        self._translator = translator
        self._bridge = log_bridge
        self._min_level = _LEVEL_ORDER["DEBUG"]
        self._autoscroll = True
        self._build_ui()
        # Nạp log đã có sẵn trong bộ đệm (log phát trước khi mở trang).
        for entry in self._bridge.snapshot():
            self._append_entry(entry, refresh=False)
        self._view.moveCursor(self._view.textCursor().MoveOperation.End)
        self._bridge.record_emitted.connect(self._on_record)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel(self._translator.translate("log.title"))
        title.setStyleSheet(f"font-size:{_m.FONT_SIZE_TITLE}px;font-weight:600;")
        root.addWidget(title)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel(self._translator.translate("log.min_level")))
        self._level_combo = QComboBox()
        self._level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._level_combo.setCurrentText("DEBUG")
        self._level_combo.setToolTip(self._translator.translate("log.level_tip"))
        self._level_combo.currentTextChanged.connect(self._on_level_changed)
        toolbar.addWidget(self._level_combo)

        self._pause_btn = QPushButton(self._translator.translate("log.pause"))
        self._pause_btn.setCheckable(True)
        self._pause_btn.setToolTip(self._translator.translate("log.pause_tip"))
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        toolbar.addWidget(self._pause_btn)

        clear_btn = QPushButton(self._translator.translate("log.clear"))
        clear_btn.setToolTip(self._translator.translate("log.clear_tip"))
        clear_btn.clicked.connect(self._on_clear)
        toolbar.addWidget(clear_btn)

        export_btn = QPushButton(self._translator.translate("log.export"))
        export_btn.setToolTip(self._translator.translate("log.export_tip"))
        export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(export_btn)
        toolbar.addStretch()
        root.addLayout(toolbar)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(20000)
        self._view.setFont(QFont("Consolas", 9))
        self._view.setStyleSheet(f"background:{_c.mono_bg()};color:{_c.mono_fg()};")
        root.addWidget(self._view, stretch=1)

    def _on_level_changed(self, text: str) -> None:
        self._min_level = _LEVEL_ORDER.get(text, 10)
        self._rebuild()

    def _on_pause_toggled(self, checked: bool) -> None:
        self._autoscroll = not checked

    def _on_clear(self) -> None:
        self._bridge.clear()
        self._view.clear()

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, self._translator.translate("log.export_dlg"), "app_log.txt", "Text (*.txt)"
        )
        if not path:
            return
        try:
            Path(path).write_text(self._view.toPlainText(), encoding="utf-8")
            logger.info("Đã xuất nhật ký ra %s", path)
            _feedback.show_success(self, self._translator.translate("log.exported"), self._translator.translate("log.saved_to").replace("{path}", str(path)))
        except OSError as exc:
            logger.error("Không thể xuất nhật ký: %s", exc)
            _feedback.show_error(self, self._translator.translate("log.export_err"), str(exc))

    def _on_record(self, entry: LogEntry) -> None:
        self._append_entry(entry, refresh=True)

    def _rebuild(self) -> None:
        self._view.clear()
        for entry in self._bridge.snapshot():
            self._append_entry(entry, refresh=False)

    def _append_entry(self, entry: LogEntry, refresh: bool) -> None:
        if _LEVEL_ORDER.get(entry.level, 20) < self._min_level:
            return
        color = _LEVEL_COLOR.get(entry.level, "#e0e0e0")
        short_name = entry.name.split(".")[-1] if entry.name else ""
        line = (
            f'<span style="color:#666">{entry.time_str}</span> '
            f'<span style="color:{color};font-weight:600">{entry.level:<7}</span> '
            f'<span style="color:#569cd6">{short_name}</span>: '
            f'<span style="color:{color}">{entry.message}</span>'
        )
        self._view.appendHtml(line)
        if refresh and self._autoscroll:
            self._view.moveCursor(self._view.textCursor().MoveOperation.End)


__all__ = ["LogPage"]
