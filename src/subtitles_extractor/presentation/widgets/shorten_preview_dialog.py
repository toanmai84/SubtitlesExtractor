"""[v3.23.629] Hộp thoại xem trước các câu AI đề xuất rút gọn — người dùng tick dòng muốn nhận."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.application.use_cases.shorten_for_speech import ShortenedLine


class ShortenPreviewDialog(QDialog):
    """Bảng ``[✓] # | Câu cũ → Câu mới | âm tiết`` — mặc định tick hết."""

    def __init__(
        self, parent: QWidget | None, title: str, intro: str, headers: Sequence[str],
        lines: Sequence[ShortenedLine],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 520)
        self._lines = list(lines)
        self.table = QTableWidget(len(self._lines), 4)
        self.table.setHorizontalHeaderLabels(list(headers))
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setWordWrap(True)
        for row, line in enumerate(self._lines):
            check = QTableWidgetItem(str(line.position + 1))
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 0, check)
            for column, text in ((1, line.before), (2, line.after),
                                 (3, f"{line.syllables_before} → {line.syllables_after}")):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, column, item)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(1, 330)
        self.table.setColumnWidth(2, 330)
        self.table.setColumnWidth(3, 90)
        self.table.resizeRowsToContents()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        note = QLabel(intro)
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(self.table, 1)
        layout.addWidget(buttons)

    def accepted_lines(self) -> list[ShortenedLine]:
        """Các dòng đang được tick."""
        return [
            line for row, line in enumerate(self._lines)
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        ]


__all__ = ["ShortenPreviewDialog"]
