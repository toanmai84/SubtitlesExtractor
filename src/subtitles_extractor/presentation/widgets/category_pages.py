"""[v3.23.674] Danh mục bên trái + trang nội dung bên phải (thay một hàng tab quá dài).

Trang Cài đặt có 11 tab ngang ("OCR / Phần cứng", "AI Ngữ nghĩa (NLP)", "Mpv nâng cao", "ROI", "Ngưỡng nhạy"…) xếp
không theo nhóm. Widget này đặt các mục vào danh sách dọc có tiêu đề nhóm, giữ API tối thiểu giống ``QTabWidget``
(``addTab``, ``count``, ``widget``, ``currentIndex``, ``setCurrentIndex``, ``tabText``).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QWidget,
)

_PAGE_ROLE = Qt.ItemDataRole.UserRole + 1


class CategoryPages(QWidget):
    """Danh sách mục (có tiêu đề nhóm) + ``QStackedWidget`` các trang."""

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None, list_width: int = 210) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._list = QListWidget(self)
        self._list.setObjectName("categoryList")
        self._list.setFixedWidth(list_width)
        self._list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._stack = QStackedWidget(self)
        layout.addWidget(self._list)
        layout.addWidget(self._stack, 1)
        self._titles: list[str] = []
        self._list.currentRowChanged.connect(self._on_row_changed)

    # ── API kiểu QTabWidget ──
    def addSection(self, title: str) -> None:  # noqa: N802 — cùng kiểu tên Qt
        """Chèn tiêu đề nhóm (không chọn được)."""
        item = QListWidgetItem(title)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
        item.setFont(font)
        self._list.addItem(item)

    def addTab(self, page: QWidget, title: str) -> int:  # noqa: N802 — cùng tên QTabWidget
        index = self._stack.addWidget(page)
        self._titles.append(title)
        item = QListWidgetItem("   " + title)
        item.setData(_PAGE_ROLE, index)
        self._list.addItem(item)
        if index == 0:
            self._list.setCurrentItem(item)
        return index

    def count(self) -> int:
        return self._stack.count()

    def widget(self, index: int) -> QWidget | None:
        return self._stack.widget(index)

    def tabText(self, index: int) -> str:  # noqa: N802 — cùng tên QTabWidget
        return self._titles[index] if 0 <= index < len(self._titles) else ""

    def currentIndex(self) -> int:  # noqa: N802 — cùng tên QTabWidget
        return self._stack.currentIndex()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 — cùng tên QTabWidget
        for row in range(self._list.count()):
            if self._list.item(row).data(_PAGE_ROLE) == index:
                self._list.setCurrentRow(row)
                return

    def _on_row_changed(self, row: int) -> None:
        item = self._list.item(row)
        if item is None:
            return
        index = item.data(_PAGE_ROLE)
        if index is None:
            return
        self._stack.setCurrentIndex(int(index))
        self.currentChanged.emit(int(index))


__all__ = ["CategoryPages"]
