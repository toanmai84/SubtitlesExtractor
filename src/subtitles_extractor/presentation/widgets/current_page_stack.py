"""[v3.23.674] Khung "chồng trang" cao theo trang ĐANG HIỆN (thay ``QStackedWidget``).

Trang TTS: ô "Cấu hình Edge TTS" (ngắn) bị kéo cao bằng ô VieNeu (dài nhất) → khoảng trống lớn phía trên tiêu đề.
``QStackedLayout`` tính chiều cao (cả theo bề ngang) bằng MAX qua mọi trang, kể cả trang ẩn. Ở đây các trang nằm trong
một layout dọc và chỉ trang hiện tại được hiện — layout bỏ qua widget ẩn nên khung cao đúng bằng trang đang chọn.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


class CurrentPageStack(QWidget):
    """Tập con API ``QStackedWidget``: ``addWidget``, ``count``, ``widget``, ``currentIndex``, ``setCurrentIndex``,
    ``currentWidget``, ``currentChanged``."""

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._pages: list[QWidget] = []
        self._current = -1

    def addWidget(self, page: QWidget) -> int:  # noqa: N802 — cùng tên QStackedWidget
        self._pages.append(page)
        self._layout.addWidget(page)
        index = len(self._pages) - 1
        if self._current < 0:
            self._current = index
            page.setVisible(True)
        else:
            page.setVisible(False)
        return index

    def count(self) -> int:
        return len(self._pages)

    def widget(self, index: int) -> QWidget | None:
        return self._pages[index] if 0 <= index < len(self._pages) else None

    def currentIndex(self) -> int:  # noqa: N802 — cùng tên QStackedWidget
        return self._current

    def currentWidget(self) -> QWidget | None:  # noqa: N802 — cùng tên QStackedWidget
        return self.widget(self._current)

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 — cùng tên QStackedWidget
        if not 0 <= index < len(self._pages) or index == self._current:
            return
        for i, page in enumerate(self._pages):
            page.setVisible(i == index)
        self._current = index
        self.currentChanged.emit(index)


__all__ = ["CurrentPageStack"]
