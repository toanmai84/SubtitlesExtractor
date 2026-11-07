"""[v3.23.677] Nút "▾" mở menu thay cho một dãy nút ít dùng.

Trang Lồng tiếng từng hiện 22 nút dưới bảng dòng (ảnh 1366×768: ba hàng nút xám khi chưa có dữ liệu). Các nút ít dùng
nay nằm trong menu, nhưng ĐỐI TƯỢNG NÚT VẪN GIỮ NGUYÊN (ẩn): mọi logic bật/tắt, nhãn đếm ("Soát người nói (3)") và
test cũ dùng nút vẫn chạy. Mỗi lần mở menu, các mục được dựng lại theo trạng thái hiện thời của nút; bấm mục = bấm nút.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtWidgets import QAbstractButton, QMenu, QToolButton, QWidget


class MenuButton(QToolButton):
    """``QToolButton`` bấm là mở menu; mục menu phản chiếu các nút (ẩn) được giao."""

    def __init__(self, title: str, buttons: Sequence[QAbstractButton], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(f"{title} ▾")
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        # Chữ đã có "▾": ẩn mũi tên nhỏ Qt tự vẽ (tránh hai mũi tên).
        self.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0px; }")
        self._buttons = list(buttons)
        for button in self._buttons:
            button.setParent(self)
            button.hide()
        self._menu = QMenu(self)
        self._menu.setToolTipsVisible(True)
        self._menu.aboutToShow.connect(self._rebuild)
        self.setMenu(self._menu)
        self._rebuild()

    @property
    def buttons(self) -> list[QAbstractButton]:
        return list(self._buttons)

    def _rebuild(self) -> None:
        self._menu.clear()
        for button in self._buttons:
            action = self._menu.addAction(button.text())
            action.setEnabled(button.isEnabled())
            action.setToolTip(button.toolTip())
            action.triggered.connect(lambda _checked=False, b=button: b.click())


__all__ = ["MenuButton"]
