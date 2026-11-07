"""[v3.23.674] Bố cục TỰ XUỐNG DÒNG cho hàng nút.

Hàng nút dựng bằng ``QHBoxLayout`` bị ép bề ngang khi cột hẹp (màn hình 1366×768): chữ trên nút bị cắt ("Quản lý giai
đoạ", "Chọn dòng bất thườ"). ``FlowLayout`` xếp nút trái → phải, hết chỗ thì xuống hàng — nút luôn đủ chữ.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget, QWidgetItem


class FlowLayout(QLayout):
    """Xếp widget theo hàng, tự xuống dòng khi hết chiều ngang (có height-for-width)."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    # ── API QLayout ──
    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 — override Qt
        self._items.append(item)

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802 — override Qt
        self.addChildWidget(widget)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 — override Qt
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 — override Qt
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 — override Qt
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 — override Qt
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 — override Qt
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 — override Qt
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 — override Qt
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 — override Qt
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                # Bố cục đặt nút theo sizeHint → bề rộng tối thiểu cũng theo sizeHint (không cắt nút rộng nhất).
                size = size.expandedTo(item.sizeHint())
        left, top, right, bottom = self.getContentsMargins()
        return size + QSize(left + right, top + bottom)

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        area = rect.adjusted(left, top, -right, -bottom)
        x, y, line_height = area.x(), area.y(), 0
        for item in self._items:
            widget = item.widget()
            if item.isEmpty() or (widget is not None and widget.isHidden()):
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > area.right() + 1 and line_height > 0:
                x = area.x()
                y += line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + bottom


def flow_row(parent: QWidget | None = None, spacing: int = 6) -> tuple[QWidget, FlowLayout]:
    """Tạo một widget chứa ``FlowLayout`` (cao theo bề ngang) để đặt vào layout dọc."""
    holder = QWidget(parent)
    layout = FlowLayout(holder, spacing)
    policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    holder.setSizePolicy(policy)
    return holder, layout


__all__ = ["FlowLayout", "flow_row"]
