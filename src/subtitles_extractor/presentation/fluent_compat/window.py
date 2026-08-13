"""FluentWindow thay thế — QMainWindow + sidebar điều hướng (PySide6 thuần, LGPL).

[v3.23.267] Thay ``qfluentwidgets.FluentWindow`` (GPL) bằng cửa sổ tự xây: một sidebar bên
trái chứa các nút điều hướng + ``QStackedWidget`` bên phải hiển thị trang tương ứng. Tái
hiện API app dùng: ``addSubInterface``, ``switchTo``, ``stackedWidget``.

Xem docs/LICENSE_ANALYSIS.md.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.presentation.fluent_compat.widgets import (
    NavigationItemPosition,
)


class _NavButton(QPushButton):
    """Nút điều hướng trong sidebar — checkable, gắn với một trang."""

    def __init__(self, text: str, icon: QIcon | None, parent: QWidget) -> None:
        super().__init__(parent)
        self.setText(text)
        if icon is not None and not icon.isNull():
            self.setIcon(icon)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("navButton", True)
        self.setMinimumHeight(40)


class FluentWindow(QMainWindow):
    """Cửa sổ chính với sidebar điều hướng (thay ``qfluentwidgets.FluentWindow``).

    API tương thích phần app dùng:
    - ``self.stackedWidget`` — ``QStackedWidget`` chứa các trang.
    - ``addSubInterface(widget, icon, text, position=...)`` — thêm trang + nút nav.
    - ``switchTo(widget)`` — chuyển sang trang cho trước.
    """

    _SIDEBAR_WIDTH = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──
        self._sidebar = QWidget(central)
        self._sidebar.setObjectName("fluentSidebar")
        self._sidebar.setFixedWidth(self._SIDEBAR_WIDTH)
        self._sidebar_layout = QVBoxLayout(self._sidebar)
        self._sidebar_layout.setContentsMargins(8, 12, 8, 12)
        self._sidebar_layout.setSpacing(4)
        # Khu TOP (mục thường) + stretch + khu BOTTOM (settings/log).
        self._top_slot = QVBoxLayout()
        self._top_slot.setSpacing(4)
        self._bottom_slot = QVBoxLayout()
        self._bottom_slot.setSpacing(4)
        self._sidebar_layout.addLayout(self._top_slot)
        self._sidebar_layout.addStretch(1)
        self._sidebar_layout.addLayout(self._bottom_slot)
        root.addWidget(self._sidebar)

        # ── Khu nội dung ──
        self.stackedWidget = QStackedWidget(central)
        root.addWidget(self.stackedWidget, 1)

        self.setCentralWidget(central)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._button_for_widget: dict[QWidget, _NavButton] = {}

    def addSubInterface(  # noqa: N802 — giữ tên API qfluentwidgets
        self,
        widget: QWidget,
        icon: object = None,
        text: str = "",
        position: NavigationItemPosition = NavigationItemPosition.TOP,
    ) -> None:
        """Thêm một trang vào stack + nút điều hướng vào sidebar.

        Args:
            widget: Trang nội dung (QWidget).
            icon: FluentIcon hoặc QIcon (gọi ``.icon()`` nếu có).
            text: Nhãn nút điều hướng.
            position: TOP (mặc định) hay BOTTOM trong sidebar.
        """
        # widget cần objectName để switchTo/route hoạt động ổn định.
        if not widget.objectName():
            widget.setObjectName(text or f"page_{self.stackedWidget.count()}")
        self.stackedWidget.addWidget(widget)

        qicon = self._resolve_icon(icon)
        button = _NavButton(text, qicon, self._sidebar)
        button.clicked.connect(lambda: self.switchTo(widget))
        self._nav_group.addButton(button)
        self._button_for_widget[widget] = button

        if position == NavigationItemPosition.BOTTOM:
            self._bottom_slot.addWidget(button)
        else:
            self._top_slot.addWidget(button)

        # Trang đầu tiên được chọn mặc định.
        if self.stackedWidget.count() == 1:
            button.setChecked(True)
            self.stackedWidget.setCurrentWidget(widget)

    def switchTo(self, widget: QWidget) -> None:  # noqa: N802 — giữ tên API
        """Chuyển sang trang cho trước + cập nhật trạng thái nút nav."""
        self.stackedWidget.setCurrentWidget(widget)
        button = self._button_for_widget.get(widget)
        if button is not None:
            button.setChecked(True)

    @staticmethod
    def _resolve_icon(icon: object) -> QIcon | None:
        if icon is None:
            return None
        if isinstance(icon, QIcon):
            return icon
        # FluentIcon (shim) có .icon()
        icon_method = getattr(icon, "icon", None)
        if callable(icon_method):
            result = icon_method()
            if isinstance(result, QIcon):
                return result
        return None


__all__ = ["FluentWindow"]
