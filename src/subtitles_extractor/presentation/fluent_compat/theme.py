"""Hệ thống theme + InfoBar thay thế qfluentwidgets bằng Qt thuần (PySide6, LGPL).

[v3.23.267] Thay ``setTheme``/``Theme``/``themeColor``/``isDarkTheme`` và ``InfoBar`` của
qfluentwidgets (GPL) bằng cài đặt tự viết. Xem docs/LICENSE_ANALYSIS.md.

- Theme sáng/tối: lưu trạng thái toàn cục + áp QSS lên QApplication.
- themeColor: màu accent mặc định (xanh dương Fluent).
- InfoBar: toast góc màn hình tự tắt sau vài giây (thay InfoBar của qfluentwidgets).
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class Theme(Enum):
    """Chế độ theme (thay ``qfluentwidgets.Theme``)."""

    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


# Trạng thái theme toàn cục (module-level, đơn giản như qfluentwidgets).
_current_theme: Theme = Theme.AUTO
_ACCENT_COLOR = QColor(0, 120, 212)  # xanh dương Fluent mặc định


def setTheme(theme: Theme) -> None:  # noqa: N802 — giữ tên API qfluentwidgets
    """Đặt theme hiện tại (thay ``qfluentwidgets.setTheme``).

    Lưu trạng thái để :func:`isDarkTheme` phản ánh đúng. Việc áp QSS chi tiết do lớp theme
    ứng dụng (presentation/theme) đảm nhiệm — hàm này giữ nguồn sự thật về chế độ.
    """
    global _current_theme
    _current_theme = theme


def isDarkTheme() -> bool:  # noqa: N802 — giữ tên API qfluentwidgets
    """True nếu đang ở theme tối (thay ``qfluentwidgets.isDarkTheme``).

    AUTO dò theo bảng màu hệ thống (nền cửa sổ tối => dark).
    """
    if _current_theme == Theme.DARK:
        return True
    if _current_theme == Theme.LIGHT:
        return False
    # AUTO: dò theo palette hệ thống.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return False
    window = app.palette().color(app.palette().ColorRole.Window)
    # Độ sáng thấp => theme tối.
    return window.lightness() < 128


def themeColor() -> QColor:  # noqa: N802 — giữ tên API qfluentwidgets
    """Màu accent hiện tại (thay ``qfluentwidgets.themeColor``)."""
    return QColor(_ACCENT_COLOR)


def setThemeColor(color: QColor | str) -> None:  # noqa: N802 — giữ tên API
    """Đặt màu accent (thay ``qfluentwidgets.setThemeColor``)."""
    global _ACCENT_COLOR
    _ACCENT_COLOR = QColor(color)


class InfoBarPosition(Enum):
    """Vị trí hiển thị InfoBar (thay enum qfluentwidgets)."""

    TOP = "top"
    TOP_RIGHT = "top_right"
    TOP_LEFT = "top_left"
    BOTTOM = "bottom"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"
    NONE = "none"


class _ToastWidget(QFrame):
    """Toast tự tắt — nền tảng cho InfoBar."""

    _PALETTE: ClassVar[dict[str, tuple[str, str]]] = {
        "success": ("#0f7b0f", "#ffffff"),
        "info": ("#0078d4", "#ffffff"),
        "warning": ("#9d5d00", "#ffffff"),
        "error": ("#c42b1c", "#ffffff"),
    }

    def __init__(
        self,
        kind: str,
        title: str,
        content: str,
        parent: QWidget | None,
        duration_ms: int,
    ) -> None:
        super().__init__(parent)
        bg, fg = self._PALETTE.get(kind, self._PALETTE["info"])
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border-radius: 6px; }}"
            f"QLabel {{ color: {fg}; background: transparent; }}"
            f"QPushButton {{ color: {fg}; background: transparent; border: none; "
            f"font-weight: bold; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.setSpacing(10)
        text = title if not content else f"{title} — {content}" if title else content
        label = QLabel(text, self)
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        close_btn = QPushButton("✕", self)
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 90))
        self.setGraphicsEffect(shadow)

        self.adjustSize()
        self._reposition()
        if duration_ms > 0:
            QTimer.singleShot(duration_ms, self.close)

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        margin = 20
        x = parent.width() - self.width() - margin
        y = margin
        self.move(max(0, x), max(0, y))

    def showEvent(self, event) -> None:  # noqa: N802 — override Qt
        self._reposition()
        super().showEvent(event)


class InfoBar:
    """Toast thông báo (thay ``qfluentwidgets.InfoBar``).

    Giữ API tĩnh: ``InfoBar.success(title, content, parent=..., duration=...)`` và các
    cho ``info``/``warning``/``error``. Các tham số qfluentwidgets thừa (``orient``,
    ``position``, ``isClosable``) được chấp nhận và bỏ qua để không phải sửa code gọi.
    """

    @staticmethod
    def _show(
        kind: str,
        title: str = "",
        content: str = "",
        *,
        parent: QWidget | None = None,
        duration: int = 4000,
        **_ignored,
    ) -> _ToastWidget | None:
        if parent is None:
            return None
        toast = _ToastWidget(kind, title, content, parent, duration)
        toast.show()
        toast.raise_()
        return toast

    @classmethod
    def success(cls, title: str = "", content: str = "", **kwargs) -> _ToastWidget | None:
        return cls._show("success", title, content, **kwargs)

    @classmethod
    def info(cls, title: str = "", content: str = "", **kwargs) -> _ToastWidget | None:
        return cls._show("info", title, content, **kwargs)

    @classmethod
    def warning(cls, title: str = "", content: str = "", **kwargs) -> _ToastWidget | None:
        return cls._show("warning", title, content, **kwargs)

    @classmethod
    def error(cls, title: str = "", content: str = "", **kwargs) -> _ToastWidget | None:
        # Lỗi hiển thị lâu hơn mặc định nếu caller không chỉ định.
        kwargs.setdefault("duration", 6000)
        return cls._show("error", title, content, **kwargs)


__all__ = [
    "InfoBar",
    "InfoBarPosition",
    "Theme",
    "isDarkTheme",
    "setTheme",
    "setThemeColor",
    "themeColor",
]
