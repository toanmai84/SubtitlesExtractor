"""Widget thay thế qfluentwidgets bằng Qt thuần (PySide6, LGPL — an toàn thương mại).

[v3.23.267] qfluentwidgets là GPL v3 (thương mại phải mua license). App bỏ hẳn, tự xây lại
bằng Qt thuần. Module này cung cấp các widget cơ bản với TÊN GIỐNG qfluentwidgets để code
gọi không phải đổi nhiều — phần lớn là widget Qt tiêu chuẩn đặt bí danh, cộng
chút style cho gần Fluent.

Xem docs/LICENSE_ANALYSIS.md và docs/PYSIDE6_MIGRATION_PLAN.md.
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ── Widget cơ bản: alias trực tiếp sang Qt thuần ─────────────────────────────
# qfluentwidgets chỉ thêm thẩm mỹ; hành vi & API giống hệt widget Qt gốc.
CheckBox = QCheckBox
ComboBox = QComboBox
LineEdit = QLineEdit
Slider = QSlider
SpinBox = QSpinBox
DoubleSpinBox = QDoubleSpinBox
ProgressBar = QProgressBar
TextEdit = QTextEdit
ScrollArea = QScrollArea


class ToolButton(QToolButton):
    """Nút công cụ (thay ``qfluentwidgets.ToolButton``).

    qfluentwidgets cho phép ``ToolButton(icon)`` với icon là FluentIcon/QIcon. QToolButton
    không nhận icon trong constructor -> shim tự tách và gọi setIcon.
    """

    def __init__(self, icon: object = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if icon is not None:
            self.setIcon(_resolve_qicon(icon))


def _resolve_qicon(icon: object):
    """Chuyển FluentIcon (shim) hoặc QIcon thành QIcon."""
    from PySide6.QtGui import QIcon

    if isinstance(icon, QIcon):
        return icon
    icon_method = getattr(icon, "icon", None)
    if callable(icon_method):
        result = icon_method()
        if isinstance(result, QIcon):
            return result
    return QIcon()


class PushButton(QPushButton):
    """Nút bấm thường (thay ``qfluentwidgets.PushButton``).

    qfluentwidgets cho phép ``PushButton(icon, text)``, ``PushButton(text)`` hoặc
    ``PushButton(icon, text, parent)``. Icon có thể là FluentIcon (shim) hoặc QIcon.
    """

    def __init__(self, *args, **kwargs) -> None:
        from PySide6.QtGui import QIcon

        # Tách icon FluentIcon nếu là đối số đầu và KHÔNG phải QIcon/str.
        if args and not isinstance(args[0], (str, QIcon)):
            first = args[0]
            icon_method = getattr(first, "icon", None)
            if callable(icon_method):
                resolved = _resolve_qicon(first)
                super().__init__(resolved, *args[1:], **kwargs)
                return
        super().__init__(*args, **kwargs)


class PrimaryPushButton(PushButton):
    """Nút nhấn mạnh (accent). Tô màu nổi bật để thay style 'primary' của Fluent."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # objectName để QSS theme tô màu accent (xem theme.py).
        self.setProperty("primary", True)


# ── Nhãn phân cấp typography (Fluent) → QLabel + cỡ chữ ──────────────────────
class CaptionLabel(QLabel):
    """Nhãn chú thích nhỏ (thay ``qfluentwidgets.CaptionLabel``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setProperty("labelClass", "caption")


class StrongBodyLabel(QLabel):
    """Nhãn thân đậm (thay ``qfluentwidgets.StrongBodyLabel``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setProperty("labelClass", "strongBody")


class BodyLabel(QLabel):
    """Nhãn thân thường."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setProperty("labelClass", "body")


# ── HeaderCardWidget (thẻ có tiêu đề) → QGroupBox có bố cục sẵn ───────────────
class HeaderCardWidget(QFrame):
    """Thẻ có tiêu đề (thay ``qfluentwidgets.HeaderCardWidget``).

    qfluentwidgets HeaderCardWidget có ``.headerLabel`` (tiêu đề) và ``.viewLayout`` (khu
    nội dung). Tái hiện đúng hai thuộc tính đó để code con thêm widget vào view.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("cardClass", "header")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(16, 12, 16, 16)
        self._root.setSpacing(8)
        # [v3.23.290] Header la LAYOUT NGANG (headerLabel + cho nut phu nhu toggle).
        # qfluentwidgets that co headerLayout; SectionCard collapsible them nut mui
        # ten vao day. Truoc chi co headerLabel -> nut toggle khong co cho -> ket.
        self.headerLayout = QHBoxLayout()
        self.headerLayout.setContentsMargins(0, 0, 0, 0)
        self.headerLabel = QLabel(self)
        self.headerLabel.setProperty("labelClass", "cardHeader")
        self.headerLayout.addWidget(self.headerLabel)
        self.headerLayout.addStretch(1)
        self._root.addLayout(self.headerLayout)
        self._view = QWidget(self)
        self.viewLayout = QVBoxLayout(self._view)
        self.viewLayout.setContentsMargins(0, 0, 0, 0)
        self._root.addWidget(self._view)

    def setTitle(self, text: str) -> None:  # noqa: N802 — giữ API qfluentwidgets
        """Đặt tiêu đề thẻ (API tương thích qfluentwidgets)."""
        self.headerLabel.setText(text)

    def getTitle(self) -> str:  # noqa: N802 — giữ API qfluentwidgets
        """Lấy tiêu đề thẻ (API tương thích qfluentwidgets)."""
        return self.headerLabel.text()


class CardWidget(QFrame):
    """Thẻ đơn giản không tiêu đề (thay ``qfluentwidgets.CardWidget``)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("cardClass", "simple")
        self.setFrameShape(QFrame.Shape.StyledPanel)


class SimpleCardWidget(CardWidget):
    """Bí danh CardWidget (một số nơi qfluentwidgets tách tên)."""


# ── Bố cục dạng group (fallback) ─────────────────────────────────────────────
GroupHeaderCardWidget = HeaderCardWidget
GroupBox = QGroupBox


class NavigationItemPosition(Enum):
    """Vị trí mục điều hướng (thay enum của qfluentwidgets)."""

    TOP = "top"
    SCROLL = "scroll"
    BOTTOM = "bottom"


__all__ = [
    "BodyLabel",
    "CaptionLabel",
    "CardWidget",
    "CheckBox",
    "ComboBox",
    "DoubleSpinBox",
    "GroupBox",
    "GroupHeaderCardWidget",
    "HeaderCardWidget",
    "LineEdit",
    "NavigationItemPosition",
    "PrimaryPushButton",
    "ProgressBar",
    "PushButton",
    "ScrollArea",
    "SimpleCardWidget",
    "Slider",
    "SpinBox",
    "StrongBodyLabel",
    "TextEdit",
    "ToolButton",
]
