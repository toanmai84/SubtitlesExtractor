"""Bảo vệ widget khỏi việc cuộn chuột vô tình làm thay đổi giá trị.

Vấn đề UX kinh điển của Qt: QSpinBox, QDoubleSpinBox, QComboBox, QSlider sẽ
thay đổi giá trị khi con trỏ chuột vô tình lướt qua trong lúc cuộn trang — dù
người dùng KHÔNG click vào.

Giải pháp:
    1. focusPolicy → StrongFocus (wheel không tự chiếm focus).
    2. eventFilter chặn wheel khi widget CHƯA focus, và CHUYỂN TIẾP lên scroll
       area cha → trang vẫn cuộn bình thường.
    3. Khi widget ĐÃ focus (chủ động click/tab vào), wheel chỉnh giá trị bình thường.

Lưu ý: KHÔNG bảo vệ QScrollBar — đó chính là thanh cuộn, cần wheel để cuộn.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
    QWidget,
)

_GUARDED_TYPES: tuple[type[QWidget], ...] = (
    QAbstractSpinBox,
    QComboBox,
    QAbstractSlider,
)


def _find_scroll_area(widget: QWidget) -> QAbstractScrollArea | None:
    """Tìm QAbstractScrollArea tổ tiên gần nhất (để chuyển tiếp cuộn trang)."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return parent
        parent = parent.parentWidget()
    return None


class _WheelGuard(QObject):
    """Chặn wheel đổi giá trị khi widget chưa focus, chuyển tiếp lên scroll area."""

    def __init__(self) -> None:
        super().__init__()
        self._forwarding = False  # cờ chống đệ quy

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel:
            if self._forwarding:
                return False
            if isinstance(obj, QWidget) and not obj.hasFocus():
                scroll = _find_scroll_area(obj)
                if scroll is not None:
                    self._forwarding = True
                    try:
                        QApplication.sendEvent(scroll.viewport(), event)
                    finally:
                        self._forwarding = False
                return True
        return super().eventFilter(obj, event)


_GUARD_SINGLETON: _WheelGuard | None = None


def _get_guard() -> _WheelGuard:
    """Trả singleton :class:`_WheelGuard`, tạo lại nếu object C++ đã bị xoá.

    Module-level QObject có thể bị huỷ theo vòng đời QApplication (vd giữa các test,
    hoặc khi truy cập trước khi app khởi tạo). Getter này tự hồi phục để bền vững.
    """
    global _GUARD_SINGLETON
    guard = _GUARD_SINGLETON
    if guard is not None:
        try:
            guard.parent()  # chạm object C++ để phát hiện đã bị xoá
            return guard
        except RuntimeError:
            pass  # object C++ đã bị xoá -> tạo lại bên dưới
    _GUARD_SINGLETON = _WheelGuard()
    return _GUARD_SINGLETON


def protect_scroll_widgets(root: QWidget) -> int:
    """Bảo vệ mọi widget cuộn-nhạy-cảm trong ``root`` (đệ quy). Bỏ qua QScrollBar.

    Returns:
        Số widget đã bảo vệ.
    """
    from PySide6.QtCore import Qt

    widgets: list[QWidget] = []
    if isinstance(root, _GUARDED_TYPES) and not isinstance(root, QScrollBar):
        widgets.append(root)
    # [v3.23.267] PySide6 findChildren chỉ nhận MỘT type (PyQt6 nhận tuple). Lặp từng type
    # rồi gộp, khử trùng lặp theo id để giữ hành vi cũ.
    seen: set[int] = set()
    for guarded_type in _GUARDED_TYPES:
        for w in root.findChildren(guarded_type):
            if isinstance(w, QScrollBar) or id(w) in seen:
                continue
            seen.add(id(w))
            widgets.append(w)

    count = 0
    guard = _get_guard()
    for w in widgets:
        w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        w.installEventFilter(guard)
        count += 1
    return count


__all__ = ["protect_scroll_widgets"]
