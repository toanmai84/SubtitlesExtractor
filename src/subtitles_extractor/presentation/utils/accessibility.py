"""[v3.23.71 — Giai đoạn 6 Accessibility] Tiện ích đặt tên truy cập (accessible name).

Các nút CHỈ-ICON (vd play/pause, prev/next, hiện/ẩn mật khẩu) không có nhãn chữ, nên công
nghệ hỗ trợ (screen reader) không đọc được gì khi người dùng khiếm thị focus vào. Đặt
``accessibleName`` cung cấp mô tả văn bản cho những điều khiển đó.

Thiết kế theo SRP: hàm thuần tuý chỉ gán thuộc tính truy cập, không chứa logic nghiệp vụ;
dễ kiểm thử và tái dùng trên mọi trang.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

__all__ = ["set_accessible_name"]


def set_accessible_name(widget: QWidget, name: str, *, set_tooltip: bool = True) -> None:
    """Đặt ``accessibleName`` cho ``widget`` để trình đọc màn hình mô tả được.

    Args:
        widget: Điều khiển cần gán tên truy cập (thường là nút chỉ-icon).
        name: Mô tả ngắn gọn, rõ nghĩa (vd "Phát/Tạm dừng", "Hiện/ẩn mật khẩu").
        set_tooltip: Nếu True và widget CHƯA có tooltip, đặt luôn tooltip cùng nội dung —
            vừa hỗ trợ người khiếm thị, vừa hữu ích cho mọi người khi rê chuột. Không ghi
            đè tooltip có sẵn.
    """
    widget.setAccessibleName(name)
    if set_tooltip and not widget.toolTip():
        widget.setToolTip(name)
