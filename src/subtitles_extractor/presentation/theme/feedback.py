"""Lớp tiện ích thông báo (toast) dùng chung cho toàn giao diện.

[v3.23.61 — Giai đoạn 2 tái thiết UI/UX] Trước đây mỗi trang tự dựng ``InfoBar`` với vị
trí/thời lượng riêng, dẫn tới trải nghiệm thông báo không nhất quán. Module này gom về một
nguồn duy nhất: cùng vị trí (trên cùng), cùng thang thời lượng theo mức độ quan trọng, cùng
kiểu icon mặc định của qfluentwidgets.

Mỗi hàm nhận ``parent`` là widget cha để toast bám đúng cửa sổ/trang hiện hành.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget
from subtitles_extractor.presentation.fluent_compat import InfoBar, InfoBarPosition

__all__ = ["show_success", "show_info", "show_warning", "show_error"]

# Thời lượng (ms) theo mức độ: thông tin/thành công đọc nhanh; cảnh báo/lỗi cần lâu hơn.
_DURATION_SUCCESS = 5000
_DURATION_INFO = 4000
_DURATION_WARNING = 6000
_DURATION_ERROR = 8000

_POSITION = InfoBarPosition.TOP


def show_success(parent: QWidget, title: str, content: str = "") -> None:
    """Hiển thị thông báo thành công (toast màu xanh, trên cùng)."""
    InfoBar.success(
        title=title, content=content, parent=parent,
        position=_POSITION, duration=_DURATION_SUCCESS,
    )


def show_info(parent: QWidget, title: str, content: str = "") -> None:
    """Hiển thị thông báo thông tin."""
    InfoBar.info(
        title=title, content=content, parent=parent,
        position=_POSITION, duration=_DURATION_INFO,
    )


def show_warning(parent: QWidget, title: str, content: str = "") -> None:
    """Hiển thị cảnh báo (hiển thị lâu hơn để người dùng kịp đọc)."""
    InfoBar.warning(
        title=title, content=content, parent=parent,
        position=_POSITION, duration=_DURATION_WARNING,
    )


def show_error(parent: QWidget, title: str, content: str = "") -> None:
    """Hiển thị lỗi (hiển thị lâu nhất)."""
    InfoBar.error(
        title=title, content=content, parent=parent,
        position=_POSITION, duration=_DURATION_ERROR,
    )
