"""Hộp thoại chọn/lưu file AN TOÀN trên Windows (tránh treo do Native Shell Dialog).

Trên Windows, hộp thoại file mặc định của hệ điều hành (Native Shell Dialog) chạy
trên môi trường COM riêng và có thể tranh chấp tài nguyên với luồng giao diện Qt khi
ứng dụng đang giữ nhiều object đồ hoạ/GPU — gây deadlock (treo) ngay trước khi cửa sổ
kịp hiện. Cách phòng tránh đã được kiểm chứng là **không dùng Native Dialog** mà dùng
hộp thoại Qt thuần (cờ ``DontUseNativeDialog``).

Hai hàm dưới đây gói cách làm an toàn đó để mọi nơi trong ứng dụng dùng nhất quán:
luôn ép Qt-only và xử lý nốt các sự kiện còn tồn (``processEvents``) ngay trước khi
mở — giải toả "cặn" event loop để giao diện ở trạng thái rảnh, giảm nguy cơ xung đột.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QFileDialog, QWidget


def _drain_event_loop() -> None:
    """Xử lý nốt các sự kiện đang chờ để giao diện rảnh trước khi mở hộp thoại."""
    app = QCoreApplication.instance()
    if app is not None:
        app.processEvents()


def open_file(
    parent: QWidget | None, caption: str, directory: str = "", file_filter: str = "",
) -> str:
    """Chọn một file để mở (Qt-only). Trả về đường dẫn, hoặc chuỗi rỗng nếu huỷ."""
    _drain_event_loop()
    path, _ = QFileDialog.getOpenFileName(
        parent, caption, directory, file_filter,
        options=QFileDialog.Option.DontUseNativeDialog,
    )
    return path


def save_file(
    parent: QWidget | None, caption: str, directory: str = "", file_filter: str = "",
) -> str:
    """Chọn đường dẫn để lưu file (Qt-only). Trả về đường dẫn, hoặc rỗng nếu huỷ."""
    _drain_event_loop()
    path, _ = QFileDialog.getSaveFileName(
        parent, caption, directory, file_filter,
        options=QFileDialog.Option.DontUseNativeDialog,
    )
    return path


def choose_directory(parent: QWidget | None, caption: str, directory: str = "") -> str:
    """Chọn một thư mục (Qt-only). Trả về đường dẫn, hoặc rỗng nếu huỷ."""
    _drain_event_loop()
    return QFileDialog.getExistingDirectory(
        parent, caption, directory,
        options=QFileDialog.Option.DontUseNativeDialog,
    )


__all__ = ["open_file", "save_file", "choose_directory"]
