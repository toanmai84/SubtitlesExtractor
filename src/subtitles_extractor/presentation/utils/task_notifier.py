"""[v3.23.682] Báo khi một việc dài xong / lỗi lúc người dùng đang ở cửa sổ khác.

Trích xuất cả phim 2 giờ 19 phút chạy ~4 giờ trên CPU; người dùng chuyển sang việc khác và chỉ biết đã xong khi quay lại
nhìn. Nay: cửa sổ ứng dụng không giữ focus → nháy nút trên thanh tác vụ (``QApplication.alert``) và hiện bong bóng ở
khay hệ thống (nếu hệ điều hành có khay). Đang nhìn ứng dụng thì không làm phiền (đã có InfoBar/hộp thoại sẵn).
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QWidget

logger = logging.getLogger(__name__)

_TRAY_PROPERTY = "_se_task_tray"
#: Thời gian bong bóng hiện trên màn hình (ms).
BALLOON_MS = 8000


def _tray(app: QApplication) -> QSystemTrayIcon | None:
    """Một biểu tượng khay dùng chung cho cả ứng dụng (tạo lười); ``None`` khi hệ thống không có khay."""
    tray = app.property(_TRAY_PROPERTY)
    if isinstance(tray, QSystemTrayIcon):
        return tray
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return None
    tray = QSystemTrayIcon(app.windowIcon(), app)
    tray.setToolTip(app.applicationDisplayName() or app.applicationName() or "Subtitles Extractor")
    tray.show()
    app.setProperty(_TRAY_PROPERTY, tray)
    return tray


def notify_task_finished(widget: QWidget | None, title: str, message: str, *, failed: bool = False) -> bool:
    """Báo việc xong khi cửa sổ chứa ``widget`` không giữ focus. Trả ``True`` nếu đã báo.

    Không bao giờ ném lỗi: thông báo hỏng không được làm hỏng luồng xử lý kết quả.
    """
    app = QApplication.instance()
    if not isinstance(app, QApplication) or widget is None:
        return False
    try:
        window = widget.window()
        if window is None or window.isActiveWindow():
            return False
        QApplication.alert(window, 0)
        tray = _tray(app)
        if tray is not None:
            icon = QSystemTrayIcon.MessageIcon.Warning if failed else QSystemTrayIcon.MessageIcon.Information
            tray.showMessage(title, (message or "")[:300], icon, BALLOON_MS)
        return True
    except RuntimeError as exc:          # widget đã bị xoá phía C++
        logger.debug("Không báo được việc xong: %s", exc)
        return False


__all__ = ["BALLOON_MS", "notify_task_finished"]
