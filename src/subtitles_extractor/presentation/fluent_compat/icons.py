"""FluentIcon thay thế — dùng icon chuẩn Qt (thay ``qfluentwidgets.FluentIcon``).

[v3.23.267] qfluentwidgets FluentIcon là bộ icon SVG riêng (GPL). Thay bằng icon chuẩn của
Qt (``QStyle.StandardPixmap``) — miễn phí, không phụ thuộc GPL. Icon Qt tiêu
chuẩn không đẹp bằng Fluent nhưng đủ dùng và hợp lệ thương mại.

API tương thích: ``FluentIcon.VIDEO.icon()`` trả về ``QIcon`` dùng trong
``setIcon``.
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle


class FluentIcon(Enum):
    """Bộ icon thay thế FluentIcon — ánh xạ sang ``QStyle.StandardPixmap``.

    Mỗi thành viên giữ tên chuẩn Qt để lấy icon hệ thống. Với icon không có tương đương
    trực tiếp trong Qt, chọn cái gần nghĩa nhất (đủ nhận diện chức năng).
    """

    VIDEO = "SP_MediaPlay"
    EDIT = "SP_FileDialogDetailedView"
    LANGUAGE = "SP_FileDialogListView"
    MICROPHONE = "SP_MediaVolume"
    CODE = "SP_FileDialogContentsView"
    FOLDER = "SP_DirIcon"
    HISTORY = "SP_BrowserReload"
    SETTING = "SP_FileDialogDetailedView"
    PLAY = "SP_MediaPlay"
    PAUSE = "SP_MediaPause"
    VIEW = "SP_FileDialogInfoView"
    LINK = "SP_DirLinkIcon"
    CARE_LEFT_SOLID = "SP_ArrowLeft"
    CARE_RIGHT_SOLID = "SP_ArrowRight"

    def icon(self) -> QIcon:
        """Trả về ``QIcon`` tương ứng (giống ``FluentIcon.X.icon()``)."""
        app = QApplication.instance()
        if app is None:
            return QIcon()
        style = app.style()
        pixmap_enum = getattr(QStyle.StandardPixmap, self.value, None)
        if pixmap_enum is None:
            return QIcon()
        return style.standardIcon(pixmap_enum)

    def qicon(self) -> QIcon:
        """Bí danh của :meth:`icon`."""
        return self.icon()


__all__ = ["FluentIcon"]
