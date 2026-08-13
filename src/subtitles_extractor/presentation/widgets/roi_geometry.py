"""Ánh xạ toạ độ giữa khung hiển thị và khung video (thuần, KHÔNG phụ thuộc Qt).

VÌ SAO tách riêng
=================
Vùng ROI quyết định phần ảnh nào được đưa vào OCR — sai vài điểm ảnh là hỏng chất
lượng trích xuất. Trước đây phép toán này bị **lặp lại ở 3 chỗ** trong
``mpv_video_widget.py`` (``_widget_to_video_roi``, ``_video_to_widget_rect``,
``_video_box_to_widget_rect`` và một bản nội tuyến nữa) — vi phạm DRY và rất dễ lệch
nhau khi sửa.

Tách thành module thuần cho phép:
    * **Kiểm thử được đầy đủ** (không cần Qt/màn hình) — kể cả tính chất khứ hồi.
    * Dùng chung cho cả widget mpv cũ lẫn widget PyAV mới.

Mô hình hiển thị
----------------
Video được thu phóng GIỮ NGUYÊN TỈ LỆ và căn giữa trong khung, nên thường có viền đen
hai bên (pillarbox) hoặc trên dưới (letterbox)::

    scale    = min(khung_rộng / video_rộng, khung_cao / video_cao)
    offset_x = (khung_rộng - video_rộng × scale) / 2
    offset_y = (khung_cao  - video_cao  × scale) / 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

# Kích thước tối thiểu của một ROI hợp lệ (điểm ảnh video).
_MIN_ROI_SIZE: Final[int] = 1


@dataclass(frozen=True, slots=True)
class DisplayGeometry:
    """Thông số thu phóng/căn giữa của video bên trong khung hiển thị.

    Attributes:
        scale: Hệ số thu phóng (khung / video). ``0.0`` nếu không hợp lệ.
        offset_x: Lề trái (điểm ảnh khung) do căn giữa.
        offset_y: Lề trên (điểm ảnh khung) do căn giữa.
        video_width: Chiều rộng video gốc.
        video_height: Chiều cao video gốc.
    """

    scale: float
    offset_x: float
    offset_y: float
    video_width: int
    video_height: int

    @property
    def is_valid(self) -> bool:
        """``True`` khi thông số dùng được để ánh xạ."""
        return (
            self.scale > 0.0 and self.video_width > 0 and self.video_height > 0
        )

    @property
    def displayed_rect(self) -> tuple[int, int, int, int]:
        """Vùng khung mà video thực sự chiếm: ``(x, y, rộng, cao)``."""
        if not self.is_valid:
            return (0, 0, 0, 0)
        x1 = int(round(self.offset_x))
        y1 = int(round(self.offset_y))
        x2 = int(round(self.offset_x + self.video_width * self.scale))
        y2 = int(round(self.offset_y + self.video_height * self.scale))
        return (x1, y1, x2 - x1, y2 - y1)


def compute_display_geometry(
    *, video_width: int, video_height: int, widget_width: int, widget_height: int
) -> DisplayGeometry:
    """Tính thông số thu phóng + căn giữa cho video trong khung.

    Args:
        video_width: Chiều rộng video gốc (px).
        video_height: Chiều cao video gốc (px).
        widget_width: Chiều rộng khung hiển thị (px).
        widget_height: Chiều cao khung hiển thị (px).

    Returns:
        :class:`DisplayGeometry`; ``is_valid`` là ``False`` nếu tham số không hợp lệ
        (kích thước <= 0) — tầng gọi nên bỏ qua thay vì tính tiếp.
    """
    if video_width <= 0 or video_height <= 0 or widget_width <= 0 or widget_height <= 0:
        return DisplayGeometry(0.0, 0.0, 0.0, max(0, video_width), max(0, video_height))

    scale = min(widget_width / video_width, widget_height / video_height)
    offset_x = (widget_width - video_width * scale) / 2.0
    offset_y = (widget_height - video_height * scale) / 2.0
    return DisplayGeometry(scale, offset_x, offset_y, video_width, video_height)


def video_rect_to_widget(
    rect: tuple[int, int, int, int], geometry: DisplayGeometry
) -> tuple[int, int, int, int]:
    """Đổi hình chữ nhật từ toạ độ VIDEO sang toạ độ KHUNG hiển thị.

    Args:
        rect: ``(x, y, rộng, cao)`` theo toạ độ video.
        geometry: Thông số hiển thị hiện tại.

    Returns:
        ``(x, y, rộng, cao)`` theo toạ độ khung; ``(0, 0, 0, 0)`` nếu thông số không
        hợp lệ.

    Notes:
        Quy đổi qua HAI GÓC rồi mới trừ ra chiều rộng/cao (thay vì nhân riêng chiều
        rộng) — cách này khớp với biên hiển thị thật và tránh dồn sai số làm lệch 1px.
    """
    if not geometry.is_valid:
        return (0, 0, 0, 0)
    x, y, width, height = rect
    x1 = int(round(geometry.offset_x + x * geometry.scale))
    y1 = int(round(geometry.offset_y + y * geometry.scale))
    x2 = int(round(geometry.offset_x + (x + width) * geometry.scale))
    y2 = int(round(geometry.offset_y + (y + height) * geometry.scale))
    return (x1, y1, x2 - x1, y2 - y1)


def widget_rect_to_video(
    rect: tuple[int, int, int, int], geometry: DisplayGeometry
) -> tuple[int, int, int, int] | None:
    """Đổi hình chữ nhật từ toạ độ KHUNG sang toạ độ VIDEO, có cắt biên.

    Phần nằm ngoài vùng video (viền đen) bị cắt bỏ; kết quả luôn nằm trong khung ảnh.

    Args:
        rect: ``(x, y, rộng, cao)`` theo toạ độ khung (thường là vùng người dùng kéo).
        geometry: Thông số hiển thị hiện tại.

    Returns:
        ``(x, y, rộng, cao)`` theo toạ độ video; ``None`` nếu vùng chọn không giao
        với ảnh (người dùng kéo hoàn toàn trên viền đen).
    """
    if not geometry.is_valid:
        return None

    x, y, width, height = rect
    if width <= 0 or height <= 0:
        return None

    # Cắt theo vùng video thực sự hiển thị.
    disp_x, disp_y, disp_w, disp_h = geometry.displayed_rect
    left = max(x, disp_x)
    top = max(y, disp_y)
    right = min(x + width, disp_x + disp_w)
    bottom = min(y + height, disp_y + disp_h)
    if right <= left or bottom <= top:
        return None

    scale = geometry.scale
    x1 = int(round((left - geometry.offset_x) / scale))
    y1 = int(round((top - geometry.offset_y) / scale))
    x2 = int(round((right - geometry.offset_x) / scale))
    y2 = int(round((bottom - geometry.offset_y) / scale))

    # Kẹp vào khung ảnh, đảm bảo rộng/cao >= 1.
    video_width, video_height = geometry.video_width, geometry.video_height
    x1 = max(0, min(x1, video_width - _MIN_ROI_SIZE))
    y1 = max(0, min(y1, video_height - _MIN_ROI_SIZE))
    x2 = max(x1 + _MIN_ROI_SIZE, min(x2, video_width))
    y2 = max(y1 + _MIN_ROI_SIZE, min(y2, video_height))

    return (x1, y1, x2 - x1, y2 - y1)


__all__ = [
    "DisplayGeometry",
    "compute_display_geometry",
    "video_rect_to_widget",
    "widget_rect_to_video",
]
