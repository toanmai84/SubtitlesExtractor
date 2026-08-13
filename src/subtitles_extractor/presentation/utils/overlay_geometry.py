"""Tiện ích hình học cho overlay (logic thuần, không phụ thuộc PyQt — testable).

Dùng để "kẹp biên" toạ độ bounding box trước khi vẽ, chống treo GUI / tràn bộ nhớ
khi OCR nhận nhầm mảng màu thành box khổng lồ (hàng triệu pixel).
"""

from __future__ import annotations

#: Lề an toàn (px) cho phép box nhô ra ngoài khung hiển thị trước khi bị kẹp.
SAFE_RENDER_MARGIN_PX = 4000


def clamp_box_coords(
    x: int,
    y: int,
    width: int,
    height: int,
    canvas_width: int,
    canvas_height: int,
    margin: int = SAFE_RENDER_MARGIN_PX,
) -> tuple[int, int, int, int]:
    """Kẹp một box ``(x, y, width, height)`` vào khung hiển thị + lề an toàn.

    Mọi cạnh bị giới hạn trong ``[-margin, canvas + margin]`` để QPainter/MPV-OSD
    không bao giờ phải vẽ một hình chữ nhật rộng hàng triệu pixel (gây treo cứng).

    Args:
        x, y, width, height: Toạ độ box gốc (có thể âm hoặc khổng lồ).
        canvas_width, canvas_height: Kích thước vùng vẽ.
        margin: Lề an toàn cho phép vượt biên.

    Returns:
        Tuple ``(x, y, width, height)`` đã kẹp; width/height luôn ``>= 0``.
    """
    low_x, high_x = -margin, canvas_width + margin
    low_y, high_y = -margin, canvas_height + margin

    x1 = max(low_x, min(x, high_x))
    y1 = max(low_y, min(y, high_y))
    x2 = max(low_x, min(x + width, high_x))
    y2 = max(low_y, min(y + height, high_y))

    return int(x1), int(y1), max(0, int(x2 - x1)), max(0, int(y2 - y1))
