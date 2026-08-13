"""[v3.23.75] Builder chuỗi stylesheet tập trung (design tokens → CSS string).

Gom các chuỗi ``setStyleSheet`` lặp lại (đặc biệt nhãn chú thích nhỏ) về MỘT nơi, dùng
hằng số cỡ chữ trong :mod:`metrics` và hàm màu trong :mod:`colors`. Mục tiêu: tuân thủ DRY
và "không hardcode" — chỉnh cỡ chữ/màu chú thích một chỗ thay vì rải rác khắp các trang.

Các hàm ở đây là HÀM THUẦN: kết quả chỉ phụ thuộc tham số (và token theme hiện hành),
không gây hiệu ứng phụ — dễ kiểm thử.
"""

from __future__ import annotations

from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.theme import metrics as _m

__all__ = ["caption_style", "mono_label_style"]


def caption_style(color: str | None = None) -> str:
    """Trả về chuỗi stylesheet cho nhãn chú thích nhỏ (caption/hint).

    Args:
        color: Mã màu CSS cho chữ. Nếu ``None`` (mặc định), dùng màu chữ mờ
            (:func:`colors.on_surface_muted`) theo theme hiện hành.

    Returns:
        Chuỗi dạng ``"font-size:<N>px;color:<màu>;"`` — đặt vào ``QWidget.setStyleSheet``.
    """
    text_color = color if color is not None else _c.on_surface_muted()
    return f"font-size:{_m.FONT_SIZE_CAPTION}px;color:{text_color};"


def mono_label_style(font_size: int | None = None) -> str:
    """Trả về chuỗi stylesheet cho nhãn đơn sắc (mã thời gian, bộ đếm…).

    Args:
        font_size: Cỡ chữ (px). Nếu ``None`` (mặc định), dùng :data:`metrics.FONT_SIZE_SMALL`.

    Returns:
        Chuỗi dạng ``"font-family: Consolas, monospace; font-size: <N>px;"``.
    """
    size = font_size if font_size is not None else _m.FONT_SIZE_SMALL
    return f"font-family: Consolas, monospace; font-size: {size}px;"
