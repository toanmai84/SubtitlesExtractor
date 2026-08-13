"""Tiện ích tính ĐỘ TƯƠNG PHẢN màu theo chuẩn WCAG 2.x (accessibility — Giai đoạn 6).

Module thuần (pure functions), KHÔNG phụ thuộc Qt/qfluentwidgets — chỉ thao tác trên chuỗi
màu hex ``#rrggbb`` — nên dễ kiểm thử và tái dùng. Dùng để kiểm các cặp (chữ, nền) trong
bảng màu theme có đạt ngưỡng tương phản tối thiểu cho người khiếm thị / ánh sáng kém.

Tham chiếu công thức: WCAG 2.1, Success Criterion 1.4.3 (Contrast Minimum).
"""

from __future__ import annotations

__all__ = [
    "AA_LARGE_RATIO",
    "AA_NORMAL_RATIO",
    "contrast_ratio",
    "hex_to_rgb",
    "meets_wcag_aa",
    "relative_luminance",
]

# Ngưỡng WCAG 2.1 mức AA.
AA_NORMAL_RATIO = 4.5  # Chữ thường (< 18pt, hoặc < 14pt đậm).
AA_LARGE_RATIO = 3.0  # Chữ lớn (>= 18pt, hoặc >= 14pt đậm).


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Chuyển chuỗi màu hex ``#rrggbb`` (hoặc ``#rgb``) thành bộ ba (r, g, b) 0–255.

    Args:
        hex_color: Chuỗi màu, ví dụ ``"#e8e8e8"`` hoặc ``"#fff"``.

    Returns:
        Bộ ba số nguyên ``(r, g, b)`` trong khoảng 0–255.

    Raises:
        ValueError: Nếu chuỗi không phải định dạng hex hợp lệ (3 hoặc 6 ký tự hex).
    """
    value = hex_color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"Mã màu hex không hợp lệ: {hex_color!r}")
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError as exc:  # ký tự không phải hex
        raise ValueError(f"Mã màu hex không hợp lệ: {hex_color!r}") from exc


def _linearize_channel(channel_0_255: int) -> float:
    """Tuyến tính hoá một kênh màu sRGB (0–255) theo công thức WCAG."""
    c = channel_0_255 / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """Tính độ sáng tương đối (relative luminance) của màu theo WCAG, trong [0, 1].

    Args:
        hex_color: Chuỗi màu hex.

    Returns:
        Độ sáng tương đối: 0.0 (đen tuyền) đến 1.0 (trắng tuyền).
    """
    r, g, b = hex_to_rgb(hex_color)
    return (
        0.2126 * _linearize_channel(r)
        + 0.7152 * _linearize_channel(g)
        + 0.0722 * _linearize_channel(b)
    )


def contrast_ratio(color_a: str, color_b: str) -> float:
    """Tính tỉ lệ tương phản giữa hai màu theo WCAG, trong [1.0, 21.0].

    Thứ tự tham số không quan trọng (tỉ lệ đối xứng).

    Args:
        color_a: Màu thứ nhất (hex).
        color_b: Màu thứ hai (hex).

    Returns:
        Tỉ lệ tương phản, từ 1.0 (giống hệt) đến 21.0 (đen trên trắng).
    """
    lum_a = relative_luminance(color_a)
    lum_b = relative_luminance(color_b)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def meets_wcag_aa(foreground: str, background: str, *, large_text: bool = False) -> bool:
    """Kiểm cặp (chữ, nền) có đạt ngưỡng tương phản WCAG mức AA hay không.

    Args:
        foreground: Màu chữ (hex).
        background: Màu nền (hex).
        large_text: True nếu là chữ lớn (ngưỡng 3.0 thay vì 4.5).

    Returns:
        True nếu tỉ lệ tương phản >= ngưỡng AA tương ứng.
    """
    threshold = AA_LARGE_RATIO if large_text else AA_NORMAL_RATIO
    return contrast_ratio(foreground, background) >= threshold
