"""Thang đo khoảng cách & bo góc nhất quán cho toàn giao diện (thang cơ sở 4px).

[v3.23.63 — Giai đoạn 4 tái thiết UI/UX] Trước đây khoảng cách (spacing/margin) được đặt
bằng pixel tuỳ tiện (2, 3, 6, 10…), khiến giao diện không đều giữa các trang. Module này
cung cấp một THANG ĐO DUY NHẤT dựa trên bội số 4 — chuẩn mực phổ biến trong thiết kế giao
diện (Material, Fluent) — để mọi nơi dùng chung, dễ căn chỉnh và đồng nhất.

Đây là các hằng số thuần (không phụ thuộc Qt), dễ kiểm thử và tái dùng.
"""

from __future__ import annotations

__all__ = [
    "SPACING_NONE",
    "SPACING_XS",
    "SPACING_SM",
    "SPACING_MD",
    "SPACING_LG",
    "SPACING_XL",
    "SPACING_XXL",
    "RADIUS_SM",
    "RADIUS_MD",
    "RADIUS_LG",
    "FONT_SIZE_CAPTION",
    "FONT_SIZE_SMALL",
    "FONT_SIZE_BODY",
    "FONT_SIZE_TITLE",
    "FONT_SIZE_HEADING",
    "snap_to_scale",
]

# ── Thang khoảng cách (đơn vị pixel, bội số 4) ───────────────────────────────
SPACING_NONE: int = 0
SPACING_XS: int = 4
SPACING_SM: int = 8
SPACING_MD: int = 12
SPACING_LG: int = 16
SPACING_XL: int = 24
SPACING_XXL: int = 32

# ── Bán kính bo góc ──────────────────────────────────────────────────────────
RADIUS_SM: int = 4
RADIUS_MD: int = 8
RADIUS_LG: int = 12

# ── Thang cỡ chữ (đơn vị pixel, dùng trong stylesheet) ───────────────────────
# Gom các giá trị font-size từng hardcode rải rác để chỉnh ở MỘT nơi (hỗ trợ
# tinh chỉnh accessibility sau này). Giá trị giữ nguyên hiện trạng để không đổi giao diện.
FONT_SIZE_CAPTION: int = 11  # nhãn chú thích / gợi ý nhỏ
FONT_SIZE_SMALL: int = 13  # văn bản phụ
FONT_SIZE_BODY: int = 16  # thân nội dung nhấn
FONT_SIZE_TITLE: int = 18  # tiêu đề phụ
FONT_SIZE_HEADING: int = 20  # tiêu đề lớn

_SCALE: tuple[int, ...] = (
    SPACING_NONE, SPACING_XS, SPACING_SM, SPACING_MD,
    SPACING_LG, SPACING_XL, SPACING_XXL,
)


def snap_to_scale(value: int) -> int:
    """Làm tròn một giá trị pixel về bậc gần nhất trong thang khoảng cách.

    Hữu ích khi cần chuẩn hoá một giá trị cũ (vd 6, 10) về bội số 4 gần nhất mà vẫn giữ ý
    đồ ban đầu. Với giá trị nằm giữa hai bậc, ưu tiên bậc nhỏ hơn để giao diện gọn hơn.

    Args:
        value: Giá trị pixel cần chuẩn hoá (không âm).

    Returns:
        Giá trị gần nhất trong thang đo. Giá trị vượt bậc lớn nhất sẽ trả về bậc lớn nhất.

    Raises:
        ValueError: Nếu ``value`` âm.
    """
    if value < 0:
        raise ValueError("Giá trị khoảng cách không được âm.")
    if value >= _SCALE[-1]:
        return _SCALE[-1]
    best = _SCALE[0]
    best_distance = abs(value - best)
    for step in _SCALE[1:]:
        distance = abs(value - step)
        # '<' (không '<=') để khi cách đều thì giữ bậc NHỎ hơn (đã duyệt trước).
        if distance < best_distance:
            best = step
            best_distance = distance
    return best
