"""Bảng màu ngữ nghĩa thích ứng theo theme (sáng/tối) cho toàn bộ giao diện.

[v3.23.59 — Giai đoạn 1 tái thiết UI/UX] Trước đây nhiều màu được hardcode (vd
``#1e1e1e``, ``#e0e0e0``) giả định nền tối — khi người dùng dùng chế độ Sáng thì giao diện
bị vỡ (chữ tối trên nền tối). Module này cung cấp các hàm trả về MÀU THEO THEME hiện tại,
để mọi nơi dùng chung một nguồn màu nhất quán và tự thích ứng sáng/tối.

[v3.23.71 — Giai đoạn 6 accessibility] Dữ liệu hoá bảng màu thành hai dict
:data:`LIGHT_PALETTE`/:data:`DARK_PALETTE` (DRY, không lặp điều kiện ``isDarkTheme()``
ở mỗi hàm) và giúp kiểm toán tương phản WCAG đọc trực tiếp giá trị của CẢ HAI theme. API
công khai (các hàm trả hex) giữ nguyên 100%.

Các hàm trả chuỗi màu hex (``#rrggbb``) để chèn trực tiếp vào stylesheet Qt. Màu nhấn lấy
từ ``themeColor()`` của qfluentwidgets để đồng bộ với màu chủ đề người dùng chọn.
"""

from __future__ import annotations

from subtitles_extractor.presentation.fluent_compat import isDarkTheme, themeColor

__all__ = [
    "LIGHT_PALETTE",
    "DARK_PALETTE",
    "accent",
    "on_accent",
    "surface",
    "surface_variant",
    "on_surface",
    "on_surface_muted",
    "border",
    "mono_bg",
    "mono_fg",
    "success",
    "warning",
    "danger",
    "info",
    "danger_bg",
    "secondary",
    "muted_italic",
]

# Bảng màu cho từng theme. Khoá = tên token ngữ nghĩa; giá trị = hex ``#rrggbb``.
# Lưu ý: ``accent`` KHÔNG nằm ở đây vì lấy động từ ``themeColor()`` (màu người dùng chọn).
DARK_PALETTE: dict[str, str] = {
    "on_accent": "#ffffff",
    "surface": "#202020",
    "surface_variant": "#2b2b2b",
    "on_surface": "#e8e8e8",
    "on_surface_muted": "#9a9a9a",
    "border": "#3a3a3a",
    "mono_bg": "#1a1a1a",
    "mono_fg": "#e0e0e0",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "danger_bg": "#3f1515",
    "info": "#58a6ff",
    "secondary": "#a371f7",
    "muted_italic": "#7a7a7a",
}

LIGHT_PALETTE: dict[str, str] = {
    "on_accent": "#ffffff",
    "surface": "#ffffff",
    "surface_variant": "#f3f3f3",
    "on_surface": "#1a1a1a",
    "on_surface_muted": "#6b6b6b",
    "border": "#e0e0e0",
    "mono_bg": "#f6f8fa",
    "mono_fg": "#24292f",
    "success": "#1a7f37",
    "warning": "#9a6700",
    "danger": "#cf222e",
    "danger_bg": "#ffebe9",
    "info": "#0969da",
    "secondary": "#8250df",
    "muted_italic": "#888888",
}


def _pick(token: str) -> str:
    """Trả màu của ``token`` theo theme hiện tại (tối/sáng)."""
    return (DARK_PALETTE if isDarkTheme() else LIGHT_PALETTE)[token]


def accent() -> str:
    """Màu nhấn chủ đạo (theo màu theme người dùng chọn)."""
    return themeColor().name()


def on_accent() -> str:
    """Màu chữ/biểu tượng đặt trên nền màu nhấn (luôn tương phản cao)."""
    return _pick("on_accent")


def surface() -> str:
    """Màu nền bề mặt chính (thẻ/panel)."""
    return _pick("surface")


def surface_variant() -> str:
    """Màu nền phụ (vùng nhấn nhẹ, hàng xen kẽ)."""
    return _pick("surface_variant")


def on_surface() -> str:
    """Màu chữ chính trên bề mặt."""
    return _pick("on_surface")


def on_surface_muted() -> str:
    """Màu chữ phụ/nhạt (chú thích, placeholder)."""
    return _pick("on_surface_muted")


def border() -> str:
    """Màu viền nhẹ giữa các vùng."""
    return _pick("border")


def mono_bg() -> str:
    """Nền vùng mã/JSON (đơn sắc, tương phản với chữ mono)."""
    return _pick("mono_bg")


def mono_fg() -> str:
    """Màu chữ trong vùng mã/JSON."""
    return _pick("mono_fg")


def success() -> str:
    """Màu trạng thái thành công."""
    return _pick("success")


def warning() -> str:
    """Màu trạng thái cảnh báo."""
    return _pick("warning")


def danger() -> str:
    """Màu trạng thái lỗi/nguy hiểm."""
    return _pick("danger")


def danger_bg() -> str:
    """Nền nhấn cho vùng lỗi (vd ô JSON khi parse thất bại)."""
    return _pick("danger_bg")


def info() -> str:
    """Màu trạng thái thông tin."""
    return _pick("info")


def secondary() -> str:
    """Màu nhấn phụ (vd tác vụ STT/transcribe) — tông tím, phân biệt với nhấn chính."""
    return _pick("secondary")


def muted_italic() -> str:
    """Màu chữ rất nhạt cho gợi ý/placeholder dạng in nghiêng."""
    return _pick("muted_italic")
