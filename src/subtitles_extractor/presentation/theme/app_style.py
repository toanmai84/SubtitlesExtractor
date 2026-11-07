"""[v3.23.674] Bảng kiểu (QSS) chung của ứng dụng — chỉ những gì cần thống nhất toàn app.

Trước đây nút "chính" (``PrimaryPushButton`` đặt thuộc tính ``primary``) và nút điều hướng (``navButton``) có đánh dấu
nhưng KHÔNG có quy tắc QSS nào dùng tới → nút "Bắt đầu…" trông y như mọi nút khác, nút điều hướng căn giữa và không
phân nhóm. Bảng này chỉ nhắm các thuộc tính đó (không đổi giao diện các widget khác).

Màu chữ phụ (tiêu đề nhóm, nút chính đang khoá) chọn tường minh theo nền sáng/tối: ``palette(mid)`` cho tương phản
~1,2-1,8:1 ở nền sáng và gần như vô hình ở nền tối.
"""

from __future__ import annotations

ACCENT = "#0f6cbd"
ACCENT_HOVER = "#115ea3"
ACCENT_PRESSED = "#0c3b5e"
#: (chữ tiêu đề nhóm, nền nút khoá, chữ nút khoá) theo nền sáng / tối — tương phản chữ ≥ 4,5:1.
_MUTED = {False: ("#5c5c5c", "#d9d9d9", "#595959"), True: ("#b3b3b3", "#3a3a3a", "#b8b8b8")}


def build_app_stylesheet(dark: bool | None = None) -> str:
    """Trả về QSS chung.

    Args:
        dark: Nền tối hay sáng; ``None`` = dò theo theme hiện hành.
    """
    if dark is None:
        try:
            from subtitles_extractor.presentation.fluent_compat import isDarkTheme

            dark = bool(isDarkTheme())
        except (ImportError, RuntimeError):
            dark = False
    section_fg, disabled_bg, disabled_fg = _MUTED[bool(dark)]
    return f"""
QPushButton[primary="true"] {{
    background-color: {ACCENT};
    color: #ffffff;
    border: 1px solid {ACCENT_HOVER};
    border-radius: 5px;
    padding: 5px 10px;
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton[primary="true"]:pressed {{ background-color: {ACCENT_PRESSED}; }}
QPushButton[primary="true"]:disabled {{
    background-color: {disabled_bg};
    color: {disabled_fg};
    border-color: {disabled_bg};
}}
QWidget#fluentSidebar {{ background-color: palette(window); }}
QLabel#navSection {{
    color: {section_fg};
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px 2px 10px;
}}
QPushButton[navButton="true"] {{
    text-align: left;
    padding: 6px 12px;
    border: 1px solid transparent;
    border-radius: 6px;
    background-color: transparent;
}}
QPushButton[navButton="true"]:hover {{ background-color: palette(midlight); }}
QPushButton[navButton="true"]:checked {{
    background-color: palette(highlight);
    color: palette(highlighted-text);
    font-weight: 600;
}}
"""


__all__ = ["ACCENT", "build_app_stylesheet"]
