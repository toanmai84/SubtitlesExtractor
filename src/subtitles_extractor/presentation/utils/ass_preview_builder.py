"""Dựng nội dung phụ đề ASS cho khung xem trước (preview) — HÀM THUẦN.

Tách khỏi ``editor_page.py`` (God Object 2.6k dòng) phần logic dựng chuỗi ASS
không phụ thuộc Qt, để: (1) giảm tải file presentation; (2) kiểm thử headless
được bằng pytest. Phần còn lại trong ``editor_page`` chỉ lo lấy tham số từ widget
rồi gọi các hàm này.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from subtitles_extractor.presentation.utils.time_format import seconds_to_ass

# Thẻ HTML đơn giản (<...>) — loại sau khi đã đổi các thẻ định dạng cần giữ.
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class AssPreviewStyle:
    """Tham số style cho phụ đề preview (đọc từ Settings của người dùng)."""

    font_name: str = "Arial"
    font_size: int = 48
    color_primary: str = "&H00FFFFFF"
    color_outline: str = "&H00000000"
    color_background: str = "&H99000000"
    margin_vertical: int = 25
    play_res_x: int = 1920
    play_res_y: int = 1080


def build_ass_header(style: AssPreviewStyle) -> str:
    """Dựng phần ``[Script Info]`` + ``[V4+ Styles]`` + tiêu đề ``[Events]``.

    Args:
        style: Tham số style preview.

    Returns:
        Chuỗi header ASS hoàn chỉnh (kết thúc bằng dòng Format của Events).
    """
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {style.play_res_x}\n"
        f"PlayResY: {style.play_res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font_name},{style.font_size},{style.color_primary},"
        f"&H000000FF,{style.color_outline},{style.color_background},0,0,0,0,"
        f"100,100,0,0,1,2,1,2,10,10,{style.margin_vertical},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )


def escape_text_to_ass(text: str) -> str:
    """Chuyển văn bản hiển thị sang dạng an toàn cho dòng Dialogue ASS.

    - Chuẩn hoá xuống dòng về ``\\n`` rồi sang ``\\N`` (ngắt dòng ASS).
    - Thoát ``{`` ``}`` (ký tự điều khiển override ASS).
    - Đổi ``<b>``/``<i>`` sang ``{\\b1}``/``{\\i1}`` (giữ định dạng cơ bản).
    - Loại mọi thẻ HTML còn lại.

    Args:
        text: Văn bản hiển thị (có thể chứa thẻ HTML đơn giản, nhiều dòng).

    Returns:
        Chuỗi đã thoát, sẵn sàng nối vào dòng ``Dialogue``.
    """
    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("<b>", r"{\b1}")
        .replace("</b>", r"{\b0}")
        .replace("<i>", r"{\i1}")
        .replace("</i>", r"{\i0}")
    )
    without_html = _HTML_TAG_PATTERN.sub("", normalized)
    return without_html.replace("\n", r"\N")


def render_dialogue_line(text: str, start_sec: float, end_sec: float) -> str:
    """Dựng MỘT dòng ``Dialogue`` ASS từ văn bản + mốc thời gian.

    Args:
        text: Văn bản hiển thị (chưa thoát).
        start_sec: Thời điểm bắt đầu (giây).
        end_sec: Thời điểm kết thúc (giây).

    Returns:
        Dòng ``Dialogue: ...`` hoàn chỉnh (không kèm ký tự xuống dòng cuối).
    """
    escaped = escape_text_to_ass(text)
    return (
        f"Dialogue: 0,{seconds_to_ass(start_sec)},{seconds_to_ass(end_sec)},"
        f"Default,,0,0,0,,{escaped}"
    )
