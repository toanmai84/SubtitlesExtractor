"""Tiện ích làm sạch text phụ đề khi hiển thị trong UI.

Phụ đề có thể chứa các tag định dạng (ASS ``{\\an8}``, HTML ``<b>``,
ngắt dòng ``\\N``) — khi hiển thị raw lên QTableView, các tag bị lộ ra
gây mất thẩm mỹ. Module này cung cấp regex pre-compiled để xoá nhanh
trong delegate paint.

Tham khảo: SubtitleEdit/Aegisub coding pattern.
"""

from __future__ import annotations

import re

# Tag ASS dạng ``{\an8}``, ``{\b1}``, ``{\fs20}``, ``{\c&H00FFFF&}``…
# Dùng non-greedy ``*?`` để khớp đúng từng cặp ``{...}``.
ASS_TAG_REGEX: re.Pattern[str] = re.compile(r"\{.*?\}")

# Tag HTML cơ bản dùng trong phụ đề (``<b>``, ``<i>``, ``<u>``, ``<font ...>``…).
HTML_TAG_REGEX: re.Pattern[str] = re.compile(r"<[^>]+>")


def clean_subtitle_text_for_display(raw_text: str) -> str:
    """Xoá tag ASS/HTML và chuyển ``\\N`` → newline thật.

    Dùng trong QStyledItemDelegate.paint để hiển thị text đẹp mà KHÔNG
    sửa đổi data gốc — tag vẫn được giữ nguyên trong subtitle entity.

    Args:
        raw_text: Text gốc (có thể chứa tag).

    Returns:
        Text đã làm sạch — vẫn giữ Unicode CJK/diacritic.

    Examples:
        >>> clean_subtitle_text_for_display("{\\an8}Xin chào\\N<b>thế giới</b>")
        'Xin chào\\nthế giới'
    """
    cleaned = HTML_TAG_REGEX.sub("", raw_text)
    cleaned = ASS_TAG_REGEX.sub("", cleaned)
    return cleaned.replace("\\N", "\n")


__all__ = [
    "ASS_TAG_REGEX",
    "HTML_TAG_REGEX",
    "clean_subtitle_text_for_display",
]
