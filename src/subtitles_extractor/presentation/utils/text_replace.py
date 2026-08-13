"""Tìm & thay thế văn bản AN TOÀN với thẻ định dạng (ASS override / HTML).

Tách khỏi ``editor_page`` (God Object) phần logic find-replace tinh tế: phải bảo
vệ các thẻ ``{...}`` (ASS override) và ``<...>`` (HTML) khỏi bị sửa nhầm khi người
dùng thay thế nội dung. Hàm thuần → kiểm thử headless dễ dàng (logic regex dễ lỗi).
"""

from __future__ import annotations

import re

# Thẻ cần BẢO VỆ: HTML (<...>) và ASS override ({...}).
_PROTECTED_TAG_PATTERN = re.compile(r"<[^>]+>|\{[^}]*\}")
# Token giữ chỗ dùng ký tự NUL (không xuất hiện trong phụ đề bình thường).
_PLACEHOLDER_TEMPLATE = "\x00TAG{}\x00"


def replace_in_text_safe(
    term: str, replacement: str, source: str, count: int
) -> str:
    """Thay thế ``term`` bằng ``replacement`` trong ``source``, GIỮ NGUYÊN thẻ.

    Quy trình: (1) trích mọi thẻ định dạng ra placeholder; (2) thay thế trên phần
    văn bản thuần (không phân biệt hoa thường); (3) khôi phục thẻ. Nhờ vậy việc
    thay thế không bao giờ làm hỏng cú pháp ``{\\b1}`` hay ``<i>``.

    Args:
        term: Chuỗi cần tìm (so khớp văn bản thuần, literal — đã escape regex).
        replacement: Chuỗi thay thế.
        source: Văn bản gốc (có thể chứa thẻ ASS/HTML).
        count: Số lần thay tối đa (``0`` = thay tất cả, theo quy ước ``re.sub``).

    Returns:
        Văn bản sau khi thay thế, thẻ định dạng được bảo toàn.
    """
    protected_tags: list[str] = []

    def _extract(match: re.Match[str]) -> str:
        protected_tags.append(match.group(0))
        return _PLACEHOLDER_TEMPLATE.format(len(protected_tags) - 1)

    safe_source = _PROTECTED_TAG_PATTERN.sub(_extract, source)

    pattern_str = re.escape(term)
    try:
        replaced = re.sub(
            pattern_str, lambda _m: replacement, safe_source,
            count=count, flags=re.IGNORECASE,
        )
    except re.error:
        # Phòng xa: nếu vì lý do nào đó pattern lỗi, dùng compiled tách bước.
        compiled = re.compile(pattern_str, flags=re.IGNORECASE)
        replaced = compiled.sub(lambda _m: replacement, safe_source, count=count)

    for index, tag in enumerate(protected_tags):
        replaced = replaced.replace(_PLACEHOLDER_TEMPLATE.format(index), tag)
    return replaced
