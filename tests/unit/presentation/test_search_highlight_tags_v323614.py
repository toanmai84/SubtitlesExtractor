"""[v3.23.614] Tô sáng từ tìm kiếm trong bảng phụ đề không được phá thẻ in nghiêng/đậm.

Bảng phụ đề vẽ chữ bằng HTML (để hiện <i>, <b>, <font color>). Tô sáng thay chuỗi tìm
kiếm ở MỌI chỗ, kể cả bên trong thẻ: tìm "i" biến ``<i>`` thành ``<<span…>i</span>>`` —
dòng in nghiêng hiện ra lẫn mã HTML vỡ.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from subtitles_extractor.presentation.widgets.rich_text_delegate import (
    _compile_html_string,
)


@pytest.mark.parametrize(
    ("raw", "query"),
    [
        ("<i>Đi thôi</i>", "i"),
        ("<b>bó tay</b>", "b"),
        ('<font color="#ff0000">Chạy</font>', "font"),
        ('<font color="#ff0000">Chạy</font>', "color"),
    ],
)
def test_tags_survive_highlighting(raw: str, query: str) -> None:
    html = _compile_html_string(raw, False, "#000000", query)
    for tag in ("<i>", "</i>", "<b>", "</b>", '<font color="#ff0000">', "</font>"):
        if tag in raw:
            assert tag in html


def test_text_is_still_highlighted() -> None:
    html = _compile_html_string("<i>Đi thôi</i>", False, "#000000", "thôi")
    assert ">thôi</span>" in html
