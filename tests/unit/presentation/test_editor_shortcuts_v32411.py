"""[v3.23.111] Test bảng phím tắt trình chỉnh sửa (hàm thuần, không cần GUI)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def _stub_translator():
    """Bộ dịch giả lập: trả về phần sau dấu chấm của khoá (đủ cho test cấu trúc)."""
    class _Stub:
        def translate(self, key: str, **kwargs: object) -> str:
            return key.split(".", 1)[-1]

    return _Stub()


def test_shortcuts_html_contains_key_bindings() -> None:
    from subtitles_extractor.presentation.pages.editor_page import (
        _EDITOR_SHORTCUTS,
        editor_shortcuts_html,
    )

    html = editor_shortcuts_html(_stub_translator())
    # Các phím then chốt phải xuất hiện để người dùng khám phá được.
    for key in ("Ctrl+M", "Ctrl+T", "Alt+Insert", "Delete", "J / L", "[ / ]", "F11"):
        assert key in html, f"Thiếu phím tắt: {key}"
    # Có đủ nhóm và mỗi nhóm có ít nhất 1 phím.
    assert len(_EDITOR_SHORTCUTS) >= 4
    assert all(rows for rows in _EDITOR_SHORTCUTS.values())


def test_shortcuts_html_is_wellformed_table() -> None:
    from subtitles_extractor.presentation.pages.editor_page import editor_shortcuts_html

    html = editor_shortcuts_html(_stub_translator())
    # Số thẻ mở/đóng <table> cân nhau (không hỏng cấu trúc).
    assert html.count("<table") == html.count("</table>")
    assert "<h3>" in html
