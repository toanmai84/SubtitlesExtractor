"""Test hàm thuần strip_formatting_tags (thay test introspection lỗi thời)."""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_editor_service import (
    strip_formatting_tags,
)


class TestStripFormattingTags:
    def test_removes_ass_override(self) -> None:
        assert strip_formatting_tags("{\\an8}Xin chào") == "Xin chào"

    def test_removes_html(self) -> None:
        assert strip_formatting_tags("<b>Đậm</b> thường") == "Đậm thường"

    def test_preserves_newline_marker_N(self) -> None:
        # CỐT LÕI: KHÔNG được xoá \N (ngắt dòng ASS).
        assert strip_formatting_tags("{\\an8}Dòng 1\\NDòng 2") == "Dòng 1\\NDòng 2"

    def test_preserves_real_newline(self) -> None:
        assert strip_formatting_tags("A\n{\\i1}B") == "A\nB"

    def test_multiple_braces(self) -> None:
        assert strip_formatting_tags("{\\pos(1,2)}A{\\c&H}B") == "AB"

    def test_no_tags_unchanged(self) -> None:
        assert strip_formatting_tags("Bình thường") == "Bình thường"

    def test_empty(self) -> None:
        assert strip_formatting_tags("") == ""
