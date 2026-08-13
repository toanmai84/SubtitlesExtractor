"""Test [v3.20 Đợt 5] module thuần text_replace (tách từ editor_page)."""

from __future__ import annotations

from subtitles_extractor.presentation.utils.text_replace import replace_in_text_safe


class TestReplaceInTextSafe:
    def test_basic_replace_all(self) -> None:
        assert replace_in_text_safe("a", "X", "aaa", 0) == "XXX"

    def test_case_insensitive(self) -> None:
        assert replace_in_text_safe("abc", "X", "ABC abc Abc", 0) == "X X X"

    def test_count_limits_replacements(self) -> None:
        assert replace_in_text_safe("a", "X", "aaaa", 2) == "XXaa"

    def test_ass_override_tags_preserved(self) -> None:
        # Nội dung BÊN TRONG cặp {...} (vd tên chuyển động) không bị thay; còn
        # text giữa các thẻ thì có. Ở đây 'pos' nằm trong {\pos(...)} → giữ nguyên.
        out = replace_in_text_safe("pos", "XYZ", r"{\pos(10,20)}pos ở đây", 0)
        assert out == r"{\pos(10,20)}XYZ ở đây"

    def test_html_tags_preserved(self) -> None:
        out = replace_in_text_safe("x", "Y", "<span class='x'>x</span>", 0)
        # Thuộc tính class='x' nằm trong thẻ → giữ; chỉ 'x' nội dung bị thay.
        assert out == "<span class='x'>Y</span>"

    def test_no_match_returns_source(self) -> None:
        src = r"không đổi {\i1}gì{\i0}"
        assert replace_in_text_safe("zzz", "Q", src, 0) == src

    def test_special_regex_chars_treated_literally(self) -> None:
        # term chứa ký tự đặc biệt regex phải được so khớp literal.
        assert replace_in_text_safe("a.b", "X", "a.b aXb", 0) == "X aXb"

    def test_replacement_with_backslash_is_literal(self) -> None:
        # replacement chứa '\\' không được hiểu là group reference.
        assert replace_in_text_safe("a", r"\1", "a", 0) == r"\1"
