"""Test [v3.20 Đợt 5] module thuần ass_preview_builder (tách từ editor_page)."""

from __future__ import annotations

from subtitles_extractor.presentation.utils.ass_preview_builder import (
    AssPreviewStyle,
    build_ass_header,
    escape_text_to_ass,
    render_dialogue_line,
)


class TestEscapeText:
    def test_newlines_become_ass_break(self) -> None:
        assert escape_text_to_ass("dòng 1\ndòng 2") == r"dòng 1\Ndòng 2"
        assert escape_text_to_ass("a\r\nb") == r"a\Nb"

    def test_braces_escaped(self) -> None:
        assert escape_text_to_ass("{x}") == r"\{x\}"

    def test_bold_italic_converted(self) -> None:
        out = escape_text_to_ass("<b>đậm</b> <i>nghiêng</i>")
        assert r"{\b1}đậm{\b0}" in out and r"{\i1}nghiêng{\i0}" in out

    def test_other_html_tags_stripped(self) -> None:
        assert escape_text_to_ass("<span style='x'>chữ</span>") == "chữ"


class TestRenderDialogueLine:
    def test_format_and_timestamps(self) -> None:
        line = render_dialogue_line("xin chào", 1.5, 3.2)
        assert line.startswith("Dialogue: 0,0:00:01.50,0:00:03.20,Default,,0,0,0,,")
        assert line.endswith("xin chào")

    def test_multiline_text(self) -> None:
        line = render_dialogue_line("a\nb", 0.0, 1.0)
        assert line.endswith(r"a\Nb")


class TestBuildHeader:
    def test_header_contains_style_and_resolution(self) -> None:
        header = build_ass_header(
            AssPreviewStyle(font_name="Roboto", font_size=60, play_res_x=720, play_res_y=1280)
        )
        assert "PlayResX: 720" in header and "PlayResY: 1280" in header
        assert "Style: Default,Roboto,60," in header
        assert header.rstrip().endswith("Effect, Text")

    def test_defaults(self) -> None:
        header = build_ass_header(AssPreviewStyle())
        assert "PlayResX: 1920" in header and "Style: Default,Arial,48," in header
