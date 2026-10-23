"""[v3.23.610] Phụ đề chữ phải-sang-trái (Ả Rập, Do Thái, Ba Tư…) xuất đúng chiều.

Trang Dịch cho chọn đích tiếng Ả Rập. Trình phát coi mỗi dòng SRT là đoạn trái-sang-phải
nếu không có dấu định hướng, nên dấu câu cuối câu (``?``, ``.``) nhảy sang đầu dòng.
gemini-srt-translator bọc dòng có chữ RTL chiếm ưu thế bằng RLE … PDF; ở đây bọc TỪNG
dòng (ký tự xuống dòng kết thúc mọi lớp định hướng theo chuẩn Unicode Bidi) và gỡ lại khi
nạp tệp để sửa tiếp không mang theo ký tự ẩn.
"""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.application.services.bidi_marks import (
    mark_rtl_lines,
    strip_rtl_marks,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.subtitle.exporters.ass_exporter import AssExporter
from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import SrtExporter
from subtitles_extractor.infrastructure.subtitle.importers.ass_importer import AssImporter
from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import SrtImporter

RLE, PDF = "‫", "‬"
ARABIC = "كيف حالك؟"
HEBREW = "מה שלומך?"


def _event(text: str) -> SubtitleEvent:
    return SubtitleEvent(index=1, text=text, interval=TimeInterval(1.0, 3.0))


class TestMarks:
    def test_each_rtl_line_is_wrapped_on_its_own(self) -> None:
        assert mark_rtl_lines(f"{ARABIC}\n{HEBREW}") == (
            f"{RLE}{ARABIC}{PDF}\n{RLE}{HEBREW}{PDF}"
        )

    def test_left_to_right_lines_are_untouched(self) -> None:
        for text in ("Xin chào!", "你好吗", "OK 123", ""):
            assert mark_rtl_lines(text) == text

    def test_mixed_line_follows_the_dominant_script(self) -> None:
        assert mark_rtl_lines("مرحبا Tom") == f"{RLE}مرحبا Tom{PDF}"
        assert mark_rtl_lines("Hello Tom مرحبا") == "Hello Tom مرحبا"

    def test_marking_twice_does_not_stack(self) -> None:
        once = mark_rtl_lines(ARABIC)
        assert mark_rtl_lines(once) == once

    def test_strip_restores_the_text(self) -> None:
        assert strip_rtl_marks(mark_rtl_lines(f"{ARABIC}\nOK")) == f"{ARABIC}\nOK"


def test_srt_export_marks_rtl_lines(tmp_path: Path) -> None:
    out = SrtExporter().export([_event(ARABIC)], tmp_path / "ar.srt")
    assert f"{RLE}{ARABIC}{PDF}" in out.read_text(encoding="utf-8-sig")


def test_ass_export_marks_rtl_lines(tmp_path: Path) -> None:
    out = AssExporter().export([_event(f"{ARABIC}\n{HEBREW}")], tmp_path / "ar.ass")
    assert f"{RLE}{ARABIC}{PDF}\\N{RLE}{HEBREW}{PDF}" in out.read_text(encoding="utf-8")


def test_reimporting_an_exported_file_gives_back_clean_text(tmp_path: Path) -> None:
    srt = SrtExporter().export([_event(ARABIC)], tmp_path / "ar.srt")
    ass = AssExporter().export([_event(ARABIC)], tmp_path / "ar.ass")
    assert SrtImporter().import_from(srt)[0].text == ARABIC
    assert AssImporter().import_from(ass)[0].text == ARABIC
