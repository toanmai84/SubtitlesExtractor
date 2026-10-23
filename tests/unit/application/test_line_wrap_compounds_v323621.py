"""[v3.23.621] Ngắt dòng phụ đề không cắt đôi từ ghép / treo lượng từ (ca thật từ phim mẫu)."""
from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_line_wrapper import wrap_subtitle_line

CASES = [
    ("thậm chí có thể tranh vị trí Thánh nữ tông môn", "vị\ntrí"),
    ("quyền hạn thực lực đỉnh cao trong một ngày.", "đỉnh\ncao"),
    ("không nên bị chuyện tình cảm nhi nữ quấy nhiễu.", "tình\ncảm"),
    ("toàn quyền sử dụng sức mạnh đỉnh cao một ngày.", "sức\nmạnh"),
    ("Ta đã nhẫn nhịn các ngươi hai mươi năm rồi.", "các\nngươi"),
]


@pytest.mark.parametrize(("text", "bad"), CASES)
def test_no_split_inside_compound(text: str, bad: str) -> None:
    wrapped = wrap_subtitle_line(text)
    assert "\n" in wrapped and bad not in wrapped
    assert wrapped.replace("\n", " ") == text
    assert all(len(part) <= 42 for part in wrapped.split("\n"))


def test_punctuation_break_still_preferred() -> None:
    assert wrap_subtitle_line("Muội nhất định phải hạnh phúc đấy, được không?") == (
        "Muội nhất định phải hạnh phúc đấy,\nđược không?")
