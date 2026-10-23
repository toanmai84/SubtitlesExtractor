"""[v3.23.441] Test ngắt dòng phụ đề cân đối."""
from __future__ import annotations
from subtitles_extractor.application.services.subtitle_line_wrapper import wrap_subtitle_line


def test_short_line_unchanged():
    assert wrap_subtitle_line("Xin chào bạn") == "Xin chào bạn"


def test_already_wrapped_unchanged():
    t = "dòng một\ndòng hai"
    assert wrap_subtitle_line(t) == t


def test_long_line_split_into_two():
    t = "Các người chẳng phải treo thưởng một trăm triệu để cứu mạng"
    w = wrap_subtitle_line(t)
    lines = w.split("\n")
    assert len(lines) == 2
    assert all(len(l) <= 42 for l in lines)
    assert " ".join(lines) == t  # không mất chữ


def test_breaks_before_conjunction():
    t = "Ngài không nói gì thì coi như đồng ý rồi nhé."  # 45c
    w = wrap_subtitle_line(t)
    # nên ngắt để 'thì' đứng đầu dòng 2 (điểm ngắt tự nhiên)
    assert "\nthì" in w


def test_no_space_single_word_unchanged():
    t = "x" * 50  # không có khoảng trắng
    assert wrap_subtitle_line(t) == t


def test_balanced_split():
    t = "một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba"
    w = wrap_subtitle_line(t)
    a, b = w.split("\n")
    assert abs(len(a) - len(b)) <= 12  # tương đối cân đối


def test_no_word_broken():
    t = "Các người chẳng phải treo thưởng một trăm triệu để cứu mạng người"
    w = wrap_subtitle_line(t)
    for line in w.split("\n"):
        # mỗi dòng bắt đầu/kết thúc bằng từ nguyên vẹn (không cắt giữa từ)
        assert not line.startswith(" ") and not line.endswith(" ")
    assert w.replace("\n", " ") == t
