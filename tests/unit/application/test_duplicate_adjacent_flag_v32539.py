"""[v3.23.139] Test cờ ``duplicate_adjacent`` — phát hiện lệch-gộp ngữ nghĩa.

Khi một câu bị tách qua nhiều dòng, model lite đôi khi dồn cả câu vào dòng đầu rồi KÉO nội
dung dòng sau lên -> hai dòng liền kề dịch TRÙNG dù NGUỒN KHÁC. Cờ này khoanh vùng lỗi mà
cờ đếm/độ dài không bắt được (số dòng vẫn khớp).
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)


def _ev(index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(index=index, start_sec=index, end_sec=index + 1, text=text)


def test_detects_adjacent_duplicate_from_shift() -> None:
    # Tái hiện đúng ca Easter Island (dòng 7-8).
    src = [
        _ev(7, "The moai."),
        _ev(8, "Moais are incredible."),
    ]
    dst = [
        _ev(7, "Moai thật kinh ngạc."),
        _ev(8, "Moai thật kinh ngạc."),  # trùng dòng trước, nguồn khác -> nghi lệch
    ]
    qf = detect_quality_flags(src, dst)
    assert qf["counts"]["duplicate_adjacent"] == 1
    assert qf["duplicate_adjacent"][0]["indices"] == [7, 8]


def test_ignores_duplicate_when_source_also_identical() -> None:
    # Hai dòng nguồn GIỐNG nhau (vd điệp khúc) -> dịch giống là ĐÚNG, không cờ.
    src = [_ev(1, "Run!"), _ev(2, "Run!")]
    dst = [_ev(1, "Chạy đi!"), _ev(2, "Chạy đi!")]
    qf = detect_quality_flags(src, dst)
    assert qf["counts"]["duplicate_adjacent"] == 0


def test_ignores_speaker_tag_when_comparing() -> None:
    # Nhãn người nói khác nhau nhưng nội dung trùng vẫn bị bắt (bỏ tag trước khi so).
    src = [_ev(1, "A."), _ev(2, "B.")]
    dst = [_ev(1, "Xin chào."), _ev(2, "[Nam:] Xin chào.")]
    qf = detect_quality_flags(src, dst)
    assert qf["counts"]["duplicate_adjacent"] == 1


def test_no_false_positive_on_distinct_lines() -> None:
    src = [_ev(1, "The moai."), _ev(2, "Moais are incredible.")]
    dst = [_ev(1, "Tượng Moai."), _ev(2, "Moai thật kinh ngạc.")]
    qf = detect_quality_flags(src, dst)
    assert qf["counts"]["duplicate_adjacent"] == 0


def test_empty_lines_not_flagged() -> None:
    src = [_ev(1, "X"), _ev(2, "Y")]
    dst = [_ev(1, ""), _ev(2, "")]
    qf = detect_quality_flags(src, dst)
    assert qf["counts"]["duplicate_adjacent"] == 0
