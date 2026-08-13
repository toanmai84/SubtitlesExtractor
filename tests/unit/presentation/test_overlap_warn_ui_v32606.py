"""[v3.23.206] Test nhãn cảnh báo "Lấn" trong bảng kết quả + bộ lọc "Có vấn đề".

Nghiệm thu 2 video thật xác nhận chuỗi fix v204-v205 (im lặng đầu 160ms -> 40ms, 0 câu
câm, RMS đều). Ca còn lại kiểu #62 (thoại ~4.9s / khung 1.28s): đã nén kịch trần user
vẫn dư 1.26s tràn — KHÔNG phải bug (thiết kế "không mất nội dung") nhưng người dùng cần
NHÌN THẤY để chủ động rút gọn bản dịch. Cải tiến: trạng thái "🔊 Lấn Xs" khi
``overlap_s > _OVERLAP_WARN_S`` (0.5s) + lọt bộ lọc "Có vấn đề".
"""

from __future__ import annotations

import pathlib

_SRC = pathlib.Path(
    "src/subtitles_extractor/presentation/pages/tts_page.py"
).read_text(encoding="utf-8")


def test_overlap_warn_threshold_defined() -> None:
    assert "_OVERLAP_WARN_S = 0.5" in _SRC


def test_status_label_shows_large_overlap() -> None:
    # Nhánh trạng thái phải có nhãn lấn giữa "Cắt" và "OK".
    assert "elif r.overlap_s > _OVERLAP_WARN_S:" in _SRC
    assert 'st = f"🔊 Lấn {r.overlap_s:.1f}s"' in _SRC


def test_issues_filter_includes_large_overlap() -> None:
    assert 'or ((not r.was_skipped) and r.overlap_s > _OVERLAP_WARN_S)' in _SRC


def test_filter_label_mentions_overlap() -> None:
    assert "bỏ + cắt + nhanh + lấn" in _SRC
