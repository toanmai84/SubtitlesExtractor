"""[v3.23.206] Test nhãn cảnh báo "Lấn" trong bảng kết quả + bộ lọc "Có vấn đề".

Nghiệm thu 2 video thật xác nhận chuỗi fix v204-v205 (im lặng đầu 160ms -> 40ms, 0 câu
câm, RMS đều). Ca còn lại kiểu #62 (thoại ~4.9s / khung 1.28s): đã nén kịch trần user
vẫn dư 1.26s tràn — KHÔNG phải bug (thiết kế "không mất nội dung") nhưng người dùng cần
NHÌN THẤY để chủ động rút gọn bản dịch. Cải tiến: trạng thái "🔊 Lấn Xs" khi
``overlap_s > _OVERLAP_WARN_S`` (0.5s) + lọt bộ lọc "Có vấn đề".

[v3.23.547] Từ đợt Việt hoá/i18n, nhãn hiển thị KHÔNG còn viết cứng trong ``tts_page.py``
mà nằm ở hai bộ từ điển ``strings_vi.json`` / ``strings_en.json``. Bộ kiểm giữ nguyên MỤC
ĐÍCH (ngưỡng + nhánh trạng thái + bộ lọc + nhãn người dùng đọc được) nhưng kiểm nhãn ở
đúng nơi nó sống bây giờ, thay vì đòi chuỗi viết cứng đã bị dời đi.
"""

from __future__ import annotations

import json
import pathlib

_SRC = pathlib.Path(
    "src/subtitles_extractor/presentation/pages/tts_page.py"
).read_text(encoding="utf-8")

_DATA_DIR = pathlib.Path("src/subtitles_extractor/data")
_STRINGS_VI: dict[str, dict[str, str]] = json.loads(
    (_DATA_DIR / "strings_vi.json").read_text(encoding="utf-8")
)
_STRINGS_EN: dict[str, dict[str, str]] = json.loads(
    (_DATA_DIR / "strings_en.json").read_text(encoding="utf-8")
)


def test_overlap_warn_threshold_defined() -> None:
    assert "_OVERLAP_WARN_S = 0.5" in _SRC


def test_status_label_shows_large_overlap() -> None:
    # Nhánh trạng thái phải có nhãn lấn giữa "Cắt" và "OK".
    assert "elif r.overlap_s > _OVERLAP_WARN_S:" in _SRC
    assert 'self._translator.translate("tts.st_overlap")' in _SRC


def test_overlap_status_string_defined_in_both_catalogues() -> None:
    """Nhãn "Lấn" phải có ở CẢ hai từ điển, kèm chỗ chèn số giây ``{s}``."""
    vi_label = _STRINGS_VI["tts"]["st_overlap"]
    en_label = _STRINGS_EN["tts"]["st_overlap"]
    assert "Lấn" in vi_label and "{s}" in vi_label
    assert "Overlap" in en_label and "{s}" in en_label


def test_issues_filter_includes_large_overlap() -> None:
    assert "or ((not r.was_skipped) and r.overlap_s > _OVERLAP_WARN_S)" in _SRC


def test_filter_label_mentions_overlap() -> None:
    """Nhãn bộ lọc "Có vấn đề" phải liệt kê đủ 4 loại, trong đó có "lấn"."""
    vi_label = _STRINGS_VI["tts"]["filt_issues"]
    assert "bỏ + cắt + nhanh + lấn" in vi_label
    assert "overlap" in _STRINGS_EN["tts"]["filt_issues"].lower()
