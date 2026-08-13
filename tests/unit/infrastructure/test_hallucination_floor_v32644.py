"""[v3.23.244] Lưới hallucination đo so với BIÊN DƯỚI THEO ÂM TIẾT.

**Bug tìm thấy trên FLAC Gemini mới (lấn 15.52s):** 12 câu lấn nặng đều là HALLUCINATION —
Gemini đọc dài gấp 3-8 lần mức tối thiểu (câu "Haiz." 1 âm tiết đọc 3.74s!). Không
phải bản dịch dài (chỉ 1/12 câu vượt ngân sách âm tiết mới).

**Vì sao lưới cũ bỏ lọt:** ``is_abnormally_long`` so audio với mô hình TRUNG BÌNH THEO KÝ
TỰ — mà mô hình đó có R²=0.07 (đã bác bỏ ở v237). Câu nhiều ký tự -> kỳ vọng trung bình bị
thổi cao -> ratio thấp -> THOÁT. Đo trên 95 câu: lưới cũ chỉ bắt 5/12 câu lấn (chạm
8.23s / 15.52s tổng lấn).

**Lời giải:** hallucination = audio dài gấp nhiều lần mức đọc GỌN NHẤT (biên dưới theo âm
tiết, R²=0.98). Ngưỡng x2.5:

* bắt **12/12** câu lấn (chạm **13.83s = 89%** tổng lấn),
* 7 câu "bắt thêm" đều đọc 2.7-3.3x biên dưới (hallucination nhẹ, khung rộng nên
  chưa lấn — retry vẫn đáng để có bản gọn hơn),
* KHÔNG đụng câu đọc bình thường (audio ≈ 1-2x biên dưới).

Khi retry ra bản gọn hơn (``shorter_take``), phần lấn này giảm mạnh ở lần chạy sau.
"""

from __future__ import annotations

import pathlib

import pytest

from subtitles_extractor.infrastructure.tts.timing_math import (
    ABNORMAL_VS_FLOOR_RATIO,
    GEMINI_MIN_BASE_S,
    GEMINI_MIN_PER_SYLLABLE_S,
    is_abnormally_long,
    is_abnormally_long_vs_floor,
)

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")


def _bat(duration_s: float, syllables: int) -> bool:
    return is_abnormally_long_vs_floor(
        duration_s,
        syllables,
        min_base_s=GEMINI_MIN_BASE_S,
        min_per_syllable_s=GEMINI_MIN_PER_SYLLABLE_S,
    )


# ── 12 câu THẬT hallucination (âm tiết, audio) — lưới mới phải bắt HẾT ───────
@pytest.mark.parametrize(
    ("syllables", "audio_s", "nhan"),
    [
        (4, 3.28, "Chúc mừng ký chủ,"),
        (3, 3.12, "Vương Kiến Cường"),
        (1, 2.70, "Tuổi:"),
        (1, 2.50, "Không"),
        (2, 3.60, "Tu vi:"),
        (3, 3.64, "thật đúng là"),
        (3, 3.02, "Giá một viên"),
        (1, 3.74, "Haiz."),
        (4, 6.40, "Đừng dùng đạo đức."),
        (5, 6.30, "nói về ta thế nào,"),
        (5, 4.64, "khi đi lại trong tông,"),
        (3, 5.76, "Chỉ cần cháu"),
    ],
)
def test_bat_het_cau_hallucination(syllables: int, audio_s: float, nhan: str) -> None:
    assert _bat(audio_s, syllables) is True, nhan


# ── Lưới cũ bỏ lọt phần lớn (đó là lý do phải đổi) ──────────────────────────
def test_luoi_cu_bo_lot() -> None:
    # "thật đúng là": 12 ký tự, audio 3.64s. Lưới cũ (trung bình/ký tự) THOÁT.
    from subtitles_extractor.infrastructure.tts.timing_math import (
        GEMINI_BASE_OVERHEAD_S,
        GEMINI_PER_CHAR_S,
    )

    cu = is_abnormally_long(
        3.64, 12, base_overhead_s=GEMINI_BASE_OVERHEAD_S, per_char_s=GEMINI_PER_CHAR_S
    )
    assert cu is False  # lưới cũ bỏ lọt
    assert _bat(3.64, 3) is True  # lưới mới bắt được


# ── Câu đọc bình thường KHÔNG bị bắt ────────────────────────────────────────
def test_cau_binh_thuong_khong_bi_bat() -> None:
    # Audio ≈ 1-2x biên dưới là nhịp đọc bình thường của Gemini.
    # 3 âm tiết, biên dưới ≈ 0.89s. Đọc 1.5s (1.7x) -> bình thường.
    assert _bat(1.5, 3) is False
    # 5 âm tiết, biên dưới ≈ 1.33s. Đọc 2.0s (1.5x) -> bình thường.
    assert _bat(2.0, 5) is False


def test_nguong_x25() -> None:
    assert pytest.approx(2.5) == ABNORMAL_VS_FLOOR_RATIO
    # Ngay tại ngưỡng: 3 âm tiết biên dưới 0.888s, x2.5 = 2.22s.
    assert _bat(2.21, 3) is False
    assert _bat(2.24, 3) is True


def test_khong_bat_khi_thieu_du_lieu() -> None:
    assert _bat(0.0, 3) is False
    assert _bat(3.0, 0) is False


# ── Adapter: cả hai vòng retry truyền syllable_count ────────────────────────
def test_adapter_truyen_so_am_tiet() -> None:
    assert _GEMINI_SRC.count("syllable_count=dem_am_tiet(text)") == 2


def test_adapter_dung_luoi_moi() -> None:
    assert "is_abnormally_long_vs_floor(" in _GEMINI_SRC
    # [v3.23.262] Gemini nay dùng THUẦN âm tiết (bỏ nhánh fallback ký tự). Text không có
    # âm tiết (rỗng/toàn dấu) -> không coi là hallucination.
    assert "syllable_count > 0 and is_abnormally_long_vs_floor(" in _GEMINI_SRC
