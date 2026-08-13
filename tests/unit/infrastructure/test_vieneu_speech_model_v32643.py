"""[v3.23.243] Hằng số biên dưới ĐÃ ĐO cho VieNeu — hoàn tất cả BA engine.

Đây là mảnh cuối. Trước đây Edge (v242) và Gemini (v237) đã có hằng số đo thật; VieNeu vẫn
dùng giá trị tạm 0.20/0.20. Phiên này có FLAC VieNeu (giọng Doan) -> đo trên 95 câu:

    do_dai_toi_thieu = 0.245 + 0.196 x so_am_tiet     R² = **0.983**

BỨC TRANH ĐẦY ĐỦ CẢ BA ENGINE:

========  ==============  ================  ======
engine    vào câu (s)     mỗi âm tiết (s)   R²
========  ==============  ================  ======
Edge      0.132           0.217             0.995
Gemini    0.231           0.217             0.89
VieNeu    0.245           0.196             0.983
========  ==============  ================  ======

Quan sát: VieNeu vào câu CHẬM NHẤT (0.245) nhưng mỗi âm tiết NHANH NHẤT (0.196). Điều này
có thể lật giả định "Gemini chậm nhất" (nền tảng của ngân sách dịch v239) — nên phải kiểm.

**Kết quả kiểm: giả định v239 VẪN ĐÚNG.** Điểm cắt VieNeu/Gemini ở n=0.67 âm tiết; với
mọi câu thật (n>=1) Gemini luôn chậm nhất hoặc bằng. Ngay câu 1 âm tiết: Gemini 0.448s vs
VieNeu 0.441s (Gemini vẫn chậm hơn 7ms). Vậy ngân sách dịch theo Gemini an toàn cho cả ba,
nay xác nhận bằng BA engine đo thật.

**KHÔNG thêm sàn vật lý cho VieNeu** (tránh over-engineering): VieNeu không có lưới TRÀN
KHUNG như Gemini, nên câu dưới sàn (ngắn) không bị lưới hallucination bắt -> không có
retry vô ích để mà lọc. Đo thật: 1 câu dưới sàn (#78 "trở thành"), 0 câu retry oan.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.tts.timing_math import (
    EDGE_MIN_BASE_S,
    EDGE_MIN_PER_SYLLABLE_S,
    GEMINI_MIN_BASE_S,
    GEMINI_MIN_PER_SYLLABLE_S,
    VIENEU_MIN_BASE_S,
    VIENEU_MIN_PER_SYLLABLE_S,
    min_speech_seconds,
)


def test_hang_so_vieneu_da_do() -> None:
    assert pytest.approx(0.245) == VIENEU_MIN_BASE_S
    assert pytest.approx(0.196) == VIENEU_MIN_PER_SYLLABLE_S


def test_vieneu_vao_cau_cham_nhat() -> None:
    # Chi phí cố định vào câu: VieNeu cao nhất trong ba engine.
    assert VIENEU_MIN_BASE_S > GEMINI_MIN_BASE_S > EDGE_MIN_BASE_S


def test_vieneu_moi_am_tiet_nhanh_nhat() -> None:
    # Nhưng mỗi âm tiết lại nhanh nhất.
    assert VIENEU_MIN_PER_SYLLABLE_S < GEMINI_MIN_PER_SYLLABLE_S
    assert VIENEU_MIN_PER_SYLLABLE_S < EDGE_MIN_PER_SYLLABLE_S


def test_gemini_van_cham_nhat_voi_moi_cau_that() -> None:
    """Nền tảng của ngân sách dịch (v239): Gemini là engine chậm nhất.

    Dù VieNeu vào câu chậm hơn, Gemini vẫn cần audio dài nhất ở MỌI câu >= 1 âm tiết —
    xác nhận ngân sách dịch theo Gemini an toàn cho cả ba engine.
    """
    for n in range(1, 15):
        g = min_speech_seconds(n, GEMINI_MIN_BASE_S, GEMINI_MIN_PER_SYLLABLE_S)
        v = min_speech_seconds(n, VIENEU_MIN_BASE_S, VIENEU_MIN_PER_SYLLABLE_S)
        e = min_speech_seconds(n, EDGE_MIN_BASE_S, EDGE_MIN_PER_SYLLABLE_S)
        assert g >= v, f"{n} âm tiết: Gemini phải >= VieNeu"
        assert g >= e, f"{n} âm tiết: Gemini phải >= Edge"


def test_diem_cat_vieneu_gemini_duoi_mot_am_tiet() -> None:
    # VieNeu chỉ chậm hơn Gemini ở n < 0.67 — không tồn tại câu thật nào.
    diem_cat = (VIENEU_MIN_BASE_S - GEMINI_MIN_BASE_S) / (
        GEMINI_MIN_PER_SYLLABLE_S - VIENEU_MIN_PER_SYLLABLE_S
    )
    assert diem_cat < 1.0


def test_ba_engine_deu_co_hang_so_rieng() -> None:
    # Cột mốc: cả ba engine nay đều đo thật, không còn engine nào dùng giá trị tạm.
    bo_ba = {
        (EDGE_MIN_BASE_S, EDGE_MIN_PER_SYLLABLE_S),
        (GEMINI_MIN_BASE_S, GEMINI_MIN_PER_SYLLABLE_S),
        (VIENEU_MIN_BASE_S, VIENEU_MIN_PER_SYLLABLE_S),
    }
    assert len(bo_ba) == 3  # ba cặp phân biệt
