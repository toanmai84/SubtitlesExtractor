"""[v3.23.242] Hằng số biên dưới ĐÃ ĐO cho Edge (giọng vi-VN-NamMinhNeural).

Trước đây chỉ Gemini có hằng số biên dưới thật; Edge/VieNeu dùng giá trị TẠM (0.20/0.20).
Phiên này có FLAC Edge thật -> đo được, thay giá trị tạm cho Edge.

Đo trên 95 câu Edge thật (độ dài audio thô = thời lượng x tốc độ dùng, gom p10 mỗi nhóm
âm tiết):

    do_dai_toi_thieu = 0.132 + 0.217 x so_am_tiet     R² = **0.995**

R² còn cao hơn Gemini (0.89) vì Edge tất định — gần như không ngân/kéo dài ngẫu nhiên. Hai
kết quả đối chiếu thú vị:

* **Hệ số mỗi âm tiết TRÙNG Gemini** (0.217 vs 0.217) — cùng một nhịp đọc cơ bản.
* **Chi phí vào câu Edge THẤP hơn** (0.13s vs Gemini 0.23s) — Edge vào câu nhanh
  hơn 100ms.

Hệ quả đã kiểm chứng bằng số:

1. **9 câu "cần nén thêm" KHÔNG phải bug** (đã kết luận đúng ở v233). 8/9 câu nằm TRÊN sàn
   vật lý -> nén DSP nhẹ là hợp lệ (Edge API chỉ trả hơi dài). Chỉ 1 câu ("trở thành",
   khung 0.20s) dưới sàn — chính là dòng phụ đề vụn, TTS không cứu được.
2. **Gemini VẪN là engine chậm nhất** -> ngân sách dịch theo Gemini (v239) là an toàn
   cho Edge. Edge cần audio ngắn hơn Gemini ~100ms cho cùng số âm tiết.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.tts.timing_math import (
    EDGE_MIN_BASE_S,
    EDGE_MIN_PER_SYLLABLE_S,
    GEMINI_MIN_BASE_S,
    GEMINI_MIN_PER_SYLLABLE_S,
    min_speech_seconds,
)


def test_hang_so_edge_da_do() -> None:
    assert pytest.approx(0.13) == EDGE_MIN_BASE_S
    assert pytest.approx(0.217) == EDGE_MIN_PER_SYLLABLE_S


def test_edge_va_cau_nhanh_hon_gemini() -> None:
    # Chi phí cố định vào câu: Edge thấp hơn Gemini (đo thật).
    assert EDGE_MIN_BASE_S < GEMINI_MIN_BASE_S


def test_cung_nhip_doc_moi_am_tiet() -> None:
    # Hệ số mỗi âm tiết gần trùng nhau — cùng nhịp đọc cơ bản.
    assert pytest.approx(GEMINI_MIN_PER_SYLLABLE_S, abs=0.01) == EDGE_MIN_PER_SYLLABLE_S


def test_gemini_van_la_engine_cham_nhat() -> None:
    """Ngân sách dịch theo engine chậm nhất (v239) — xác nhận bằng số Edge thật.

    Với MỌI số âm tiết, Gemini cần audio dài hơn Edge -> ngân sách tính theo Gemini luôn
    an toàn cho Edge (Edge đọc kịp thoải mái).
    """
    for n in (1, 3, 5, 7, 10):
        edge = min_speech_seconds(n, EDGE_MIN_BASE_S, EDGE_MIN_PER_SYLLABLE_S)
        gemini = min_speech_seconds(n, GEMINI_MIN_BASE_S, GEMINI_MIN_PER_SYLLABLE_S)
        assert gemini > edge, f"{n} âm tiết: Gemini phải chậm hơn Edge"


def test_san_edge_thap_hon_gemini() -> None:
    # Sàn vật lý = audio ngắn nhất (1 âm tiết). Edge vào câu nhanh -> sàn thấp hơn.
    san_edge = min_speech_seconds(1, EDGE_MIN_BASE_S, EDGE_MIN_PER_SYLLABLE_S)
    san_gemini = min_speech_seconds(1, GEMINI_MIN_BASE_S, GEMINI_MIN_PER_SYLLABLE_S)
    assert san_edge < san_gemini
    assert san_edge == pytest.approx(0.347, abs=0.01)
