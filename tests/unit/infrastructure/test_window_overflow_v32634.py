"""[v3.23.234] Tách HAI câu hỏi khác nhau — sửa hồi quy do chính v232 gây ra.

**Câu hỏi treo đã đóng.** FLAC Gemini thật cho thấy lặng đầu còn sót chỉ **0ms**
(median), so với 40ms của VieNeu/Edge -> ``trim_edge_silence`` đã cắt sạch. Chi phí cố
định 0.84s/câu của Gemini là **NHỊP ĐỌC THẬT**, không phải khoảng lặng thừa.

**Hồi quy do v232.** Tôi từng gọi 6 câu bị lưới bắt là "retry oan" và nới ngưỡng cho khớp
nhịp đọc chậm của Gemini. Nhưng những lần retry đó KHÔNG oan — ``shorter_take`` giữ bản
ngắn nhất trong N lần, ép model đọc gọn cho vừa khung (đo trên log: rút được 15-20%). Bỏ
chúng đi:

=========================  ============  ===================
Gemini                     nén median    lấn tổng
=========================  ============  ===================
v228 (lưới VieNeu, chặt)   1.30          6.75s
v233 (lưới Gemini, lỏng)   **1.64**      **18.61s** (x2.8)
=========================  ============  ===================

**Nguyên nhân gốc: gộp nhầm hai câu hỏi hoàn toàn khác nhau.**

1. *"Audio có bất thường so với NHỊP ĐỌC của engine không?"* -> hallucination. Phụ thuộc
   engine (VieNeu 0.30s/câu, Gemini 0.84s/câu).
2. *"Audio có VỪA KHUNG PHỤ ĐỀ không?"* -> tràn khung. **Không liên quan gì** tới việc
   engine đọc nhanh hay chậm.

Lưới VieNeu bắt được câu tràn khung chỉ vì ĂN MAY (nó chặt nên bắt bừa nhiều thứ). v234
hỏi đúng câu hỏi (2) bằng ``exceeds_window_even_compressed``.

Kết quả mô phỏng trên 95 câu Gemini thật:

* 23 câu lấn nặng: v233 bắt **2/23**, v234 bắt **23/23**.
* 54 câu vốn đã vừa khung: lưới VieNeu retry oan 3 câu, v234 retry oan **0**.
"""

from __future__ import annotations

import pathlib

import pytest

from subtitles_extractor.infrastructure.tts.timing_math import (
    GEMINI_BASE_OVERHEAD_S,
    GEMINI_PER_CHAR_S,
    exceeds_window_even_compressed,
    is_abnormally_long,
)

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")

# Ca THẬT gây lấn nặng nhất (văn bản, số ký tự, audio, khung, lấn thực tế)
_CA_TRAN_KHUNG = [
    ("ta sẽ không cho cô", 18, 4.24, 0.72, 1.40),
    ("tất nhiên sẽ chịu", 17, 4.46, 0.84, 1.39),
    ("Giới tính", 9, 3.30, 0.48, 1.17),
    ("Đã là một", 9, 2.88, 0.44, 1.00),
    ("Chỉ cần cháu", 12, 3.30, 0.56, 0.99),
]


# ── Câu hỏi (2): audio có vừa khung không? ──────────────────────────────────
@pytest.mark.parametrize(
    ("text", "n_char", "duration_s", "window_s", "_lan"), _CA_TRAN_KHUNG
)
def test_bat_duoc_cau_tran_khung(
    text: str, n_char: int, duration_s: float, window_s: float, _lan: float
) -> None:
    # Lưới hallucination (mô hình Gemini) BỎ SÓT — vì với nhịp đọc chậm của Gemini thì
    # những câu này KHÔNG bất thường. Chúng chỉ đơn giản là quá dài so với khung.
    assert exceeds_window_even_compressed(duration_s, window_s, 2.0) is True, text


def test_cau_vua_khung_khong_bi_dung_toi() -> None:
    # Nén 2.0x là vừa -> KHÔNG lấy mẫu lại (chống retry oan, tiết kiệm API).
    assert exceeds_window_even_compressed(2.40, 1.30, 2.0) is False
    assert exceeds_window_even_compressed(1.20, 1.50, 2.0) is False
    # Vừa khít ngưỡng cũng không được báo động giả.
    assert exceeds_window_even_compressed(2.00, 1.00, 2.0) is False


def test_khung_khong_ro_thi_bo_qua() -> None:
    # available_s = 0 (không rõ khung) -> không được suy đoán bừa.
    assert exceeds_window_even_compressed(5.0, 0.0, 2.0) is False


def test_tran_nen_cang_cao_cang_it_phai_lay_mau_lai() -> None:
    # Cùng một câu: cho phép nén mạnh hơn -> vừa khung -> khỏi retry.
    assert exceeds_window_even_compressed(3.0, 1.0, 2.0) is True
    assert exceeds_window_even_compressed(3.0, 1.0, 3.0) is False


# ── Hai câu hỏi phải TÁCH BẠCH ──────────────────────────────────────────────
def test_hai_luoi_do_hai_thu_khac_nhau() -> None:
    """Câu "Giới tính": 9 ký tự, Gemini đọc 3.30s, khung 0.48s.

    * Hallucination? KHÔNG — với nhịp Gemini (0.84 + 0.05*9 = 1.29s kỳ vọng), 3.30s chỉ
      gấp 2.6x, dưới ngưỡng 3.0x.
    * Vừa khung? KHÔNG — nén kịch trần 2.0x còn 1.65s, vẫn tràn 1.17s sang câu sau.

    Hai câu trả lời khác nhau cho cùng một audio. Gộp chúng làm một là gốc rễ của hồi quy
    v232.
    """
    hall = is_abnormally_long(
        3.30, 9, base_overhead_s=GEMINI_BASE_OVERHEAD_S, per_char_s=GEMINI_PER_CHAR_S
    )
    tran = exceeds_window_even_compressed(3.30, 0.48, 2.0)
    assert hall is False  # không phải hallucination
    assert tran is True  # nhưng vẫn phải lấy mẫu lại


def test_van_bat_hallucination_that_du_khung_rong() -> None:
    # Ca thảm hoạ: "Tu vi," ngân 7.68s. Khung rộng 5s -> không tràn, nhưng VẪN bất thường.
    assert exceeds_window_even_compressed(7.68, 5.0, 2.0) is False
    assert (
        is_abnormally_long(
            7.68,
            6,
            base_overhead_s=GEMINI_BASE_OVERHEAD_S,
            per_char_s=GEMINI_PER_CHAR_S,
        )
        is True
    )


# ── Adapter: cả HAI vòng retry dùng chung một hàm quyết định ────────────────
def test_ca_hai_vong_retry_dung_chung_ham_quyet_dinh() -> None:
    # Định nghĩa + 2 lời gọi (Standard và Native) = 3 lần xuất hiện.
    assert _GEMINI_SRC.count("_ly_do_lay_mau_lai") == 3


def test_vong_retry_nhan_duoc_khung_that() -> None:
    # Không truyền khung thì lưới (2) vô dụng — đây là lỗi dễ mắc nhất khi nối dây.
    assert _GEMINI_SRC.count("available_s=effective_available_seconds(") == 2


def test_log_noi_ro_ly_do_lay_mau_lai() -> None:
    # Log cũ chỉ nói "audio DÀI BẤT THƯỜNG" cho mọi trường hợp -> chẩn đoán sai suốt 2
    # phiên. Nay phải phân biệt rõ hai nguyên nhân.
    assert "audio DÀI BẤT THƯỜNG so với nhịp đọc Gemini" in _GEMINI_SRC
    assert "audio TRÀN KHUNG dù đã nén hết cỡ" in _GEMINI_SRC
