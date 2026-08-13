"""[v3.23.235] Dừng lấy mẫu lại khi đã hết cải thiện — cắt retry vô vọng của Gemini.

**Nghiệm thu v234 (Gemini, FLAC thật): ĐẠT.**

=========================  ============  ==========
Gemini                     nén median    lấn tổng
=========================  ============  ==========
v228 (lưới VieNeu, ăn may) 1.30          6.75s
v233 (lưới lỏng — hồi quy) 1.64          18.61s
**v234 (tách 2 câu hỏi)**  **1.30**      **5.60s**
=========================  ============  ==========

Lấn còn thấp hơn cả v228. Nhưng lộ ra cái giá: phiên TTS kéo từ 3.5 phút lên **7.5 phút**.

**Bug: retry vô vọng.** Nhiều câu có khung phụ đề NHỎ HƠN cả nhịp đọc chuẩn của engine::

    "biến thành"    10 ký tự, khung 0.20s — kỳ vọng 1.34s, nén 2.0x còn 0.67s
    "Chỉ cần cháu"  12 ký tự, khung 0.56s — kỳ vọng 1.44s, nén 2.0x còn 0.72s

Không bản nào vừa được, nên vòng retry chạy hết 10 lượt rồi mới chịu thua. Tệ hơn: bản
ngắn nhất thường đã tìm được từ lượt 2-3, các lượt sau lặp lại y hệt::

    "gần như"       0.84s ở lượt 3, rồi 7 lượt sau đều đúng 0.84s
    "Chỉ cần cháu"  1.24s ở lượt 2, rồi 8 lượt sau không lượt nào ngắn hơn

Giải pháp: đếm số lượt **LIÊN TIẾP** không cải thiện (không cắt cứng theo số lượt — chuỗi
cải thiện thường đứt quãng). Mô phỏng trên 10 ca thật:

===============  =================  =========================
patience         lượt gọi API       audio dài thêm (median)
===============  =================  =========================
10 (hiện tại)    100                —
2                38 (-62%)          +0.00s, nhưng max **+5.44s**
**3**            **54 (-46%)**      **+0.00s** (max +1.48s)
4                68 (-32%)          +0.00s (max +1.48s)
===============  =================  =========================

**ĐÁNH ĐỔI ĐÃ CHẤP NHẬN:** ca "biến thành" (khung 0.20s) sẽ dừng ở lượt 7 với bản 4.12s,
bỏ lỡ bản 2.64s vốn chỉ xuất hiện ở lượt thứ 10. Chấp nhận vì với khung 0.20s
thì **cả 4.12s lẫn 2.64s đều tràn vô vọng** — chênh lệch đó không cứu được gì, trong khi
46% lượt gọi API tiết kiệm được là thật. Ngay cả patience=4 cũng không giữ nổi ca này; chỉ
chạy đủ 10 lượt mới lấy được, tức phải trả giá bằng toàn bộ phần tiết kiệm.

**Kiểm toán engine-parity:** VieNeu cũng dùng ``shorter_take``, nhưng lưới của nó CHỈ bắt
hallucination — không có lưới "tràn khung". Bệnh retry-vô-vọng sinh ra từ lưới tràn khung
(thêm ở v234, chỉ Gemini), nên VieNeu không mắc. Không áp mù sang đó.
"""

from __future__ import annotations

import pathlib

from subtitles_extractor.infrastructure.tts.audio_utils import (
    RESAMPLE_PATIENCE,
    cap_nhat_ban_tot_nhat,
)

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")


def _chay_vong(
    ban_thu: list[float], patience: int = RESAMPLE_PATIENCE
) -> tuple[float, int]:
    """Mô phỏng vòng lấy mẫu lại; trả (độ dài bản giữ lại, số lượt đã gọi)."""
    best, streak = float("inf"), 0
    for luot, d in enumerate(ban_thu, 1):
        best, streak, nen_dung = cap_nhat_ban_tot_nhat(best, d, streak, patience)
        if nen_dung:
            return best, luot
    return best, len(ban_thu)


# ── Cắt được các ca lặp lại vô ích ──────────────────────────────────────────
def test_dung_som_khi_lap_lai_vo_ich() -> None:
    # Ca thật "gần như": 0.84s từ lượt 3, rồi 7 lượt sau y hệt.
    ban = [1.04, 1.04, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84]
    best, luot = _chay_vong(ban)
    assert best == 0.84  # vẫn giữ đúng bản ngắn nhất
    assert luot == 6  # dừng ở lượt 6 thay vì 10 -> tiết kiệm 4 lượt gọi API


def test_ca_that_chi_can_chau() -> None:
    # 1.24s tìm được ở lượt 2; 8 lượt sau không lượt nào ngắn hơn.
    ban = [10.68, 1.24, 10.68, 1.24, 1.24, 10.68, 1.80, 3.20, 3.20, 3.20]
    best, luot = _chay_vong(ban)
    assert best == 1.24
    assert luot == 5  # dừng sớm, không mất gì


# ── Nhưng KHÔNG giết ca cải thiện muộn ──────────────────────────────────────
def test_dem_lien_tiep_cho_chuoi_cai_thien_dut_quang() -> None:
    """Chuỗi cải thiện thường ĐỨT QUÃNG — đó là lý do không cắt cứng theo số lượt.

    Ca thật "biến thành": cải thiện ở lượt 1 (8.08s) rồi lượt 4 (4.12s), xen giữa là hai
    lượt tệ hơn. Cắt cứng ở lượt 3 sẽ dừng khi mới có 8.08s; đếm LIÊN TIẾP thì bộ đếm được
    reset ở lượt 4 và vòng lặp chạy tiếp tới 4.12s.
    """
    ban = [8.08, 11.76, 11.76, 4.12, 4.12, 6.08, 4.12, 6.08, 4.12, 2.64]
    best, luot = _chay_vong(ban)
    assert best == 4.12  # vượt qua được "vùng trũng" ở lượt 2-3
    assert luot == 7  # dừng khi thực sự hết cải thiện
    # ĐÁNH ĐỔI: bỏ lỡ bản 2.64s ở lượt 10. Chấp nhận — khung câu này chỉ 0.20s, cả 4.12s
    # lẫn 2.64s đều tràn vô vọng, nên chênh lệch đó không cứu được gì.
    assert min(ban) == 2.64


def test_chuoi_dut_quang_reset_bo_dem() -> None:
    # Cải thiện ở lượt 4 -> bộ đếm phải RESET, không được cộng dồn từ trước.
    best, streak, dung = cap_nhat_ban_tot_nhat(5.0, 6.0, 0)
    assert (best, streak, dung) == (5.0, 1, False)
    best, streak, dung = cap_nhat_ban_tot_nhat(best, 7.0, streak)
    assert streak == 2 and dung is False
    best, streak, dung = cap_nhat_ban_tot_nhat(best, 3.0, streak)  # cải thiện!
    assert (best, streak, dung) == (3.0, 0, False)


def test_dat_nguong_kien_nhan_thi_dung() -> None:
    best, streak, dung = 2.0, 2, False
    best, streak, dung = cap_nhat_ban_tot_nhat(best, 2.0, streak)
    assert streak == RESAMPLE_PATIENCE
    assert dung is True


def test_cai_thien_khong_dang_ke_van_tinh_la_khong_cai_thien() -> None:
    # Ngắn hơn 0.5ms không phải cải thiện thật -> không được reset bộ đếm.
    _best, streak, _dung = cap_nhat_ban_tot_nhat(2.000, 1.9995, 1)
    assert streak == 2


def test_lan_dau_luon_duoc_nhan() -> None:
    best, streak, dung = cap_nhat_ban_tot_nhat(float("inf"), 4.0, 0)
    assert (best, streak, dung) == (4.0, 0, False)


# ── Adapter: áp cho CẢ HAI vòng retry ───────────────────────────────────────
def test_ca_hai_vong_retry_deu_dung_som() -> None:
    assert _GEMINI_SRC.count("cap_nhat_ban_tot_nhat(") == 2
    assert _GEMINI_SRC.count("no_improve_streak = 0") == 2


def test_log_giai_thich_ly_do_dung() -> None:
    # Người dùng phải hiểu vì sao câu đó không vừa khung, thay vì thấy 10 dòng cảnh báo
    # giống hệt nhau rồi tưởng app treo.
    assert "Khung câu quá hẹp so với nhịp đọc engine" in _GEMINI_SRC
    assert "Dừng, dùng bản ngắn " in _GEMINI_SRC
