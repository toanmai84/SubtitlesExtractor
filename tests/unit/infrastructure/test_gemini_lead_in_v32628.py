"""[v3.23.228] BUG PARITY: Gemini phớt lờ cài đặt "Ăn gian đầu" của người dùng.

Rà soát chéo pipeline ba engine phát hiện: ``lead_in`` xuất hiện 5 lần trong Edge, 4 lần
trong VieNeu, và **0 lần trong Gemini**. Nghĩa là cùng một phụ đề với cùng cài đặt UI
("Ăn gian đầu = 0.25s"), Gemini có khung hẹp hơn tới 0.25s so với hai engine kia -> nén
giọng mạnh hơn và chồng tiếng nhiều hơn, mà người dùng không hề biết vì sao.

Mô phỏng trên 95 câu thật (cùng phụ đề, cùng audio, chỉ khác có/không lead-in):

===========================  ==========  =========  ==========
Gemini                       nén median  nén max    câu >1.8x
===========================  ==========  =========  ==========
TRƯỚC (v227)                 1.30        **2.00**   **4**
SAU  (v228)                  1.30        1.78       **0**
===========================  ==========  =========  ==========

30/95 câu được ăn gian, 28 câu đọc chậm lại (đỡ gấp). Hai câu vốn bị nén KỊCH TRẦN 2.0x
(tan formant) nay về mức an toàn.

Lead-in dùng đúng công thức của VieNeu v219 (``needed_lead_s``): CHỈ dời sớm những câu
thật sự cần khung rộng hơn — tránh lặp lại bug v218 (dời sớm cả loạt -> toàn bộ tiếng lệch
trước khẩu hình).
"""

from __future__ import annotations

import inspect
import pathlib

from subtitles_extractor.infrastructure.tts import gemini_tts_adapter
from subtitles_extractor.infrastructure.tts.timing_math import (
    effective_available_seconds,
    lead_in_seconds,
    total_speed_ratio,
)

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")


# ── Gemini nay CÓ lead-in, ở CẢ HAI nhánh ───────────────────────────────────
def test_gemini_dung_lead_in_dung_chung() -> None:
    assert "lead_in_seconds" in _GEMINI_SRC
    # Cả hai nhánh (Standard TTS và Native Audio) đều phải áp — bỏ sót một nhánh nghĩa là
    # người dùng vẫn gặp bug tuỳ model họ chọn.
    assert _GEMINI_SRC.count("play_start_sec") >= 4
    assert _GEMINI_SRC.count("prev_audio_end") >= 4


def test_gemini_ghi_master_tai_moc_phat_that() -> None:
    # Ghi vào ``start_sec`` thay vì ``play_start_sec`` là bug: lead-in tính rồi nhưng
    # KHÔNG áp vào master -> khung nới ra trên giấy, tiếng vẫn phát đúng mốc cũ.
    assert "int(play_start_sec * sr)" in _GEMINI_SRC
    assert "ss = int(start_sec * sr)" not in _GEMINI_SRC


def test_gemini_bao_cao_moc_da_doi() -> None:
    # SRT xuất ra phải khớp tiếng thật, và UI đếm "Dời mốc" mới đúng.
    assert "adjusted_start_sec=play_start_sec" in _GEMINI_SRC


def test_process_standard_nhan_moc_ket_thuc_cau_truoc() -> None:
    sig = inspect.signature(gemini_tts_adapter.GeminiTTSAdapter._process_standard)
    assert "prev_audio_end_sec" in sig.parameters


def test_lead_in_chi_an_gian_khi_that_su_can() -> None:
    """Đây là bài học v218/v219 — không được lặp lại ở Gemini.

    v218 dời sớm 250ms cho 93/95 câu (kể cả câu vốn đã vừa khung) -> gần như TOÀN BỘ tiếng
    vang trước khẩu hình. v219 sửa bằng ``needed_lead_s``: chỉ câu nào cần khung rộng hơn
    mới được dời.
    """
    # Câu vốn đã vừa khung -> needed_lead = 0 -> KHÔNG dời, dù có khoảng lặng phía trước.
    assert lead_in_seconds(10.0, 5.0, max_lead_s=0.25, needed_lead_s=0.0) == 0.0
    # Câu chật -> được dời, nhưng không quá trần.
    assert 0.0 < lead_in_seconds(10.0, 5.0, max_lead_s=0.25, needed_lead_s=0.5) <= 0.25
    # Không có khoảng lặng thật phía trước -> KHÔNG được đè lên câu trước.
    assert lead_in_seconds(10.0, 10.0, max_lead_s=0.25, needed_lead_s=0.5) == 0.0


# ── Tác động định lượng: hai câu kịch trần được cứu ─────────────────────────
def test_lead_in_cuu_cau_bi_nen_kich_tran() -> None:
    """Ca thật (khung hẹp 0.64s): không lead-in -> nén KỊCH TRẦN 2.0x (tan formant).

    Có lead-in 0.25s (khoảng lặng phía trước đủ rộng) -> khung 0.89s -> nén về mức an
    toàn. Mô phỏng trên 95 câu thật: Gemini từ **4 câu nén > 1.8x (2 câu kịch trần)** về
    **0 câu**.
    """
    voice_dur, base_speed, max_speed = 1.30, 1.3, 2.5
    start, end, next_start = 123.96, 124.60, 124.68
    prev_end = 123.50  # audio câu trước kết thúc sớm -> có khoảng lặng THẬT

    base_av = effective_available_seconds(start, end, next_start)
    khong_lead = total_speed_ratio(voice_dur, base_av, base_speed, max_speed)

    lead = lead_in_seconds(
        start, prev_end, max_lead_s=0.25,
        needed_lead_s=max(0.0, voice_dur / base_speed - base_av),
    )
    co_lead = total_speed_ratio(voice_dur, base_av + lead, base_speed, max_speed)

    assert lead > 0.0
    assert khong_lead >= 1.99  # KỊCH TRẦN chất lượng
    assert co_lead < 1.8  # thoát khỏi vùng nguy hiểm
