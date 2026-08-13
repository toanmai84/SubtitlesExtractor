"""[v3.23.232] Mô hình độ dài giọng phải theo TỪNG ENGINE + khôi phục fix v230.

**Sai lầm dữ liệu ở v231 (đã sửa).** Phiên trước tôi phân tích một bộ FLAC+CSV và tưởng là
Edge, trong khi ``Engine: Gemini TTS`` ghi rõ trong file cấu hình. Hậu quả: số liệu
"chồng tiếng 6.75s / nén max 2.00x" của **Gemini** bị đem so với "0.95s / 1.62x" của
**Edge**, rồi kết luận sai rằng fix v230 gây hồi quy -> hoàn nguyên oan một fix đúng.
v232 khôi phục lại v230; việc nghiệm thu nó vẫn cần một bộ FLAC **Edge** thật.

**Bug thật, tìm ra khi phân tích lại cho đúng engine.** Hồi quy trên 95 câu Gemini::

    VieNeu : do_dai = 0.302 + 0.0486 x n   (sai số chuẩn 0.273s)
    Gemini : do_dai = 0.839 + 0.0499 x n   (sai số chuẩn 0.725s)

Chi phí cố định mỗi câu của Gemini là **0.84s** so với 0.30s của VieNeu (**+178%**) —
model native audio vào câu chậm rãi hơn hẳn; tốc độ mỗi ký tự thì gần như y hệt (+3%).

Lưới "audio dài bất thường" (v221/v227) lại dùng hằng số VieNeu cho MỌI engine -> với
Gemini nó bắt **6/95 câu OAN**: "Tu vi,", "Haiz,", "Khụ khụ.", "Ừm.", "Thanh Linh Đan,",
"Chú nói đi." Mỗi câu = một lượt **retry gọi API TỐN TIỀN**, và bản "ngắn nhất" được chọn
lại có thể là bản đọc vội, kém tự nhiên.
"""

from __future__ import annotations

import pathlib

import pytest

from subtitles_extractor.infrastructure.tts.timing_math import (
    GEMINI_BASE_OVERHEAD_S,
    GEMINI_PER_CHAR_S,
    SPEECH_BASE_OVERHEAD_S,
    SPEECH_PER_CHAR_S,
    expected_speech_seconds,
    generation_time_cap_seconds,
    is_abnormally_long,
)

# Sáu ca THẬT bị lưới bắt oan khi dùng hằng số VieNeu cho Gemini:
# (văn bản, số ký tự, độ dài Gemini sinh ra)
_CA_GEMINI_BINH_THUONG = [
    ("Tu vi,", 6, 2.42),
    ("Thanh Linh Đan,", 15, 3.42),
    ("Haiz,", 5, 2.64),
    ("Khụ khụ.", 8, 2.78),
    ("Ừm.", 3, 2.04),
    ("Chú nói đi.", 11, 2.92),
]


def _ab_gemini(duration_s: float, char_count: int) -> bool:
    return is_abnormally_long(
        duration_s,
        char_count,
        base_overhead_s=GEMINI_BASE_OVERHEAD_S,
        per_char_s=GEMINI_PER_CHAR_S,
    )


# ── Mô hình Gemini khác hẳn VieNeu ở CHI PHÍ CỐ ĐỊNH ────────────────────────
def test_gemini_vao_cau_cham_hon_nhieu() -> None:
    assert GEMINI_BASE_OVERHEAD_S > SPEECH_BASE_OVERHEAD_S * 2.5  # +178%
    # Nhưng tốc độ đọc từng ký tự thì gần như y hệt.
    assert pytest.approx(SPEECH_PER_CHAR_S, abs=0.005) == GEMINI_PER_CHAR_S


def test_ky_vong_gemini_luon_dai_hon_vieneu() -> None:
    for n in (3, 8, 15, 30):
        vieneu = expected_speech_seconds(n)
        gemini = expected_speech_seconds(n, GEMINI_BASE_OVERHEAD_S, GEMINI_PER_CHAR_S)
        assert gemini > vieneu


# ── 6 ca thật: KHÔNG được báo động giả nữa ──────────────────────────────────
@pytest.mark.parametrize(("text", "n_char", "duration_s"), _CA_GEMINI_BINH_THUONG)
def test_khong_con_retry_oan(text: str, n_char: int, duration_s: float) -> None:
    # Với hằng số VieNeu: bị bắt OAN -> tốn một lượt gọi API.
    assert is_abnormally_long(duration_s, n_char) is True, text
    # Với hằng số Gemini: nhận ra đây là nhịp đọc BÌNH THƯỜNG của engine này.
    assert _ab_gemini(duration_s, n_char) is False, text


def test_tiet_kiem_dung_6_luot_goi_api() -> None:
    bat_oan = sum(
        1 for _, n, d in _CA_GEMINI_BINH_THUONG if is_abnormally_long(d, n)
    )
    bat_dung = sum(1 for _, n, d in _CA_GEMINI_BINH_THUONG if _ab_gemini(d, n))
    assert bat_oan == 6  # hằng số VieNeu -> 6 lượt gọi API lãng phí
    assert bat_dung == 0  # hằng số Gemini -> không câu nào bị bắt oan


# ── Nhưng ca thảm hoạ THẬT vẫn phải bị chặn ─────────────────────────────────
def test_van_bat_duoc_hallucination_that() -> None:
    # Ca thảm hoạ kiểu "Kiều Kiều à," -> 32.10s: vượt xa 3x kỳ vọng của CẢ HAI mô hình.
    assert _ab_gemini(32.10, 12) is True
    # Câu ngắn ngân lê thê (gấp 4 lần nhịp Gemini) vẫn bị bắt.
    assert _ab_gemini(4.20, 3) is True


def test_tran_sinh_cung_theo_mo_hinh_engine() -> None:
    # Trần thời lượng sinh cũng phải nới theo nhịp chậm của Gemini, nếu không sẽ cắt cụt
    # những câu Gemini đọc bình thường.
    cap_vieneu = generation_time_cap_seconds(15)
    cap_gemini = generation_time_cap_seconds(
        15, base_overhead_s=GEMINI_BASE_OVERHEAD_S, per_char_s=GEMINI_PER_CHAR_S
    )
    assert cap_gemini > cap_vieneu
    # "Thanh Linh Đan," (15 ký tự) Gemini đọc 3.42s -> phải nằm DƯỚI trần, không bị chặn.
    assert cap_gemini > 3.42


# ── Adapter dùng đúng mô hình của mình ──────────────────────────────────────
def test_gemini_adapter_dung_hang_so_rieng() -> None:
    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    # [v3.23.234] Hai vòng retry (Standard + Native) nay gọi CHUNG một hàm quyết định
    # (``_ly_do_lay_mau_lai``), nên hằng số chỉ còn xuất hiện MỘT lần — đúng DRY. Điều
    # cần bảo đảm là cả hai vòng đều đi qua hàm đó (test riêng ở v32634).
    # [v3.23.237] Chỉ còn MỘT chỗ dùng mô hình trung bình: lưới hallucination. Lưới sàn
    # vật lý nay dùng mô hình BIÊN DƯỚI THEO ÂM TIẾT (``GEMINI_MIN_*``) — hai câu hỏi khác
    # nhau thì cần hai mô hình khác nhau:
    #   * "audio có bất thường so với nhịp TRUNG BÌNH?" -> hallucination
    #   * "audio NGẮN NHẤT có vừa khung không?"        -> biên dưới
    # [v3.23.262] Gemini BỎ hoàn toàn mô hình trung bình theo ký tự — hallucination nay
    # THUẦN âm tiết (biên dưới ``GEMINI_MIN_*``). Không còn dùng GEMINI_BASE_OVERHEAD_S.
    assert src.count("base_overhead_s=GEMINI_BASE_OVERHEAD_S") == 0
    assert "min_base_s=GEMINI_MIN_BASE_S" in src
    assert src.count("_ly_do_lay_mau_lai") == 3  # 1 định nghĩa + 2 lời gọi


def test_vieneu_van_giu_mo_hinh_cua_no() -> None:
    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
    ).read_text(encoding="utf-8")
    # VieNeu KHÔNG được vô tình dùng hằng số Gemini.
    assert "GEMINI_BASE_OVERHEAD_S" not in src


# ── Khôi phục fix v230 (lý do hoàn nguyên ở v231 đã bị bác bỏ) ──────────────
def test_da_khoi_phuc_fix_v230() -> None:
    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/edge_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "needed_early = max(" in src
    assert "early = min(lipsync_early, needed_early)" in src
    # Ghi lại lịch sử ngay tại code để không ai hoàn nguyên lần nữa vì lý do sai.
    # (Kiểm theo từ khoá rời, không theo cả câu — comment có thể được xuống dòng lại.)
    assert "lấy nhầm từ phiên" in src
    assert "Engine: Gemini TTS" in src
