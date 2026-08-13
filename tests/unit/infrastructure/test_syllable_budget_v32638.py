"""[v3.23.238] Ngân sách DỊCH chuyển từ ký tự sang ÂM TIẾT.

Tiếp nối v237: tiếng Việt đơn âm tiết nên thời gian đọc tỉ lệ với số âm tiết, không phải
số ký tự (mô hình ký tự R²=0.07, mô hình âm tiết biên dưới R²=0.89). Ngân sách gửi cho
model dịch vì thế cũng phải tính theo âm tiết.

**Nhưng model dịch đếm KÝ TỰ đầu ra**, nên chuỗi đúng là:

    khung -> ngân sách ÂM TIẾT (đúng vật lý) -> quy sang KÝ TỰ (để gửi model)

Quy đổi: median **4.17 ký tự/âm tiết** (đo trên bản dịch Việt thật).

**Chọn ``ref_speed`` là QUYẾT ĐỊNH ĐÁNH ĐỔI, không phải con số đo được.** Ngân sách càng
chặt -> TTS càng ít chồng tiếng, nhưng model càng bị ép cắt nghĩa (bẫy v222). Đo trên 95
dòng thật, số dòng bị ép NGẮN HƠN bản dịch hiện tại (vốn đã tốt):

========  ===================  ============================================
ref       dòng bị ép           đánh giá
========  ===================  ============================================
1.3       25/95                cắt cả "Cháu không biết." -> quá thô bạo
1.5       15/95
**1.7**   **5/95**             **Toan chọn** — chỉ siết dòng dài bất thường
2.0       1/95                 gần như không ràng buộc gì
========  ===================  ============================================

Năm dòng bị siết ở ref=1.7 đều là dòng CHẬT THẬT, và chỉ siết nhẹ 1-5 ký tự::

    khung 0.20s  "Trở thành"              9 ký tự -> ngân sách 4
    khung 0.52s  "Thanh Linh Đan"        14 ký tự -> ngân sách 13
    khung 0.76s  "cháu phải sớm đột phá," 22 ký tự -> ngân sách 21
"""

from __future__ import annotations

import pathlib

import pytest

from subtitles_extractor.infrastructure.tts.timing_math import (
    CHARS_PER_SYLLABLE,
    SYLLABLE_BUDGET_REF_SPEED,
    readable_syllable_budget,
    syllable_budget_to_chars,
)

_DICH_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/translation/gemini_translation_adapter.py"
).read_text(encoding="utf-8")


# ── Ngân sách âm tiết ───────────────────────────────────────────────────────
def test_ngan_sach_tang_theo_khung() -> None:
    assert readable_syllable_budget(0.5) < readable_syllable_budget(1.0)
    assert readable_syllable_budget(1.0) < readable_syllable_budget(1.8)


def test_luon_cho_it_nhat_mot_am_tiet() -> None:
    # Khung 0.20s không đọc nổi gì, nhưng ngân sách 0 thì model không dịch được -> sàn 1.
    assert readable_syllable_budget(0.20) >= 1
    assert readable_syllable_budget(0.0) >= 1
    assert readable_syllable_budget(-1.0) >= 1


def test_ref_cang_cao_ngan_sach_cang_rong() -> None:
    chat = readable_syllable_budget(1.0, ref_speed=1.3)
    rong = readable_syllable_budget(1.0, ref_speed=2.0)
    assert rong > chat


def test_ref_giu_can_can_5_dong() -> None:
    # Toan chọn cán cân "cân bằng" = ~5/95 dòng bị ép. [v3.23.239] ngân sách nay dùng hằng
    # số biên dưới GEMINI (đo thật) thay hằng số tạm 0.20/0.20; hằng số Gemini chặt hơn
    # nên ref nới 1.7 -> 1.9 để GIỮ NGUYÊN cán cân 5 dòng. Đổi số này = đổi cán cân.
    assert pytest.approx(1.9) == SYLLABLE_BUDGET_REF_SPEED


# ── Quy đổi âm tiết -> ký tự ────────────────────────────────────────────────
def test_quy_doi_sang_ky_tu() -> None:
    assert pytest.approx(4.2, abs=0.1) == CHARS_PER_SYLLABLE
    assert syllable_budget_to_chars(1) == 4
    assert syllable_budget_to_chars(10) == 42
    # Không bao giờ trả 0 — model cần ít nhất một ký tự để dịch.
    assert syllable_budget_to_chars(0) >= 1


# ── Năm ca THẬT bị siết ở ref=1.7 (đúng đối tượng) ──────────────────────────
@pytest.mark.parametrize(
    ("khung_s", "ban_dich", "so_ky_tu"),
    [
        (0.20, "Trở thành", 9),
        (0.44, "Đã là một", 9),
        (0.52, "Thanh Linh Đan", 14),
        (0.64, "đã nói ta thế nào,", 18),
        (0.76, "cháu phải sớm đột phá,", 22),
    ],
)
def test_dong_chat_that_bi_siet(khung_s: float, ban_dich: str, so_ky_tu: int) -> None:
    ngan_sach = syllable_budget_to_chars(readable_syllable_budget(khung_s))
    assert ngan_sach < so_ky_tu, ban_dich  # đúng: đây là dòng cần siết


def test_dong_binh_thuong_khong_bi_dung_toi() -> None:
    """Ở ref=1.3 câu này bị ép còn 8 ký tự — cắt vào nghĩa. Ở ref=1.7 thì không."""
    ngan_sach = syllable_budget_to_chars(readable_syllable_budget(0.60))
    assert ngan_sach >= len("Cháu không biết.")


# ── Tầng dịch dùng đúng chuỗi âm tiết -> ký tự ──────────────────────────────
def test_tang_dich_dung_ngan_sach_am_tiet() -> None:
    assert "syllable_budget_to_chars(readable_syllable_budget(usable_s))" in _DICH_SRC
    # Không được gọi thẳng ngân sách theo ký tự nữa.
    assert "return readable_char_budget(usable_s)" not in _DICH_SRC
