"""[v3.23.223] NGHĨA THẮNG GIỚI HẠN — sửa hồi quy chất lượng dịch do v222 gây ra.

v222 sửa đúng công thức ngân sách ký tự, nhưng phạm hai sai lầm cùng lúc:

1. Đặt ``TRANSLATION_REF_SPEED = 1.0`` — "dòng đạt ngân sách sẽ đọc ở tốc độ tự nhiên,
   không nén chút nào". Quá thận trọng: trần siết tới **55/95 dòng**.
2. Nâng chỉ thị prompt thành "TRẦN CỨNG", kèm câu mời gọi tai hại: *"với dòng có
   max_chars rất nhỏ, hãy dịch thành mảnh câu cực gọn — đó là điều ĐÚNG"*.

Kết quả đo trên bản dịch thật do model sinh ra sau v222 (ngắn đi 19%):

* "cháu đã nói về chú thế nào," -> **"Chú?"** (mất sạch nghĩa)
* "Không phải chú không cho cháu cơ hội." -> **"Cho cơ hội,"** (mất phủ định kép, đảo ý)
* "trong lòng nghĩ về chú ra sao," -> **"Chú nghĩ gì?"** (đảo chủ thể)
* "Trở thành" -> **"Thành"** (cụt lủn, vô nghĩa)

TTS đạt chỉ số đẹp nhất từ trước tới nay (95% câu ở base 1.30x, lấn 0.00s) nhưng đó là
đánh đổi SAI. Nguyên tắc: người xem thà nghe giọng hơi gấp mà HIỂU ĐÚNG, còn hơn nghe rõ
mà nghĩa sai.

Mục tiêu đúng không phải "không nén" mà là "không nén QUÁ 2.0x" (ngưỡng tan formant).
Thang đo v216: nén 1.1x mất 5% sắc phụ âm | 1.6x mất 8% | 2.0x mất 19% -> nén tới 1.6x
vẫn nghe tốt, nên đặt trần ở đó.
"""

from __future__ import annotations

import pathlib

from subtitles_extractor.infrastructure.tts.timing_math import (
    QUALITY_STRETCH_CAP,
    TRANSLATION_REF_SPEED,
    readable_char_budget,
)

_TRANSLATOR_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/translation/gemini_translation_adapter.py"
).read_text(encoding="utf-8")


# ── Trần đặt ở mức "nén vẫn nghe tốt", không phải "không nén" ────────────────
def test_ref_speed_dat_o_muc_nen_van_nghe_tot() -> None:
    # Không được quay lại 1.0 (v222 — ép cắt nghĩa), cũng không được chạm trần chất lượng.
    assert TRANSLATION_REF_SPEED > 1.0, "ref_speed=1.0 ép model cắt nghĩa (hồi quy v222)"
    assert TRANSLATION_REF_SPEED < QUALITY_STRETCH_CAP, (
        "ref_speed phải nằm DƯỚI trần tan formant 2.0x — nếu bằng, mọi dòng đạt ngân "
        "sách vẫn bị nén tới ngưỡng méo giọng."
    )


def test_ngan_sach_rong_hon_han_v222() -> None:
    # Cùng khung, ngân sách mới phải rộng hơn đáng kể so với ref_speed=1.0 của v222.
    for khung in (0.64, 1.0, 1.5, 2.0, 3.0):
        v222 = readable_char_budget(khung, ref_speed=1.0)
        nay = readable_char_budget(khung)
        assert nay > v222, f"khung {khung}s: ngân sách không rộng hơn v222"


def test_ca_that_72_khong_con_bi_ep_cat_nghia() -> None:
    """#72 "cháu đã nói về chú thế nào," (27 ký tự), khung hiệu dụng 0.64s.

    v222 cấp 6 ký tự -> model dịch thành "Chú?" (mất sạch nghĩa).
    Nay ngân sách rộng hơn: dòng vẫn chật, nhưng model có chỗ giữ được ý.
    """
    ngan_sach = readable_char_budget(0.64)
    assert ngan_sach > 6, "vẫn cấp 6 ký tự -> model buộc phải cắt nghĩa như v222"
    assert ngan_sach >= 12


# ── Prompt: NGHĨA thắng giới hạn ────────────────────────────────────────────
def test_prompt_uu_tien_nghia_tren_gioi_han() -> None:
    assert "NGHĨA LUÔN THẮNG GIỚI HẠN" in _TRANSLATOR_SRC
    assert "GIỮ NGHĨA và chấp" in _TRANSLATOR_SRC


def test_prompt_cam_cac_kieu_cat_nghia_da_ghi_nhan() -> None:
    # Bốn dạng lỗi ĐO ĐƯỢC ở bản dịch v222 — phải bị cấm tường minh.
    for cam in ("bỏ từ phủ định", "đảo người nói/người nghe", "mảnh cụt vô nghĩa"):
        assert cam in _TRANSLATOR_SRC, cam


def test_prompt_co_vi_du_phan_dien_tu_ca_that() -> None:
    # Ví dụ phản diện mạnh hơn mô tả trừu tượng — dùng chính ca đã hỏng.
    assert "VÍ DỤ PHẢN DIỆN" in _TRANSLATOR_SRC
    assert "Cho cơ hội," in _TRANSLATOR_SRC  # ca mất phủ định kép
    assert "Cách rút gọn ĐÚNG" in _TRANSLATOR_SRC  # có nêu cách làm ĐÚNG


def test_prompt_khong_con_ngon_tu_moi_goi_cat_xen() -> None:
    # Câu tai hại của v222 phải biến mất hoàn toàn.
    assert "mảnh câu cực gọn" not in _TRANSLATOR_SRC
    assert "TRẦN CỨNG" not in _TRANSLATOR_SRC
