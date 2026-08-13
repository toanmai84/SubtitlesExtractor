"""[v3.23.224] Lưới phát hiện DỊCH THIẾU + bỏ chỉ thị ép độ dài theo câu gốc CJK.

Bằng chứng đo được (chuỗi v222 -> v223, cùng video, 95 dòng):

* Model chỉ dùng **52% ngân sách ký tự** được cấp -> ``max_chars`` KHÔNG còn là ràng buộc.
* Các dòng bị cắt nghĩa đều còn THỪA chỗ: #62 dùng 14/34, #68 dùng 18/43, #73 dùng 12/26.
* Thủ phạm: chỉ thị *"NHẮM độ dài (số âm tiết) TƯƠNG ĐƯƠNG hoặc NGẮN HƠN câu gốc"* — câu
  gốc là tiếng Trung, cô đọng hơn tiếng Việt về BẢN CHẤT (6 chữ Hán thường cần 9-12 âm
  tiết tiếng Việt). Chỉ thị ra lệnh làm điều bất khả thi; cách duy nhất để tuân thủ là
  CẮT NGHĨA.

Lưới phát hiện không dùng hằng số đoán mò: nó so mỗi dòng với TRUNG VỊ tỉ lệ độ dài của
CHÍNH bộ phim đó (cùng cặp ngôn ngữ, cùng văn phong).
"""

from __future__ import annotations

import pathlib

from subtitles_extractor.application.services.under_translation_guard import (
    find_under_translated,
)

_TRANSLATOR_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/translation/gemini_translation_adapter.py"
).read_text(encoding="utf-8")


def _job(extra: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """Nền một job dịch bình thường (tỉ lệ Việt/CJK ~2.5x) + các dòng cần kiểm."""
    # 9 ký tự CJK -> 26 ký tự Việt (tỉ lệ ~2.9x, sát thực tế đo được)
    binh_thuong = [
        (i, "你今天来这里做什么", "Hôm nay cháu tới đây làm gì") for i in range(1, 16)
    ]
    return binh_thuong + extra


# ── Bắt đúng ca thật đã hỏng ────────────────────────────────────────────────
def test_bat_duoc_ca_that_bi_cat_nghia() -> None:
    # #72: "cháu đã nói về chú thế nào," (27) bị rút thành "Chú?" (4).
    suspects = find_under_translated(_job([(72, "我是怎么说你的", "Chú?")]))
    assert [s.index for s in suspects] == [72]
    assert suspects[0].severity < 0.3  # ngắn hơn 30% mặt bằng phim


def test_bat_duoc_ca_mat_phu_dinh_kep() -> None:
    # #91: "Không phải chú không cho cháu cơ hội." -> "Cho cơ hội,"
    suspects = find_under_translated(
        _job([(91, "不是叔叔不给你机会", "Cho cơ hội,")])
    )
    assert 91 in [s.index for s in suspects]


def test_khong_bao_dong_gia_voi_ban_dich_du_nghia() -> None:
    # Chính dòng đó, dịch ĐẦY ĐỦ -> không được cảnh báo.
    suspects = find_under_translated(
        _job([(91, "不是叔叔不给你机会", "Không phải chú không cho cháu cơ hội.")])
    )
    assert suspects == []


def test_cau_ngan_tu_nhien_khong_bi_bao_dong() -> None:
    # "Ừm.", "Hả?" là dịch ĐÚNG của thán từ ngắn — gốc quá ngắn nên bị loại khỏi mẫu.
    suspects = find_under_translated(_job([(88, "嗯", "Ừm."), (56, "啊", "Hả?")]))
    assert suspects == []


# ── Kỷ luật thống kê ────────────────────────────────────────────────────────
def test_mau_qua_nho_thi_khong_ket_luan() -> None:
    # Dưới ngưỡng mẫu tối thiểu, trung vị không đáng tin -> im lặng, không đoán bừa.
    assert find_under_translated([(1, "你今天来这里做什么", "Ừ")]) == []


def test_tu_hieu_chinh_theo_tung_job() -> None:
    """Cùng một bản dịch có thể BÌNH THƯỜNG ở job này và ĐÁNG NGỜ ở job kia.

    Đó là điểm mấu chốt: không có hằng số phổ quát cho tỉ lệ độ dài Việt/CJK — nó phụ
    thuộc cặp ngôn ngữ, thể loại, văn phong. Lưới lấy chính bộ phim làm chuẩn.
    """
    goc = "你最近好吗"  # 5 ký tự — đủ dài để vào mẫu thống kê

    # Job A: mặt bằng dịch RẤT dài -> bản dịch cụt là bất thường.
    job_dai = [
        (i, goc, "Dạo này cháu khoẻ không, mọi việc ổn chứ") for i in range(1, 16)
    ]
    bat = find_under_translated([*job_dai, (99, goc, "Khoẻ chứ")])
    assert [x.index for x in bat] == [99]

    # Job B: mặt bằng dịch NGẮN (phụ đề cô đọng) -> ĐÚNG bản dịch đó lại BÌNH THƯỜNG.
    job_ngan = [(i, goc, "Khoẻ không") for i in range(1, 16)]
    assert find_under_translated([*job_ngan, (99, goc, "Khoẻ chứ")]) == []


def test_sap_xep_theo_muc_nang() -> None:
    suspects = find_under_translated(
        _job([(10, "我是怎么说你的", "Chú?"), (11, "我是怎么说你的", "Cháu nói về chú,")])
    )
    assert suspects[0].index == 10  # nặng nhất đứng đầu


# ── Prompt: đã gỡ chỉ thị ép độ dài theo câu gốc ────────────────────────────
def test_prompt_khong_con_ep_do_dai_theo_cau_goc() -> None:
    # Chỉ thị này ra lệnh làm điều BẤT KHẢ THI -> model chỉ tuân thủ được bằng cắt nghĩa.
    assert "TƯƠNG ĐƯƠNG hoặc NGẮN HƠN câu gốc" not in _TRANSLATOR_SRC
    assert "KHÔNG ÉP ĐỘ DÀI THEO CÂU GỐC" in _TRANSLATOR_SRC


def test_prompt_yeu_cau_dung_du_ngan_sach() -> None:
    # Model chỉ dùng 52% ngân sách -> ngân sách là để DÙNG, không phải để tiết kiệm.
    assert "DÙNG ĐỦ NGÂN SÁCH" in _TRANSLATOR_SRC
    assert "DỪNG rút gọn" in _TRANSLATOR_SRC


def test_prompt_cam_dao_huong_hanh_dong() -> None:
    # #72/#73 bị ĐẢO chủ ngữ <-> tân ngữ: lỗi nặng hơn câu dài, phải cấm tường minh.
    assert "GIỮ NGUYÊN HƯỚNG HÀNH ĐỘNG" in _TRANSLATOR_SRC
    assert "hoán đổi CHỦ NGỮ với TÂN NGỮ" in _TRANSLATOR_SRC
