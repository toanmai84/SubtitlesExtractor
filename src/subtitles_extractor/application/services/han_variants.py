"""[v3.23.615] So khớp chữ Hán không phân biệt phồn thể/giản thể (chỉ để SO, không sửa).

PaddleOCR đọc lẫn chữ phồn vào phụ đề giản thể khi nét chữ mờ. Dữ liệu OCR thật 14/10
(phim "天骄女友退婚" tập 1): câu "楚瑶已筑基圆满" có 10/12 khung đọc thành "楚瑤"
(瑤 phồn) và bản phụ đề giữ "楚瑤". Bảng nhân vật (do model viết) ghi "楚瑶", nên mọi
phép so tên đúng từng chữ đều trượt: chẩn đoán xếp nữ chính vào "chỉ nhận ra qua hình"
dù tên cô có trong thoại, khối nhắc thuật ngữ theo lô bỏ sót thuật ngữ viết khác dạng,
và bước thay tên còn sót trong bản dịch không nhận ra "楚瑤".

KHÔNG tự đổi chữ trong phụ đề: đo trên phụ đề thật, phụ đề chuẩn cũng dùng lẫn phồn/giản
(xem ``text_correction.clean_edge_noise``). Hàm ở đây chỉ dùng hai phía của một phép so.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _fold_table() -> dict[int, str]:
    """Bảng ``str.translate`` phồn → giản (nạp trễ: tránh vòng import gói pipeline)."""
    from subtitles_extractor.application.services.subtitle_pipeline.constants import (
        HAN_TRADITIONAL_TO_SIMPLIFIED,
    )

    return str.maketrans(HAN_TRADITIONAL_TO_SIMPLIFIED)


def fold_han_variants(text: str) -> str:
    """Đưa chữ Hán phồn thể về giản thể để so khớp (hàm thuần).

    Args:
        text: Chuỗi bất kỳ; chữ ngoài bảng phồn → giản (kể cả chữ Latin) giữ nguyên.

    Returns:
        Chuỗi cùng độ dài, chữ phồn trong bảng đã thành chữ giản.
    """
    return text.translate(_fold_table()) if text else text


__all__ = ["fold_han_variants"]
