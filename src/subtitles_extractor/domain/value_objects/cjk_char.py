"""[v3.23.619] Nhận biết ký tự CJK — quy tắc nghiệp vụ thuần, thuộc tầng domain.

Trước đây ``domain/entities/ocr_frame_result.py`` import hàm này từ
``application/services/cjk_utils.py`` (domain phụ thuộc ngược lên application, trái quy tắc
``presentation → application → domain``), lại import bên trong hàm nên trả chi phí tra
``sys.modules`` ở mỗi lần nối box. ``cjk_utils`` nay dùng lại hàm ở đây.
"""

from __future__ import annotations


def is_cjk_char(char: str) -> bool:
    """``True`` nếu ký tự thuộc CJK / Hangul / Kana."""
    if len(char) != 1:
        return False
    # Giữ nguyên check ord() cho hàm này để tối ưu tốc độ khi chỉ check 1 ký tự duy nhất
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xAC00 <= code <= 0xD7AF
        or 0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0xF900 <= code <= 0xFAFF
    )


__all__ = ["is_cjk_char"]
