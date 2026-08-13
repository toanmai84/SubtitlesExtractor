"""Ánh xạ ngôn ngữ → phiên bản OCR cần chỉ định tường minh.

LỊCH SỬ — MỘT GIẢ THUYẾT SAI ĐÃ ĐƯỢC GỠ (v3.23.324)
====================================================
v3.23.303 cho rằng ``latin``/``cyrillic``/``arabic``/``devanagari`` là mã ngôn ngữ hợp
lệ nhưng "chỉ có model dưới PP-OCRv5", nên thêm bảng ép ``ocr_version="PP-OCRv5"``.
Log build thực tế chứng minh giả thuyết đó SAI — vẫn báo::

    No models are available for lang='latin' and ocr_version='PP-OCRv5'

Đối chiếu mã nguồn paddleocr 3.7 (``_utils/langs.py`` + ``_pipelines/ocr.py``) cho thấy
nguyên nhân thật: đây là **TÊN NHÓM HỆ CHỮ VIẾT**, không phải mã ngôn ngữ. PaddleOCR
nhận mã **thành viên** (vd ``fr``) rồi mới tự suy ra model nhóm
(``latin_PP-OCRv5_mobile_rec``). Bản thân ``"latin"`` KHÔNG nằm trong ``LATIN_LANGS``
nên truyền vào luôn thất bại — không ``ocr_version`` nào cứu được.

Đã sửa tận gốc ở ``UI_LANGUAGE_CHOICES``: thay 4 mã nhóm bằng mã thành viên thật.

Bảng dưới đây nay TRỐNG. Giữ lại module vì cơ chế "một số ngôn ngữ cần chỉ định
``ocr_version``" vẫn có thể cần trong tương lai, và để các nơi gọi không phải sửa.
"""

from __future__ import annotations

from typing import Final

#: Ngôn ngữ cần chỉ định ``ocr_version`` tường minh. Hiện KHÔNG có mục nào —
#: xem lịch sử ở docstring module.
SCRIPT_GROUP_OCR_VERSIONS: Final[dict[str, str]] = {}


def resolve_script_group_version(language: str) -> str | None:
    """Trả về ``ocr_version`` cần chỉ định cho ``language``, nếu có.

    Args:
        language: Mã ngôn ngữ PaddleOCR (vd ``"fr"``, ``"ch"``).

    Returns:
        Chuỗi phiên bản khi ngôn ngữ cần chỉ định tường minh; ``None`` để PaddleOCR tự
        phân giải (trường hợp mặc định hiện nay).
    """
    return SCRIPT_GROUP_OCR_VERSIONS.get(language.strip().lower())


__all__ = ["SCRIPT_GROUP_OCR_VERSIONS", "resolve_script_group_version"]
