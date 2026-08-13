"""Ngôn ngữ OCR — canh giữ mã hợp lệ (cập nhật v3.23.324).

⚠️ BÀI HỌC: bộ test này TRƯỚC ĐÂY khẳng định một điều SAI.

v3.23.303 giả định ``latin``/``cyrillic``/``arabic``/``devanagari`` là mã ngôn ngữ hợp
lệ "chỉ có model dưới PP-OCRv5", và test khi đó chỉ kiểm bảng ánh xạ khớp với giả định
đó — nên nó PASS trong khi tính năng vẫn hỏng ngoài thực tế.

Log build thật bác bỏ giả định::

    No models are available for lang='latin' and ocr_version='PP-OCRv5'

Đối chiếu mã nguồn paddleocr 3.7 cho nguyên nhân thật: đó là **tên nhóm hệ chữ viết**,
không phải mã ngôn ngữ. Nay test kiểm ĐÚNG thứ quan trọng: mọi mã trong UI phải là mã
PaddleOCR dùng được, và KHÔNG được là tên nhóm.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.embedded_ocr_language import (
    AUTO_LANGUAGE,
    UI_LANGUAGE_CHOICES,
)
from subtitles_extractor.infrastructure.ocr.script_group_versions import (
    SCRIPT_GROUP_OCR_VERSIONS,
    resolve_script_group_version,
)

#: Tên NHÓM hệ chữ viết — KHÔNG bao giờ được dùng làm mã ngôn ngữ.
_SCRIPT_GROUP_NAMES = frozenset(
    {"latin", "cyrillic", "arabic", "devanagari", "eslav"}
)


def _ui_codes() -> list[str]:
    return [code for _label, code in UI_LANGUAGE_CHOICES if code and code != AUTO_LANGUAGE]


@pytest.mark.parametrize("group_name", sorted(_SCRIPT_GROUP_NAMES))
def test_ui_never_offers_script_group_as_language(group_name: str) -> None:
    """LỖI ĐÃ SỬA: tên nhóm không phải mã ngôn ngữ — truyền vào là PaddleOCR báo lỗi.

    PaddleOCR nhận mã THÀNH VIÊN (vd ``fr``) rồi tự suy ra model nhóm; bản thân
    ``"latin"`` không nằm trong ``LATIN_LANGS`` nên luôn thất bại.
    """
    assert group_name not in _ui_codes()


def test_ui_language_codes_are_unique() -> None:
    codes = _ui_codes()
    assert len(codes) == len(set(codes))


def test_ui_still_covers_cjk_and_vietnamese() -> None:
    """Ngôn ngữ cốt lõi của dự án phải luôn có mặt."""
    codes = set(_ui_codes())
    for essential in ("ch", "chinese_cht", "japan", "korean", "vi", "en"):
        assert essential in codes


def test_ui_covers_each_major_script_via_member_language() -> None:
    """Mỗi hệ chữ viết lớn phải tiếp cận được qua ÍT NHẤT một mã thành viên."""
    codes = set(_ui_codes())
    # Đại diện đã đối chiếu với paddleocr 3.7: fr∈LATIN, ru∈CYRILLIC,
    # ar∈ARABIC, hi∈DEVANAGARI.
    assert codes & {"fr", "de", "es", "pt", "id"}, "thiếu đại diện chữ Latin"
    assert "ru" in codes, "thiếu đại diện chữ Kirin"
    assert "ar" in codes, "thiếu đại diện chữ Ả-Rập"
    assert "hi" in codes, "thiếu đại diện Devanagari"


def test_version_override_table_is_empty_by_design() -> None:
    """Bảng ép ``ocr_version`` nay trống — giả thuyết cũ đã được gỡ."""
    assert SCRIPT_GROUP_OCR_VERSIONS == {}


@pytest.mark.parametrize("language", ["fr", "ru", "ar", "hi", "ch", "japan", ""])
def test_no_language_needs_forced_version(language: str) -> None:
    """Không ngôn ngữ nào cần ép phiên bản — để PaddleOCR tự phân giải."""
    assert resolve_script_group_version(language) is None
