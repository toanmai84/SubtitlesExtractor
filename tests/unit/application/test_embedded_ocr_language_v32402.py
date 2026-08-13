"""[v3.23.102] Test ánh xạ ngôn ngữ track -> mã PaddleOCR cho OCR phụ đề nhúng."""

from __future__ import annotations

from subtitles_extractor.application.services.embedded_ocr_language import (
    AUTO_LANGUAGE,
    UI_LANGUAGE_CHOICES,
    describe_paddle_lang,
    resolve_paddle_lang,
)


def test_resolve_common_languages() -> None:
    assert resolve_paddle_lang("eng") == "en"
    assert resolve_paddle_lang("jpn") == "japan"
    assert resolve_paddle_lang("kor") == "korean"
    assert resolve_paddle_lang("chi") == "ch"
    assert resolve_paddle_lang("zho") == "ch"
    assert resolve_paddle_lang("vie") == "vi"
    assert resolve_paddle_lang("rus") == "cyrillic"
    assert resolve_paddle_lang("ara") == "arabic"
    assert resolve_paddle_lang("fra") == "latin"
    assert resolve_paddle_lang("deu") == "latin"


def test_resolve_is_case_insensitive_and_trims() -> None:
    assert resolve_paddle_lang("ENG") == "en"
    assert resolve_paddle_lang("  Jpn ") == "japan"


def test_resolve_unknown_returns_none() -> None:
    assert resolve_paddle_lang("") is None
    assert resolve_paddle_lang("xyz") is None
    assert resolve_paddle_lang("zzz") is None


def test_ui_choices_first_is_auto() -> None:
    assert UI_LANGUAGE_CHOICES[0][1] == AUTO_LANGUAGE == ""
    codes = [code for _label, code in UI_LANGUAGE_CHOICES]
    assert "en" in codes and "ch" in codes and "japan" in codes and "korean" in codes


def test_describe_paddle_lang() -> None:
    assert describe_paddle_lang("") == "mặc định"
    assert describe_paddle_lang(None) == "mặc định"
    assert "en" in describe_paddle_lang("en")


def test_unified_model_coverage() -> None:
    # [v3.23.103] PP-OCRv6 hợp nhất xử lý Trung/Nhật/Anh/Latin/Việt -> dùng engine chính.
    from subtitles_extractor.application.services.embedded_ocr_language import (
        is_covered_by_unified_model,
    )

    assert is_covered_by_unified_model("en")
    assert is_covered_by_unified_model("japan")
    assert is_covered_by_unified_model("ch")
    assert is_covered_by_unified_model("latin")
    assert is_covered_by_unified_model("vi")
    # Ngoài vùng phủ -> cần engine riêng
    assert not is_covered_by_unified_model("korean")
    assert not is_covered_by_unified_model("cyrillic")
    assert not is_covered_by_unified_model("arabic")
    assert not is_covered_by_unified_model(None)
