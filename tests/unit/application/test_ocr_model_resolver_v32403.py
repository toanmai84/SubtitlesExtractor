"""[v3.23.103] Test resolver tên model OCR — đặc biệt nhánh PP-OCRv6 mới."""

from __future__ import annotations

from subtitles_extractor.application.services.ocr_model_resolver import (
    resolve_ocr_model_names,
)
from subtitles_extractor.infrastructure.settings.application_settings import OcrSettings


def test_default_is_ppocrv6_medium() -> None:
    settings = OcrSettings()
    assert settings.version == "PP-OCRv6_medium"
    det, rec = resolve_ocr_model_names(settings)
    assert det == "PP-OCRv6_medium_det"
    assert rec == "PP-OCRv6_medium_rec"


def test_resolve_ppocrv6_tiers() -> None:
    for tier in ("medium", "small", "tiny"):
        settings = OcrSettings(version=f"PP-OCRv6_{tier}",
                               detection_model="PP-OCRv6_medium_det",
                               recognition_model="PP-OCRv6_medium_rec")
        det, rec = resolve_ocr_model_names(settings)
        assert det == f"PP-OCRv6_{tier}_det"
        assert rec == f"PP-OCRv6_{tier}_rec"


def test_resolve_keeps_ppocrv5() -> None:
    settings = OcrSettings(version="PP-OCRv5_server",
                           detection_model="PP-OCRv6_medium_det",
                           recognition_model="PP-OCRv6_medium_rec")
    det, rec = resolve_ocr_model_names(settings)
    assert det == "PP-OCRv5_server_det"
    assert rec == "PP-OCRv5_server_rec"


def test_custom_model_name_preserved() -> None:
    # Tên model tuỳ chỉnh (không thuộc tập default) phải được giữ nguyên.
    settings = OcrSettings(version="PP-OCRv6_medium",
                           detection_model="my_finetuned_det",
                           recognition_model="my_finetuned_rec")
    det, rec = resolve_ocr_model_names(settings)
    assert det == "my_finetuned_det"
    assert rec == "my_finetuned_rec"


def test_default_models_for_version_covers_v6_and_others() -> None:
    # [v3.23.104] Nguồn sự thật chung cho Re-OCR override (không rơi v6 về v5).
    from subtitles_extractor.application.services.ocr_model_resolver import (
        default_models_for_version,
    )

    assert default_models_for_version("PP-OCRv6_medium") == (
        "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec")
    assert default_models_for_version("PP-OCRv6_small") == (
        "PP-OCRv6_small_det", "PP-OCRv6_small_rec")
    assert default_models_for_version("PP-OCRv6_tiny") == (
        "PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec")
    assert default_models_for_version("PP-OCRv5_server") == (
        "PP-OCRv5_server_det", "PP-OCRv5_server_rec")
    assert default_models_for_version("PP-OCRv4") == ("PP-OCRv4_det", "PP-OCRv4_rec")
    # Giá trị lạ -> mặc định an toàn v6 medium (KHÔNG rơi về v5_mobile như bug cũ)
    assert default_models_for_version("unknown_xyz") == (
        "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec")
