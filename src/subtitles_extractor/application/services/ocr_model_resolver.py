"""Tiện ích xác định tên model OCR từ cài đặt — tránh lặp logic.

Logic này từng tồn tại đồng thời tại:
    * ``ApplicationContainer._build_ocr_config``
    * ``ExtractPageViewModel._build_request`` (thông qua ``_resolve_ocr_models``)

Gom vào đây để đảm bảo DRY: chỉ một nơi duy nhất ánh xạ ``version`` →
``(det_model, rec_model)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subtitles_extractor.infrastructure.settings.application_settings import (
        OcrSettings,
    )

# Tên model mặc định — khi settings chứa đúng tên này, ta hiểu user chưa
# tuỳ chỉnh và sẽ dùng default theo phiên bản.
_DEFAULT_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "PP-OCRv6_medium_det",
        "PP-OCRv6_medium_rec",
        "PP-OCRv6_small_det",
        "PP-OCRv6_small_rec",
        "PP-OCRv6_tiny_det",
        "PP-OCRv6_tiny_rec",
        "PP-OCRv5_mobile_det",
        "PP-OCRv5_mobile_rec",
        "PP-OCRv5_server_det",
        "PP-OCRv5_server_rec",
        "PP-OCRv4_det",
        "PP-OCRv4_rec",
    }
)


def default_models_for_version(version: str) -> tuple[str, str]:
    """Ánh xạ chuỗi ``version`` -> ``(det_default, rec_default)`` mặc định của phiên bản.

    Một nguồn sự thật duy nhất cho cả pipeline chính lẫn Re-OCR Studio (tránh map tay sai
    khiến PP-OCRv6 bị rơi về PP-OCRv5).
    """
    if version == "PP-OCRv6_medium":
        return "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"
    if version == "PP-OCRv6_small":
        return "PP-OCRv6_small_det", "PP-OCRv6_small_rec"
    if version == "PP-OCRv6_tiny":
        return "PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"
    if version == "PP-OCRv5_server":
        return "PP-OCRv5_server_det", "PP-OCRv5_server_rec"
    if version == "PP-OCRv5_mobile":
        return "PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"
    if version == "PP-OCRv4":
        return "PP-OCRv4_det", "PP-OCRv4_rec"
    # Mặc định an toàn: PP-OCRv6 medium (mới nhất, đa ngôn ngữ, chính xác cao).
    return "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"


def resolve_ocr_model_names(ocr_settings: OcrSettings) -> tuple[str, str]:
    """Trả về ``(detection_model_name, recognition_model_name)`` từ cài đặt.

    Quy tắc:
        1. Xác định model mặc định theo ``version`` (qua ``default_models_for_version``).
        2. Nếu ``detection_model`` / ``recognition_model`` trong settings là một
           trong ``_DEFAULT_MODEL_NAMES`` (tức là user chưa tuỳ chỉnh), dùng
           default của ``version`` hiện tại.
        3. Nếu user đã đặt tên model tùy chỉnh (không trong tập default),
           dùng đúng tên đó — cho phép trỏ tới local fine-tuned model.

    Args:
        ocr_settings: Snapshot cài đặt OCR từ :class:`ApplicationSettings`.

    Returns:
        Tuple ``(det_model_name, rec_model_name)`` sẵn sàng truyền vào
        :class:`OcrEngineConfig`.
    """
    det_default, rec_default = default_models_for_version(ocr_settings.version)

    det = (
        det_default
        if ocr_settings.detection_model in _DEFAULT_MODEL_NAMES
        else ocr_settings.detection_model
    )
    rec = (
        rec_default
        if ocr_settings.recognition_model in _DEFAULT_MODEL_NAMES
        else ocr_settings.recognition_model
    )
    return det, rec


__all__ = ["default_models_for_version", "resolve_ocr_model_names"]
