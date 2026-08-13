"""Adapter cho OCR engine — hiện thực Paddle, sẵn sàng mở rộng EasyOCR…"""

from __future__ import annotations

from subtitles_extractor.infrastructure.ocr.bundled_models import (
    configure_bundled_paddle_models,
)
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter
from subtitles_extractor.infrastructure.ocr.result_parser import parse_paddle_result

__all__ = [
    "PaddleOcrAdapter",
    "configure_bundled_paddle_models",
    "parse_paddle_result",
]
