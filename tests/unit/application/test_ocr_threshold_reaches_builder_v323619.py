"""[v3.23.619] Ngưỡng OCR mặc định không được chặn mất các quy tắc tin cậy của bước gộp câu.

Phim mẫu tập 1: câu "瑶瑶" thứ hai hiện 14 khung (tin cậy 0.53-0.81); với ngưỡng OCR mặc
định cũ 0.75 chỉ 5 khung tới được bước gộp câu (0.2s) → bị loại như mảnh chớp, dù quy tắc
"câu hai chữ đứng yên ≥ 0.4s chỉ cần 0.55" (v3.23.616) lẽ ra giữ lại.
"""
from __future__ import annotations

from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    _STABLE_TWO_CJK_MIN_CONFIDENCE as BUILDER_LOWEST_CONFIDENCE,
)
from subtitles_extractor.infrastructure.settings.application_settings import OcrSettings


def test_default_ocr_threshold_lets_builder_rules_work() -> None:
    assert OcrSettings().score_threshold <= BUILDER_LOWEST_CONFIDENCE
