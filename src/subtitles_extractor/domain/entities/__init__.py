"""Các entity của tầng nghiệp vụ."""

from __future__ import annotations

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
    Polygon,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata

__all__ = [
    "OcrFrameResult",
    "OcrTextBox",
    "Polygon",
    "SubtitleEvent",
    "VideoMetadata",
]
