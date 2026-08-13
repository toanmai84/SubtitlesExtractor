"""Các value object bất biến của tầng nghiệp vụ."""

from __future__ import annotations

from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.device_kind import (
    DeviceKind,
    PrecisionMode,
    SubtitleFormat,
)
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

__all__ = [
    "Confidence",
    "DeviceKind",
    "PrecisionMode",
    "Roi",
    "SubtitleFormat",
    "TimeInterval",
]
