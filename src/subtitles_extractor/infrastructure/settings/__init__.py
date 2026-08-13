"""Cấu hình & kho lưu trữ thiết lập."""

from __future__ import annotations

from subtitles_extractor.infrastructure.settings.application_settings import (
    ApplicationSettings,
    FrameSamplingSettings,
    HardwareSettings,
    OcrSettings,
    PostProcessSettings,
    UiSettings,
)
from subtitles_extractor.infrastructure.settings.json_file_repository import (
    JsonFileRepository,
)
from subtitles_extractor.infrastructure.settings.qsettings_repository import (
    QSettingsRepository,
)
from subtitles_extractor.infrastructure.settings.settings_service import (
    SettingsService,
)

__all__ = [
    "ApplicationSettings",
    "FrameSamplingSettings",
    "HardwareSettings",
    "JsonFileRepository",
    "OcrSettings",
    "PostProcessSettings",
    "QSettingsRepository",
    "SettingsService",
    "UiSettings",
]
