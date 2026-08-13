"""Các Protocol định nghĩa hợp đồng cho tầng infrastructure."""

from __future__ import annotations

from subtitles_extractor.domain.ports.auto_roi_detector_port import (
    AutoRoiDetectorPort,
)
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplerPort,
    FrameSamplingConfig,
    SampledFrame,
)
from subtitles_extractor.domain.ports.hardsub_detector_port import (
    HardsubDetectionResult,
    HardsubDetectorPort,
)
from subtitles_extractor.domain.ports.ocr_engine_port import (
    OcrEngineConfig,
    OcrEnginePort,
)
from subtitles_extractor.domain.ports.progress_reporter_port import (
    NullProgressReporter,
    ProgressReporterPort,
)
from subtitles_extractor.domain.ports.settings_repository_port import (
    SettingsRepositoryPort,
)
from subtitles_extractor.domain.ports.subtitle_exporter_port import (
    SubtitleExporterPort,
)
from subtitles_extractor.domain.ports.subtitle_importer_port import (
    SubtitleImporterPort,
)
from subtitles_extractor.domain.ports.translator_port import TranslatorPort
from subtitles_extractor.domain.ports.video_metadata_reader_port import (
    VideoMetadataReaderPort,
)
from subtitles_extractor.domain.ports.video_player_port import VideoPlayerPort

__all__ = [
    "AutoRoiDetectorPort",
    "FrameSamplerPort",
    "FrameSamplingConfig",
    "HardsubDetectionResult",
    "HardsubDetectorPort",
    "NullProgressReporter",
    "OcrEngineConfig",
    "OcrEnginePort",
    "ProgressReporterPort",
    "SampledFrame",
    "SettingsRepositoryPort",
    "SubtitleExporterPort",
    "SubtitleImporterPort",
    "TranslatorPort",
    "VideoMetadataReaderPort",
    "VideoPlayerPort",
]
