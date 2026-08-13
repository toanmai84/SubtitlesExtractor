"""Use case "Phát hiện hardsub" — kiểm tra video có hardsub trước khi extract."""

from __future__ import annotations

import logging

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.ports.hardsub_detector_port import (
    HardsubDetectionResult,
    HardsubDetectorPort,
)

logger = logging.getLogger(__name__)


class DetectHardsubUseCase:
    """Thin wrapper trên :class:`HardsubDetectorPort` để giữ tách lớp."""

    def __init__(self, detector: HardsubDetectorPort) -> None:
        self._detector = detector

    def execute(
        self, metadata: VideoMetadata, max_samples: int = 12
    ) -> HardsubDetectionResult:
        result = self._detector.detect(metadata, max_samples=max_samples)
        logger.info(
            "Phát hiện hardsub: %s (confidence=%.2f, frames=%d).",
            "có" if result.has_hardsub else "không",
            result.confidence,
            result.sample_frames_used,
        )
        return result


__all__ = ["DetectHardsubUseCase"]
