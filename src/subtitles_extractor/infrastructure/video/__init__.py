"""Adapter cho I/O video — metadata reader, frame sampler, detector, player."""

from __future__ import annotations

from subtitles_extractor.infrastructure.video.decoders.opencv_frame_sampler import (
    OpenCvFrameSampler,
)
from subtitles_extractor.infrastructure.video.gradient_hardsub_detector import (
    GradientHardsubDetector,
)
from subtitles_extractor.infrastructure.video.mpv_metadata_reader import (
    MpvMetadataReader,
)
from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
    OcrBasedAutoRoiDetector,
)
from subtitles_extractor.infrastructure.video.opencv_metadata_reader import (
    OpenCvMetadataReader,
)
from subtitles_extractor.infrastructure.video.perceptual_hash import (
    compute_phash,
    hamming_distance,
    pixel_diff_ratio,
)

__all__ = [
    "GradientHardsubDetector",
    "MpvMetadataReader",
    "OcrBasedAutoRoiDetector",
    "OpenCvFrameSampler",
    "OpenCvMetadataReader",
    "compute_phash",
    "hamming_distance",
    "pixel_diff_ratio",
]
