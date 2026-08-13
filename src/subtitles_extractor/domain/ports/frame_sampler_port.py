"""Hợp đồng lấy mẫu khung hình từ video."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.value_objects.roi import Roi

if TYPE_CHECKING:
    import numpy as np

@dataclass(frozen=True, slots=True)
class SampledFrame:
    frame_index: int
    timestamp_sec: float
    image_rgb: np.ndarray
    is_duplicate: bool = False
    is_error: bool = False

@dataclass(frozen=True, slots=True)
class FrameSamplingConfig:
    sample_step_sec: float = 0.25
    phash_distance_threshold: int = 5
    pixel_diff_threshold: float = 0.01
    skip_intro_sec: float = 0.0
    skip_outro_sec: float = 0.0

    apply_median_blend: bool = False
    median_blend_frames: int = 3

    #[CẢI TIẾN]: Ủy quyền toàn bộ Tiền xử lý sang VRAM (CuPy)
    vram_upscale_small_text: bool = False
    vram_upscale_target_height_px: int = 96
    vram_add_border: bool = False
    vram_border_thickness_px: int = 8
    vram_sharpen: bool = False
    vram_contrast_factor: float = 1.0

@runtime_checkable
class FrameSamplerPort(Protocol):
    def iter_frames(
        self, metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig,
    ) -> Iterator[SampledFrame]: ...

__all__ = ["FrameSamplerPort", "FrameSamplingConfig", "SampledFrame"]
