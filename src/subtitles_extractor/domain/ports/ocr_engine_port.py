"""Hợp đồng OCR engine — tách hoàn toàn khỏi PaddleOCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.domain.value_objects.device_kind import (
    DeviceKind,
    PrecisionMode,
)

if TYPE_CHECKING:
    import numpy as np

@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    upscale_small_text: bool = True
    upscale_target_height_px: int = 96
    add_white_border: bool = True
    border_thickness_px: int = 8
    apply_sharpen: bool = False
    apply_contrast_boost: bool = False
    contrast_factor: float = 1.20
    apply_clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 8

@dataclass(frozen=True, slots=True)
class OcrEngineConfig:
    device: DeviceKind = DeviceKind.GPU
    detection_model_name: str = "PP-OCRv6_medium_det"
    recognition_model_name: str = "PP-OCRv6_medium_rec"
    language: str = "ch"

    limit_side_len: int = 0
    limit_type: str = "min"
    det_thresh: float = 0.3
    det_box_thresh: float = 0.6
    det_unclip_ratio: float = 1.5

    score_threshold: float | None = None
    batch_size: int = 16

    use_textline_orientation: bool = False
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False

    enable_mkldnn: bool = False
    use_tensorrt: bool = False
    precision: PrecisionMode = PrecisionMode.FP32
    parallel_workers: int = 4

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class OcrEnginePort(Protocol):
    @property
    def is_initialized(self) -> bool: ...

    def initialize(self) -> None: ...

    def release(self) -> None: ...

    def infer(
        self,
        image_rgb: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> OcrFrameResult: ...

    def infer_batch(
        self,
        images_rgb: list[np.ndarray],
        frame_indices: list[int],
        timestamps_sec: list[float],
    ) -> list[OcrFrameResult]: ...

__all__ = ["OcrEngineConfig", "OcrEnginePort", "PreprocessConfig"]
