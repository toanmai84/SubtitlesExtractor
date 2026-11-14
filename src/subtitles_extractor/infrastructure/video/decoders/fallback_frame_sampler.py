"""[v3.23.723] Bộ lấy mẫu NVDEC có đường lùi sang PyAV — không bao giờ mất phụ đề vì GPU.

Trước bản này, khi đã chọn NVDEC mà giải mã / xử lý GPU hỏng (codec NVDEC không hỗ trợ, thiếu NVRTC cho CuPy, lỗi
CUDA giữa chừng), lượt trích xuất hoặc thất bại hẳn, hoặc — tệ hơn — mọi khung lỗi bị BỎ IM LẶNG (bộ chuyển màu trả
``None``) và phụ đề ra rỗng mà không báo gì. Nay:

* NVDEC ném ``VideoDecodeError`` trước khung đầu → chạy PyAV từ đầu;
* hỏng giữa chừng → PyAV tiếp tục từ ngay sau khung cuối NVDEC đã phát (đánh số khung nối tiếp);
* bước tiền xử lý đã giao cho VRAM (``vram_*``) được làm lại trên CPU bằng CÙNG mã (``apply_gpu_preprocessing_pipeline``
  với NumPy — khớp bản VRAM tới ±1–2 mức xám), nên ảnh đưa vào OCR vẫn như cài đặt.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import VideoDecodeError
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig, SampledFrame
from subtitles_extractor.domain.value_objects.boundary_track import BoundaryTrack
from subtitles_extractor.domain.value_objects.roi import Roi

logger = logging.getLogger(__name__)

#: Tiếp tục sau khung cuối NVDEC đã phát một khoảng rất nhỏ (PyAV lọc ``mốc < skip_intro``).
_RESUME_EPSILON_SEC = 1e-3


def _apply_vram_steps_on_cpu(image: np.ndarray, config: FrameSamplingConfig, buffers: dict[str, Any]) -> np.ndarray:
    from subtitles_extractor.infrastructure.ocr.preprocessing.gpu_image_filters import (
        apply_gpu_preprocessing_pipeline,
    )

    return np.ascontiguousarray(apply_gpu_preprocessing_pipeline(
        image, buffers,
        upscale_small_text=config.vram_upscale_small_text,
        upscale_target_height_px=config.vram_upscale_target_height_px,
        sharpen=config.vram_sharpen,
        contrast_factor=config.vram_contrast_factor,
        add_border=config.vram_add_border,
        border_thickness_px=config.vram_border_thickness_px,
        apply_clahe=config.vram_apply_clahe,
        clahe_clip_limit=config.vram_clahe_clip_limit,
        clahe_tile_size=config.vram_clahe_tile_size,
        xp=np,
    ))


class FallbackFrameSampler:
    """Bọc bộ lấy mẫu GPU (``primary``) bằng bộ PyAV (``fallback_factory()``) khi GPU hỏng."""

    def __init__(self, primary: Any, fallback_factory: Callable[[], Any]) -> None:
        self._primary = primary
        self._fallback_factory = fallback_factory
        self.last_boundary_track: BoundaryTrack | None = None
        #: Lý do lần gần nhất phải lùi sang PyAV (``None`` = NVDEC chạy trọn).
        self.last_fallback_reason: str | None = None
        #: [v3.23.723] Bộ giải mã đã thật sự chạy lượt vừa xong (ghi vào ``meta.run``): tên bộ chính, tên bộ dự phòng, hoặc
        #: ``"<chính>→<dự phòng>@<giây>"`` khi lùi giữa chừng.
        self.effective_sampler: str = type(primary).__name__
        #: Mốc (giây) chuyển sang bộ dự phòng giữa chừng, ``None`` nếu không.
        self.fallback_at_sec: float | None = None

    def iter_frames(self, metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig) -> Iterator[SampledFrame]:
        from subtitles_extractor.infrastructure.video.decoders.pynvvideocodec_frame_sampler import (
            vram_preprocessing_enabled,
        )

        self.last_boundary_track = None
        self.last_fallback_reason = None
        self.fallback_at_sec = None
        primary_name = type(self._primary).__name__
        self.effective_sampler = primary_name
        last_ts: float | None = None
        emitted = 0
        try:
            for frame in self._primary.iter_frames(metadata, roi, config):
                last_ts = frame.timestamp_sec
                emitted += 1
                yield frame
            self.last_boundary_track = getattr(self._primary, "last_boundary_track", None)
            return
        except VideoDecodeError as exc:
            self.last_fallback_reason = str(exc)
            if last_ts is None:
                logger.warning("NVDEC không dùng được cho %s (%s) — giải mã bằng PyAV.", metadata.path.name, exc)
            else:
                logger.warning(
                    "NVDEC hỏng giữa chừng ở %.3f s (%s) — PyAV giải mã tiếp từ mốc này.", last_ts, exc,
                )

        # [v3.23.723] Tiếp tục ĐÚNG lưới lấy mẫu: khung kế tiếp tới lượt là ``last_ts + bước`` (trước: ``last_ts + 1 ms``
        # → lấy thêm khung ngay sau và lệch lưới một khung về sau).
        step = max(0.001, float(config.sample_step_sec))
        resume = config if last_ts is None else dataclasses.replace(
            config, skip_intro_sec=max(config.skip_intro_sec, last_ts + max(step - _RESUME_EPSILON_SEC, _RESUME_EPSILON_SEC)),
        )
        fallback = self._fallback_factory()
        fallback_name = type(fallback).__name__
        if last_ts is None:
            self.effective_sampler = fallback_name
        else:
            self.fallback_at_sec = round(last_ts, 3)
            self.effective_sampler = f"{primary_name}→{fallback_name}@{last_ts:.3f}"
        redo_vram = vram_preprocessing_enabled(config)
        buffers: dict[str, Any] = {}
        for frame in fallback.iter_frames(metadata, roi, resume):
            image = frame.image_rgb
            if redo_vram and image is not None and image.size:
                image = _apply_vram_steps_on_cpu(image, config, buffers)
            yield dataclasses.replace(frame, frame_index=frame.frame_index + emitted, image_rgb=image)
        # Dải ranh giới chỉ đúng khi một bộ chạy trọn lượt.
        self.last_boundary_track = getattr(fallback, "last_boundary_track", None) if last_ts is None else None


__all__ = ["FallbackFrameSampler"]
