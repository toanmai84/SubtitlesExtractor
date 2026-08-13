"""Adapter :class:`FrameSamplerPort` dựa trên :class:`cv2.VideoCapture`.

Chiến lược VFR-safe & High Performance:
    1. Smart Seek-or-Grab: Dùng `grab()` thay vì `set()` nếu khoảng cách seek ngắn.
    2. Timestamp thực = ``CAP_PROP_POS_MSEC / 1000`` sau ``cap.retrieve()``.
    3. Loại trùng bằng pHash 64-bit + Pixel Diff bằng cv2 thuần.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import cv2
import numpy as np

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import (
    VideoNotFoundError,
)
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    SampledFrame,
)
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.video.perceptual_hash import (
    compute_phash,
    hamming_distance,
    pixel_diff_ratio,
)

logger = logging.getLogger(__name__)

# Ngưỡng (ms) để quyết định dùng grab() (nhanh) thay vì set() (chậm do phải tìm Keyframe)
_SMART_GRAB_THRESHOLD_MS: float = 1200.0


class OpenCvFrameSampler:
    """Lấy mẫu khung hình bằng OpenCV — Tối ưu hóa hiệu năng cao."""

    def iter_frames(
        self,
        metadata: VideoMetadata,
        roi: Roi | None,
        config: FrameSamplingConfig,
    ) -> Iterator[SampledFrame]:
        capture = cv2.VideoCapture(str(metadata.path))
        if not capture.isOpened():
            raise VideoNotFoundError(
                f"OpenCV không mở được tệp video: {metadata.path}."
            )

        try:
            yield from self._iterate(capture, metadata, roi, config)
        finally:
            capture.release()

    def _iterate(
        self,
        capture: cv2.VideoCapture,
        metadata: VideoMetadata,
        roi: Roi | None,
        config: FrameSamplingConfig,
    ) -> Iterator[SampledFrame]:
        skip_intro_ms = max(0, int(round(config.skip_intro_sec * 1000)))
        skip_outro_ms = max(0, int(round(config.skip_outro_sec * 1000)))
        step_ms = max(int(round(config.sample_step_sec * 1000)), 1)
        total_ms = int(round(metadata.duration_sec * 1000)) - skip_outro_ms
        last_hash: int | None = None
        last_image: np.ndarray | None = None
        frame_counter: int = 0

        logger.info(
            "OpenCV High-Perf decode: %s, step=%.3fs, range=%.1f-%.1fs.",
            metadata.path.name,
            config.sample_step_sec,
            skip_intro_ms / 1000.0,
            total_ms / 1000.0,
        )

        timestamp_ms = float(skip_intro_ms)
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

        while timestamp_ms <= total_ms:
            current_pos_ms = capture.get(cv2.CAP_PROP_POS_MSEC)

            # Thuật toán Smart Seek-or-Grab
            if current_pos_ms >= 0 and 0 < timestamp_ms - current_pos_ms <= _SMART_GRAB_THRESHOLD_MS:
                # Nếu khoảng cách gần, grab() cày lướt bỏ qua frame cho đến khi đạt mục tiêu
                while current_pos_ms < timestamp_ms:
                    if not capture.grab():
                        break
                    current_pos_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                success, frame_bgr = capture.retrieve()
            else:
                # Khoảng cách xa hoặc thụt lùi, bắt buộc phải dùng set() tới I-Frame
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
                success, frame_bgr = capture.read()
                current_pos_ms = capture.get(cv2.CAP_PROP_POS_MSEC)

            if not success or frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim < 3:
                # [Corrupted Frame Crash] Video tải từ web có thể có frame hỏng →
                # OpenCV trả mảng rỗng/sai chiều. Bỏ qua thay vì để sập khi cắt ROI.
                logger.debug(
                    "Frame hỏng/rỗng tại %.2fs — bỏ qua.",
                    timestamp_ms / 1000.0,
                )
                timestamp_ms += step_ms
                continue

            actual_pos_ms = current_pos_ms if current_pos_ms >= 0 else timestamp_ms
            actual_ts_sec = actual_pos_ms / 1000.0

            cropped_bgr = self._apply_roi(frame_bgr, roi)
            image_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

            current_hash = compute_phash(image_rgb)
            if not self._should_keep_with_hash(
                image_rgb, current_hash, last_hash, last_image, config
            ):
                yield SampledFrame(
                    frame_index=frame_counter,
                    timestamp_sec=actual_ts_sec,
                    image_rgb=np.empty(0),
                    is_duplicate=True,
                )
                frame_counter += 1
                timestamp_ms += step_ms
                continue

            yield SampledFrame(
                frame_index=frame_counter,
                timestamp_sec=actual_ts_sec,
                image_rgb=image_rgb,
                is_duplicate=False,
            )
            last_hash = current_hash
            last_image = image_rgb
            frame_counter += 1
            timestamp_ms += step_ms

    @staticmethod
    def _apply_roi(frame_bgr: np.ndarray, roi: Roi | None) -> np.ndarray:
        if roi is None:
            return frame_bgr
        height, width = frame_bgr.shape[:2]
        clipped = roi.clip_to(width, height)
        return frame_bgr[
            clipped.y : clipped.y2,
            clipped.x : clipped.x2,
        ]

    @staticmethod
    def _should_keep_with_hash(
        image_rgb: np.ndarray,
        current_hash: int,
        last_hash: int | None,
        last_image: np.ndarray | None,
        config: FrameSamplingConfig,
    ) -> bool:
        if last_hash is None or last_image is None:
            return True
        if hamming_distance(current_hash, last_hash) > config.phash_distance_threshold:
            return True
        if pixel_diff_ratio(image_rgb, last_image) > config.pixel_diff_threshold:
            return True
        return False


__all__ = ["OpenCvFrameSampler"]
