"""Adapter :class:`HardsubDetectorPort` — phát hiện hardsub bằng gradient analysis.

CẢI TIẾN:
    1. [LOGIC FIX] Tính L1-norm Magnitude chuẩn (không làm suy giảm 50% độ gắt của nét chữ).
    2. [QUALITY FIX] Chống nhiễu giả mạo (False Positive) từ nền phức tạp (cỏ, nước) bằng Morphological Filtering, chỉ bắt các cụm Text thật sự.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import VideoDecodeError
from subtitles_extractor.domain.ports.hardsub_detector_port import (
    HardsubDetectionResult,
)

logger = logging.getLogger(__name__)

# Tỉ lệ vùng nửa dưới so với chiều cao — phụ đề thường ở 30% dưới cùng.
_BOTTOM_REGION_RATIO: float = 0.30
# Ngưỡng gradient để được coi là cạnh sắc nét.
_EDGE_MAGNITUDE_THRESHOLD: int = 60
# Tỉ lệ pixel cạnh trên tổng vùng để được coi là "có text" trong frame.
_FRAME_HAS_TEXT_RATIO: float = 0.012
# Tỉ lệ frame có text trên tổng frame mẫu để kết luận hardsub.
_HARDSUB_DECISION_THRESHOLD: float = 0.40


class GradientHardsubDetector:
    """Phát hiện hardsub bằng phân tích gradient vùng đáy khung hình."""

    def detect(
        self,
        metadata: VideoMetadata,
        max_samples: int = 12,
    ) -> HardsubDetectionResult:
        capture = cv2.VideoCapture(str(metadata.path))
        if not capture.isOpened():
            raise VideoDecodeError(
                f"OpenCV không mở được tệp video: {metadata.path}."
            )

        try:
            sampling_points = self._compute_sampling_points(metadata, max_samples)
            edge_ratios = self._collect_edge_ratios(capture, sampling_points)
        finally:
            capture.release()

        if not edge_ratios:
            return HardsubDetectionResult(
                has_hardsub=False,
                confidence=0.0,
                sample_frames_used=0,
                reason="Không đọc được frame nào để phân tích.",
            )

        frames_with_text = sum(
            1 for ratio in edge_ratios if ratio >= _FRAME_HAS_TEXT_RATIO
        )
        text_frame_ratio = frames_with_text / len(edge_ratios)
        confidence = min(1.0, text_frame_ratio / _HARDSUB_DECISION_THRESHOLD * 0.85)
        has_hardsub = text_frame_ratio >= _HARDSUB_DECISION_THRESHOLD
        reason = (
            f"{frames_with_text}/{len(edge_ratios)} frame có cạnh đặc trưng text "
            f"ở vùng đáy ({text_frame_ratio:.0%})."
        )
        return HardsubDetectionResult(
            has_hardsub=has_hardsub,
            confidence=confidence,
            sample_frames_used=len(edge_ratios),
            reason=reason,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_sampling_points(
        metadata: VideoMetadata, max_samples: int
    ) -> list[float]:
        """Trả về danh sách timestamp giây trải đều, tránh đầu/cuối."""
        if metadata.duration_sec <= 0 or max_samples <= 0:
            return []
        # Bỏ 5% đầu và 5% cuối — thường là logo/credit không phải phụ đề.
        usable_start = metadata.duration_sec * 0.05
        usable_end = metadata.duration_sec * 0.95
        if usable_end <= usable_start:
            return [metadata.duration_sec / 2.0]
        step = (usable_end - usable_start) / max(max_samples, 1)
        return [usable_start + step * i for i in range(max_samples)]

    @staticmethod
    def _collect_edge_ratios(
        capture: cv2.VideoCapture, timestamps_sec: list[float]
    ) -> list[float]:
        """Tính tỉ lệ pixel cạnh trong vùng nửa dưới của mỗi frame mẫu."""
        ratios: list[float] = []
        for ts_sec in timestamps_sec:
            capture.set(cv2.CAP_PROP_POS_MSEC, ts_sec * 1000.0)
            success, frame_bgr = capture.read()
            if not success or frame_bgr is None:
                continue
            ratios.append(_compute_bottom_edge_ratio(frame_bgr))
        return ratios


def _compute_bottom_edge_ratio(frame_bgr: np.ndarray) -> float:
    """Tỉ lệ pixel có gradient cao trong vùng đáy khung hình."""
    height, _width = frame_bgr.shape[:2]
    bottom_start = int(height * (1.0 - _BOTTOM_REGION_RATIO))
    bottom_region = frame_bgr[bottom_start:, :]
    if bottom_region.size == 0:
        return 0.0

    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

    # Tính đạo hàm bậc 1 theo X và Y
    sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)

    # [LOGIC FIX]: Sử dụng trọng số 1.0 để giữ nguyên L1-Norm (Độ gắt thực tế)
    magnitude = cv2.addWeighted(abs_x, 1.0, abs_y, 1.0, 0)

    # Phân ngưỡng (Thresholding) lấy các cạnh sắc nét
    _, edge_mask = cv2.threshold(magnitude, _EDGE_MAGNITUDE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # [QUALITY FIX]: Áp dụng Morphological Close để lọc nhiễu nền.
    # Kernel 9x3 cực kỳ nhạy bén trong việc gom các ký tự rời rạc thành một khối chữ ngang,
    # đồng thời xóa sổ các điểm ảnh nhiễu rải rác.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    text_blobs = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)

    edge_pixels = int(np.count_nonzero(text_blobs))
    total_pixels = gray.size
    return float(edge_pixels) / total_pixels if total_pixels else 0.0


__all__ = ["GradientHardsubDetector"]
