"""Use case "Tự phát hiện Multi-ROI"."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.ports.auto_roi_detector_port import (
    AutoRoiDetectorPort,
)
from subtitles_extractor.domain.value_objects.roi import Roi

if TYPE_CHECKING:
    from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig

logger = logging.getLogger(__name__)

# Callback nhận ``(current, total, message)`` — giống Port progress nhưng
# không yêu cầu phía gọi phải tạo full ProgressReporter.
ProgressCallback = Callable[[int, int, str], None]


class DetectAutoRoiUseCase:
    """Wrapper trên :class:`AutoRoiDetectorPort` để giữ tách lớp.

    Args:
        detector: Adapter detect ROI (hiện thực :class:`AutoRoiDetectorPort`).
    """

    def __init__(self, detector: AutoRoiDetectorPort) -> None:
        self._detector = detector

    def execute(
        self,
        metadata: VideoMetadata,
        step_ms: int = 1000,
        batch_size: int = 8,
        vram_flags: FrameSamplingConfig | None = None,
    ) -> Roi | list[Roi] | None:
        """Chạy detect đầy đủ — bao gồm cả review UI nếu adapter bật.

        Args:
            vram_flags: Cờ VRAM preprocessing từ user settings để Auto-ROI
                dùng đúng GPU pipeline với extract chính. Truyền ``None``
                khi không có VRAM preprocessing.

        Returns:
            ``Roi`` đơn lẻ, danh sách ``Roi``, hoặc ``None`` nếu không tìm
            thấy ROI hợp lệ.
        """
        # Dùng dynamic dispatch để không break các adapter khác không có vram_flags.
        detect_fn = getattr(self._detector, "detect", None)
        if callable(detect_fn):
            try:
                result = detect_fn(metadata, step_ms=step_ms, batch_size=batch_size, vram_flags=vram_flags)
            except TypeError:
                # Adapter cũ không nhận vram_flags — fallback không truyền.
                result = detect_fn(metadata, step_ms=step_ms, batch_size=batch_size)
        else:
            result = None

        if result is None:
            logger.info("Không tìm được ROI tự động cho %s.", metadata.filename)
        elif isinstance(result, list):
            logger.info("Đề xuất %d ROI cho %s.", len(result), metadata.filename)
        return result

    def analyze_only(
        self,
        metadata: VideoMetadata,
        step_ms: int = 1000,
        batch_size: int = 8,
        progress_callback: ProgressCallback | None = None,
        vram_flags: FrameSamplingConfig | None = None,
    ) -> object | None:
        """Chỉ phân tích — không hiển thị UI review.

        Trả về kết quả ``AnalysisResult`` (hoặc ``None`` nếu không có ROI)
        để caller tự xử lý UI. Giúp tách thread phân tích khỏi UI thread.

        Args:
            vram_flags: Cờ VRAM preprocessing từ user settings.

        Returns:
            Kết quả phân tích (kiểu ``AnalysisResult`` của adapter) hoặc
            ``None`` nếu adapter không hỗ trợ.
        """
        delegate = getattr(self._detector, "analyze_only", None)
        if callable(delegate):
            try:
                return delegate(metadata, step_ms, batch_size, progress_callback, vram_flags=vram_flags)
            except TypeError:
                # Fallback cho adapter cũ không có vram_flags param.
                return delegate(metadata, step_ms, batch_size, progress_callback)
        return None

    def supports_analyze_only(self) -> bool:
        """``True`` nếu adapter hỗ trợ tách analyze khỏi review."""
        return callable(getattr(self._detector, "analyze_only", None))


__all__ = ["DetectAutoRoiUseCase", "ProgressCallback"]
