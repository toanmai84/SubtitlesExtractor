"""Hợp đồng phát hiện hardsub trên video.

Adapter sẽ phân tích vài frame mẫu để quyết định xem video có khả năng
chứa phụ đề khắc cứng (hardsub) hay không. Hữu ích để cảnh báo người
dùng nếu họ đang cố OCR một video không có phụ đề.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata


@dataclass(frozen=True, slots=True)
class HardsubDetectionResult:
    """Kết quả phát hiện hardsub.

    Attributes:
        has_hardsub:        Đúng nếu confidence ≥ ngưỡng quyết định.
        confidence:         Điểm tin cậy ``∈ [0.0, 1.0]``.
        sample_frames_used: Số frame đã phân tích.
        reason:             Lý do quyết định (để hiển thị/debug).
    """

    has_hardsub: bool
    confidence: float
    sample_frames_used: int
    reason: str = ""


@runtime_checkable
class HardsubDetectorPort(Protocol):
    """Phát hiện video có hardsub hay không thông qua sampling."""

    def detect(
        self,
        metadata: VideoMetadata,
        max_samples: int = 12,
    ) -> HardsubDetectionResult:
        """Phân tích ``max_samples`` frame trải đều theo thời gian.

        Raises:
            VideoDecodeError: Khi không đọc được frame.
        """
        ...


__all__ = ["HardsubDetectionResult", "HardsubDetectorPort"]
