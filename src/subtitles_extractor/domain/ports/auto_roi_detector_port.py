"""Hợp đồng tự phát hiện Multi-ROI phụ đề trên video."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.value_objects.roi import Roi

# Callback báo cáo tiến độ ``(current, total, message)``.
ProgressCallback = Callable[[int, int, str], None]


@runtime_checkable
class AutoRoiDetectorPort(Protocol):
    """Hợp đồng phát hiện vùng ROI có khả năng chứa phụ đề.

    Hai phương thức chính:
        * :meth:`detect` — chạy đầy đủ, có thể block để hiện UI review.
        * :meth:`analyze_only` — chỉ chạy phân tích heavy-CPU, trả về kết
          quả thô để UI hiển thị riêng (chạy được trên QThread).
    """

    def detect(
        self,
        metadata: VideoMetadata,
        step_ms: int = 1000,
        batch_size: int = 8,
    ) -> Roi | list[Roi] | None:
        """Chạy detect đầy đủ. Trả về ``None`` nếu không tìm thấy ROI."""
        ...

    def analyze_only(
        self,
        metadata: VideoMetadata,
        step_ms: int = 1000,
        batch_size: int = 8,
        progress_callback: ProgressCallback | None = None,
    ) -> object | None:
        """Chỉ phân tích, không UI. Trả về ``AnalysisResult`` của adapter."""
        ...


__all__ = ["AutoRoiDetectorPort", "ProgressCallback"]
