"""Hợp đồng kho lưu trữ trạng thái video."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.entities.video_state import VideoState


@runtime_checkable
class VideoStateRepositoryPort(Protocol):
    """Đọc/ghi trạng thái video (ROI, các thiết lập cục bộ) theo đường dẫn file."""

    def get(self, video_path: str) -> VideoState | None:
        """Lấy trạng thái đã lưu, trả về None nếu chưa từng lưu."""
        ...

    def save(self, state: VideoState) -> None:
        """Lưu hoặc cập nhật (Upsert) trạng thái video."""
        ...

__all__ = ["VideoStateRepositoryPort"]
