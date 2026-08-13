"""Hợp đồng kho lưu trữ phụ đề vào Database."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent


@runtime_checkable
class SubtitleRepositoryPort(Protocol):
    """Đọc/ghi danh sách SubtitleEvent vào cơ sở dữ liệu."""

    def load_events(self, video_path: str) -> list[SubtitleEvent] | None:
        """Đọc danh sách phụ đề đã lưu. Trả về None nếu chưa từng trích xuất."""
        ...

    def save_events(self, video_path: str, events: list[SubtitleEvent]) -> None:
        """Lưu toàn bộ danh sách phụ đề (Ghi đè bản cũ)."""
        ...

    def has_events(self, video_path: str) -> bool:
        """Kiểm tra nhanh xem video này đã có dữ liệu phụ đề trong DB chưa."""
        ...

__all__ = ["SubtitleRepositoryPort"]
