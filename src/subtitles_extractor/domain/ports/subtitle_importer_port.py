"""Hợp đồng đọc lại tệp phụ đề đã có (SRT/ASS) thành :class:`SubtitleEvent`.

Cho phép Editor mở tệp .srt/.ass có sẵn để chỉnh sửa, hoặc nạp lại kết
quả lần trước.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent


@runtime_checkable
class SubtitleImporterPort(Protocol):
    """Đọc tệp phụ đề thành danh sách :class:`SubtitleEvent`."""

    @property
    def file_extension(self) -> str:
        """Phần mở rộng (gồm dấu chấm), ví dụ ``".srt"``."""
        ...

    def import_from(self, source_path: Path) -> list[SubtitleEvent]:
        """Đọc và parse tệp.

        Raises:
            FileNotFoundError: Khi tệp không tồn tại.
            ValueError:        Khi nội dung không đúng định dạng.
        """
        ...


__all__ = ["SubtitleImporterPort"]
