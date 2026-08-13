"""Hợp đồng xuất tệp phụ đề ra đĩa.

Mỗi adapter (SRT/ASS/VTT…) hiện thực giao diện này. Adapter phải
ghi nguyên tử (atomic write) — ghi vào tệp tạm rồi rename, để không
để lại tệp dở dang nếu bị ngắt giữa chừng.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent


@runtime_checkable
class SubtitleExporterPort(Protocol):
    """Ghi danh sách :class:`SubtitleEvent` ra một định dạng cụ thể."""

    @property
    def file_extension(self) -> str:
        """Phần mở rộng (gồm dấu chấm, ví dụ ``".srt"``)."""
        ...

    def export(
        self,
        events: Sequence[SubtitleEvent],
        output_path: Path,
    ) -> Path:
        """Ghi ``events`` ra ``output_path``.

        Args:
            events:      Danh sách câu phụ đề đã sắp xếp tăng theo thời gian.
            output_path: Đường dẫn tệp đích (sẽ ghi đè nếu tồn tại).

        Returns:
            Đường dẫn tuyệt đối của tệp đã ghi.

        Raises:
            SubtitleExportError: Khi ghi đĩa thất bại.
        """
        ...


__all__ = ["SubtitleExporterPort"]
