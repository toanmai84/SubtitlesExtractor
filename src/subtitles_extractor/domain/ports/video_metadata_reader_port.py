"""Hợp đồng đọc metadata video — tầng domain định nghĩa, infrastructure cài đặt.

Adapter sẽ được hiện thực bằng OpenCV/PyAV/FFmpeg ở tầng infrastructure
nhưng tầng application chỉ phụ thuộc vào hợp đồng này.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata


@runtime_checkable
class VideoMetadataReaderPort(Protocol):
    """Đọc metadata cơ bản của một tệp video.

    Hiện thực phải:
        * Bắt :class:`FileNotFoundError` và raise lại
          :class:`subtitles_extractor.domain.exceptions.VideoNotFoundError`.
        * Bắt mọi lỗi giải mã/parse và raise
          :class:`subtitles_extractor.domain.exceptions.VideoDecodeError`.
    """

    def read(self, video_path: Path) -> VideoMetadata:
        """Đọc metadata video. Raises ``VideoNotFoundError``/``VideoDecodeError``."""
        ...


__all__ = ["VideoMetadataReaderPort"]
