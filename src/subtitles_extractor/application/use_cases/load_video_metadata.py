"""Use case "Đọc metadata video".

Đây là use case đơn giản nhất — chỉ delegate sang
:class:`VideoMetadataReaderPort` và xử lý lỗi nhất quán. Mục đích chính
là ngăn tầng presentation gọi trực tiếp adapter.
"""

from __future__ import annotations

import logging
from pathlib import Path

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import VideoNotFoundError
from subtitles_extractor.domain.ports.video_metadata_reader_port import (
    VideoMetadataReaderPort,
)

logger = logging.getLogger(__name__)


class LoadVideoMetadataUseCase:
    """Đọc thông tin của một tệp video.

    Args:
        reader: Adapter hiện thực :class:`VideoMetadataReaderPort`.
    """

    def __init__(self, reader: VideoMetadataReaderPort) -> None:
        self._reader = reader

    def execute(self, video_path: Path) -> VideoMetadata:
        """Đọc metadata.

        Raises:
            VideoNotFoundError: Khi tệp không tồn tại.
            VideoDecodeError:   Khi không đọc được header.
        """
        if not video_path.exists():
            raise VideoNotFoundError(
                f"Không tìm thấy tệp video: {video_path}."
            )
        metadata = self._reader.read(video_path)
        logger.info(
            "Đã đọc metadata: %s (%dx%d, %.2f fps, %.2fs).",
            metadata.filename,
            metadata.width,
            metadata.height,
            metadata.fps,
            metadata.duration_sec,
        )
        return metadata


__all__ = ["LoadVideoMetadataUseCase"]
