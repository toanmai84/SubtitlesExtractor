"""Use case "Xuất phụ đề đã chỉnh sửa" — sử dụng bởi editor.

Cho phép tầng presentation chỉnh sửa danh sách
:class:`SubtitleEvent` rồi ghi ra tệp mà không phải biết exporter cụ thể.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_exporter_port import (
    SubtitleExporterPort,
)
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat

logger = logging.getLogger(__name__)


class ExportSubtitlesUseCase:
    """Ghi danh sách phụ đề đã chỉnh sửa ra tệp."""

    def __init__(self, exporters: dict[str, SubtitleExporterPort]) -> None:
        self._exporters = exporters

    def execute(
        self,
        events: Sequence[SubtitleEvent],
        output_path: Path,
        output_format: SubtitleFormat,
    ) -> Path:
        """Trả về đường dẫn tệp đã ghi.

        Raises:
            KeyError: Khi định dạng yêu cầu không có exporter.
            SubtitleExportError: Khi ghi đĩa thất bại.
        """
        try:
            exporter = self._exporters[output_format.value]
        except KeyError as exc:
            raise KeyError(
                f"Định dạng phụ đề {output_format!r} chưa được đăng ký. "
                f"Khả dụng: {sorted(self._exporters)}."
            ) from exc
        result_path = exporter.export(events, output_path)
        logger.info("Đã xuất %d câu phụ đề → %s.", len(events), result_path)
        return result_path


__all__ = ["ExportSubtitlesUseCase"]
