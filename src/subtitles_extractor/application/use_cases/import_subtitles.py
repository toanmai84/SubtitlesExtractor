"""Use case "Đọc tệp phụ đề có sẵn" — phục vụ Editor."""

from __future__ import annotations

import logging
from pathlib import Path

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_importer_port import (
    SubtitleImporterPort,
)

logger = logging.getLogger(__name__)


class ImportSubtitlesUseCase:
    """Đọc tệp ``.srt``/``.ass`` có sẵn thành danh sách event."""

    def __init__(self, importers: dict[str, SubtitleImporterPort]) -> None:
        self._importers = importers

    def execute(self, source_path: Path) -> list[SubtitleEvent]:
        suffix = source_path.suffix.lower().lstrip(".")
        try:
            importer = self._importers[suffix]
        except KeyError as exc:
            raise KeyError(
                f"Định dạng {suffix!r} không được hỗ trợ. "
                f"Định dạng khả dụng: {sorted(self._importers)}."
            ) from exc
        events = importer.import_from(source_path)
        logger.info("Đã nạp %d câu phụ đề từ %s.", len(events), source_path.name)
        return events


__all__ = ["ImportSubtitlesUseCase"]
