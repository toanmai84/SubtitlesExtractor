"""Adapter đọc tệp phụ đề từ định dạng cụ thể."""

from __future__ import annotations

from subtitles_extractor.infrastructure.subtitle.importers.ass_importer import (
    AssImporter,
)
from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import (
    SrtImporter,
)

__all__ = ["AssImporter", "SrtImporter"]
