"""Adapter cho I/O phụ đề — exporters và importers."""

from __future__ import annotations

from subtitles_extractor.infrastructure.subtitle.atomic_save import (
    atomic_write_text,
)
from subtitles_extractor.infrastructure.subtitle.exporters import (
    AssExporter,
    SrtExporter,
)
from subtitles_extractor.infrastructure.subtitle.importers import (
    AssImporter,
    SrtImporter,
)

__all__ = [
    "AssExporter",
    "AssImporter",
    "SrtExporter",
    "SrtImporter",
    "atomic_write_text",
]
