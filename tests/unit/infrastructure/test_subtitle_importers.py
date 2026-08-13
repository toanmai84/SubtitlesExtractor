"""Test :class:`SrtImporter` và :class:`AssImporter` — round-trip với exporter."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.subtitle.exporters.ass_exporter import (
    AssExporter,
)
from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import (
    SrtExporter,
)
from subtitles_extractor.infrastructure.subtitle.importers.ass_importer import (
    AssImporter,
)
from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import (
    SrtImporter,
)


@pytest.fixture
def sample_events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(
            index=1,
            text="Câu thứ nhất",
            interval=TimeInterval(0.0, 2.5),
            confidence=Confidence(0.95),
        ),
        SubtitleEvent(
            index=2,
            text="Câu thứ hai\nDòng dưới",
            interval=TimeInterval(3.0, 5.123),
            confidence=Confidence(0.88),
        ),
        SubtitleEvent(
            index=3,
            text="Câu cuối — có dấu, dấu phẩy, ý nghĩa",
            interval=TimeInterval(10.0, 12.0),
            confidence=Confidence(0.99),
        ),
    ]


class TestSrtImporter:
    def test_import_round_trip(
        self, sample_events: list[SubtitleEvent], tmp_path: Path
    ) -> None:
        target = tmp_path / "test.srt"
        SrtExporter().export(sample_events, target)

        events = SrtImporter().import_from(target)
        assert len(events) == len(sample_events)
        for original, imported in zip(sample_events, events, strict=True):
            assert imported.text == original.text
            assert imported.start_sec == pytest.approx(original.start_sec, abs=1e-3)
            assert imported.end_sec == pytest.approx(original.end_sec, abs=1e-3)

    def test_import_handles_bom(self, tmp_path: Path) -> None:
        target = tmp_path / "bom.srt"
        # Tệp có BOM ở đầu.
        target.write_bytes(
            "\ufeff1\n00:00:00,000 --> 00:00:01,000\nXin chào\n".encode("utf-8")
        )
        events = SrtImporter().import_from(target)
        assert len(events) == 1
        assert events[0].text == "Xin chào"

    def test_import_handles_dot_separator(self, tmp_path: Path) -> None:
        target = tmp_path / "dot.srt"
        target.write_text(
            "1\n00:00:01.500 --> 00:00:03.000\nDùng dấu chấm\n", encoding="utf-8"
        )
        events = SrtImporter().import_from(target)
        assert events[0].start_sec == 1.5

    def test_import_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SrtImporter().import_from(tmp_path / "missing.srt")

    def test_import_empty_returns_empty(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.srt"
        target.write_text("", encoding="utf-8")
        assert SrtImporter().import_from(target) == []


class TestAssImporter:
    def test_import_round_trip(
        self, sample_events: list[SubtitleEvent], tmp_path: Path
    ) -> None:
        target = tmp_path / "test.ass"
        AssExporter().export(sample_events, target)

        events = AssImporter().import_from(target)
        assert len(events) == len(sample_events)
        for original, imported in zip(sample_events, events, strict=True):
            assert imported.text == original.text
            # ASS chỉ chính xác đến centisecond.
            assert imported.start_sec == pytest.approx(original.start_sec, abs=0.01)
            assert imported.end_sec == pytest.approx(original.end_sec, abs=0.01)

    def test_import_skips_override_blocks(self, tmp_path: Path) -> None:
        target = tmp_path / "ov.ass"
        content = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\an8}Xin chào
"""
        target.write_text(content, encoding="utf-8")
        events = AssImporter().import_from(target)
        assert len(events) == 1
        assert events[0].text == "Xin chào"
