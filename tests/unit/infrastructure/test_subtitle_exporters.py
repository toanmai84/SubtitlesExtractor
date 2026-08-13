"""Test atomic write helper và SRT/ASS exporter."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.subtitle.atomic_save import (
    atomic_write_text,
)
from subtitles_extractor.infrastructure.subtitle.exporters.ass_exporter import (
    AssExporter,
)
from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import (
    SrtExporter,
)


class TestAtomicWriteText:
    def test_writes_content_to_target(self, tmp_path: Path) -> None:
        target = tmp_path / "test.txt"
        atomic_write_text(target, "Xin chào")
        assert target.read_text(encoding="utf-8") == "Xin chào"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c.txt"
        atomic_write_text(target, "nested")
        assert target.read_text(encoding="utf-8") == "nested"

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "existing.txt"
        target.write_text("cũ", encoding="utf-8")
        atomic_write_text(target, "mới")
        assert target.read_text(encoding="utf-8") == "mới"

    def test_no_temp_file_remains(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(target, "abc")
        # Không nên còn .tmp trong thư mục.
        leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []


@pytest.fixture
def sample_events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(
            index=1,
            text="Xin chào thế giới!",
            interval=TimeInterval(0.0, 2.5),
            confidence=Confidence(0.95),
        ),
        SubtitleEvent(
            index=2,
            text="Dòng một\nDòng hai",
            interval=TimeInterval(3.0, 5.123),
            confidence=Confidence(0.88),
        ),
    ]


class TestSrtExporter:
    def test_writes_well_formed_srt(
        self, tmp_path: Path, sample_events: list[SubtitleEvent]
    ) -> None:
        exporter = SrtExporter()
        target = tmp_path / "out.srt"
        result = exporter.export(sample_events, target)
        assert result == target.resolve()

        content = target.read_text(encoding="utf-8")
        # 2 block ngăn cách bằng dòng trống.
        assert "1\n00:00:00,000 --> 00:00:02,500\nXin chào thế giới!" in content
        assert "2\n00:00:03,000 --> 00:00:05,123\nDòng một\nDòng hai" in content

    def test_extension(self) -> None:
        assert SrtExporter().file_extension == ".srt"

    def test_empty_event_list(self, tmp_path: Path) -> None:
        exporter = SrtExporter()
        target = tmp_path / "empty.srt"
        exporter.export([], target)
        assert target.exists()
        # [v3.14.4] File ghi bằng utf-8-sig (có BOM) — đọc bằng utf-8-sig để bỏ BOM.
        assert target.read_text(encoding="utf-8-sig") == ""


class TestAssExporter:
    def test_writes_header_and_events(
        self, tmp_path: Path, sample_events: list[SubtitleEvent]
    ) -> None:
        exporter = AssExporter()
        target = tmp_path / "out.ass"
        exporter.export(sample_events, target)
        content = target.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content
        # Newline trong text được escape thành \\N.
        assert "Dòng một\\NDòng hai" in content
        # Format thời gian H:MM:SS.cc.
        assert "Dialogue: 0,0:00:00.00,0:00:02.50" in content

    def test_extension(self) -> None:
        assert AssExporter().file_extension == ".ass"
