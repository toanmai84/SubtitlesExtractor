"""Unit test cho Phần 2 — Bảo vệ dữ liệu & I/O (v3.14.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.subtitle.atomic_save import (
    _replace_with_retry,
    atomic_write_text,
)
from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import SrtExporter
from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import _parse_srt


class TestAtomicReplaceRetry:
    def test_succeeds_first_try(self, tmp_path: Path) -> None:
        src = tmp_path / "a.tmp"; src.write_text("x")
        dst = tmp_path / "a.srt"
        _replace_with_retry(str(src), str(dst))
        assert dst.read_text() == "x"

    def test_retries_then_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        import os
        calls = {"n": 0}
        real_replace = os.replace

        def flaky(s, d):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("WinError 32: file đang bị khoá")
            return real_replace(s, d)

        monkeypatch.setattr(os, "replace", flaky)
        monkeypatch.setattr("time.sleep", lambda _s: None)  # bỏ chờ thật
        src = tmp_path / "b.tmp"; src.write_text("y")
        dst = tmp_path / "b.srt"
        _replace_with_retry(str(src), str(dst), initial_delay_sec=0.01)
        assert calls["n"] == 3 and dst.read_text() == "y"

    def test_gives_up_after_max(self, tmp_path: Path, monkeypatch) -> None:
        import os
        monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError("locked")))
        monkeypatch.setattr("time.sleep", lambda _s: None)
        with pytest.raises(PermissionError):
            _replace_with_retry("x", "y", max_attempts=3, initial_delay_sec=0.01)


class TestUtf8SigBom:
    def test_atomic_write_adds_bom(self, tmp_path: Path) -> None:
        target = tmp_path / "vn.srt"
        atomic_write_text(target, "Xin chào", encoding="utf-8-sig")
        assert target.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM


class TestMillisecondPrecision:
    def test_single_digit_ms_is_500(self) -> None:
        srt = "1\n00:00:01,5 --> 00:00:02,0\nHello\n"
        events = list(_parse_srt(srt))
        assert events[0].interval.start_sec == pytest.approx(1.5)

    def test_two_digit_ms(self) -> None:
        srt = "1\n00:00:00,05 --> 00:00:01,25\nHi\n"
        events = list(_parse_srt(srt))
        assert events[0].interval.start_sec == pytest.approx(0.05)  # "05"→"050"=50ms
        assert events[0].interval.end_sec == pytest.approx(1.25)

    def test_standard_three_digit(self) -> None:
        srt = "1\n00:00:01,500 --> 00:00:02,750\nHi\n"
        events = list(_parse_srt(srt))
        assert events[0].interval.start_sec == pytest.approx(1.5)
        assert events[0].interval.end_sec == pytest.approx(2.75)


class TestRobustParser:
    def test_handles_extra_blank_lines(self) -> None:
        srt = "1\n00:00:01,000 --> 00:00:02,000\nDòng A\n\n\n\n2\n00:00:03,000 --> 00:00:04,000\nDòng B\n"
        events = list(_parse_srt(srt))
        assert len(events) == 2
        assert events[0].text == "Dòng A" and events[1].text == "Dòng B"

    def test_handles_zero_width_space(self) -> None:
        srt = "1\n00:00:01,000 --> 00:00:02,000\n\u200bDòng A\u200b\n2\n00:00:03,000 --> 00:00:04,000\nDòng B\n"
        events = list(_parse_srt(srt))
        assert len(events) == 2
        assert events[0].text == "Dòng A"

    def test_multiline_text_preserved(self) -> None:
        srt = "1\n00:00:01,000 --> 00:00:02,000\nDòng 1\nDòng 2\n"
        events = list(_parse_srt(srt))
        assert events[0].text == "Dòng 1\nDòng 2"

    def test_full_round_trip_with_exporter(self, tmp_path: Path) -> None:
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.confidence import Confidence
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        events = [
            SubtitleEvent(1, "Câu một", TimeInterval(1.5, 2.5), Confidence(1.0)),
            SubtitleEvent(2, "Câu hai", TimeInterval(3.0, 4.25), Confidence(1.0)),
        ]
        target = tmp_path / "rt.srt"
        SrtExporter().export(events, target)
        from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import SrtImporter
        loaded = SrtImporter().import_from(target)
        assert len(loaded) == 2
        assert loaded[0].interval.start_sec == pytest.approx(1.5)
        assert loaded[1].interval.end_sec == pytest.approx(4.25)
