"""[v3.23.362] Kiểm thử trích track ASS/SSA nhúng theo ĐỊNH DẠNG GỐC (SE-style)."""
from __future__ import annotations

from pathlib import Path

from subtitles_extractor.domain.ports.embedded_subtitle_port import EmbeddedSubtitleTrack
from subtitles_extractor.infrastructure.video.ffmpeg_embedded_subtitle_adapter import (
    FfmpegEmbeddedSubtitleAdapter,
)

_SAMPLE_ASS = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname
Style: Default,Arial

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\i1}Xin chào{\\i0} thế giới
Dialogue: 0,0:00:04.00,0:00:06.50,Default,,0,0,0,,Dòng một\\NDòng hai
"""


def test_ass_track_extracted_natively(monkeypatch) -> None:
    adapter = FfmpegEmbeddedSubtitleAdapter()

    def fake_run(command: list[str], label: str, timeout: float | None = None) -> None:
        # ASS phải dùng '-c:s copy' (sao chép nguyên, không chuyển đổi).
        assert command[command.index("-c:s") + 1] == "copy"
        Path(command[-1]).write_text(_SAMPLE_ASS, encoding="utf-8")

    monkeypatch.setattr(adapter, "_run_ffmpeg", fake_run)
    track = EmbeddedSubtitleTrack(track_index=0, codec="ass", is_bitmap=False)
    result = adapter._extract_text(Path("/fake.mkv"), track)

    assert len(result.events) == 2
    # Override block {\i1}{\i0} bị xoá, text sạch.
    assert result.events[0].text == "Xin chào thế giới"
    # \N chuyển thành xuống dòng thật.
    assert result.events[1].text == "Dòng một\nDòng hai"
    assert result.events[0].start_sec == 1.0
    assert result.events[0].end_sec == 3.0


def test_ass_empty_falls_back_to_srt(monkeypatch) -> None:
    adapter = FfmpegEmbeddedSubtitleAdapter()
    codecs_tried: list[str] = []

    def fake_run(command: list[str], label: str, timeout: float | None = None) -> None:
        codec = command[command.index("-c:s") + 1]
        codecs_tried.append(codec)
        if codec == "copy":
            Path(command[-1]).write_text("", encoding="utf-8")  # ASS rỗng
        else:
            Path(command[-1]).write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8"
            )

    monkeypatch.setattr(adapter, "_run_ffmpeg", fake_run)
    track = EmbeddedSubtitleTrack(track_index=0, codec="ass", is_bitmap=False)
    result = adapter._extract_text(Path("/fake.mkv"), track)

    # Thử ASS copy trước, rỗng → rơi về SRT.
    assert codecs_tried == ["copy", "srt"]
    assert len(result.events) == 1


def test_subrip_track_uses_srt_path(monkeypatch) -> None:
    adapter = FfmpegEmbeddedSubtitleAdapter()
    codecs_tried: list[str] = []

    def fake_run(command: list[str], label: str, timeout: float | None = None) -> None:
        codecs_tried.append(command[command.index("-c:s") + 1])
        Path(command[-1]).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8"
        )

    monkeypatch.setattr(adapter, "_run_ffmpeg", fake_run)
    track = EmbeddedSubtitleTrack(track_index=0, codec="subrip", is_bitmap=False)
    result = adapter._extract_text(Path("/fake.mkv"), track)

    # subrip KHÔNG đi đường ASS copy — chỉ SRT.
    assert codecs_tried == ["srt"]
    assert len(result.events) == 1
