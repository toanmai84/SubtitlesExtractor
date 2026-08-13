"""[v3.23.170] Test trích phụ đề VĂN BẢN: timeout riêng + loại data/attachment stream.

Lỗi thực tế (ảnh người dùng): "ffmpeg quá thời gian khi trích phụ đề văn bản" với file
lớn (Blu-ray remux). Nguyên nhân: ffmpeg phải demux tuần tự để gom subtitle stream trải
cả phim; timeout chung 120s không đủ. Fix: timeout RIÊNG dài hơn cho trích text + loại
data/attachment (font) khỏi output. Test bằng mock subprocess (không cần ffmpeg thật).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.domain.ports.embedded_subtitle_port import (
    EmbeddedSubtitleTrack,
)
from subtitles_extractor.infrastructure.video.ffmpeg_embedded_subtitle_adapter import (
    FfmpegEmbeddedSubtitleAdapter,
    VideoDecodeError,
)


def _track() -> EmbeddedSubtitleTrack:
    return EmbeddedSubtitleTrack(
        track_index=0, codec="subrip", language="vie", title="", is_bitmap=False
    )


def test_text_extract_uses_longer_timeout(tmp_path, monkeypatch) -> None:
    adapter = FfmpegEmbeddedSubtitleAdapter(
        subprocess_timeout_sec=120.0, text_extract_timeout_sec=900.0
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["timeout"] = kwargs.get("timeout")
        # Giả lập ffmpeg ghi SRT hợp lệ ra đường dẫn cuối trong command.
        Path(command[-1]).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nXin chào\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"x")

    result = adapter._extract_text(video, _track())
    assert result.is_bitmap is False
    assert captured["timeout"] == 900.0  # dùng timeout RIÊNG, không phải 120
    # Lệnh phải LOẠI data + attachment stream.
    command = captured["command"]
    assert "-map" in command and "-0:d?" in command and "-0:t?" in command


def test_text_extract_timeout_raises_friendly_error(tmp_path, monkeypatch) -> None:
    adapter = FfmpegEmbeddedSubtitleAdapter(text_extract_timeout_sec=5.0)

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"x")

    with pytest.raises(VideoDecodeError, match="quá thời gian"):
        adapter._extract_text(video, _track())


def test_default_text_timeout_longer_than_general() -> None:
    adapter = FfmpegEmbeddedSubtitleAdapter()
    assert adapter._text_extract_timeout > adapter._timeout
