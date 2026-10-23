"""[v3.23.617] Nối các tập khác tốc độ khung bằng ffmpeg THẬT: không được mất khung hình.

Tái hiện lỗi người dùng báo ("video sau khi nối bị thiếu khung hình ở một số phân đoạn"):
concat demuxer + ``-c copy`` với tập 25 fps rồi tập 30 fps cho tệp ra có khoảng trống hàng
chục giây không có khung nào, mà ffmpeg vẫn trả mã 0.
"""
from __future__ import annotations

import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video import ffmpeg_concat_adapter as adapter

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Cần ffmpeg + ffprobe",
)


def _clip(path: Path, *, fps: int, seconds: float, codec: str = "libx264") -> Path:
    """Clip có hình + tiếng, time base mặc định của muxer MP4 theo fps (giống video thật)."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=s=160x284:r={fps}:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
         "-c:v", codec, "-pix_fmt", "yuv420p", "-g", str(fps), "-bf", "2",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, timeout=120,
    )
    return path


def _video_stamps(path: Path) -> list[float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    return sorted(float(t.strip(",")) for t in out.split() if t.strip(",") not in ("", "N/A"))


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    return float(out.strip())


def test_mixed_frame_rates_are_normalized_without_frame_gaps(tmp_path: Path) -> None:
    clips = [
        _clip(tmp_path / "01.mp4", fps=25, seconds=3),
        _clip(tmp_path / "02.mp4", fps=30, seconds=3),
        _clip(tmp_path / "03.mp4", fps=25, seconds=2),
    ]
    output = adapter.concatenate_videos(clips, tmp_path / "tron_bo.mkv")

    stamps = _video_stamps(output)
    max_gap = max(b - a for a, b in pairwise(stamps))
    assert max_gap < 0.1, f"Khoảng trống không có khung hình {max_gap:.2f}s"
    # Tổng 8s ở 25 fps (số đông) → ~200 khung, hình–tiếng không lệch.
    assert 195 <= len(stamps) <= 205
    assert abs(_duration(output) - 8.0) < 0.3
    assert abs(stamps[-1] - 8.0) < 0.2


def test_identical_episodes_still_use_fast_stream_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = [_clip(tmp_path / f"{i:02d}.mp4", fps=25, seconds=2) for i in (1, 2, 3)]
    normalized_calls: list[int] = []
    original = adapter._concat_normalized
    monkeypatch.setattr(
        adapter, "_concat_normalized",
        lambda *a, **k: normalized_calls.append(1) or original(*a, **k),
    )
    output = adapter.concatenate_videos(clips, tmp_path / "tron_bo.mkv")
    assert normalized_calls == []
    assert abs(_duration(output) - 6.0) < 0.3


def test_broken_stream_copy_output_is_redone_by_normalizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lớp bảo vệ thứ hai: thông số đọc được giống nhau nhưng tệp sao chép luồng vẫn hỏng."""
    clips = [_clip(tmp_path / f"{i:02d}.mp4", fps=25, seconds=2) for i in (1, 2)]
    monkeypatch.setattr(adapter, "_max_video_packet_gap", lambda _p: 41.5)
    normalized_calls: list[int] = []
    original = adapter._concat_normalized
    monkeypatch.setattr(
        adapter, "_concat_normalized",
        lambda *a, **k: normalized_calls.append(1) or original(*a, **k),
    )
    adapter.concatenate_videos(clips, tmp_path / "tron_bo.mkv")
    assert normalized_calls == [1]


def test_long_filter_graph_goes_through_script_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bộ nhiều tập: filter dài vượt giới hạn dòng lệnh Windows → đọc từ tệp."""
    clips = [
        _clip(tmp_path / "01.mp4", fps=25, seconds=1),
        _clip(tmp_path / "02.mp4", fps=30, seconds=1),
    ]
    monkeypatch.setattr(adapter, "_INLINE_FILTER_MAX_CHARS", 10)
    output = adapter.concatenate_videos(clips, tmp_path / "tron_bo.mkv")
    assert abs(_duration(output) - 2.0) < 0.3


def test_ffmpeg_version_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        def __init__(self, line: str) -> None:
            self.stdout = line + "\n"

    for line, expected in [
        ("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023", 6),
        ("ffmpeg version n7.1 Copyright (c) 2000-2024", 7),
        ("ffmpeg version N-118000-gabc-20250101 Copyright", 99),
    ]:
        monkeypatch.setattr(adapter.subprocess, "run", lambda *a, _l=line, **k: _Result(_l))
        assert adapter._ffmpeg_major_version("ffmpeg") == expected
