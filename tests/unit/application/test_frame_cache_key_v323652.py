"""[v3.23.652] Cache khung Re-OCR không được trúng giữa hai video trùng tên khác thư mục / tệp bị thay."""

from __future__ import annotations

import os
from pathlib import Path

from subtitles_extractor.application.use_cases.extract_subtitles import _video_identity


def test_same_name_different_folder(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = tmp_path / "a" / "tap01.mp4"
    second = tmp_path / "b" / "tap01.mp4"
    first.write_bytes(b"x" * 10)
    second.write_bytes(b"x" * 10)
    assert _video_identity(first) != _video_identity(second)


def test_replaced_file_changes_identity(tmp_path: Path) -> None:
    video = tmp_path / "tap01.mp4"
    video.write_bytes(b"x" * 10)
    before = _video_identity(video)
    video.write_bytes(b"y" * 12)
    os.utime(video, ns=(1, 2))
    assert _video_identity(video) != before


def test_missing_file_does_not_raise(tmp_path: Path) -> None:
    assert "nope.mp4" in _video_identity(tmp_path / "nope.mp4")
