"""Test cho :mod:`infrastructure.media.ffmpeg_locator` (v3.23.297).

Kiểm chứng định vị ffmpeg/ffprobe ưu tiên bundle-first: env override (SUBEXT_FFMPEG_DIR)
-> bản nhúng -> PATH. Đảm bảo bản standalone dùng được ffmpeg đã nhúng thay vì phụ
thuộc PATH hệ thống.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.media import ffmpeg_locator


def _make_fake_exe(directory: Path, name: str) -> Path:
    """Tạo file 'thực thi' giả trong ``directory`` (đặt bit X trên POSIX)."""
    tool = "ffmpeg" if name == "ffmpeg" else "ffprobe"
    filename = f"{tool}.exe" if os.name == "nt" else tool
    path = directory / filename
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_find_executable_in_dir_hit(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path, "ffmpeg")
    found = ffmpeg_locator.find_executable_in_dir(tmp_path, "ffmpeg")
    assert found == str(exe)


def test_find_executable_in_dir_miss(tmp_path: Path) -> None:
    assert ffmpeg_locator.find_executable_in_dir(tmp_path, "ffmpeg") is None


def test_env_override_takes_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SUBEXT_FFMPEG_DIR trỏ đúng thư mục -> dùng bản đó."""
    exe = _make_fake_exe(tmp_path, "ffmpeg")
    monkeypatch.setenv("SUBEXT_FFMPEG_DIR", str(tmp_path))
    assert ffmpeg_locator.find_ffmpeg() == str(exe)


def test_env_override_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _make_fake_exe(tmp_path, "ffprobe")
    monkeypatch.setenv("SUBEXT_FFMPEG_DIR", str(tmp_path))
    assert ffmpeg_locator.find_ffprobe() == str(exe)


def test_env_override_nonexistent_falls_through(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """SUBEXT_FFMPEG_DIR trỏ thư mục không tồn tại -> bỏ qua, không crash."""
    monkeypatch.setenv("SUBEXT_FFMPEG_DIR", "/khong/ton/tai/xyz")
    # Không assert giá trị (tuỳ máy CI có ffmpeg trên PATH hay không) — chỉ cần KHÔNG lỗi.
    result = ffmpeg_locator.find_ffmpeg()
    assert result is None or isinstance(result, str)


def test_no_env_no_bundle_uses_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Không env, không frozen -> uỷ quyền PATH (shutil.which)."""
    monkeypatch.delenv("SUBEXT_FFMPEG_DIR", raising=False)
    monkeypatch.setattr(
        ffmpeg_locator.shutil, "which", lambda tool: f"/usr/bin/{tool}"
    )
    assert ffmpeg_locator.find_ffmpeg() == "/usr/bin/ffmpeg"
    assert ffmpeg_locator.find_ffprobe() == "/usr/bin/ffprobe"


def test_path_miss_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBEXT_FFMPEG_DIR", raising=False)
    monkeypatch.setattr(ffmpeg_locator.shutil, "which", lambda tool: None)
    assert ffmpeg_locator.find_ffmpeg() is None
