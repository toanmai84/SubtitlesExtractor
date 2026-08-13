"""Test tìm libmpv DLL đa vị trí (mpv_dll_manager, v3.23.299).

Kiểm chứng app tìm được libmpv DLL người dùng CHÉP SẴN (env / gốc dự án / cạnh exe)
thay vì bắt buộc tải runtime từ GitHub. Không assert thư mục ưu-tiên-thấp cụ thể
"thắng" (gốc dự án luôn kiểm trước download-dir); test bộ tìm tĩnh + ưu tiên env.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video.mpv_dll_manager import MpvDllManager


def _write_fake_dll(directory: Path, name: str = "libmpv-2.dll", size: int = 2_000_000) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"MZ" + b"\x00" * (size - 2))
    return path


def test_find_dll_in_dir_valid(tmp_path: Path) -> None:
    dll = _write_fake_dll(tmp_path)
    assert MpvDllManager._find_dll_in_dir(tmp_path) == dll


def test_find_dll_in_dir_too_small_ignored(tmp_path: Path) -> None:
    _write_fake_dll(tmp_path, size=1000)
    assert MpvDllManager._find_dll_in_dir(tmp_path) is None


def test_find_dll_in_dir_empty(tmp_path: Path) -> None:
    assert MpvDllManager._find_dll_in_dir(tmp_path) is None


def test_find_dll_in_dir_accepts_mpv2_name(tmp_path: Path) -> None:
    dll = _write_fake_dll(tmp_path, name="mpv-2.dll")
    assert MpvDllManager._find_dll_in_dir(tmp_path) == dll


def test_env_override_takes_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dll = _write_fake_dll(tmp_path)
    monkeypatch.setenv("SUBEXT_MPV_DIR", str(tmp_path))
    manager = MpvDllManager(app_data_dir=tmp_path / "appdata")
    assert manager._find_existing_dll() == dll


def test_env_override_mpv2_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dll = _write_fake_dll(tmp_path, name="mpv-2.dll")
    monkeypatch.setenv("SUBEXT_MPV_DIR", str(tmp_path))
    manager = MpvDllManager(app_data_dir=tmp_path / "appdata")
    assert manager._find_existing_dll() == dll


def test_env_empty_dir_no_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUBEXT_MPV_DIR", str(tmp_path / "empty"))
    manager = MpvDllManager(app_data_dir=tmp_path / "appdata")
    result = manager._find_existing_dll()
    assert result is None or isinstance(result, Path)


def test_find_existing_returns_path_or_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBEXT_MPV_DIR", raising=False)
    manager = MpvDllManager(app_data_dir=tmp_path / "appdata")
    result = manager._find_existing_dll()
    assert result is None or isinstance(result, Path)
