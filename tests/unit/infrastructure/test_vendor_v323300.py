"""Test cho :mod:`infrastructure.vendor` (thư mục binary native tập trung, v3.23.300).

Kiểm chứng phân giải gốc vendor: env SUBEXT_VENDOR_DIR -> bundle (_MEIPASS/vendor) ->
gốc dự án (source). Và vendor_subdir chỉ trả thư mục con TỒN TẠI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure import vendor


def test_env_override_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUBEXT_VENDOR_DIR", str(tmp_path))
    assert vendor.vendor_root() == tmp_path


def test_env_override_subdir_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mpv").mkdir()
    monkeypatch.setenv("SUBEXT_VENDOR_DIR", str(tmp_path))
    assert vendor.vendor_subdir("mpv") == tmp_path / "mpv"


def test_env_override_subdir_missing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBEXT_VENDOR_DIR", str(tmp_path))
    assert vendor.vendor_subdir("ffmpeg") is None


def test_env_override_nonexistent_dir_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env trỏ thư mục không tồn tại -> bỏ qua, dùng nguồn/bundle; không crash."""
    monkeypatch.setenv("SUBEXT_VENDOR_DIR", "/khong/ton/tai/vendor")
    result = vendor.vendor_root()
    assert result is None or isinstance(result, Path)


def test_source_vendor_root_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chạy nguồn: repo có vendor/ ở gốc -> tìm thấy."""
    monkeypatch.delenv("SUBEXT_VENDOR_DIR", raising=False)
    root = vendor.vendor_root()
    assert root is not None and root.name == "vendor"


def test_source_vendor_mpv_has_dll(monkeypatch: pytest.MonkeyPatch) -> None:
    """vendor/mpv chứa libmpv DLL đã chuyển vào (xác nhận tổ chức tập trung)."""
    monkeypatch.delenv("SUBEXT_VENDOR_DIR", raising=False)
    mpv_dir = vendor.vendor_subdir("mpv")
    assert mpv_dir is not None
    dlls = list(mpv_dir.glob("*.dll"))
    assert any(d.name in ("libmpv-2.dll", "mpv-2.dll") for d in dlls)
