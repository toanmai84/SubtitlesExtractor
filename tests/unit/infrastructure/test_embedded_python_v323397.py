"""[v3.23.397] Test resolver Python embeddable (thuần, không phụ thuộc đóng gói)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.process.embedded_python import (
    find_bundled_python,
    resolve_installer_python,
)


def test_find_bundled_none_when_not_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert find_bundled_python() is None


def test_find_bundled_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert find_bundled_python() is None


def test_find_bundled_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "vendor" / "python_embed" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert find_bundled_python() == str(exe)


def test_resolve_prefers_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "vendor" / "python_embed" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    # Có cả hệ thống lẫn nhúng → phải chọn NHÚNG.
    assert resolve_installer_python("C:/Python/python.exe") == str(exe)


def test_resolve_falls_back_to_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert resolve_installer_python("C:/Python/python.exe") == "C:/Python/python.exe"


def test_resolve_none_when_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert resolve_installer_python(None) is None
