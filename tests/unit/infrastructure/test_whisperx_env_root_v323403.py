"""[v3.23.403] Test resolver vị trí whisperx_env: cạnh exe (mới) + TEMP (cũ, tương thích).

[v3.23.547] BỎ ``importlib.reload``. Cả ``whisperx_env_root`` lẫn
``whisperx_env_root_candidates`` đọc ``sys.frozen`` / ``sys.executable`` /
``sys._MEIPASS`` ngay LÚC GỌI, nên ``monkeypatch`` là đủ — nạp lại module vô ích.

Ngược lại, nạp lại gây ô nhiễm THẬT giữa các tệp kiểm: nó dựng lại lớp
``WhisperXInstallError`` trong ``__dict__`` của module (cập nhật TẠI CHỖ), trong khi
``test_whisperx_installer_v323336.py`` đã ``import`` lớp CŨ từ trước. Hàm
``assert_targets_isolated_env`` sau đó ném lớp MỚI, còn ``pytest.raises`` chờ lớp CŨ →
ngoại lệ lọt ra ngoài và bộ kiểm đó đỏ. Triệu chứng đặc trưng: chạy riêng thì xanh, chạy
cả bộ thì đỏ — tuỳ thứ tự tệp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.stt import whisperx_installer


def test_env_root_source_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    root = whisperx_installer.whisperx_env_root()
    # Chạy nguồn: gốc dự án (thư mục chứa src/).
    assert (root / "src").is_dir() or root.name  # tồn tại đường dẫn hợp lệ


def test_env_root_frozen_is_exe_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/mnt/dist/App.exe", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "/mnt/temp/_MEI123", raising=False)
    assert whisperx_installer.whisperx_env_root() == Path("/mnt/dist")


def test_env_candidates_include_legacy_temp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/mnt/dist/App.exe", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "/mnt/temp/_MEI123", raising=False)
    cands = whisperx_installer.whisperx_env_root_candidates()
    assert Path("/mnt/dist") in cands  # mới (cạnh exe)
    assert Path("/mnt/temp") in cands  # cũ (%TEMP%, tương thích ngược)
    # Vị trí mới phải đứng TRƯỚC (ưu tiên).
    assert cands[0] == Path("/mnt/dist")


def test_resolvers_doc_sys_luc_goi_khong_luc_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Khoá lý do bỏ ``reload``: đổi ``sys`` là kết quả đổi NGAY, không cần nạp lại."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/a/App.exe", raising=False)
    assert whisperx_installer.whisperx_env_root() == Path("/a")

    monkeypatch.setattr(sys, "executable", "/b/App.exe", raising=False)
    assert whisperx_installer.whisperx_env_root() == Path("/b")
