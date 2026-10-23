"""[v3.23.643] Ngoại lệ chưa bắt + số phiên bản nhất quán."""

from __future__ import annotations

import re
import sys
import threading
import tomllib
from pathlib import Path

import pytest
from loguru import logger

import subtitles_extractor
from subtitles_extractor.infrastructure.logging import crash_hooks

ROOT = Path(__file__).resolve().parents[3]


def test_version_consistency_v323643() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    serializer = (
        ROOT / "src/subtitles_extractor/infrastructure/serializers/raw_ocr_serializer.py"
    ).read_text(encoding="utf-8")
    declared = re.search(r'app_version: str = "([^"]+)"', serializer)
    assert declared is not None
    assert subtitles_extractor.__version__ == project == declared.group(1)


@pytest.fixture
def sink():
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(str(message)), level="CRITICAL")
    saved = (sys.excepthook, threading.excepthook, crash_hooks._INSTALLED)
    # hook "cũ" im lặng để pytest(-qt) không coi ngoại lệ cố ý là lỗi của test
    sys.excepthook = lambda *_a: None
    threading.excepthook = lambda _a: None
    crash_hooks._INSTALLED = False
    yield messages
    logger.remove(handler_id)
    sys.excepthook, threading.excepthook, crash_hooks._INSTALLED = saved


def test_thread_and_main_exceptions_are_logged(sink: list[str], tmp_path: Path) -> None:
    crash_hooks.install_crash_hooks(tmp_path)

    def boom() -> None:
        raise ValueError("lỗi trong luồng nền")

    thread = threading.Thread(target=boom, name="worker-x")
    thread.start()
    thread.join()
    assert any("worker-x" in m and "lỗi trong luồng nền" in m for m in sink)

    try:
        raise RuntimeError("lỗi trong slot")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())  # type: ignore[arg-type]
    assert any("lỗi trong slot" in m for m in sink)
    assert (tmp_path / "fault.log").exists()


def test_install_twice_is_a_noop(sink: list[str], tmp_path: Path) -> None:
    crash_hooks.install_crash_hooks(tmp_path)
    first = sys.excepthook
    crash_hooks.install_crash_hooks(tmp_path)
    assert sys.excepthook is first


# ---------------------------------------------------------------------------
# [v3.23.650] fault.log hữu ích thật: cất bản cũ, cài lại sau paddle, dấu phiên, ghi Qt fatal đồng bộ.
# ---------------------------------------------------------------------------
def _fresh_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crash_hooks, "_INSTALLED", False)
    monkeypatch.setattr(crash_hooks, "_FAULT_FILE", None)
    monkeypatch.setattr(crash_hooks, "_PREVIOUS_CRASH", None)
    monkeypatch.setattr(crash_hooks, "_SESSION_MARKER", None)
    monkeypatch.setattr(sys, "excepthook", lambda *_a: None)
    monkeypatch.setattr(threading, "excepthook", lambda _a: None)


def test_non_empty_fault_log_is_archived_and_reported_once_v323650(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_hooks(monkeypatch)
    (tmp_path / "fault.log").write_text("Fatal Python error: Segmentation fault\n", encoding="utf-8")
    crash_hooks.install_crash_hooks(tmp_path)
    archived = list(tmp_path.glob("fault_*.log"))
    assert len(archived) == 1 and "Segmentation" in archived[0].read_text(encoding="utf-8")
    info = crash_hooks.pop_previous_crash()
    assert info is not None and info["fault_log"] == archived[0]
    assert crash_hooks.pop_previous_crash() is None          # chỉ báo một lần
    crash_hooks.end_session()


def test_empty_fault_log_is_not_reported_v323650(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_hooks(monkeypatch)
    (tmp_path / "fault.log").write_text("", encoding="utf-8")
    crash_hooks.install_crash_hooks(tmp_path)
    assert crash_hooks.pop_previous_crash() is None
    assert not list(tmp_path.glob("fault_*.log"))


def test_only_five_archived_fault_logs_are_kept_v323650(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_hooks(monkeypatch)
    for i in range(7):
        (tmp_path / f"fault_20200101_00000{i}.log").write_text("x", encoding="utf-8")
    (tmp_path / "fault.log").write_text("boom", encoding="utf-8")
    crash_hooks.install_crash_hooks(tmp_path)
    assert len(list(tmp_path.glob("fault_*.log"))) == 5


def test_session_marker_detects_unclean_exit_v323650(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_hooks(monkeypatch)
    crash_hooks.begin_session(tmp_path)                       # phiên 1: không gọi end_session (bị kill)
    assert (tmp_path / "session.running").exists()
    (tmp_path / "session.running").write_text("999999 1", encoding="utf-8")   # PID không còn tồn tại
    monkeypatch.setattr(crash_hooks, "_SESSION_MARKER", None)
    crash_hooks.begin_session(tmp_path)                       # phiên 2 thấy dấu còn lại
    assert crash_hooks.pop_previous_crash() is not None
    crash_hooks.end_session()
    assert not (tmp_path / "session.running").exists()
    crash_hooks.begin_session(tmp_path)                       # phiên 3 sau lần thoát sạch
    assert crash_hooks.pop_previous_crash() is None
    crash_hooks.end_session()


def test_rearm_and_note_fatal_write_to_fault_log_v323650(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import faulthandler

    _fresh_hooks(monkeypatch)
    crash_hooks.install_crash_hooks(tmp_path)
    faulthandler.disable()                                    # mô phỏng paddle đè trình xử lý
    crash_hooks.rearm()
    assert faulthandler.is_enabled()
    crash_hooks.note_fatal("QThread: Destroyed while thread is still running")
    assert "Destroyed while thread" in (tmp_path / "fault.log").read_text(encoding="utf-8")


def test_marker_of_a_live_other_instance_is_not_reported_as_crash_v323650(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os as _os

    _fresh_hooks(monkeypatch)
    (tmp_path / "session.running").write_text(f"{_os.getppid()} 1", encoding="utf-8")   # tiến trình cha đang sống
    crash_hooks.begin_session(tmp_path)
    assert crash_hooks.pop_previous_crash() is None
    assert (tmp_path / "session.running").read_text(encoding="utf-8").startswith(str(_os.getppid()))  # không đè
