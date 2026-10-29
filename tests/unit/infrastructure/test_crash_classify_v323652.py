"""[v3.23.652] Không báo "lần trước bị tắt đột ngột" khi chỉ có ngoại lệ đã được xử lý hoặc lỗi sau khi đóng sạch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.logging import crash_hooks
from subtitles_extractor.infrastructure.logging.crash_hooks import classify_fault_text

LUA = "Windows fatal exception: code 0xe24c4a02\n\nThread 0x1 (most recent call first):\n  File \"mpv.py\", line 689 in _event_generator\n"
AV_MPV = "Windows fatal exception: access violation\n\nThread 0x2 (most recent call first):\n  File \"mpv.py\", line 1058 in name\n"
AV_OTHER = "Windows fatal exception: access violation\n\nThread 0x3 (most recent call first):\n  File \"editor_page.py\", line 1 in x\n"
# Định dạng thật với all_threads=True: luồng sự kiện mpv luôn có mặt, luồng gây lỗi là "Current thread".
AV_QT_ALL = ("Windows fatal exception: access violation\n\nThread 0x10 (most recent call first):\n"
             "  File \"mpv.py\", line 689 in _event_generator\n\n"
             "Current thread 0x11 (most recent call first):\n  File \"editor_page.py\", line 9 in paint\n\n"
             "Thread 0x12 (most recent call first):\n  File \"mpv.py\", line 1 in _loop\n")
AV_MPV_ALL = ("Windows fatal exception: access violation\n\n"
              "Current thread 0x11 (most recent call first):\n  File \"mpv.py\", line 1058 in name\n\n"
              "Thread 0x12 (most recent call first):\n  File \"main.py\", line 1 in main\n")
CLEAN = "[clean-exit 2026-10-26 10:00:00]\n"


@pytest.mark.parametrize(
    ("text", "fatal", "mpv", "shutdown"),
    [
        (LUA * 30, False, False, False),
        (LUA + AV_MPV, True, True, False),
        (AV_MPV + LUA + AV_OTHER, True, False, False),          # crash cuối cùng không phải mpv
        ("[Qt FATAL] QThread: Destroyed while thread is still running\n", True, False, False),
        ("Windows fatal exception: code 0xc0000409\n", True, False, False),
        ("Windows fatal exception: code 0x800706ba\n", False, False, False),
        (LUA + AV_MPV + CLEAN, False, False, False),            # AV lượt đầu đã được bắt: app vẫn chạy tới lúc đóng
        (LUA + CLEAN + AV_OTHER, False, False, True),           # lỗi lúc dọn dẹp sau khi đóng sạch
        ("Fatal Python error: Segmentation fault\n", True, False, False),
        (AV_QT_ALL, True, False, False),                        # crash ở Qt: không được tắt mpv oan
        (AV_MPV_ALL, True, True, False),
        ("Windows fatal exception: int divide by zero\n", True, False, False),
        ("Windows fatal exception: page error\n", True, False, False),
    ],
)
def test_classify_fault_text_v323652(text: str, fatal: bool, mpv: bool, shutdown: bool) -> None:
    verdict = classify_fault_text(text)
    assert (verdict.fatal, verdict.mpv_crash, verdict.shutdown_crash) == (fatal, mpv, shutdown)


@pytest.fixture
def hooks(monkeypatch: pytest.MonkeyPatch):
    for name, value in (("_INSTALLED", False), ("_FAULT_FILE", None), ("_PREVIOUS_CRASH", None),
                        ("_SESSION_MARKER", None), ("_LOG_DIR", None)):
        monkeypatch.setattr(crash_hooks, name, value)
    monkeypatch.setattr(sys, "excepthook", lambda *_a: None)
    return crash_hooks


def test_benign_fault_log_is_archived_but_not_reported_v323652(hooks, tmp_path: Path) -> None:
    (tmp_path / "fault.log").write_text(LUA * 5, encoding="utf-8")
    hooks.install_crash_hooks(tmp_path)
    assert list(tmp_path.glob("fault_*.log"))
    assert hooks.pop_previous_crash() is None
    assert not (tmp_path / "mpv_disabled.flag").exists()


def test_mark_clean_exit_writes_marker_and_ends_session_v323652(hooks, tmp_path: Path) -> None:
    import faulthandler

    hooks.install_crash_hooks(tmp_path)
    hooks.begin_session(tmp_path)
    hooks.mark_clean_exit()
    assert "[clean-exit" in (tmp_path / "fault.log").read_text(encoding="utf-8")
    assert not (tmp_path / "session.running").exists()
    assert not faulthandler.is_enabled()
