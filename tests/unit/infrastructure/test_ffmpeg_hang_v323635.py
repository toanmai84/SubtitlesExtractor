"""[v3.23.635] ffmpeg TREO không được làm treo ứng dụng.

Phát hiện khi chạy bộ test đầy đủ: một tiến trình ffmpeg kẹt 26 phút (không in gì), vì vòng
``for line in process.stderr`` chặn tới khi ffmpeg đóng stderr nên ``timeout_sec`` và nút huỷ
đều vô tác dụng. Dùng một "ffmpeg giả" bằng Python để tái hiện (treo im lặng, in tiến độ, lỗi).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video.ffmpeg_concat_adapter import (
    VideoConcatError,
    _run_ffmpeg_concat,
)


def _fake(code: str) -> list[str]:
    return [sys.executable, "-u", "-c", code]


def test_silent_hang_is_killed_by_stall_watchdog(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(VideoConcatError, match="treo"):
        _run_ffmpeg_concat(
            _fake("import time; time.sleep(60)"), tmp_path / "o.mkv", 10.0, None, None,
            timeout_sec=120.0, stall_timeout_sec=1.0,
        )
    assert time.monotonic() - started < 15


def test_overall_timeout_still_applies(tmp_path: Path) -> None:
    with pytest.raises(VideoConcatError, match="quá thời gian"):
        _run_ffmpeg_concat(
            _fake("import time; time.sleep(60)"), tmp_path / "o.mkv", 10.0, None, None,
            timeout_sec=1.0, stall_timeout_sec=120.0,
        )


def test_cancel_works_while_ffmpeg_is_silent(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(VideoConcatError, match="huỷ"):
        _run_ffmpeg_concat(
            _fake("import time; time.sleep(60)"), tmp_path / "o.mkv", 10.0, None,
            lambda: time.monotonic() - started > 0.5, timeout_sec=120.0, stall_timeout_sec=120.0,
        )
    assert time.monotonic() - started < 15


def test_progress_and_success(tmp_path: Path) -> None:
    out = tmp_path / "o.mkv"
    code = (
        "import sys, time, pathlib\n"
        "for t in ('00:00:01.00', '00:00:05.00', '00:00:10.00'):\n"
        "    sys.stderr.write(f'frame=1 time={t} bitrate=1\\r\\n'); sys.stderr.flush(); time.sleep(0.05)\n"
        f"pathlib.Path({str(out)!r}).write_bytes(b'x')\n"
    )
    seen: list[float] = []
    result = _run_ffmpeg_concat(_fake(code), out, 10.0, seen.append, None, timeout_sec=30.0)
    assert result == out
    assert seen == sorted(seen) and seen[-1] == 1.0 and any(0.0 < v < 1.0 for v in seen)


def test_nonzero_exit_reports_last_stderr_lines(tmp_path: Path) -> None:
    code = "import sys; sys.stderr.write('Error opening encoder\\n'); sys.exit(3)"
    with pytest.raises(VideoConcatError, match=r"mã 3.*Error opening encoder"):
        _run_ffmpeg_concat(_fake(code), tmp_path / "o.mkv", 10.0, None, None, timeout_sec=30.0)


def test_finished_but_no_output_file(tmp_path: Path) -> None:
    with pytest.raises(VideoConcatError, match="không thấy tệp"):
        _run_ffmpeg_concat(_fake("pass"), tmp_path / "o.mkv", 10.0, None, None, timeout_sec=30.0)
