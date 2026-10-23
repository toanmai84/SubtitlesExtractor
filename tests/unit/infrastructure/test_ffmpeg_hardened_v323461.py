"""[v3.23.461] Runner ffmpeg chống treo: stdin=DEVNULL + kill cứng khi timeout."""
from __future__ import annotations
import subprocess
import sys
import time
import pytest
from subtitles_extractor.infrastructure.process.hidden_process import run_ffmpeg_hardened


def test_normal_exit_returns_completed():
    r = run_ffmpeg_hardened([sys.executable, "-c", "print('ok')"], timeout=10, check=True)
    assert r.returncode == 0
    assert b"ok" in r.stdout


def test_nonzero_exit_raises_called_process_error():
    with pytest.raises(subprocess.CalledProcessError) as ei:
        run_ffmpeg_hardened([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=10, check=True)
    assert ei.value.returncode == 3


def test_nonzero_exit_no_check_returns():
    r = run_ffmpeg_hardened([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=10, check=False)
    assert r.returncode == 3


def test_hang_is_killed_not_waited():
    t0 = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        run_ffmpeg_hardened([sys.executable, "-c", "import time; time.sleep(60)"], timeout=1.0, check=True)
    # Phải kill nhanh, KHÔNG chờ hết 60s
    assert time.time() - t0 < 15


def test_stderr_captured_on_error():
    r = run_ffmpeg_hardened(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(0)"],
        timeout=10, check=True,
    )
    assert b"boom" in r.stderr
