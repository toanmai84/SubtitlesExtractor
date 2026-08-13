"""Test huỷ phiên âm và sửa lỗi VieNeu GPU — v3.23.346.

HAI LỖI THẬT:

1. **Bấm Huỷ không ăn.** Adapter kiểm huỷ NGAY TRONG vòng ``for line in process.stderr``,
   nên chỉ phát hiện khi có DÒNG MỚI. WhisperX im lặng khá lâu ở pha nặng — đo được:
   bấm Huỷ ở giây 1,5 mà tới giây 8,0 mới dừng.

2. **VieNeu GPU hỏng 55 lần liên tiếp.** Log thật::

       VieNeu GPU worker thất bại: Try: `pip install transformers -U` …   (×55)

   Nguyên nhân: metadata của ``vieneu`` khai báo ``transformers`` chỉ ở extra
   ``legacy``, nên ``pip install vieneu`` TRẦN không cài nó — mà đường PyTorch (đường
   duy nhất chạy GPU) lại cần. Và mã thoát 5 không kích hoạt việc tắt GPU nên nó thử
   lại từng câu, tốn ~3 phút.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
    VIENEU_EXTRA_PACKAGES,
    VIENEU_INSTALL_SPECS,
    repair_command,
)


def _adapter_source() -> str:
    import subtitles_extractor.infrastructure.stt.whisperx_adapter as module

    return Path(module.__file__).read_text(encoding="utf-8")


def _tts_source() -> str:
    import subtitles_extractor.infrastructure.tts.vieneu_tts_adapter as module

    return Path(module.__file__).read_text(encoding="utf-8")


# ── Huỷ phải phản hồi kể cả khi tiến trình con im lặng ───────────────────────
def test_stderr_read_in_separate_thread() -> None:
    """Đọc ở luồng riêng để vòng chính kiểm huỷ được định kỳ."""
    source = _adapter_source()
    assert "def _drain_stderr" in source
    assert "threading.Thread(target=_drain_stderr" in source


def test_cancel_polled_on_queue_timeout() -> None:
    """Hàng đợi rỗng (tiến trình im lặng) VẪN phải kiểm huỷ — đây là điểm sửa."""
    source = _adapter_source()
    index = source.index("except queue.Empty:")
    window = source[index : index + 400]
    assert "cancelled()" in window
    assert "process.terminate()" in window


def test_poll_interval_is_responsive() -> None:
    """Nhịp kiểm phải đủ nhỏ để người dùng thấy phản hồi gần như tức thì."""
    source = _adapter_source()
    assert "_CANCEL_POLL_SECONDS" in source
    index = source.index("_CANCEL_POLL_SECONDS: Final[float] =")
    value = float(source[index:].split("=")[1].split("\n")[0].strip())
    assert 0.05 <= value <= 0.5


def test_cancel_latency_is_small_in_practice() -> None:
    """Chạy THẬT: tiến trình con im lặng, bấm huỷ phải dừng trong dưới nửa giây."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        child = temp / "slow.py"
        child.write_text(
            "import sys, time\n"
            "print('PROGRESS 0 100 x', file=sys.stderr, flush=True)\n"
            "time.sleep(8)\n",
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [sys.executable, str(child)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def drain() -> None:
            try:
                for raw in process.stderr:  # type: ignore[union-attr]
                    lines.put(raw.rstrip("\n"))
            finally:
                lines.put(None)

        threading.Thread(target=drain, daemon=True).start()

        cancel_at = time.monotonic() + 0.5
        cancelled_at: float | None = None
        while True:
            try:
                item = lines.get(timeout=0.2)
            except queue.Empty:
                if time.monotonic() >= cancel_at:
                    cancelled_at = time.monotonic()
                    process.terminate()
                    break
                continue
            if item is None:
                break

        assert cancelled_at is not None
        assert cancelled_at - cancel_at < 0.5  # phản hồi nhanh
        process.wait(timeout=10)


# ── VieNeu GPU: cài đúng gói ─────────────────────────────────────────────────
def test_install_uses_legacy_extra() -> None:
    """``vieneu`` TRẦN không kéo transformers — phải dùng extra ``legacy``."""
    assert any("[legacy]" in spec for spec in VIENEU_INSTALL_SPECS)


def test_check_names_have_no_extras() -> None:
    """Tên dùng để KIỂM đã cài phải bỏ phần ngoặc vuông, nếu không sẽ không khớp."""
    for name in VIENEU_EXTRA_PACKAGES:
        assert "[" not in name


def test_repair_command_upgrades() -> None:
    """Môi trường cài bằng bản cũ phải sửa được mà không cần xoá đi làm lại."""
    command = repair_command("py.exe")
    assert "--upgrade" in command
    assert any("[legacy]" in part for part in command)
    assert command[0] == "py.exe"


# ── VieNeu GPU: không thử lại vô ích ─────────────────────────────────────────
def test_failure_streak_counter_exists() -> None:
    """Mã 5 (lỗi chạy) trước đây không tắt GPU -> thử lại cả 55 câu."""
    source = _tts_source()
    assert "_GPU_FAILURE_LIMIT" in source
    assert "self._gpu_failure_streak += 1" in source


def test_streak_resets_on_success() -> None:
    """Chỉ hỏng LIÊN TIẾP mới đáng tắt GPU — một câu lỗi lẻ thì không."""
    source = _tts_source()
    assert "self._gpu_failure_streak = 0" in source


def test_failure_limit_is_reasonable() -> None:
    """Đủ nhỏ để không lãng phí, đủ lớn để chịu được một câu lỗi lẻ."""
    source = _tts_source()
    index = source.index("_GPU_FAILURE_LIMIT: Final[int] =")
    value = int(source[index:].split("=")[1].split("\n")[0].strip())
    assert 2 <= value <= 10


def test_streak_initialised_in_constructor() -> None:
    assert "self._gpu_failure_streak: int = 0" in _tts_source()


def test_fixed_errors_still_disable_immediately() -> None:
    """Thiếu thư viện / không có GPU là lỗi CỐ ĐỊNH — tắt ngay, khỏi đếm.

    [v3.23.349] Với tiến trình thường trú, lỗi này lộ ra ở bước bắt tay chứ không phải
    qua mã thoát từng câu.
    """
    source = _tts_source()
    assert "không báo sẵn sàng" in source
    assert "_disable_gpu_worker" in source
