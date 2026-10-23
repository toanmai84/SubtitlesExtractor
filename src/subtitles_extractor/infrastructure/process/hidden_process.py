"""Chạy tiến trình con mà KHÔNG bật cửa sổ console (Windows).

VÌ SAO cần module này
=====================
Ứng dụng gọi tiến trình con ở rất nhiều nơi: ffmpeg, ffprobe, pip, worker TTS/STT.
Trên Windows, mỗi lời gọi thiếu cờ ``CREATE_NO_WINDOW`` sẽ **bật một cửa sổ cmd đen**
nhấp nháy rồi tắt. Với TTS chạy 55 câu thì đó là 55 lần nhấp nháy — vừa khó chịu vừa
làm người dùng tưởng ứng dụng lỗi.

Rà soát tự động tìm được **11 lời gọi** thiếu cờ này. Rải rác mỗi tệp một kiểu (có nơi
dùng ``_subprocess_flags``, nơi ``_hidden_console_kwargs``, nơi không có gì) nên rất dễ
sót khi thêm lời gọi mới.

Module này gom về MỘT chỗ. Mọi lời gọi tiến trình con nên dùng nó.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

#: Cờ Windows ẩn cửa sổ console của tiến trình con. Không tồn tại trên nền tảng khác.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def no_window_kwargs() -> dict[str, Any]:
    """Tham số bổ sung để tiến trình con không bật cửa sổ.

    Returns:
        Dict truyền vào ``subprocess.run``/``Popen`` bằng ``**``. Rỗng trên nền tảng
        không phải Windows.
    """
    if sys.platform != "win32":
        return {}
    return {"creationflags": _CREATE_NO_WINDOW}


def run_hidden(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Chạy ``subprocess.run`` mà không bật cửa sổ console.

    Args:
        command: Lệnh và tham số.
        **kwargs: Tham số khác truyền thẳng cho ``subprocess.run``.

    Returns:
        Kết quả tiến trình.
    """
    return subprocess.run(command, **no_window_kwargs(), **kwargs)


def popen_hidden(command: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
    """Mở ``subprocess.Popen`` mà không bật cửa sổ console.

    Args:
        command: Lệnh và tham số.
        **kwargs: Tham số khác truyền thẳng cho ``subprocess.Popen``.

    Returns:
        Tiến trình đã mở.
    """
    return subprocess.Popen(command, **no_window_kwargs(), **kwargs)


def _kill_process_tree(proc: "subprocess.Popen[Any]") -> None:
    """[v3.23.461] Kết liễu CHẮC CHẮN tiến trình con (và con cháu). Trên Windows, ffmpeg treo
    trong lời gọi driver GPU có thể KHÔNG chết bằng ``TerminateProcess`` đơn lẻ → dùng
    ``taskkill /F /T`` (force + cây tiến trình). Nền khác: ``kill()``."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                creationflags=_CREATE_NO_WINDOW,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_ffmpeg_hardened(
    command: list[str], timeout: float, check: bool = True
) -> subprocess.CompletedProcess[Any]:
    """[v3.23.461] Chạy ffmpeg CHỐNG TREO tối đa cho app GUI.

    Ba lớp chống treo:
    1. ``stdin=DEVNULL`` — ffmpeg nhận EOF ngay, KHÔNG kẹt chờ nhập (kết hợp cờ ``-nostdin``).
    2. ``communicate(timeout)`` — rút CẢ stdout/stderr song song (không deadlock buffer) + có
       hạn giờ.
    3. Khi timeout → ``taskkill /F /T`` kết liễu CHẮC CHẮN (kể cả ffmpeg kẹt trong driver GPU),
       KHÔNG để app chờ vô hạn / bắt người dùng tự kill ffmpeg.

    Args:
        command: Lệnh ffmpeg (nên có sẵn ``-nostdin``).
        timeout: Hạn giờ (giây) — quá thì kill cứng.
        check: True → ném ``CalledProcessError`` khi mã thoát ≠ 0 (như ``subprocess.run``).

    Returns:
        ``CompletedProcess`` (stdout/stderr là bytes).

    Raises:
        subprocess.TimeoutExpired: Nếu quá hạn (đã kill tiến trình).
        subprocess.CalledProcessError: Nếu ``check`` và mã thoát ≠ 0.
    """
    proc = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **no_window_kwargs(),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        # Rút nốt output còn lại sau khi kill (có hạn giờ để chính bước dọn KHÔNG treo).
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            stdout, stderr = b"", b""
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode, command, output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


__all__ = [
    "no_window_kwargs",
    "popen_hidden",
    "run_hidden",
    "run_ffmpeg_hardened",
]
