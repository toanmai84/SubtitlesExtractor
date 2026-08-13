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


__all__ = ["no_window_kwargs", "popen_hidden", "run_hidden"]
