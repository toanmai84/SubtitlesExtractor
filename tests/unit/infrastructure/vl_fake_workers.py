"""Tiến trình con giả cho test bộ đọc PaddleOCR-VL tiến trình riêng (phải là hàm cấp mô-đun — spawn nạp theo tên)."""

from __future__ import annotations

import os
import time
from typing import Any


def echo_worker(conn: Any, prepare: bool) -> None:
    conn.send(("ready", os.getpid()))
    while True:
        message = conn.recv()
        if message[0] == "quit":
            return
        conn.send(("ok", [f"{crop.shape[1]}x{crop.shape[0]}" for crop in message[1]]))


def hang_worker(conn: Any, prepare: bool) -> None:
    conn.send(("ready", os.getpid()))
    conn.recv()
    time.sleep(600)                                  # lệnh GPU không bao giờ xong


def crash_worker(conn: Any, prepare: bool) -> None:
    conn.send(("ready", os.getpid()))
    conn.recv()
    os._exit(3)                                       # sập (lỗi native) giữa lúc đọc


def error_worker(conn: Any, prepare: bool) -> None:
    conn.send(("ready", os.getpid()))
    while True:
        message = conn.recv()
        if message[0] == "quit":
            return
        conn.send(("error", "RuntimeError: CUDA error 700"))


class _NoChardet:
    """Như bản exe: không có gói ``chardet`` (LGPL, không đóng gói)."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == "chardet" or name.startswith("chardet."):
            raise ModuleNotFoundError("No module named 'chardet'")
        return None


def exe_like_setup_worker(conn: Any, prepare: bool) -> None:
    """Điều kiện exe cửa sổ: stdout/stderr = None, không có chardet → chuẩn bị tiến trình con rồi thử import chardet."""
    import sys

    sys.modules.pop("chardet", None)
    sys.meta_path.insert(0, _NoChardet())
    sys.stdout = None
    sys.stderr = None
    from subtitles_extractor.composition.child_process_setup import (
        prepare_child_process,
    )

    prepare_child_process()
    import chardet

    conn.send(("ready", os.getpid()))
    conn.recv()
    conn.send(("ok", [type(sys.stderr).__name__, chardet.detect("门没关".encode("utf-8"))["encoding"] or ""]))


def failing_start_worker(conn: Any, prepare: bool) -> None:
    from subtitles_extractor.infrastructure.ocr.paddle_vl_process import _send_quietly

    try:
        raise ModuleNotFoundError("No module named 'chardet'")
    except ModuleNotFoundError as exc:
        _send_quietly(conn, ("error", f"{type(exc).__name__}: {exc}"))
