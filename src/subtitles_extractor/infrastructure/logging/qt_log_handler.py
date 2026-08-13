"""Cầu nối log: thu log từ Loguru/stdlib vào bộ đệm vòng + phát Qt signal.

Trang Log đăng ký nhận ``record_emitted`` để hiển thị log theo thời gian thực.
Bộ đệm vòng (ring buffer) giữ N bản ghi gần nhất để trang Log nạp lại đầy đủ
khi mở (kể cả log phát ra trước khi trang được tạo).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class LogEntry:
    """Một dòng log đã chuẩn hoá để hiển thị."""

    time_str: str
    level: str
    name: str
    message: str


class QtLogBridge(QObject):
    """QObject phát Qt signal mỗi khi có log mới + giữ bộ đệm gần nhất.

    Thread-safe: sink của Loguru có thể chạy trên luồng worker; signal Qt được
    phát an toàn qua cơ chế hàng đợi của Qt tới luồng GUI.
    """

    record_emitted = Signal(object)  # phát LogEntry

    def __init__(self, capacity: int = 5000) -> None:
        super().__init__()
        self._buffer: deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def loguru_sink(self, message: object) -> None:
        """Sink gắn vào Loguru (``logger.add(bridge.loguru_sink, ...)``).

        Args:
            message: Đối tượng message của Loguru (có thuộc tính ``record``).
        """
        try:
            record = message.record  # type: ignore[attr-defined]
            entry = LogEntry(
                time_str=record["time"].strftime("%H:%M:%S.%f")[:-3],
                level=record["level"].name,
                name=record["name"] or "",
                message=record["message"],
            )
        except (AttributeError, KeyError, TypeError):
            entry = LogEntry(
                time_str="", level="INFO", name="", message=str(message).rstrip()
            )
        with self._lock:
            self._buffer.append(entry)
        self.record_emitted.emit(entry)

    def snapshot(self) -> list[LogEntry]:
        """Trả về bản sao toàn bộ log đang giữ trong bộ đệm (cũ → mới)."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Xoá bộ đệm log."""
        with self._lock:
            self._buffer.clear()


__all__ = ["QtLogBridge", "LogEntry"]
