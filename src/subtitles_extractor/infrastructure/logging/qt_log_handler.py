"""Cầu nối log: thu log từ Loguru/stdlib vào bộ đệm vòng + phát Qt signal.

Trang Log đăng ký nhận ``record_emitted`` để hiển thị log theo thời gian thực.
Bộ đệm vòng (ring buffer) giữ N bản ghi gần nhất để trang Log nạp lại đầy đủ
khi mở (kể cả log phát ra trước khi trang được tạo).
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Signal

_MAX_TRACEBACK_CHARS = 4000
"""Trần độ dài traceback nhét vào một dòng log GUI — tránh phình bộ đệm/đơ bảng khi lỗi lồng sâu."""

_SINK_FORMAT = ""
"""Định dạng sink: rỗng → Loguru chỉ dựng phần ``{exception}`` nó tự nối vào cuối.

Phần đó được dựng NGAY ở luồng ghi log, TRƯỚC hàng đợi: với ``enqueue=True`` Loguru pickle
bản ghi và bỏ khung stack của exception, nên dựng lại ở sink chỉ còn "ValueError: …".
"""


def _shorten_traceback(text: str) -> str:
    """Cắt phần GIỮA traceback quá dài; giữ đầu và cuối (dòng cuối là loại lỗi)."""
    if len(text) <= _MAX_TRACEBACK_CHARS:
        return text
    head = text[: _MAX_TRACEBACK_CHARS // 2]
    tail = text[-_MAX_TRACEBACK_CHARS // 2 :]
    return f"{head}\n… (đã cắt bớt traceback) …\n{tail}"


def _format_exception(exception: object) -> str:
    """Dựng chuỗi traceback từ ``record["exception"]`` của Loguru (không bao giờ ném lỗi).

    Args:
        exception: Bộ ba ``(type, value, traceback)`` do Loguru cung cấp.

    Returns:
        Traceback dạng văn bản, đã cắt bớt nếu quá dài.
    """
    try:
        exc_type, exc_value, exc_tb = exception  # type: ignore[misc]
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()
    except (TypeError, ValueError, AttributeError):
        return str(exception)
    return _shorten_traceback(text)


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

    def attach(self, loguru_logger: Any, level: str = "DEBUG") -> int:
        """Gắn sink vào Loguru theo đúng cách giữ được traceback.

        ``enqueue=True`` để log phát ra từ chính slot của trang Nhật ký không chạy lồng
        vào sink đang chạy (Loguru khoá theo handler). ``diagnose=False``: không chép
        giá trị biến (có thể là khoá API) vào bản log người dùng gửi đi.

        Args:
            loguru_logger: ``loguru.logger``.
            level: Cấp thấp nhất đưa lên trang Nhật ký.

        Returns:
            Mã handler (để ``logger.remove``).
        """
        return int(
            loguru_logger.add(
                self.loguru_sink, level=level, format=_SINK_FORMAT, enqueue=True,
                backtrace=False, diagnose=False,
            )
        )

    def loguru_sink(self, message: object) -> None:
        """Sink gắn vào Loguru (dùng :meth:`attach` để gắn).

        Args:
            message: Đối tượng message của Loguru (có thuộc tính ``record``).
        """
        try:
            record = message.record  # type: ignore[attr-defined]
            text = record["message"]
            # [v3.23.486] KÈM TRACEBACK: trước đây sink chỉ lấy ``record["message"]`` nên mọi
            # crash trên trang Nhật ký (và file người dùng xuất ra từ đó) chỉ còn MỘT dòng
            # "Lỗi không lường trước..." — không đủ để chẩn đoán. File log gốc của Loguru vẫn có
            # traceback, nhưng người dùng thường gửi bản xuất từ trang Nhật ký.
            exception = record.get("exception")
            if exception is not None:
                # [v3.23.613] Ưu tiên traceback Loguru đã dựng trước hàng đợi (xem
                # _SINK_FORMAT); dựng lại từ bản ghi chỉ là đường lùi khi gắn kiểu khác.
                rendered = str(message).strip()
                details = (
                    _shorten_traceback(rendered)
                    if rendered.startswith("Traceback")
                    else _format_exception(exception)
                )
                text = f"{text}\n{details}"
            entry = LogEntry(
                time_str=record["time"].strftime("%H:%M:%S.%f")[:-3],
                level=record["level"].name,
                name=record["name"] or "",
                message=text,
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
