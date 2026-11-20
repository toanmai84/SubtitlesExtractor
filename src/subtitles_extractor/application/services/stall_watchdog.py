"""[v3.23.746] Đồng hồ canh TREO của một lượt chạy dài: không có tiến độ quá lâu → ghi ngăn xếp MỌI luồng vào nhật ký.

Log 18/11 21:34 (phim 大乾悍刀行 84 tập): lượt "Xuất dữ liệu thô OCR" dừng sau dòng "Nạp bộ đọc dòng PaddleOCR-VL-1.6-0.9B
(gpu)" hơn 4 giờ, GPU 3 %, VRAM 7,6/8,0 GB — nhật ký không cho biết luồng nào kẹt ở hàm nào. ``faulthandler`` chỉ ghi khi
SẬP, không ghi khi TREO. Đồng hồ này lấy ngăn xếp bằng ``sys._current_frames()`` dưới GIL (an toàn — không đọc ngăn xếp
luồng khác khi không giữ GIL như ``faulthandler`` trên Windows), ghi ở mức WARNING để nhật ký người dùng gửi có đủ.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any, Final

logger = logging.getLogger(__name__)

#: Không có tiến độ chừng này giây → ghi ngăn xếp lần đầu.
STALL_DUMP_SEC: Final[float] = 180.0
#: Còn kẹt → ghi lại sau mỗi khoảng này (tối đa :data:`MAX_DUMPS` lần mỗi lần kẹt).
REPEAT_DUMP_SEC: Final[float] = 600.0
MAX_DUMPS: Final[int] = 3


def thread_stacks() -> str:
    """Ngăn xếp Python của mọi luồng (hàm thuần trên trạng thái hiện tại; gọi dưới GIL)."""
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():  # noqa: SLF001 — API chuẩn để lấy ngăn xếp mọi luồng
        stack = "".join(traceback.format_stack(frame))
        parts.append(f"--- Luồng {names.get(ident, '?')} ({ident}) ---\n{stack}")
    return "\n".join(parts)


class StallWatchdog:
    """``touch(thông điệp)`` mỗi lần có tiến độ; im quá :data:`STALL_DUMP_SEC` → ghi ngăn xếp mọi luồng vào nhật ký."""

    def __init__(self, *, stall_sec: float = STALL_DUMP_SEC, repeat_sec: float = REPEAT_DUMP_SEC,
                 clock: Callable[[], float] = time.monotonic, dump: Callable[[str, float, str], None] | None = None,
                 ) -> None:
        self._stall_sec = stall_sec
        self._repeat_sec = repeat_sec
        self._clock = clock
        self._dump = dump or _log_dump
        self._lock = threading.Lock()
        self._last = clock()
        self._last_message = ""
        self._dumps = 0
        self._next_dump_at = self._last + stall_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def touch(self, message: str = "") -> None:
        with self._lock:
            self._last = self._clock()
            self._next_dump_at = self._last + self._stall_sec
            self._dumps = 0
            if message:
                self._last_message = message

    def check(self) -> bool:
        """Kiểm một lần (luồng canh gọi định kỳ; test gọi trực tiếp). ``True`` nếu vừa ghi ngăn xếp."""
        with self._lock:
            now = self._clock()
            if now < self._next_dump_at or self._dumps >= MAX_DUMPS:
                return False
            self._dumps += 1
            self._next_dump_at = now + self._repeat_sec
            idle, last_message = now - self._last, self._last_message
        self._dump(last_message, idle, thread_stacks())
        return True

    def start(self) -> StallWatchdog:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="stall-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(5.0):
            try:
                self.check()
            except Exception as exc:  # noqa: BLE001 — đồng hồ canh không được phá lượt chạy
                logger.debug("Đồng hồ canh treo lỗi: %s", exc)

    def __enter__(self) -> StallWatchdog:
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


def _log_dump(last_message: str, idle_sec: float, stacks: str) -> None:
    logger.warning(
        "Lượt chạy không có tiến độ %.0f s (bước cuối: %s) — ngăn xếp các luồng để tìm chỗ treo:\n%s",
        idle_sec, last_message or "?", stacks,
    )


class WatchedReporter:
    """Bọc bộ báo tiến độ: mỗi ``report`` là một nhịp sống cho :class:`StallWatchdog`; mọi thứ khác chuyển nguyên."""

    def __init__(self, inner: Any, watchdog: StallWatchdog) -> None:
        self._inner = inner
        self._watchdog = watchdog

    def report(self, *args: Any, **kwargs: Any) -> Any:
        message = next((a for a in reversed(args) if isinstance(a, str)), "")
        self._watchdog.touch(message)
        return self._inner.report(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


__all__ = ["MAX_DUMPS", "STALL_DUMP_SEC", "StallWatchdog", "WatchedReporter", "thread_stacks"]
