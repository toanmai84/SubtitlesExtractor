"""[v3.23.650] Dừng mọi luồng nền còn chạy khi đóng ứng dụng.

``MainWindow.closeEvent`` từng chỉ chờ trích xuất / Re-OCR / đo tốc độ / dò ROI. Các luồng khác (xuất bản video,
ghép video, cài lõi OCR/CUDA/mpv/VieNeu GPU/WhisperX, rút gọn AI, nạp giọng VieNeu…) là ``QThread(self)`` con của
trang hoặc chỉ được Python giữ; cửa sổ bị huỷ khi chúng còn chạy ⇒ Qt in "QThread: Destroyed while thread is still
running" rồi ``abort()`` — ứng dụng biến mất không để lại dấu vết, ffmpeg/pip con bị bỏ rơi.

Thứ tự an toàn: (1) yêu cầu huỷ mềm trên worker (``cancel``/``request_cancel`` — giết tiến trình con nếu có),
(2) ``quit`` + chờ, (3) quá hạn thì ``terminate`` (chỉ ở lúc thoát) rồi chờ ngắn.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable

from PySide6.QtCore import QObject, QThread

from subtitles_extractor.presentation.qt_compat import is_deleted

logger = logging.getLogger(__name__)

# Tên thuộc tính của trang giữ worker/luồng có thể còn chạy lúc đóng.
WORKER_ATTRS: tuple[str, ...] = (
    "_worker", "_concat_worker", "_cuda_worker", "_paddle_worker", "_mpv_install_worker",
    "_gpu_install_worker", "_stt_install_worker", "_shorten_worker", "_vieneu_voice_loader",
)
THREAD_ATTRS: tuple[str, ...] = (
    "_thread", "_concat_thread", "_cuda_thread", "_paddle_thread", "_mpv_install_thread",
    "_gpu_install_thread", "_stt_install_thread", "_shorten_worker", "_vieneu_voice_loader",
)

__all__ = ["WORKER_ATTRS", "THREAD_ATTRS", "request_cancel_workers", "running_threads", "stop_threads"]


def _alive(obj: object) -> bool:
    return obj is not None and not is_deleted(obj)


def request_cancel_workers(pages: Iterable[object]) -> int:
    """Gọi ``cancel()``/``request_cancel()`` trên mọi worker của các trang. Trả về số worker đã được yêu cầu huỷ."""
    count = 0
    for page in pages:
        for name in WORKER_ATTRS:
            worker = getattr(page, name, None)
            if not _alive(worker):
                continue
            for method in ("cancel", "request_cancel"):
                fn = getattr(worker, method, None)
                if callable(fn):
                    with contextlib.suppress(Exception):
                        fn()
                        count += 1
                    break
    return count


def running_threads(root: QObject | None, pages: Iterable[object] = ()) -> list[QThread]:
    """Mọi QThread còn chạy: con của ``root`` + luồng/worker mà các trang giữ bằng tham chiếu Python."""
    found: dict[int, QThread] = {}
    if _alive(root):
        for thread in root.findChildren(QThread):  # type: ignore[union-attr]
            if _alive(thread) and thread.isRunning():
                found[id(thread)] = thread
    for page in pages:
        for name in THREAD_ATTRS:
            thread = getattr(page, name, None)
            if isinstance(thread, QThread) and _alive(thread) and thread.isRunning():
                found[id(thread)] = thread
    return list(found.values())


def stop_threads(
    threads: Iterable[QThread], *, wait_ms: int = 4000, kill_wait_ms: int = 1000,
    pump: Callable[[], None] | None = None,
) -> int:
    """Dừng các luồng; trả về số luồng phải ``terminate``."""
    killed = 0
    for thread in threads:
        if not _alive(thread) or not thread.isRunning():
            continue
        with contextlib.suppress(RuntimeError):
            thread.requestInterruption()
            thread.quit()
        if pump is not None:
            pump()
        if thread.wait(wait_ms):
            continue
        logger.warning("Luồng nền %r không dừng trong %d ms — buộc terminate lúc thoát.", thread, wait_ms)
        with contextlib.suppress(RuntimeError):
            thread.terminate()
            thread.wait(kill_wait_ms)
        killed += 1
    return killed
