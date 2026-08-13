"""Service seek video preview persistent — event-driven, không poll loop.

CẢI TIẾN ĐỘT PHÁ (V3.37 - Thread Safe & Event Throttling):
    1. [CRITICAL FIX] Ngăn chặn Crash Segfault: Đảm bảo _reader.close() luôn chạy
       trên đúng Worker Thread thông qua cơ chế Shutdown Signal.
    2. [PERFORMANCE] Chống ngập lụt Event Loop: Áp dụng cờ trạng thái _is_decoding
       để bỏ qua các Signal dư thừa khi người dùng kéo Slider quá nhanh.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage

if TYPE_CHECKING:
    from subtitles_extractor.presentation.workers.seek_worker import (
        PersistentVideoReader,
    )

logger = logging.getLogger(__name__)

_SHUTDOWN_WAIT_MS: int = 10000

def _ndarray_to_qimage(image_rgb: np.ndarray) -> QImage:
    height, width, _ = image_rgb.shape
    contiguous = np.ascontiguousarray(image_rgb)
    qimage = QImage(
        contiguous.tobytes(),
        width,
        height,
        3 * width,
        QImage.Format.Format_RGB888,
    ).copy()
    return qimage


class PreviewSeekService(QObject):
    frame_ready = Signal(QImage, int, int, float)
    failed = Signal(str)

    _seek_requested = Signal()
    # Signal chuyên dụng để dọn dẹp tài nguyên an toàn trên Worker Thread
    _stop_requested = Signal()

    def __init__(
        self,
        video_path: Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(None)

        self._video_path = video_path
        self._reader: PersistentVideoReader | None = None

        self._lock = threading.Lock()
        self._pending_timestamp_sec: float | None = None
        self._is_shutting_down: bool = False

        # Cờ chống ngập lụt Event Queue
        self._is_decoding: bool = False

        self._thread = QThread()

    def start(self) -> None:
        if self._thread.isRunning():
            return

        self.moveToThread(self._thread)
        self._thread.started.connect(self._initialize)
        self._seek_requested.connect(
            self._on_seek_requested, Qt.ConnectionType.QueuedConnection
        )
        self._stop_requested.connect(
            self._on_stop_requested, Qt.ConnectionType.QueuedConnection
        )

        self._thread.start()

    def request_seek(self, timestamp_sec: float) -> None:
        with self._lock:
            if self._is_shutting_down:
                return
            self._pending_timestamp_sec = timestamp_sec

            # [PERFORMANCE FIX] Nếu Thread đang bận decode, không gửi thêm Signal.
            # Vòng lặp trong _on_seek_requested sẽ tự động lấy timestamp mới nhất sau khi xong.
            if self._is_decoding:
                return
            self._is_decoding = True

        self._seek_requested.emit()

    def stop(self) -> None:
        with self._lock:
            if self._is_shutting_down:
                return
            self._is_shutting_down = True

        if self._thread.isRunning():
            # Gửi lệnh yêu cầu Worker Thread tự dọn dẹp và thoát
            self._stop_requested.emit()
            if not self._thread.wait(_SHUTDOWN_WAIT_MS):
                logger.warning(
                    "PreviewSeekService: worker thread không dừng trong %dms — Bỏ qua để tránh deadlock.",
                    _SHUTDOWN_WAIT_MS,
                )
        else:
            # Nếu Thread chưa kịp chạy mà đã gọi stop, dọn dẹp luôn
            if self._reader is not None:
                self._reader.close()
                self._reader = None

    @Slot()
    def _initialize(self) -> None:
        from subtitles_extractor.presentation.workers.seek_worker import (
            PersistentVideoReader,
        )
        try:
            self._reader = PersistentVideoReader(self._video_path)
        except (OSError, RuntimeError, ImportError) as exc:
            logger.warning(
                "PreviewSeekService không mở được video %s: %s.",
                self._video_path.name, exc,
            )
            self.failed.emit(f"Không mở được video: {exc}.")

    @Slot()
    def _on_seek_requested(self) -> None:
        """Vòng lặp tự tiêu thụ timestamp để tránh nghẽn Event Queue."""
        while True:
            with self._lock:
                if self._is_shutting_down:
                    self._is_decoding = False
                    return
                timestamp_sec = self._pending_timestamp_sec
                self._pending_timestamp_sec = None

            # Hết việc để làm -> Xóa cờ bận và thoát Slot
            if timestamp_sec is None:
                with self._lock:
                    self._is_decoding = False
                return

            if self._reader is not None:
                self._decode_and_emit(timestamp_sec)

    @Slot()
    def _on_stop_requested(self) -> None:
        """Thực thi an toàn trên Worker Thread: Đóng reader và kết thúc Event Loop."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._thread.quit()

    def _decode_and_emit(self, timestamp_sec: float) -> None:
        try:
            frame_rgb = self._reader.seek(timestamp_sec)
        except (RuntimeError, OSError) as exc:
            logger.debug(
                "PreviewSeekService decode lỗi tại %.3fs: %s.",
                timestamp_sec, exc,
            )
            return

        if frame_rgb is None:
            return

        try:
            qimage = _ndarray_to_qimage(frame_rgb)
        except (ValueError, AttributeError) as exc:
            logger.debug("Convert ndarray → QImage lỗi: %s.", exc)
            return

        height, width = frame_rgb.shape[:2]
        self.frame_ready.emit(qimage, width, height, timestamp_sec)

__all__ = ["PreviewSeekService"]
