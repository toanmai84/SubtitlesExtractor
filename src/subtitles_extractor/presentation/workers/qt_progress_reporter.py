"""Adapter :class:`ProgressReporterPort` chuyển tiến độ qua Qt signal.

Worker thread chạy use case, gọi ``report()`` ⇒ phát ``progress_changed``
signal ⇒ UI thread nhận và cập nhật progress bar. Cờ huỷ là một
:class:`threading.Event` đơn giản — UI gọi ``request_cancel()`` để bật cờ.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal


class QtProgressReporter(QObject):
    """Hiện thực :class:`ProgressReporterPort` cho ngữ cảnh PyQt.

    Signals:
        progress_changed (int, int, str): ``(current, total, message)``.
    """

    progress_changed = Signal(int, int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancel_event = threading.Event()

    # ── Port API ────────────────────────────────────────────────────────

    def report(self, current: int, total: int, message: str = "") -> None:
        self.progress_changed.emit(current, total, message)

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    # ── UI helpers ──────────────────────────────────────────────────────

    def request_cancel(self) -> None:
        """Gọi từ UI thread để báo worker dừng."""
        self._cancel_event.set()

    def reset(self) -> None:
        """Xoá cờ huỷ — gọi trước khi bắt đầu phiên mới."""
        self._cancel_event.clear()


__all__ = ["QtProgressReporter"]
