"""Worker tải libmpv DLL trên QThread để UI không bị treo."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.infrastructure.video.mpv_dll_manager import (
    MpvDllError,
    MpvDllManager,
)

logger = logging.getLogger(__name__)


class MpvDllDownloadWorker(QObject):
    """Worker tải libmpv DLL.

    Signals:
        progress (int, int):  bytes_downloaded, total_bytes.
        finished (object):    :class:`MpvDllStatus` khi xong.
        failed (str):         Message lỗi.
    """

    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, app_data_dir: Path) -> None:
        super().__init__()
        self._app_data_dir = app_data_dir

    def run(self) -> None:
        manager = MpvDllManager(self._app_data_dir)
        try:
            status = manager.download_and_install(
                progress_callback=self._on_progress
            )
        except MpvDllError as exc:
            logger.exception("Tải libmpv thất bại.")
            self.failed.emit(str(exc))
            return
        except (OSError, RuntimeError) as exc:
            logger.exception("Lỗi hệ thống khi tải libmpv.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
            return
        self.finished.emit(status)

    def _on_progress(self, downloaded: int, total: int) -> None:
        self.progress.emit(downloaded, total)


__all__ = ["MpvDllDownloadWorker"]
