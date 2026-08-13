"""Worker QThread chạy TTS trên nền."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Signal

from subtitles_extractor.application.use_cases.generate_tts import GenerateTTSUseCase
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSCancelledError,
    TTSGenerationError,
    TTSRequest,
    TTSSegmentResult,
    TTSUnavailableError,
)

logger = logging.getLogger(__name__)


class TTSWorker(QThread):
    """Chạy GenerateTTSUseCase trên thread nền.

    Signals:
        progress_changed: (ratio: float, label: str)
        finished_ok:      list[TTSSegmentResult] khi hoàn tất.
        failed:           str thông điệp lỗi.
        cancelled:        khi người dùng huỷ.
    """

    progress_changed = Signal(float, str)   # (0.0..1.0, label)
    finished_ok = Signal(object)             # list[TTSSegmentResult]
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        use_case: GenerateTTSUseCase,
        request: TTSRequest,
        output_path: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._use_case = use_case
        self._request = request
        self._output_path = output_path
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        """Yêu cầu dừng TTS sớm."""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            results: list[TTSSegmentResult] = self._use_case.execute(
                request=self._request,
                output_path=self._output_path,
                progress_cb=self._on_progress,
                cancel_cb=self._cancel_event.is_set,
            )
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.finished_ok.emit(results)
        except TTSCancelledError:
            logger.info("TTSWorker: người dùng huỷ.")
            self.cancelled.emit()
        except TTSUnavailableError as exc:
            logger.warning("TTSWorker unavailable: %s", exc)
            self.failed.emit(str(exc))
        except TTSGenerationError as exc:
            logger.warning("TTSWorker gen error: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("TTSWorker lỗi không xác định.")
            self.failed.emit(f"Lỗi không xác định: {exc}")

    def _on_progress(self, ratio: float, label: str) -> None:
        if not self._cancel_event.is_set():
            self.progress_changed.emit(ratio, label)


__all__ = ["TTSWorker"]
