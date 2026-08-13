"""Worker chạy trích phụ đề NHÚNG trên QThread (không chặn UI).

Hai tác vụ tách biệt:
  * :class:`ListEmbeddedTracksWorker` — liệt kê track (nhanh, ffprobe).
  * :class:`ExtractEmbeddedWorker` — trích 1 track; nếu bitmap thì OCR (có thể lâu),
    phát ``progress`` và hỗ trợ huỷ mềm qua cờ.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.use_cases.extract_embedded_subtitles import (
    ExtractEmbeddedRequest,
    ExtractEmbeddedSubtitlesUseCase,
)
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.domain.ports.embedded_subtitle_port import EmbeddedSubtitleTrack

logger = logging.getLogger(__name__)


class ListEmbeddedTracksWorker(QObject):
    """Liệt kê track phụ đề nhúng (ffprobe) — nhanh, không cần progress."""

    finished = Signal(object)  # list[EmbeddedSubtitleTrack]
    failed = Signal(str)

    def __init__(
        self, use_case: ExtractEmbeddedSubtitlesUseCase, video_path: Path
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._video_path = video_path

    def run(self) -> None:
        try:
            tracks = self._use_case.list_tracks(self._video_path)
            self.finished.emit(tracks)
        except SubtitlesExtractorError as exc:
            logger.error("Tác vụ nền embedded_extract_worker thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Tác vụ nền embedded_extract_worker thất bại.")
            self.failed.emit(f"Lỗi hệ thống khi đọc phụ đề nhúng: {exc}")


class ExtractEmbeddedWorker(QObject):
    """Trích một track phụ đề nhúng; OCR nếu là bitmap."""

    progress = Signal(int, int, str)
    finished = Signal(object)  # list[SubtitleEvent]
    failed = Signal(str)

    def __init__(
        self,
        use_case: ExtractEmbeddedSubtitlesUseCase,
        video_path: Path,
        track: EmbeddedSubtitleTrack,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._video_path = video_path
        self._track = track
        self._is_cancelled = False

    def request_cancel(self) -> None:
        """Yêu cầu dừng mềm (kiểm giữa các ảnh OCR bitmap)."""
        self._is_cancelled = True

    def run(self) -> None:
        try:
            response = self._use_case.execute(
                ExtractEmbeddedRequest(video_path=self._video_path, track=self._track),
                progress_callback=self._on_progress,
                cancellation_check=lambda: self._is_cancelled,
            )
            self.finished.emit(response.events)
        except SubtitlesExtractorError as exc:
            logger.error("Tác vụ nền embedded_extract_worker thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Tác vụ nền embedded_extract_worker thất bại.")
            self.failed.emit(f"Lỗi hệ thống khi trích phụ đề nhúng: {exc}")

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.emit(current, total, message)


__all__ = ["ListEmbeddedTracksWorker", "ExtractEmbeddedWorker"]
