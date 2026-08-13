"""Worker phiên âm giọng nói (WhisperX) trên QThread — không chặn UI."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.use_cases.transcribe_speech import (
    TranscribeSpeechRequest,
    TranscribeSpeechUseCase,
)
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

logger = logging.getLogger(__name__)


class TranscribeSpeechWorker(QObject):
    """Phiên âm media thành phụ đề; phát progress, hỗ trợ huỷ mềm."""

    progress = Signal(int, int, str)
    finished = Signal(object)  # list[SubtitleEvent]
    raw_ready = Signal(object, str, str)  # raw_segments, detected_language, model_size
    failed = Signal(str)

    def __init__(
        self,
        use_case: TranscribeSpeechUseCase,
        media_path: Path,
        config: TranscriptionConfig,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._media_path = media_path
        self._config = config
        self._is_cancelled = False

    def request_cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            result = self._use_case.execute(
                TranscribeSpeechRequest(media_path=self._media_path, config=self._config),
                progress_callback=self._on_progress,
                cancellation_check=lambda: self._is_cancelled,
            )
            # Phát raw TRƯỚC để UI có thể xuất dữ liệu thô phục vụ hiệu chuẩn.
            if getattr(result, "raw_segments", None):
                self.raw_ready.emit(
                    result.raw_segments,
                    result.detected_language,
                    self._config.model_size,
                )
            self.finished.emit(result.events)
        except SubtitlesExtractorError as exc:
            # [v3.23.340] GHI LOG trước khi phát tín hiệu. Trước đây lỗi chỉ hiện trên
            # màn hình rồi biến mất — trang Nhật ký TRỐNG TRƠN nên không cách nào chẩn
            # đoán khi người dùng báo lỗi.
            logger.error("Phiên âm thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Lỗi hệ thống khi phiên âm.")
            self.failed.emit(f"Lỗi hệ thống khi phiên âm: {exc}")

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.emit(current, total, message)


__all__ = ["TranscribeSpeechWorker"]
