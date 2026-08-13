"""Use case phiên âm giọng nói thành phụ đề (Speech-to-Text).

Lớp mỏng điều phối :class:`SpeechToTextPort`. Tách khỏi adapter để dễ test và để
tầng presentation không phụ thuộc trực tiếp vào engine cụ thể (WhisperX).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from subtitles_extractor.domain.ports.speech_to_text_port import (
    SpeechToTextPort,
    TranscriptionConfig,
    TranscriptionProgressCallback,
    TranscriptionResult,
)


@dataclass(frozen=True, slots=True)
class TranscribeSpeechRequest:
    """Yêu cầu phiên âm một media."""

    media_path: Path
    config: TranscriptionConfig


class TranscribeSpeechUseCase:
    """Phiên âm giọng nói trong audio/video thành phụ đề."""

    def __init__(self, stt_engine: SpeechToTextPort) -> None:
        self._stt_engine = stt_engine

    def is_available(self) -> bool:
        """Engine STT có sẵn sàng dùng không (để UI ẩn/hiện chức năng)."""
        return self._stt_engine.is_available()

    def engine_name(self) -> str:
        return self._stt_engine.get_engine_name()

    def execute(
        self,
        request: TranscribeSpeechRequest,
        progress_callback: TranscriptionProgressCallback | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> TranscriptionResult:
        """Phiên âm media thành phụ đề.

        Args:
            request:            Media + cấu hình phiên âm.
            progress_callback:  ``(current, total, message)`` cho UI.
            cancellation_check: Trả ``True`` để dừng sớm.

        Returns:
            Kết quả phiên âm (events + ngôn ngữ nhận diện).

        Raises:
            SpeechToTextError: Khi engine không sẵn sàng hoặc lỗi phiên âm.
        """
        return self._stt_engine.transcribe(
            request.media_path, request.config, progress_callback, cancellation_check
        )


__all__ = ["TranscribeSpeechRequest", "TranscribeSpeechUseCase"]
