"""[v3.23.629] Worker QThread: nhờ AI rút gọn các câu dịch quá dài (không chặn giao diện)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from threading import Event

from PySide6.QtCore import QThread, Signal

from subtitles_extractor.application.use_cases.shorten_for_speech import (
    ShortenForSpeechUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationCancelledError,
)
from subtitles_extractor.infrastructure.tts.tts_fit import LineFit

logger = logging.getLogger(__name__)


class ShortenWorker(QThread):
    """Chạy :class:`ShortenForSpeechUseCase`.

    Signals:
        finished_ok: ``list[ShortenedLine]`` đã hợp lệ.
        failed: thông điệp lỗi.
        cancelled: người dùng huỷ.
    """

    finished_ok = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        use_case: ShortenForSpeechUseCase,
        fits: Sequence[LineFit],
        texts: Sequence[str],
        target_lang: str,
        model_name: str,
        *,
        source_texts: Sequence[str] = (),
        speakers: Sequence[str] = (),
        addressees: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._args = (list(fits), list(texts), target_lang, model_name)
        self._kwargs = {
            "source_texts": list(source_texts), "speakers": list(speakers),
            "addressees": list(addressees),
        }
        self._cancel = Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            result = self._use_case.execute(
                *self._args, **self._kwargs, cancel_cb=self._cancel.is_set
            )
            self.finished_ok.emit(result)
        except TranslationCancelledError:
            self.cancelled.emit()
        except SubtitleTranslationError as exc:
            logger.warning("Rút gọn câu thất bại: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 — chống treo giao diện
            logger.exception("Lỗi không lường trước khi rút gọn câu.")
            self.failed.emit(f"Lỗi không xác định: {exc}")


__all__ = ["ShortenWorker"]
