"""[v3.23.679] Worker QThread: AI nghe lại các câu lồng tiếng và chấm phát âm (không chặn giao diện)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from threading import Event

from PySide6.QtCore import QThread, Signal

from subtitles_extractor.application.services.pronunciation_check import LineToCheck
from subtitles_extractor.infrastructure.tts.pronunciation_judge import (
    GeminiPronunciationJudge,
    PronunciationJudgeError,
)

logger = logging.getLogger(__name__)


class PronunciationWorker(QThread):
    """Chạy :meth:`GeminiPronunciationJudge.check`.

    Signals:
        progress: ``(đã chấm, tổng)``.
        finished_ok: ``list[PronunciationVerdict]``.
        failed: thông điệp lỗi.
    """

    progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, judge: GeminiPronunciationJudge, lines: Sequence[LineToCheck]) -> None:
        super().__init__()
        self._judge = judge
        self._lines = list(lines)
        self._cancel = Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            verdicts = self._judge.check(
                self._lines, progress=lambda done, total: self.progress.emit(done, total),
                cancelled=self._cancel.is_set,
            )
        except PronunciationJudgeError as exc:   # lỗi nghiệp vụ (thiếu key, hết hạn mức): báo người dùng
            logger.warning("Kiểm tra phát âm dừng: %s", exc)
            self.failed.emit(str(exc))
            return
        except (OSError, RuntimeError, ValueError) as exc:   # tệp audio mất / SDK lỗi bất ngờ
            logger.exception("Kiểm tra phát âm thất bại")
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(verdicts)
