"""Worker chạy :class:`ReOcrUseCase` trên QThread.

Tách khỏi :class:`ExtractSubtitlesWorker` vì:
    * Re-OCR không xuất file, không gộp Multi-ROI.
    * Re-OCR cần signal kết quả khác (``ReOcrResponse`` thay vì
      ``ExtractSubtitlesResponse``) để view model gọi đúng method
      :meth:`SubtitleEditorService.replace_events_by_uid`.

CẢI TIẾN:
    1. [CRITICAL FIX] Ngăn chặn Silent Crash: Bổ sung Catch-all Exception trong run()
       để đảm bảo UI không bao giờ bị kẹt vô tận (Hanging State) nếu engine OCR
       ném ra lỗi ngoại lệ không lường trước.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.dtos.reocr_dto import (
    ReOcrRequest,
    ReOcrResponse,
)
from subtitles_extractor.application.use_cases.reocr import ReOcrUseCase
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.presentation.workers.qt_progress_reporter import (
    QtProgressReporter,
)

logger = logging.getLogger(__name__)


class ReOcrWorker(QObject):
    """Chạy :class:`ReOcrUseCase` trên thread khác, phát kết quả qua signal.

    Signals:
        finished: ``(ReOcrResponse,)`` — phát khi pipeline hoàn tất thành công.
        failed:   ``(str,)`` — phát khi có lỗi (message tiếng Việt).
    """

    finished = Signal(object)  # ReOcrResponse
    failed = Signal(str)

    def __init__(
        self,
        use_case: ReOcrUseCase,
        request: ReOcrRequest,
        reporter: QtProgressReporter,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._request = request
        self._reporter = reporter

    def run(self) -> None:
        """Slot kết nối với ``QThread.started`` — chạy trên thread mới."""
        try:
            response: ReOcrResponse = self._use_case.execute(
                request=self._request, progress=self._reporter
            )
            self.finished.emit(response)

        except SubtitlesExtractorError as exc:
            logger.exception("Lỗi nghiệp vụ khi Re-OCR.")
            self.failed.emit(str(exc))

        except (ValueError, OSError, RuntimeError) as exc:
            # ValueError: time range không hợp lệ sau khi clip.
            # OSError: lỗi I/O (file system, network DLL).
            # RuntimeError: lỗi runtime của adapter (mpv/cv2/paddle).
            logger.exception("Lỗi hệ thống khi Re-OCR.")
            self.failed.emit(f"Lỗi hệ thống: {exc}.")

        except Exception as exc:
            # [CRITICAL FIX]: Bắt toàn bộ lỗi ngoại lệ (IndexError, TypeError, MemoryError...)
            # để ngăn Thread chết âm thầm làm UI bị treo vĩnh viễn.
            logger.exception("Lỗi ngoại lệ không lường trước khi Re-OCR.")
            self.failed.emit(f"Lỗi hệ thống nghiêm trọng: {exc}.")


__all__ = ["ReOcrWorker"]
