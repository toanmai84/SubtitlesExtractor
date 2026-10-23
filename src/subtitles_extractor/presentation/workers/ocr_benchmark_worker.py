"""[v3.23.645] Worker đo A/B OCR trên QThread (không chặn giao diện)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.dtos.extract_subtitles_dto import ExtractSubtitlesRequest
from subtitles_extractor.application.services.ocr_benchmark import (
    BenchmarkScenario,
    current_settings_values,
    recommend,
)
from subtitles_extractor.application.services.ocr_probe_window import ProbeWindow
from subtitles_extractor.application.use_cases.benchmark_ocr import BenchmarkOcrUseCase
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.presentation.workers.qt_progress_reporter import QtProgressReporter

logger = logging.getLogger(__name__)


class OcrBenchmarkWorker(QObject):
    """``finished`` phát ``(list[ScenarioResult], Recommendation | None, bool đã_huỷ)``."""

    finished = Signal(object, object, bool)
    failed = Signal(str)

    def __init__(
        self,
        use_case: BenchmarkOcrUseCase,
        request: ExtractSubtitlesRequest,
        scenarios: Sequence[BenchmarkScenario],
        window: ProbeWindow,
        reporter: QtProgressReporter,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._request = request
        self._scenarios = list(scenarios)
        self._window = window
        self._reporter = reporter

    def run(self) -> None:
        try:
            results = self._use_case.execute(self._request, self._scenarios, self._window, self._reporter)
            cancelled = self._reporter.is_cancelled()
            self.finished.emit(results, None if cancelled else recommend(results, current_settings_values(self._request)), cancelled)
        except SubtitlesExtractorError as exc:
            logger.error("Đo tốc độ OCR thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
            logger.exception("Đo tốc độ OCR thất bại.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        except Exception as exc:  # noqa: BLE001 - luồng nền: tuyệt đối không để giao diện kẹt ở trạng thái bận
            logger.exception("Đo tốc độ OCR: lỗi không lường trước.")
            self.failed.emit(f"Lỗi không lường trước: {exc}")
