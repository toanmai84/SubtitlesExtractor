"""Worker chạy các tác vụ phụ trợ trên QThread (hardsub detect, auto-ROI)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.use_cases.detect_auto_roi import DetectAutoRoiUseCase
from subtitles_extractor.application.use_cases.detect_hardsub import DetectHardsubUseCase
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError

logger = logging.getLogger(__name__)

class DetectHardsubWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, use_case: DetectHardsubUseCase, metadata: VideoMetadata, max_samples: int = 12) -> None:
        super().__init__()
        self._use_case = use_case
        self._metadata = metadata
        self._max_samples = max_samples

    def run(self) -> None:
        try:
            result = self._use_case.execute(metadata=self._metadata, max_samples=self._max_samples)
            self.finished.emit(result)
        except SubtitlesExtractorError as exc:
            logger.error("Tác vụ nền detection_workers thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Tác vụ nền detection_workers thất bại.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")

class DetectAutoRoiWorker(QObject):
    progress = Signal(int, int, str)
    analysis_ready = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(
        self,
        use_case: DetectAutoRoiUseCase,
        metadata: VideoMetadata,
        step_ms: int = 1000,
        batch_size: int = 8,
        vram_flags: object | None = None,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._metadata = metadata
        self._step_ms = step_ms
        self._batch_size = batch_size
        # vram_flags: FrameSamplingConfig | None — dùng ``object`` để tránh
        # import vòng tròn trong worker module.
        self._vram_flags = vram_flags

    def _on_progress(self, current: int, total: int, msg: str) -> None:
        self.progress.emit(current, total, msg)

    def run(self) -> None:
        try:
            if self._use_case.supports_analyze_only():
                result = self._use_case.analyze_only(
                    self._metadata,
                    step_ms=self._step_ms,
                    batch_size=self._batch_size,
                    progress_callback=self._on_progress,
                    vram_flags=self._vram_flags,
                )
                if result is None:
                    self.finished.emit(None)
                    return

                detector = getattr(self._use_case, "_detector", None)
                show_review = getattr(detector, "_show_review_ui", False)

                if show_review:
                    self.analysis_ready.emit(result)
                    return
                from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
                    _clusters_to_rois,
                )
                rois = _clusters_to_rois(result.clusters)
                self.finished.emit(rois if rois else None)
            else:
                roi = self._use_case.execute(
                    metadata=self._metadata,
                    step_ms=self._step_ms,
                    batch_size=self._batch_size,
                    vram_flags=self._vram_flags,
                )
                self.finished.emit(roi)
        except SubtitlesExtractorError as exc:
            logger.error("Tác vụ nền detection_workers thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Tác vụ nền detection_workers thất bại.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        finally:
            self.done.emit()

__all__ = ["DetectAutoRoiWorker", "DetectHardsubWorker"]
