"""Worker chạy các tác vụ phụ trợ trên QThread (hardsub detect, auto-ROI)."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.use_cases.detect_auto_roi import DetectAutoRoiUseCase
from subtitles_extractor.application.use_cases.detect_hardsub import DetectHardsubUseCase
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError

logger = logging.getLogger(__name__)


class RoiDetectionCancelled(Exception):
    """[v3.23.698] Người dùng bấm Huỷ khi đang tự dò vùng phụ đề — ném từ hàm báo tiến độ, dừng vòng lấy mẫu/OCR."""


class _CancellableDetection:
    """[v3.23.698] Huỷ giữa chừng cho worker dò vùng: bộ dò gọi hàm báo tiến độ trong vòng lấy mẫu (~mỗi 1 % khung),
    nên hàm đó ném :class:`RoiDetectionCancelled` khi đã bấm Huỷ. Trước đây nút Huỷ vẫn bật trong lúc dò nhưng không làm
    gì — lượt dò đầy đủ (~5 phút trên CPU) chạy hết rồi mới trả quyền."""

    def _init_cancel(self) -> None:
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _check_cancel(self) -> None:
        if self._cancel_event.is_set():
            raise RoiDetectionCancelled

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
        except Exception as exc:  # noqa: BLE001
            # [v3.23.650] Mọi ngoại lệ khác (ValueError/KeyError/cv2.error/MemoryError…) từng thoát khỏi run() không
            # phát finished/failed → thread.quit không bao giờ chạy, trạng thái bận kẹt vĩnh viễn, nút bị khoá.
            logger.exception("Tác vụ nền %s gặp lỗi không lường trước.", "detection_workers")
            self.failed.emit(f"Lỗi không xác định: {exc}")

class DetectAutoRoiWorker(_CancellableDetection, QObject):
    progress = Signal(int, int, str)
    analysis_ready = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()   # [v3.23.698]
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
        self._init_cancel()

    def _on_progress(self, current: int, total: int, msg: str) -> None:
        self._check_cancel()
        self.progress.emit(current, total, msg)

    def run(self) -> None:
        try:
            self._check_cancel()
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
                self._check_cancel()
                self.finished.emit(rois if rois else None)
            else:
                roi = self._use_case.execute(
                    metadata=self._metadata,
                    step_ms=self._step_ms,
                    batch_size=self._batch_size,
                    vram_flags=self._vram_flags,
                )
                self._check_cancel()
                self.finished.emit(roi)
        except RoiDetectionCancelled:
            logger.info("Đã huỷ tự dò vùng phụ đề.")
            self.cancelled.emit()
        except SubtitlesExtractorError as exc:
            logger.error("Tác vụ nền detection_workers thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Tác vụ nền detection_workers thất bại.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        except Exception as exc:  # noqa: BLE001
            # [v3.23.650] Mọi ngoại lệ khác (ValueError/KeyError/cv2.error/MemoryError…) từng thoát khỏi run() không
            # phát finished/failed → thread.quit không bao giờ chạy, trạng thái bận kẹt vĩnh viễn, nút bị khoá.
            logger.exception("Tác vụ nền %s gặp lỗi không lường trước.", "detection_workers")
            self.failed.emit(f"Lỗi không xác định: {exc}")
        finally:
            self.done.emit()

class DetectSeriesRoiWorker(_CancellableDetection, QObject):
    """[v3.23.686] Phim bộ: kiểm nhanh ROI tập trước, không khớp thì dò đầy đủ — MỘT tác vụ nền."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()   # [v3.23.698]
    done = Signal()

    def __init__(
        self,
        use_case: DetectAutoRoiUseCase,
        metadata: VideoMetadata,
        previous_roi: object | None,
        step_ms: int = 1000,
        batch_size: int = 8,
        vram_flags: object | None = None,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._metadata = metadata
        self._previous_roi = previous_roi
        self._step_ms = step_ms
        self._batch_size = batch_size
        self._vram_flags = vram_flags
        self._init_cancel()

    def _on_progress(self, current: int, total: int, msg: str) -> None:
        self._check_cancel()
        self.progress.emit(current, total, msg)

    def run(self) -> None:
        try:
            self._check_cancel()
            roi = self._use_case.detect_series_roi(
                self._metadata, self._previous_roi, step_ms=self._step_ms, batch_size=self._batch_size,
                progress_callback=self._on_progress, vram_flags=self._vram_flags,
            )
            self._check_cancel()
            self.finished.emit(roi)
        except RoiDetectionCancelled:
            logger.info("Đã huỷ dò vùng phụ đề phim bộ.")
            self.cancelled.emit()
        except SubtitlesExtractorError as exc:
            logger.error("Kiểm ROI phim bộ thất bại: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 — như DetectAutoRoiWorker: luôn phát tín hiệu kết thúc
            logger.exception("Kiểm ROI phim bộ gặp lỗi không lường trước.")
            self.failed.emit(f"Lỗi không xác định: {exc}")
        finally:
            self.done.emit()


__all__ = ["DetectAutoRoiWorker", "DetectHardsubWorker", "DetectSeriesRoiWorker"]
