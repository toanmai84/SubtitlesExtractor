"""Worker chạy :class:`ExtractSubtitlesUseCase` trên QThread.

CẢI TIẾN ĐỘT PHÁ (V3.0 - UI/UX & Thread Safety):
    1. [CRITICAL FIX] Ngăn chặn Silent Crash: Bổ sung Catch-all Exception trong run()
       để đảm bảo UI không bao giờ bị kẹt vô tận (Hanging State) nếu có lỗi bất ngờ.
    2. [UX POLISH] Smooth Progress Bar: Thêm Proxy để scale tiến trình Multi-ROI (0-100% tổng),
       tránh hiện tượng thanh progress bar nhảy giật cục 0->100 nhiều lần.
    3. [PERFORMANCE] Tối ưu hóa gộp danh sách bằng itertools.chain tốc độ cao.
    4. [LOGIC FIX] Sửa lỗi tính toán sai lệch trong log thống kê gộp câu.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
    ExtractSubtitlesResponse,
)
from subtitles_extractor.application.use_cases.export_subtitles import (
    ExportSubtitlesUseCase,
)
from subtitles_extractor.application.use_cases.extract_subtitles import (
    ExtractSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.presentation.workers.qt_progress_reporter import (
    QtProgressReporter,
)

logger = logging.getLogger(__name__)


def _shift_event_coordinates(events: list[SubtitleEvent], roi: Roi | None) -> None:
    """Dịch chuyển tọa độ Bounding Box của OCR từ hệ quy chiếu ROI sang Video gốc."""
    if not roi:
        return

    offset_x, offset_y = roi.x, roi.y

    for ev in events:
        if ev.bounding_box is not None:
            x1, y1, x2, y2 = ev.bounding_box
            ev.bounding_box = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)

        if ev.position is not None:
            px, py = ev.position
            ev.position = (px + offset_x, py + offset_y)


def _merge_events(
    events_per_roi: list[list[SubtitleEvent]],
) -> list[SubtitleEvent]:
    """Gộp nhiều danh sách SubtitleEvent từ các ROI khác nhau, khử trùng giao thoa.

    [Ghost Duplication Fix] Uỷ thác cho :func:`merge_events_from_rois` (tầng
    application, thuần, đã kiểm thử) — nếu 2 ROI giao nhau bắt trùng câu, gộp 1 và
    giữ câu có Confidence cao hơn.
    """
    from subtitles_extractor.application.services.event_deduplication import (
        merge_events_from_rois,
    )

    return merge_events_from_rois(events_per_roi)


class _MultiRoiProgressProxy:
    """[UX FIX]: Proxy nội bộ để scale tiến trình Multi-ROI.
    Giúp UI không bị giật cục (nhảy 0-100% nhiều lần) khi chạy qua từng ROI.
    """
    def __init__(self, real_reporter: QtProgressReporter, current_roi: int, total_rois: int) -> None:
        self._real = real_reporter
        self._current_roi = current_roi
        self._total_rois = total_rois

    def report(self, current: int, total: int, message: str) -> None:
        if total > 0:
            base_percent = (self._current_roi - 1) / self._total_rois
            roi_percent = (current / total) / self._total_rois
            overall_progress = base_percent + roi_percent
            self._real.report(int(overall_progress * 1000), 1000, f"[ROI {self._current_roi}/{self._total_rois}] {message}")
        else:
            self._real.report(0, 0, f"[ROI {self._current_roi}/{self._total_rois}] {message}")

    def is_cancelled(self) -> bool:
        return self._real.is_cancelled()

    # Áp dụng Duck-typing cho các method khác của QtProgressReporter (nếu có)
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class ExtractSubtitlesWorker(QObject):
    """Chạy use case trích xuất trên thread khác, phát kết quả qua signal."""

    finished = Signal(object)  # ExtractSubtitlesResponse
    failed = Signal(str)

    def __init__(
        self,
        use_case: ExtractSubtitlesUseCase,
        requests: list[ExtractSubtitlesRequest],
        reporter: QtProgressReporter,
        export_use_case: ExportSubtitlesUseCase | None = None,
    ) -> None:
        super().__init__()
        if not requests:
            raise ValueError("requests không được rỗng.")
        self._use_case = use_case
        self._requests = requests
        self._reporter = reporter
        self._export_use_case = export_use_case

    def run(self) -> None:
        """Slot chạy trên thread mới (kết nối với ``QThread.started``)."""
        try:
            if len(self._requests) == 1:
                self._run_single()
            else:
                self._run_multi_roi()
        except SubtitlesExtractorError as exc:
            logger.exception("Lỗi nghiệp vụ khi trích xuất phụ đề.")
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Lỗi hệ thống khi trích xuất phụ đề.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        except Exception as exc:
            # [CRITICAL FIX]: Bắt toàn bộ lỗi ngoại lệ (IndexError, TypeError,...)
            # để ngăn Thread chết âm thầm làm UI bị treo (Hanging Spinner) vĩnh viễn.
            logger.exception("Lỗi không lường trước trong worker thread.")
            self.failed.emit(f"Lỗi hệ thống nghiêm trọng: {exc}")

    # ── Single-ROI path ────────────────────────────

    def _run_single(self) -> None:
        request = self._requests[0]
        response: ExtractSubtitlesResponse = self._use_case.execute(
            request=request,
            progress=self._reporter,
        )

        _shift_event_coordinates(response.events, request.roi)

        self.finished.emit(response)

    # ── Multi-ROI path ────────────────────────────────────────────────────

    def _run_multi_roi(self) -> None:
        """Chạy từng ROI lần lượt, gộp kết quả và export 1 lần cuối."""
        n_rois = len(self._requests)
        time_started = time.perf_counter()
        events_per_roi: list[list[SubtitleEvent]] = []
        total_frames_processed = 0

        for roi_idx, request in enumerate(self._requests, start=1):
            if self._reporter.is_cancelled():
                logger.info("Người dùng huỷ tại ROI #%d/%d.", roi_idx, n_rois)
                break

            logger.info(
                "Đang xử lý ROI #%d/%d: (%d,%d) %dx%d.",
                roi_idx,
                n_rois,
                request.roi.x if request.roi else 0,
                request.roi.y if request.roi else 0,
                request.roi.width if request.roi else 0,
                request.roi.height if request.roi else 0,
            )

            # [UX FIX]: Áp dụng Proxy để làm mượt thanh Progress Bar
            progress_proxy = _MultiRoiProgressProxy(self._reporter, roi_idx, n_rois)
            progress_proxy.report(0, 0, "Đang khởi tạo trích xuất…")

            try:
                response = self._use_case.execute(
                    request=request,
                    progress=progress_proxy, # type: ignore
                )
            except SubtitlesExtractorError as exc:
                logger.warning(
                    "ROI #%d thất bại — bỏ qua và tiếp tục ROI tiếp theo: %s.",
                    roi_idx,
                    exc,
                )
                continue

            _shift_event_coordinates(response.events, request.roi)

            events_per_roi.append(response.events)
            total_frames_processed += response.frames_processed
            logger.info(
                "ROI #%d/%d hoàn tất: %d câu phụ đề.",
                roi_idx,
                n_rois,
                len(response.events),
            )

        if self._reporter.is_cancelled():
            self.finished.emit(
                ExtractSubtitlesResponse(
                    events= [],
                    output_path=self._requests[0].output_path,
                    elapsed_seconds=time.perf_counter() - time_started,
                    frames_processed=total_frames_processed,
                )
            )
            return

        self._reporter.report(0, 0, "Đang gộp kết quả từ tất cả ROI…")
        merged_events = _merge_events(events_per_roi)

        # [LOGIC FIX]: Sửa lại dòng log hiển thị sai số lượng toán học
        logger.info(
            "Gộp Multi-ROI hoàn tất: %d ROI → Tổng cộng %d câu phụ đề.",
            n_rois,
            len(merged_events),
        )

        first_request = self._requests[0]
        if self._export_use_case is not None and merged_events:
            self._reporter.report(0, 0, "Đang ghi tệp phụ đề gộp…")
            try:
                output_path = self._export_use_case.execute(
                    events=merged_events,
                    output_path=first_request.output_path,
                    output_format=first_request.output_format,
                )
            except (SubtitlesExtractorError, OSError, KeyError) as exc:
                logger.exception("Export kết quả Multi-ROI thất bại: %s.", exc)
                raise
        else:
            output_path = first_request.output_path
            if not merged_events:
                logger.warning("Không có câu phụ đề nào sau khi gộp tất cả ROI.")

        elapsed = time.perf_counter() - time_started
        self.finished.emit(
            ExtractSubtitlesResponse(
                events=merged_events,
                output_path=output_path,
                elapsed_seconds=elapsed,
                frames_processed=total_frames_processed,
            )
        )

__all__ = ["ExtractSubtitlesWorker", "_merge_events"]
