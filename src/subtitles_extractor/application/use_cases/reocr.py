"""Use case "Re-OCR" — chạy lại OCR cho một hoặc nhiều khoảng thời gian.

CẢI TIẾN:
    1. [BUG FIX] Sửa lỗi mất tham số VRAM_Preprocess và Median Blend khi tạo Sub-Request.
    2. [UX FIX] Proxy Progress Reporter để chống reset thanh tiến trình.
    3. [CRITICAL BUG FIX] Sửa lỗi AttributeError do đồng bộ tên biến với ExtractSubtitlesUseCase.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
    SubtitleFormat,
)
from subtitles_extractor.application.dtos.reocr_dto import (
    ReOcrRequest,
    ReOcrResponse,
    TimeRange,
    _merge_overlapping_ranges,
)
from subtitles_extractor.application.use_cases.extract_subtitles import (
    ExtractSubtitlesUseCase,
)
from subtitles_extractor.application.use_cases.load_video_metadata import (
    LoadVideoMetadataUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.exceptions import (
    SubtitlesExtractorError,
)
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
)
from subtitles_extractor.domain.ports.progress_reporter_port import (
    NullProgressReporter,
    ProgressReporterPort,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Thời lượng tối thiểu của 1 time range — quá ngắn sẽ trả 0 frame và phí thời gian.
_MIN_RANGE_DURATION_SEC: float = 0.12

# Padding 2 đầu cho mỗi range — bù trừ cho lỗi timestamp do giải mã VFR.
_RANGE_PADDING_SEC: float = 0.00


class _SubTaskProgressProxy(ProgressReporterPort):
    """Proxy pattern để ánh xạ tiến độ (0-100%) của một UseCase con
    vào đúng lát cắt tiến độ của UseCase tổng.
    """

    def __init__(
        self,
        parent: ProgressReporterPort,
        base_ratio: float,
        slice_ratio: float,
        prefix: str,
    ) -> None:
        self._parent = parent
        self._base_ratio = base_ratio
        self._slice_ratio = slice_ratio
        self._prefix = prefix

    def report(self, current: int, total: int, message: str) -> None:
        sub_ratio = (current / total) if total > 0 else 0.0
        global_ratio = self._base_ratio + (sub_ratio * self._slice_ratio)
        self._parent.report(
            int(global_ratio * 1000), 1000, f"{self._prefix}: {message}"
        )

    def is_cancelled(self) -> bool:
        return self._parent.is_cancelled()


class ReOcrUseCase:
    """Orchestrator chạy lại OCR cho các khoảng thời gian được chỉ định.

    Args:
        extract_use_case: Use case OCR toàn video — Re-OCR uỷ quyền cho nó
            xử lý từng range.
        load_metadata_use_case: Use case đọc metadata video. Nhận qua DI để
            không phải truy cập private member của ``extract_use_case``
            (vi phạm encapsulation và Law of Demeter).
    """

    def __init__(
        self,
        extract_use_case: ExtractSubtitlesUseCase,
        load_metadata_use_case: LoadVideoMetadataUseCase,
    ) -> None:
        self._extract = extract_use_case
        self._load_metadata = load_metadata_use_case

    def execute(
        self,
        request: ReOcrRequest,
        progress: ProgressReporterPort | None = None,
    ) -> ReOcrResponse:
        """Chạy Re-OCR."""
        reporter = progress or NullProgressReporter()
        time_started = time.perf_counter()

        reporter.report(0, 100, "Đang đọc metadata video…")

        metadata = self._load_metadata.execute(request.video_path)
        total_duration = metadata.duration_sec

        padded_ranges = self._pad_and_clip_ranges(request.time_ranges, total_duration)

        merged_ranges = _merge_overlapping_ranges(
            padded_ranges, request.merge_window_sec
        )

        valid_ranges = [
            r for r in merged_ranges if r.duration_sec >= _MIN_RANGE_DURATION_SEC
        ]

        if not valid_ranges:
            raise ValueError(
                "Sau khi xử lý và clip về thời lượng video, không còn khoảng thời "
                "gian hợp lệ nào để Re-OCR."
            )

        total_valid_ranges = len(valid_ranges)
        logger.info(
            "Re-OCR: %d range yêu cầu → %d range hợp lệ sau padding/merge. "
            "Tổng thời lượng cần quét: %.2fs.",
            len(request.time_ranges),
            total_valid_ranges,
            sum(r.duration_sec for r in valid_ranges),
        )

        all_new_events: list[SubtitleEvent] = []
        total_frames_processed = 0
        slice_ratio = 1.0 / total_valid_ranges
        was_cancelled = False

        for range_idx, time_range in enumerate(valid_ranges):
            if reporter.is_cancelled():
                logger.info(
                    "Người dùng huỷ tại range %d/%d.",
                    range_idx + 1,
                    total_valid_ranges,
                )
                was_cancelled = True
                break

            proxy_reporter = _SubTaskProgressProxy(
                parent=reporter,
                base_ratio=range_idx * slice_ratio,
                slice_ratio=slice_ratio,
                prefix=f"Đoạn {range_idx + 1}/{total_valid_ranges}[{time_range.start_sec:.1f}s - {time_range.end_sec:.1f}s]"
            )

            sub_request = self._build_sub_request(
                request=request,
                time_range=time_range,
                total_duration=total_duration,
            )

            try:
                sub_response = self._extract.execute(
                    request=sub_request, progress=proxy_reporter
                )
            except (
                SubtitlesExtractorError,
                OSError,
                ValueError,
                RuntimeError,
                MemoryError,
            ) as exc:
                logger.warning(
                    "Range #%d/%d Re-OCR thất bại — bỏ qua: %s.",
                    range_idx + 1,
                    total_valid_ranges,
                    exc,
                )
                continue

            all_new_events.extend(sub_response.events)
            total_frames_processed += sub_response.frames_processed

        all_new_events.sort(key=lambda event: (event.start_sec, event.end_sec))

        # Huỷ có thể xảy ra NGAY TRONG lúc extract một range chạy (không bắt được
        # ở vòng lặp vì check nằm đầu mỗi vòng) → kiểm tra lại sau vòng lặp.
        was_cancelled = was_cancelled or reporter.is_cancelled()

        elapsed = time.perf_counter() - time_started
        logger.info(
            "Re-OCR hoàn tất: %d range → %d câu mới (%d frame, %.1fs).",
            total_valid_ranges,
            len(all_new_events),
            total_frames_processed,
            elapsed,
        )

        return ReOcrResponse(
            new_events=all_new_events,
            replaced_uids=list(request.replace_uids),
            elapsed_seconds=elapsed,
            frames_processed=total_frames_processed,
            ranges_processed=total_valid_ranges,
            was_cancelled=was_cancelled,
        )

    @staticmethod
    def _pad_and_clip_ranges(
        ranges: Iterable[TimeRange], total_duration_sec: float
    ) -> list[TimeRange]:
        clipped: list[TimeRange] = []
        for time_range in ranges:
            start = max(0.0, time_range.start_sec - _RANGE_PADDING_SEC)
            end = min(total_duration_sec, time_range.end_sec + _RANGE_PADDING_SEC)
            if start < end:
                clipped.append(TimeRange(start_sec=start, end_sec=end))
        return clipped

    @staticmethod
    def _build_sub_request(
        request: ReOcrRequest,
        time_range: TimeRange,
        total_duration: float,
    ) -> ExtractSubtitlesRequest:
        skip_intro = time_range.start_sec
        skip_outro = max(0.0, total_duration - time_range.end_sec)

        # [CRITICAL FIX]: Kế thừa ĐẦY ĐỦ các tham số Tiền xử lý (GPU/CPU) từ Request tổng
        sampling = FrameSamplingConfig(
            sample_step_sec=request.sampling.sample_step_sec,
            phash_distance_threshold=request.sampling.phash_distance_threshold,
            pixel_diff_threshold=request.sampling.pixel_diff_threshold,
            skip_intro_sec=skip_intro,
            skip_outro_sec=skip_outro,
            apply_median_blend=request.sampling.apply_median_blend,
            median_blend_frames=request.sampling.median_blend_frames,
            # Bổ sung kế thừa các tham số tối ưu hóa VRAM của PyNvVideoCodec
            vram_upscale_small_text=request.sampling.vram_upscale_small_text,
            vram_upscale_target_height_px=request.sampling.vram_upscale_target_height_px,
            vram_add_border=request.sampling.vram_add_border,
            vram_border_thickness_px=request.sampling.vram_border_thickness_px,
            vram_sharpen=request.sampling.vram_sharpen,
            vram_contrast_factor=request.sampling.vram_contrast_factor,
        )

        dummy_output = request.video_path.with_suffix(".reocr.tmp")

        return ExtractSubtitlesRequest(
            video_path=request.video_path,
            output_path=dummy_output,
            output_format=SubtitleFormat.SRT,
            roi=request.roi,
            sampling=sampling,
            ocr=request.ocr,
            builder=request.builder,
            auto_tune_batch=request.auto_tune_batch,
            save_debug_frames=request.save_debug_frames,
            debug_frames_dir=request.debug_frames_dir,
            keep_temp_files=False,
            skip_export=True,
        )

__all__ = ["ReOcrUseCase"]
