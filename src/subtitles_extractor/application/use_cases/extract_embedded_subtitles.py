"""Use case trích phụ đề NHÚNG (embedded) — text-based hoặc bitmap-OCR.

Điều phối :class:`EmbeddedSubtitlePort` và (khi cần) :class:`OcrEnginePort`:
  * Track text-based → trả thẳng ``list[SubtitleEvent]``.
  * Track bitmap → OCR từng ảnh bằng PaddleOCR rồi dựng ``SubtitleEvent``.

Đầu ra đồng nhất ``list[SubtitleEvent]`` với mọi nguồn khác (hardsub, STT) để tái
dùng toàn bộ editor/export/translate/TTS phía sau.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
from loguru import logger

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.embedded_subtitle_port import (
    EmbeddedSubtitlePort,
    EmbeddedSubtitleTrack,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEnginePort
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class ExtractEmbeddedRequest:
    """Yêu cầu trích một track phụ đề nhúng."""

    video_path: Path
    track: EmbeddedSubtitleTrack


@dataclass(frozen=True, slots=True)
class ExtractEmbeddedResponse:
    """Kết quả trích: danh sách câu phụ đề + cờ đã-OCR."""

    events: list[SubtitleEvent]
    used_ocr: bool


class ExtractEmbeddedSubtitlesUseCase:
    """Trích phụ đề nhúng; OCR bitmap bằng PaddleOCR khi cần."""

    def __init__(
        self,
        embedded_port: EmbeddedSubtitlePort,
        ocr_engine: OcrEnginePort,
        ocr_batch_size: int = 16,
    ) -> None:
        self._embedded_port = embedded_port
        self._ocr_engine = ocr_engine
        # [v3.23.94] Kích thước lô OCR ảnh phụ đề bitmap (tận dụng GPU theo lô thay vì
        # OCR từng ảnh). >=1; giá trị hợp lý lấy từ cấu hình OCR (mặc định 16).
        self._ocr_batch_size = max(1, ocr_batch_size)

    def list_tracks(self, video_path: Path) -> list[EmbeddedSubtitleTrack]:
        """Liệt kê track phụ đề nhúng (uỷ quyền cho adapter)."""
        return self._embedded_port.list_tracks(video_path)

    def execute(
        self,
        request: ExtractEmbeddedRequest,
        progress_callback: ProgressCallback | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ExtractEmbeddedResponse:
        """Trích track; OCR nếu là bitmap.

        Args:
            request:            Video + track cần trích.
            progress_callback:  ``(current, total, message)`` cho UI (chỉ OCR bitmap).
            cancellation_check: Trả ``True`` để dừng sớm (chỉ OCR bitmap).

        Returns:
            Danh sách câu phụ đề + cờ đã dùng OCR.
        """
        result = self._embedded_port.extract_track(request.video_path, request.track)
        if not result.is_bitmap:
            return ExtractEmbeddedResponse(events=result.events, used_ocr=False)

        return ExtractEmbeddedResponse(
            events=self._ocr_bitmap_frames(
                result.bitmap_frames, progress_callback, cancellation_check
            ),
            used_ocr=True,
        )

    def _ocr_bitmap_frames(
        self,
        bitmap_frames,
        progress_callback: ProgressCallback | None,
        cancellation_check: Callable[[], bool] | None,
    ) -> list[SubtitleEvent]:
        if not self._ocr_engine.is_initialized:
            self._ocr_engine.initialize()

        events: list[SubtitleEvent] = []
        total = len(bitmap_frames)
        # [v3.23.94] OCR theo LÔ qua infer_batch -> một lần predict cho nhiều ảnh, tận
        # dụng GPU hiệu quả hơn nhiều so với OCR từng ảnh (giảm overhead mỗi lần gọi).
        for batch_start in range(0, total, self._ocr_batch_size):
            if cancellation_check is not None and cancellation_check():
                logger.info("Huỷ OCR phụ đề bitmap ở ảnh {}/{}.", batch_start, total)
                break

            batch_frames = bitmap_frames[batch_start : batch_start + self._ocr_batch_size]
            images_rgb: list = []
            indices: list[int] = []
            timestamps: list[float] = []
            valid_frames = []
            for offset, frame in enumerate(batch_frames):
                image_bgr = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue
                images_rgb.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                indices.append(batch_start + offset)
                timestamps.append(frame.start_sec)
                valid_frames.append(frame)

            if images_rgb:
                batch_results = self._ocr_engine.infer_batch(
                    images_rgb, indices, timestamps
                )
                for frame, ocr_result in zip(valid_frames, batch_results, strict=True):
                    text = ocr_result.get_joined_text().strip()
                    if not text:
                        continue
                    events.append(
                        SubtitleEvent(
                            index=len(events) + 1,
                            text=text,
                            interval=TimeInterval(
                                start_sec=frame.start_sec, end_sec=frame.end_sec
                            ),
                            confidence=ocr_result.mean_confidence,
                        )
                    )

            if progress_callback is not None:
                done = min(batch_start + self._ocr_batch_size, total)
                progress_callback(done, total, f"OCR ảnh phụ đề {done}/{total}…")

        if progress_callback is not None:
            progress_callback(total, total, f"Hoàn tất OCR {len(events)} câu phụ đề.")
        logger.info("OCR bitmap (theo lô) xong: {} câu từ {} ảnh.", len(events), total)
        return events


__all__ = [
    "ExtractEmbeddedRequest",
    "ExtractEmbeddedResponse",
    "ExtractEmbeddedSubtitlesUseCase",
]
