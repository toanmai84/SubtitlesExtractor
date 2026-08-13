"""Use case "Trích xuất phụ đề" — orchestrator chính của ứng dụng.

TỐI ƯU ĐỘT PHÁ (V3.13 - OOM Fast-Fail & Stability):
    * [STABILITY FIX]: Ngừng nuốt lỗi Tràn VRAM (OOM). Nếu GPU hết bộ nhớ, lập tức
      Hủy toàn bộ tiến trình thay vì cố chạy tiếp tạo ra hàng vạn dòng lỗi.
    * [CRITICAL PERF]: Thay thế bộ đệm mảng Numpy RGB thô bằng Bộ nén In-Memory JPEG.
    * [KIẾN TRÚC MỚI]: Xử lý Tịnh tiến Tọa độ VRAM ngược giúp Khớp 100% Khung Hình.
    * [BUG FIX]: Đưa import dataclasses lên Global Scope chống sập luồng khi Frame rỗng.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
    ExtractSubtitlesResponse,
)
from subtitles_extractor.application.services.subtitle_builder import (
    SubtitleBuilder,
)
from subtitles_extractor.application.use_cases.load_video_metadata import (
    LoadVideoMetadataUseCase,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.domain.exceptions import (
    OcrInferenceError,
    OcrModelLoadError,
    SubtitlesExtractorError,
)
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplerPort,
    FrameSamplingConfig,
    SampledFrame,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEnginePort
from subtitles_extractor.domain.ports.progress_reporter_port import (
    NullProgressReporter,
    ProgressReporterPort,
)
from subtitles_extractor.domain.ports.subtitle_exporter_port import (
    SubtitleExporterPort,
)
from subtitles_extractor.domain.ports.video_metadata_reader_port import (
    VideoMetadataReaderPort,
)

if TYPE_CHECKING:
    from subtitles_extractor.domain.entities.video_metadata import VideoMetadata

logger = logging.getLogger(__name__)

_AUTO_TUNE_TARGET_SECONDS: float = 1.5
_AUTO_TUNE_MIN_BATCH_SIZE: int = 1
_AUTO_TUNE_MAX_BATCH_SIZE: int = 256
_PROGRESS_BATCH_REPORT_MULTIPLIER: int = 5


@dataclass(frozen=True, slots=True)
class _CompressedSampledFrame:
    frame_index: int
    timestamp_sec: float
    image_jpeg_bytes: bytes
    is_duplicate: bool
    is_error: bool


class _ReOcrFrameCacheManager:
    def __init__(self) -> None:
        self._cached_key: str = ""
        self._cached_frames: list[_CompressedSampledFrame] = []
        self._max_cached_frames: int = 500  # Tối đa ~20 giây video @ 25fps
        self._lock: threading.RLock = threading.RLock()

    @property
    def max_cached_frames(self) -> int:
        return self._max_cached_frames

    def get(self, key: str) -> list[SampledFrame] | None:
        with self._lock:
            if self._cached_key != key or not self._cached_frames:
                return None
            cached_snapshot = list(self._cached_frames)

        logger.info("⚡ RAM Cache HIT! Giải nén %d frames từ bộ nhớ đệm...", len(cached_snapshot))
        decoded_frames: list[SampledFrame] = []
        empty_image = np.empty(0, dtype=np.uint8)

        for comp_frame in cached_snapshot:
            if comp_frame.is_duplicate or comp_frame.is_error or not comp_frame.image_jpeg_bytes:
                decoded_img = empty_image
            else:
                np_arr = np.frombuffer(comp_frame.image_jpeg_bytes, np.uint8)
                bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                decoded_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) if bgr_img is not None else empty_image

            decoded_frames.append(SampledFrame(
                frame_index=comp_frame.frame_index,
                timestamp_sec=comp_frame.timestamp_sec,
                image_rgb=decoded_img,
                is_duplicate=comp_frame.is_duplicate,
                is_error=comp_frame.is_error,
            ))
        return decoded_frames

    def put(self, key: str, frames: list[SampledFrame]) -> None:
        if len(frames) > self._max_cached_frames:
            return

        compressed_list: list[_CompressedSampledFrame] = []
        for frame in frames:
            jpeg_bytes = b""
            if not frame.is_duplicate and not frame.is_error and frame.image_rgb.size > 0:
                bgr_img = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2BGR)
                success, encoded = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success and encoded is not None:
                    jpeg_bytes = encoded.tobytes()

            compressed_list.append(_CompressedSampledFrame(
                frame_index=frame.frame_index, timestamp_sec=frame.timestamp_sec,
                image_jpeg_bytes=jpeg_bytes, is_duplicate=frame.is_duplicate, is_error=frame.is_error,
            ))

        with self._lock:
            self._cached_key = key
            self._cached_frames = compressed_list

        logger.debug("Đã nén và lưu %d frames vào RAM Cache thành công.", len(compressed_list))

    def clear(self) -> None:
        with self._lock:
            self._cached_key = ""
            self._cached_frames = []


_GLOBAL_FRAME_CACHE = _ReOcrFrameCacheManager()


def _sampling_signature(cfg: FrameSamplingConfig) -> str:
    """Chữ ký mọi tham số lấy mẫu ẢNH HƯỞNG ĐẾN PIXEL của frame.

    Dùng làm thành phần khoá cache RAM. Bất kỳ thay đổi nào ở bước lấy mẫu, ngưỡng khử trùng
    hay tiền xử lý VRAM (upscale/sharpen/contrast/border/median-blend) đều phải tạo khoá khác
    để không tái dùng frame lỗi thời. (Model OCR không nằm ở đây vì không đổi pixel.)
    """
    return "_".join(
        str(x)
        for x in (
            cfg.sample_step_sec,
            cfg.phash_distance_threshold,
            cfg.pixel_diff_threshold,
            cfg.skip_intro_sec,
            cfg.skip_outro_sec,
            cfg.apply_median_blend,
            cfg.median_blend_frames,
            cfg.vram_upscale_small_text,
            cfg.vram_upscale_target_height_px,
            cfg.vram_add_border,
            cfg.vram_border_thickness_px,
            cfg.vram_sharpen,
            cfg.vram_contrast_factor,
        )
    )


class ExtractSubtitlesUseCase:
    def __init__(
        self,
        metadata_reader: VideoMetadataReaderPort,
        frame_sampler: FrameSamplerPort,
        ocr_engine: OcrEnginePort,
        builder: SubtitleBuilder,
        exporters: dict[str, SubtitleExporterPort],
    ) -> None:
        self._load_metadata_use_case = LoadVideoMetadataUseCase(metadata_reader)
        self._frame_sampler_port = frame_sampler
        self._ocr_engine_port = ocr_engine
        self._subtitle_builder = builder
        self._subtitle_exporters = exporters

    def execute(
        self,
        request: ExtractSubtitlesRequest,
        progress: ProgressReporterPort | None = None,
    ) -> ExtractSubtitlesResponse:
        progress_reporter = progress or NullProgressReporter()
        subtitle_exporter: SubtitleExporterPort | None = None

        if not request.skip_export:
            subtitle_exporter = self._resolve_exporter(request.output_format.value)

        start_time_seconds = time.perf_counter()
        progress_reporter.report(0, 100, "Đang chuẩn bị dữ liệu…")

        video_metadata = self._load_metadata_use_case.execute(request.video_path)
        if request.roi is not None:
            video_metadata = video_metadata.replace_roi(request.roi)

        apply_configuration_method = getattr(self._ocr_engine_port, "apply_config", None)
        if callable(apply_configuration_method):
            apply_configuration_method(request.ocr)

        if not self._ocr_engine_port.is_initialized:
            progress_reporter.report(0, 100, "Đang nạp mô hình AI OCR lên thiết bị xử lý…")
            self._ocr_engine_port.initialize()

        debug_frames_directory: Path | None = None
        if request.save_debug_frames:
            debug_frames_directory = self._resolve_debug_directory(request)
            debug_frames_directory.mkdir(parents=True, exist_ok=True)

        try:
            ocr_frame_results = self._execute_ocr_inference_loop(
                video_metadata, request, progress_reporter, debug_frames_directory
            )
        finally:
            gc.collect()

        if progress_reporter.is_cancelled():
            return ExtractSubtitlesResponse(
                events=[], output_path=request.output_path, elapsed_seconds=time.perf_counter() - start_time_seconds, frames_processed=len(ocr_frame_results),
            )

        progress_reporter.report(95, 100, "Đang tối ưu không gian và gộp câu phụ đề…")

        raw_ocr_file_path: Path | None = None
        if request.raw_ocr_output_path is not None:
            raw_ocr_file_path = self._save_raw_ocr_payload(
                ocr_results=ocr_frame_results, output_file_path=request.raw_ocr_output_path,
                request=request, metadata=video_metadata,
            )

        built_subtitle_events = self._subtitle_builder.build(ocr_frame_results, roi=request.roi)
        final_output_path = request.output_path

        if not request.skip_export and subtitle_exporter is not None:
            progress_reporter.report(99, 100, "Đang lưu tệp định dạng Subtitle…")
            final_output_path = subtitle_exporter.export(built_subtitle_events, request.output_path)

        elapsed_time_seconds = time.perf_counter() - start_time_seconds
        return ExtractSubtitlesResponse(
            events=built_subtitle_events, output_path=final_output_path,
            elapsed_seconds=elapsed_time_seconds, frames_processed=len(ocr_frame_results), raw_ocr_path=raw_ocr_file_path,
        )

    def _resolve_exporter(self, format_key: str) -> SubtitleExporterPort:
        try:
            return self._subtitle_exporters[format_key]
        except KeyError as missing_key_error:
            raise KeyError(f"Định dạng {format_key!r} không được hỗ trợ.") from missing_key_error

    @staticmethod
    def _resolve_debug_directory(request: ExtractSubtitlesRequest) -> Path:
        base_directory = Path(request.debug_frames_dir) if request.debug_frames_dir else request.output_path.parent / "se_debug_frames"
        return base_directory / request.video_path.stem

    def _execute_ocr_inference_loop(
        self,
        metadata: VideoMetadata,
        request: ExtractSubtitlesRequest,
        reporter: ProgressReporterPort,
        debug_dir: Path | None,
    ) -> list[OcrFrameResult]:
        accumulated_results: list[OcrFrameResult] = []
        frame_batch_queue: list[SampledFrame] = []

        current_batch_size = max(1, request.ocr.batch_size)
        is_auto_tune_enabled = request.auto_tune_batch

        processed_batch_count = 0
        progress_report_interval = max(1, current_batch_size * _PROGRESS_BATCH_REPORT_MULTIPLIER)
        last_reported_frame_count = 0
        last_valid_ocr_result: OcrFrameResult | None = None

        expected_total_frames = max(1, int(metadata.duration_sec / request.sampling.sample_step_sec))

        for sampled_frame in self._iterate_sampled_frames(metadata, request, reporter):
            if reporter.is_cancelled():
                break

            if sampled_frame.is_error:
                if frame_batch_queue:
                    new_ocr_results, current_batch_size, progress_report_interval = self._flush_batch_and_tune_performance(
                        batch_queue=frame_batch_queue, debug_dir=debug_dir, current_batch_size=current_batch_size,
                        current_interval=progress_report_interval, batch_count=processed_batch_count,
                        auto_tune=is_auto_tune_enabled, request=request, metadata=metadata
                    )
                    processed_batch_count += 1
                    accumulated_results.extend(new_ocr_results)
                    frame_batch_queue.clear()

                accumulated_results.append(OcrFrameResult(frame_index=sampled_frame.frame_index, timestamp_sec=sampled_frame.timestamp_sec, text_boxes=[]))
                last_valid_ocr_result = None
                continue

            if sampled_frame.is_duplicate:
                if frame_batch_queue:
                    new_ocr_results, current_batch_size, progress_report_interval = self._flush_batch_and_tune_performance(
                        batch_queue=frame_batch_queue, debug_dir=debug_dir, current_batch_size=current_batch_size,
                        current_interval=progress_report_interval, batch_count=processed_batch_count,
                        auto_tune=is_auto_tune_enabled, request=request, metadata=metadata
                    )
                    processed_batch_count += 1
                    accumulated_results.extend(new_ocr_results)
                    if new_ocr_results:
                        last_valid_ocr_result = new_ocr_results[-1]
                    frame_batch_queue.clear()

                if last_valid_ocr_result is not None:
                    accumulated_results.append(OcrFrameResult(frame_index=sampled_frame.frame_index, timestamp_sec=sampled_frame.timestamp_sec, text_boxes=list(last_valid_ocr_result.text_boxes)))
                continue

            frame_batch_queue.append(sampled_frame)

            if len(frame_batch_queue) >= current_batch_size:
                new_ocr_results, current_batch_size, progress_report_interval = self._flush_batch_and_tune_performance(
                    batch_queue=frame_batch_queue, debug_dir=debug_dir, current_batch_size=current_batch_size,
                    current_interval=progress_report_interval, batch_count=processed_batch_count,
                    auto_tune=is_auto_tune_enabled, request=request, metadata=metadata
                )
                processed_batch_count += 1
                accumulated_results.extend(new_ocr_results)

                if new_ocr_results:
                    last_valid_ocr_result = new_ocr_results[-1]
                frame_batch_queue.clear()

                if len(accumulated_results) - last_reported_frame_count >= progress_report_interval:
                    reporter.report(len(accumulated_results), expected_total_frames, f"Đã quét OCR {len(accumulated_results)}/{expected_total_frames} frame…")
                    last_reported_frame_count = len(accumulated_results)

        if frame_batch_queue and not reporter.is_cancelled():
            new_ocr_results = self._execute_ocr_inference_on_batch(frame_batch_queue, debug_dir, request, metadata)
            accumulated_results.extend(new_ocr_results)
            reporter.report(len(accumulated_results), expected_total_frames, f"Đã quét xong {len(accumulated_results)} khung hình.")

        return accumulated_results

    def _flush_batch_and_tune_performance(
        self, *, batch_queue: list[SampledFrame], debug_dir: Path | None, current_batch_size: int,
        current_interval: int, batch_count: int, auto_tune: bool, request: ExtractSubtitlesRequest, metadata: VideoMetadata
    ) -> tuple[list[OcrFrameResult], int, int]:

        inference_start_time = time.perf_counter()
        inference_results = self._execute_ocr_inference_on_batch(batch_queue, debug_dir, request, metadata)
        inference_duration = time.perf_counter() - inference_start_time

        tuned_batch_size = current_batch_size
        tuned_interval = current_interval

        if auto_tune and batch_count == 1 and inference_duration > 0:
            calculated_optimal_size = int(current_batch_size * _AUTO_TUNE_TARGET_SECONDS / inference_duration)
            tuned_batch_size = max(_AUTO_TUNE_MIN_BATCH_SIZE, min(_AUTO_TUNE_MAX_BATCH_SIZE, calculated_optimal_size))
            tuned_interval = max(1, tuned_batch_size * _PROGRESS_BATCH_REPORT_MULTIPLIER)

        return inference_results, tuned_batch_size, tuned_interval

    def _iterate_sampled_frames(
        self, metadata: VideoMetadata, request: ExtractSubtitlesRequest, reporter: ProgressReporterPort,
    ) -> Iterator[SampledFrame]:

        is_reocr_task = request.skip_export and request.raw_ocr_output_path is None
        cache_key = ""

        if is_reocr_task:
            roi_str = f"{request.roi.x}_{request.roi.y}_{request.roi.width}_{request.roi.height}" if request.roi else "none"
            # [v3.23.106] Khoá cache phải bao trùm MỌI tham số ảnh hưởng pixel của frame
            # (bước lấy mẫu, ngưỡng khử trùng, toàn bộ tiền xử lý VRAM). Trước đây chỉ gồm
            # intro/outro/roi -> đổi tiền xử lý (upscale/sharpen/contrast…) vẫn trúng cache cũ
            # và phục vụ frame lỗi thời. (Model OCR không đổi pixel nên không cần đưa vào.)
            cache_key = f"{metadata.path.name}|{_sampling_signature(request.sampling)}|{roi_str}"
            cached_frames = _GLOBAL_FRAME_CACHE.get(cache_key)
            if cached_frames is not None:
                yield from cached_frames
                return

        collected_frames_for_cache: list[SampledFrame] = []
        cache_overflow = False
        for sampled_frame in self._frame_sampler_port.iter_frames(
            metadata=metadata, roi=metadata.roi, config=request.sampling,
        ):
            if reporter.is_cancelled():
                return

            # [v3.23.106] CHỈ cache khi gom được TRỌN BỘ frame (<= cap). Nếu video vượt cap ->
            # bỏ cache hẳn cho key này (không phục vụ tập frame BỊ CẮT -> tránh OCR lại thiếu
            # frame, sót phụ đề như báo cáo); lần sau sẽ giải mã đầy đủ lại.
            if is_reocr_task and not cache_overflow:
                if len(collected_frames_for_cache) < _GLOBAL_FRAME_CACHE.max_cached_frames:
                    collected_frames_for_cache.append(sampled_frame)
                else:
                    cache_overflow = True
                    collected_frames_for_cache = []

            yield sampled_frame

        if is_reocr_task and collected_frames_for_cache and not cache_overflow:
            _GLOBAL_FRAME_CACHE.put(cache_key, collected_frames_for_cache)

    def _execute_ocr_inference_on_batch(
        self, batch_queue: list[SampledFrame], debug_dir: Path | None, request: ExtractSubtitlesRequest, metadata: VideoMetadata
    ) -> list[OcrFrameResult]:
        if debug_dir is not None:
            self._save_debug_frames_to_disk(batch_queue, debug_dir)

        try:
            raw_inference_results = self._ocr_engine_port.infer_batch(
                images_rgb=[frame.image_rgb for frame in batch_queue],
                frame_indices=[frame.frame_index for frame in batch_queue],
                timestamps_sec=[frame.timestamp_sec for frame in batch_queue],
            )

            if request.export_raw_images and request.raw_ocr_output_path:
                self._save_annotated_raw_frames(batch_queue, raw_inference_results, request.raw_ocr_output_path)

            return self._reverse_vram_geometry(raw_inference_results, batch_queue, request, metadata)

        except (OcrInferenceError, OcrModelLoadError) as model_error:
            logger.exception("Lỗi suy luận OCR trên lô %d frame: %s.", len(batch_queue), model_error)
            return [OcrFrameResult(frame_index=f.frame_index, timestamp_sec=f.timestamp_sec, text_boxes=[]) for f in batch_queue]
        except (RuntimeError, MemoryError) as system_error:
            logger.exception("Lỗi OOM/System khi suy luận: %s.", system_error)
            raise SubtitlesExtractorError(f"Hệ thống cạn kiệt bộ nhớ hoặc lỗi Trình xử lý: {system_error}") from system_error

    def _reverse_vram_geometry(
        self, raw_results: list[OcrFrameResult], batch_queue: list[SampledFrame], 
        request: ExtractSubtitlesRequest, metadata: VideoMetadata
    ) -> list[OcrFrameResult]:
        """Tịnh tiến ngược Hệ quy chiếu Tọa độ VRAM về lại Khung hình Gốc."""
        if not request.sampling.vram_upscale_small_text and not request.sampling.vram_add_border:
            return raw_results
            
        orig_h = request.roi.height if request.roi else metadata.height
        border_px = request.sampling.vram_border_thickness_px if request.sampling.vram_add_border else 0
        
        reversed_results = []
        for result, frame in zip(raw_results, batch_queue, strict=True):
            prep_h = frame.image_rgb.shape[0]
            scaled_h = prep_h - 2 * border_px
            scale = scaled_h / float(orig_h) if orig_h > 0 else 1.0
            
            if abs(scale - 1.0) < 1e-4 and border_px == 0:
                reversed_results.append(result)
                continue
                
            safe_scale = max(1e-5, scale)
            new_boxes = []
            for box in result.text_boxes:
                if not box.polygon:
                    new_boxes.append(box)
                    continue
                reversed_poly = [
                    (int(round((pt[0] - border_px) / safe_scale)), int(round((pt[1] - border_px) / safe_scale)))
                    for pt in box.polygon
                ]
                new_boxes.append(dataclasses.replace(box, polygon=reversed_poly))
            reversed_results.append(dataclasses.replace(result, text_boxes=new_boxes))
            
        return reversed_results

    def _save_annotated_raw_frames(
        self,
        batch_queue: list[SampledFrame],
        ocr_results: list[OcrFrameResult],
        raw_json_path: Path,
    ) -> None:
        try:
            from PySide6.QtCore import QPoint
            from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
        except ImportError:
            return

        base_name = raw_json_path.name
        for extension in (".gz", ".json", ".seraw"):
            base_name = base_name.replace(extension, "")
        image_directory = raw_json_path.parent / f"{base_name}_ocr_images"

        input_directory = image_directory / "input"
        output_directory = image_directory / "output"
        input_directory.mkdir(parents=True, exist_ok=True)
        output_directory.mkdir(parents=True, exist_ok=True)

        box_color = QColor(0, 255, 0)
        text_shadow_color = QColor(0, 0, 0)
        text_fill_color = QColor(255, 255, 0)
        conf_fill_color = QColor(0, 255, 255)

        for sampled_frame, frame_result in zip(batch_queue, ocr_results, strict=True):
            if sampled_frame.image_rgb.size == 0:
                continue

            file_name = (
                f"frame_{sampled_frame.frame_index:06d}_"
                f"{int(sampled_frame.timestamp_sec * 1000)}ms.jpg"
            )

            image_h, image_w = sampled_frame.image_rgb.shape[:2]
            channels = (
                sampled_frame.image_rgb.shape[2]
                if sampled_frame.image_rgb.ndim == 3
                else 1
            )
            if channels != 3:
                logger.debug(
                    "Bỏ qua frame #%d: cần ảnh RGB 3 kênh, nhận được %d kênh.",
                    sampled_frame.frame_index, channels,
                )
                continue

            contiguous_rgb = np.ascontiguousarray(sampled_frame.image_rgb, dtype=np.uint8)
            qt_input_image = QImage(
                contiguous_rgb.data, image_w, image_h, image_w * channels,
                QImage.Format.Format_RGB888,
            ).copy()

            qt_input_image.save(str(input_directory / file_name), "JPG", quality=85)

            qt_output_image = qt_input_image.copy()
            painter = QPainter(qt_output_image)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                box_pen = QPen(box_color, max(2, image_h // 300))
                text_shadow_pen = QPen(text_shadow_color, 3)
                text_fill_pen = QPen(text_fill_color, 1)
                conf_shadow_pen = QPen(text_shadow_color, 2)
                conf_fill_pen = QPen(conf_fill_color, 1)

                large_font = QFont()
                large_font.setPointSize(max(12, image_h // 35))
                large_font.setBold(True)

                small_font = QFont()
                small_font.setPointSize(max(9, image_h // 60))
                small_font.setBold(True)

                for text_box in frame_result.text_boxes:
                    if not text_box.polygon:
                        continue

                    points = [QPoint(int(x), int(y)) for x, y in text_box.polygon]

                    painter.setPen(box_pen)
                    point_count = len(points)
                    for i in range(point_count):
                        painter.drawLine(points[i], points[(i + 1) % point_count])

                    anchor_x = points[0].x()
                    anchor_y = points[0].y()
                    text_y = max(0, anchor_y - 5)

                    painter.setFont(large_font)
                    painter.setPen(text_shadow_pen)
                    painter.drawText(anchor_x + 1, text_y + 1, text_box.text)
                    painter.setPen(text_fill_pen)
                    painter.drawText(anchor_x, text_y, text_box.text)

                    confidence_label = f"{float(text_box.confidence):.2f}"
                    painter.setFont(small_font)
                    painter.setPen(conf_shadow_pen)
                    painter.drawText(anchor_x, anchor_y + 15, confidence_label)
                    painter.setPen(conf_fill_pen)
                    painter.drawText(anchor_x, anchor_y + 15, confidence_label)
            finally:
                painter.end()

            qt_output_image.save(str(output_directory / file_name), "JPG", quality=85)

    @staticmethod
    def _save_raw_ocr_payload(
        ocr_results: list[OcrFrameResult], output_file_path: Path, request: ExtractSubtitlesRequest, metadata: VideoMetadata,
    ) -> Path | None:
        try:
            from datetime import datetime

            from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
                RawOcrMeta,
                save_raw_ocr,
            )
            roi_coordinates_list = [request.roi.x, request.roi.y, request.roi.width, request.roi.height] if request.roi else None
            
            # [LƯU TRỮ CẤU HÌNH TIỀN XỬ LÝ VÀ OCR VÀO JSON]
            preprocess_dict = {
                "upscale_small_text": request.ocr.preprocess.upscale_small_text or request.sampling.vram_upscale_small_text,
                "upscale_target_height_px": request.ocr.preprocess.upscale_target_height_px or request.sampling.vram_upscale_target_height_px,
                "add_white_border": request.ocr.preprocess.add_white_border or request.sampling.vram_add_border,
                "border_thickness_px": request.ocr.preprocess.border_thickness_px or request.sampling.vram_border_thickness_px,
                "apply_sharpen": request.ocr.preprocess.apply_sharpen or request.sampling.vram_sharpen,
                "apply_contrast_boost": request.ocr.preprocess.apply_contrast_boost or (request.sampling.vram_contrast_factor != 1.0),
                "contrast_factor": request.ocr.preprocess.contrast_factor or request.sampling.vram_contrast_factor,
                "apply_clahe": request.ocr.preprocess.apply_clahe,
                "clahe_clip_limit": request.ocr.preprocess.clahe_clip_limit,
                "clahe_tile_size": request.ocr.preprocess.clahe_tile_size,
                "apply_median_blend": request.sampling.apply_median_blend,
            }

            advanced_ocr_dict = {
                "limit_side_len": request.ocr.limit_side_len,
                "det_thresh": request.ocr.det_thresh,
                "box_thresh": request.ocr.det_box_thresh,
                "det_unclip_ratio": request.ocr.det_unclip_ratio,
            }

            metadata_payload = RawOcrMeta(
                video_name=request.video_path.name, video_duration_sec=metadata.duration_sec,
                frame_count=len(ocr_results), sample_step_sec=request.sampling.sample_step_sec,
                detection_model=request.ocr.detection_model_name or "", recognition_model=request.ocr.recognition_model_name or "",
                score_threshold=request.ocr.score_threshold or 0.0, saved_at=datetime.now(tz=UTC).isoformat(),
                roi_xywh=roi_coordinates_list,
                preprocess=preprocess_dict,
                advanced_ocr=advanced_ocr_dict,
            )
            save_raw_ocr(frames=ocr_results, output_path=output_file_path, meta=metadata_payload)
            return output_file_path
        except (OSError, ValueError, ImportError) as io_error:
            logger.warning("Không lưu được dữ liệu OCR thô: %s.", io_error)
            return None

    @staticmethod
    def _save_debug_frames_to_disk(
        batch_queue: list[SampledFrame], debug_directory: Path,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            logger.warning("Không import được OpenCV để lưu debug frame: %s.", exc)
            return

        for sampled_frame in batch_queue:
            if sampled_frame.image_rgb.size == 0:
                continue
            try:
                bgr_image_array = cv2.cvtColor(
                    np.ascontiguousarray(sampled_frame.image_rgb, dtype=np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
                file_name = (
                    f"frame_{sampled_frame.frame_index:06d}_"
                    f"{sampled_frame.timestamp_sec * 1000:.0f}ms.png"
                )
                cv2.imwrite(str(debug_directory / file_name), bgr_image_array)
            except (OSError, cv2.error, ValueError) as exc:
                logger.debug("Bỏ qua lưu debug frame #%d: %s.", sampled_frame.frame_index, exc)

__all__ = ["ExtractSubtitlesUseCase"]
