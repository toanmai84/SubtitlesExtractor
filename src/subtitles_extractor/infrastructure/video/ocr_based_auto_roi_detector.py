"""Adapter tự động phát hiện Multi-ROI theo pipeline DBSCAN + Variance + Review.

CẢI TIẾN v3.2 — Auto-ROI Quality & Performance:
    1. [LOGGING] Chuyển từ ``logging`` sang ``loguru.logger`` cho consistency
       với toàn project (cả 6 core module đều dùng loguru).
    2. [BUG FIX] ``.copy()`` ảnh RGB trước khi lưu top-K để tránh race
       condition với decoder reuse buffer (mpv/pyav).
    3. [PERFORMANCE] Top-K frame selection bằng heap (``heapq``) thay vì
       sort+pop mỗi frame — O(N × log K) thay O(N × K log K).
    4. [PERFORMANCE] Throttle progress callback — chỉ gọi khi % thay đổi
       hoặc mỗi 10 frame, tránh flood UI thread với Qt signals.
    5. [QUALITY] Composite image dùng ``np.median`` thay vì ``np.mean`` —
       robust với outlier (frame chuyển cảnh sáng).
    6. [QUALITY] Stratified sampling: chia video thành K bucket thời gian,
       chọn 1 frame tốt nhất mỗi bucket — đa dạng spatial-temporal.
    7. [QUALITY] Skip intro/outro cap ở 60s — video dài không bị skip quá
       nhiều.

Cải tiến lịch sử (V3.10 - Memory Guardian):
    1. [CRITICAL FIX] Ngăn chặn Crash OOM: Loại bỏ mảng lưu trữ khung hình
       khổng lồ, chỉ giữ lại Top 8 khung hình có nhiều chữ nhất trong RAM.
    2. [PERFORMANCE] Tránh lãng phí CPU: Chỉ Convert BGR đối với những
       frame được chọn.
    3. [LOGIC FIX] Đồng bộ Frame Index: Truyền Index vật lý của video vào
       cụm DBSCAN để thuật toán đánh giá độ ổn định thời gian chính xác.
"""
from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import (
    OcrInferenceError,
    OcrModelLoadError,
    VideoDecodeError,
)
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplerPort,
    FrameSamplingConfig,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEnginePort
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
    ROICluster,
)

_COMPOSITE_FRAME_COUNT: int = 8
_MAX_ITERATIONS_FOR_AUTO_ROI: int = 3000
# Cap skip intro/outro tránh skip quá nhiều với video dài.
_MAX_SKIP_INTRO_OUTRO_SEC: float = 60.0
# Throttle progress callback: gọi mỗi N frame thay vì every frame.
_PROGRESS_THROTTLE_INTERVAL: int = 10
# Stratified sampling: số bucket thời gian để chọn top-K đa dạng.
_STRATIFIED_BUCKET_COUNT: int = 8

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class AnalysisResult:
    """Kết quả phân tích ROI sau khi gom cụm DBSCAN."""
    clusters: list[ROICluster]
    composite_bgr: np.ndarray
    raw_bboxes: list[RawBBox] = field(default_factory=list)


class OcrBasedAutoRoiDetector:
    """Tự động phát hiện các vùng chứa phụ đề dựa trên lấy mẫu OCR thưa thớt."""

    def __init__(
        self,
        ocr_engine: OcrEnginePort,
        frame_sampler: FrameSamplerPort,
        show_review_ui: bool = True,
        parent_widget: object | None = None,
        analyzer_kwargs_provider: "Callable[[], dict] | None" = None,
    ) -> None:
        self._ocr_engine = ocr_engine
        self._frame_sampler = frame_sampler
        self._show_review_ui = show_review_ui
        self._parent_widget = parent_widget
        # [v3.19] Provider trả tham số tinh chỉnh BBoxAnalyzer (đọc từ Settings tại
        # thời điểm gọi, nên người dùng đổi cài đặt là có hiệu lực ngay, không cần
        # khởi động lại). None → dùng mặc định của engine.
        self._analyzer_kwargs_provider = analyzer_kwargs_provider

    def analyze_only(
        self,
        metadata: VideoMetadata,
        step_ms: int = 1000,
        batch_size: int = 8,
        progress_callback: ProgressCallback | None = None,
        vram_flags: FrameSamplingConfig | None = None,
    ) -> AnalysisResult | None:
        """Chỉ thực hiện lấy mẫu và phân tích, không hiển thị UI kiểm duyệt.

        Args:
            metadata: Metadata của video cần phân tích.
            step_ms: Bước lấy mẫu frame (milliseconds).
            batch_size: Số frame mỗi batch OCR.
            progress_callback: Callback báo tiến độ ``(cur, total, msg)``.
            vram_flags: Cấu hình VRAM preprocessing kế thừa từ user settings.
                Nếu ``None``, tắt toàn bộ VRAM preprocessing cho Auto-ROI.
                Truyền vào để kế thừa ``vram_sharpen``, ``vram_upscale_small_text``,
                ``vram_upscale_target_height_px``, ``vram_add_border``,
                ``vram_border_thickness_px``, ``vram_contrast_factor`` từ pipeline
                chính — đảm bảo Auto-ROI dùng đúng cùng GPU pipeline với extract.
        """
        top_rgb_images, all_raw_bboxes = self._sample_and_ocr(
            metadata, step_ms, batch_size, progress_callback, vram_flags=vram_flags,
        )

        if not all_raw_bboxes:
            logger.warning(
                "Không tìm thấy text trong các frame mẫu của {}.",
                metadata.filename,
            )
            return None

        if progress_callback is not None:
            progress_callback(0, 0, "Đang phân tích gom cụm (DBSCAN)...")

        analyzer_tuning_kwargs: dict = {}
        if self._analyzer_kwargs_provider is not None:
            try:
                analyzer_tuning_kwargs = dict(self._analyzer_kwargs_provider())
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Bỏ qua tham số tinh chỉnh Auto-ROI không hợp lệ: {}", exc)
        analyzer = BBoxAnalyzer(
            frame_width=metadata.width, frame_height=metadata.height,
            **analyzer_tuning_kwargs,
        )
        clusters = analyzer.analyze(all_raw_bboxes)

        if not clusters:
            logger.warning("BBoxAnalyzer không tìm được cụm ROI nào hợp lệ.")
            return None

        composite_bgr = self._build_composite_image(
            top_rgb_images, metadata.width, metadata.height
        )

        return AnalysisResult(clusters=clusters, composite_bgr=composite_bgr, raw_bboxes=all_raw_bboxes)

    @staticmethod
    def review_clusters(
        result: AnalysisResult,
        parent_widget: object | None = None,
        translator: object | None = None,
    ) -> list[Roi] | None:
        """Hiển thị giao diện cho người dùng kiểm duyệt (Human-in-the-Loop)."""
        import copy

        from PySide6.QtWidgets import QDialog

        from subtitles_extractor.presentation.widgets.multi_roi_review_dialog import (
            MultiROIReviewDialog,
        )

        cloned_clusters = [copy.deepcopy(cluster) for cluster in result.clusters]
        dialog = MultiROIReviewDialog(
            composite_bgr=result.composite_bgr,
            clusters=cloned_clusters,
            parent=parent_widget,
            translator=translator,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            kept_clusters = dialog.get_kept_clusters()
            return _clusters_to_rois(kept_clusters)
        return None

    def detect(
        self,
        metadata: VideoMetadata,
        step_ms: int = 1000,
        batch_size: int = 8,
        vram_flags: FrameSamplingConfig | None = None,
    ) -> list[Roi] | None:
        """Thực hiện chu trình tự động phát hiện đầy đủ (Phân tích → Kiểm duyệt)."""
        result = self.analyze_only(metadata, step_ms, batch_size, vram_flags=vram_flags)
        if result is None:
            return None

        if not self._show_review_ui:
            return _clusters_to_rois(result.clusters)

        return self.review_clusters(result, parent_widget=self._parent_widget)

    def detect_primary_subtitle_roi(
        self,
        metadata: VideoMetadata,
        step_ms: int = 1000,
        batch_size: int = 8,
        progress_callback: ProgressCallback | None = None,
        vram_flags: FrameSamplingConfig | None = None,
    ) -> Roi | None:
        """[Preset Tự nhận diện] Tự phát hiện rồi chọn DUY NHẤT vùng phụ đề chính.

        Khác :meth:`detect`: không hiển thị UI kiểm duyệt, không trả nhiều ROI — chạy
        phân tích toàn khung rồi chọn cụm "đậm đặc nhất" (frame-presence áp đảo) làm
        ROI duy nhất để trích xuất hardsub.

        Returns:
            ROI phụ đề chính, hoặc ``None`` nếu không phát hiện được cụm nào.
        """
        from subtitles_extractor.infrastructure.video.roi_selection import (
            select_subtitle_roi_smart,
        )

        result = self.analyze_only(
            metadata, step_ms, batch_size, progress_callback, vram_flags=vram_flags
        )
        if result is None:
            return None
        roi = select_subtitle_roi_smart(
            result.clusters, result.raw_bboxes, metadata.width, metadata.height
        )
        if roi is not None:
            logger.info(
                "Tự nhận diện ROI phụ đề chính: x={}, y={}, {}x{}.",
                roi.x, roi.y, roi.width, roi.height,
            )
        return roi

    def _sample_and_ocr(
        self,
        metadata: VideoMetadata,
        step_ms: int,
        batch_size: int,
        progress_callback: ProgressCallback | None = None,
        vram_flags: FrameSamplingConfig | None = None,
    ) -> tuple[list[np.ndarray], list[RawBBox]]:
        """Lấy mẫu các khung hình và chạy OCR để thu thập Bounding Box.

        Thuật toán:
            1. Skip intro/outro 5% của video, cap ở 60s mỗi đầu.
            2. Sample frame theo ``step_ms``, dedupe bằng pHash.
            3. Mỗi batch ``batch_size`` frame → flush qua OCR engine.
            4. Lưu top-K frame có nhiều box nhất (heap O(log K)).
            5. Stratified sampling: chia video thành buckets thời gian
               để top-K không bị bias vào 1 vùng text-heavy.
            6. Throttle progress callback mỗi ``_PROGRESS_THROTTLE_INTERVAL``
               frame để tránh flood Qt signals.

        Args:
            vram_flags: Nếu không ``None``, các cờ VRAM preprocessing
                (``vram_sharpen``, ``vram_upscale_small_text``, …) được
                merge vào ``temp_sampling_config`` để NVDEC pipeline dùng
                đúng preprocessing pipeline đã cài đặt. Nếu ``None``
                (mặc định), tất cả VRAM flags tắt — Auto-ROI chỉ dùng
                full-frame decode không tiền xử lý.
        """
        if not self._ocr_engine.is_initialized:
            self._ocr_engine.initialize()

        step_sec = step_ms / 1000.0
        # Cap skip intro/outro để không skip quá nhiều cho video dài.
        # Ví dụ: video 2h → 5% = 6 phút (quá nhiều) → cap về 60s.
        raw_skip_sec = metadata.duration_sec * 0.05
        effective_skip_sec = min(raw_skip_sec, _MAX_SKIP_INTRO_OUTRO_SEC)

        # [CRITICAL BUG FIX v3.6.1] — Kế thừa VRAM flags từ pipeline chính.
        # Trước đây FrameSamplingConfig hardcode vram_sharpen=False và
        # vram_upscale_small_text=False → log luôn "VRAM_Preprocess=False"
        # dù user đã bật GPU preprocessing trong settings.
        # Nay merge từ vram_flags (nếu có) để NVDEC dùng đúng pipeline.
        temp_sampling_config = FrameSamplingConfig(
            sample_step_sec=step_sec,
            phash_distance_threshold=64,
            pixel_diff_threshold=-1.0,
            skip_intro_sec=effective_skip_sec,
            skip_outro_sec=effective_skip_sec,
            # Merge VRAM flags từ pipeline chính (nếu có).
            vram_upscale_small_text=vram_flags.vram_upscale_small_text if vram_flags else False,
            vram_upscale_target_height_px=vram_flags.vram_upscale_target_height_px if vram_flags else 96,
            vram_add_border=vram_flags.vram_add_border if vram_flags else False,
            vram_border_thickness_px=vram_flags.vram_border_thickness_px if vram_flags else 8,
            vram_sharpen=vram_flags.vram_sharpen if vram_flags else False,
            vram_contrast_factor=vram_flags.vram_contrast_factor if vram_flags else 1.0,
        )

        all_raw_bboxes: list[RawBBox] = []

        # ── Stratified top-K: chia video thành buckets theo timestamp ──
        # Mỗi bucket giữ 1 heap min-K' = K // bucket_count slot.
        # Cuối cùng merge tất cả bucket → top K toàn cục.
        # Đảm bảo composite frame đa dạng spatial-temporal, không bị bias
        # vào 1 cụm text-heavy ngắn (vd intro credits).
        bucket_duration_sec = max(1.0, metadata.duration_sec / _STRATIFIED_BUCKET_COUNT)
        slots_per_bucket = max(
            1, _COMPOSITE_FRAME_COUNT // _STRATIFIED_BUCKET_COUNT
        )
        # Mỗi bucket là min-heap: (box_count, monotonic_id, rgb_image_copy).
        # monotonic_id cần để break tie tránh so sánh numpy array khi heap reorder.
        bucket_heaps: list[list[tuple[int, int, np.ndarray]]] = [
            [] for _ in range(_STRATIFIED_BUCKET_COUNT)
        ]
        monotonic_counter = itertools.count()

        batch_rgb_images: list[np.ndarray] = []
        batch_frame_indices: list[int] = []
        batch_timestamps_sec: list[float] = []
        total_estimated_frames = max(
            1, int((metadata.duration_sec * 0.9) / step_sec)
        )

        # Throttle progress callback — track last reported value để chỉ
        # emit signal khi % thay đổi đáng kể (giảm load Qt event loop).
        last_progress_reported: int = -1

        def _maybe_report_progress(current_iteration: int) -> None:
            """Gọi progress_callback chỉ khi đến mốc throttle."""
            if progress_callback is None:
                return
            nonlocal last_progress_reported
            # Emit mỗi N frame HOẶC khi % thay đổi >= 1%.
            should_emit = (
                current_iteration % _PROGRESS_THROTTLE_INTERVAL == 0
                or current_iteration - last_progress_reported
                >= max(1, total_estimated_frames // 100)
            )
            if not should_emit:
                return
            last_progress_reported = current_iteration
            progress_callback(
                current_iteration,
                total_estimated_frames,
                f"Đang trích xuất frame mẫu "
                f"({current_iteration}/{total_estimated_frames})...",
            )

        def _flush_ocr_batch() -> None:
            if not batch_rgb_images:
                return
            try:
                ocr_results = self._ocr_engine.infer_batch(
                    images_rgb=batch_rgb_images,
                    frame_indices=batch_frame_indices,
                    timestamps_sec=batch_timestamps_sec,
                )
            except (OcrInferenceError, OcrModelLoadError, RuntimeError) as exc:
                logger.debug(
                    "Bỏ qua batch {} frame (bắt đầu từ frame_index={}) do lỗi OCR: {}.",
                    len(batch_rgb_images),
                    batch_frame_indices[0],
                    exc,
                )
                batch_rgb_images.clear()
                batch_frame_indices.clear()
                batch_timestamps_sec.clear()
                return

            for ocr_frame_result, rgb_image in zip(
                ocr_results, batch_rgb_images, strict=True
            ):
                valid_boxes = [
                    b for b in ocr_frame_result.text_boxes
                    if b.bounding_box
                    and b.bounding_box[2] > b.bounding_box[0]
                    and b.bounding_box[3] > b.bounding_box[1]
                ]
                box_count = len(valid_boxes)
                if box_count == 0:
                    continue

                for text_box in valid_boxes:
                    x_min, y_min, x_max, y_max = text_box.bounding_box  # type: ignore[misc]
                    all_raw_bboxes.append(
                        RawBBox(
                            coord_x_min=float(x_min),
                            coord_y_min=float(y_min),
                            coord_x_max=float(x_max),
                            coord_y_max=float(y_max),
                            confidence=float(text_box.confidence),
                            frame_idx=ocr_frame_result.frame_index,
                            timestamp_sec=ocr_frame_result.timestamp_sec,
                        )
                    )

                # Stratified top-K (PERF): heappush O(log K') thay sort+pop.
                # Race condition fix: copy() để tránh decoder reuse buffer
                # ghi đè ảnh đã lưu (xảy ra với mpv/pyav khi giữ shared ref).
                bucket_idx = min(
                    _STRATIFIED_BUCKET_COUNT - 1,
                    int(ocr_frame_result.timestamp_sec / bucket_duration_sec),
                )
                heap_for_bucket = bucket_heaps[bucket_idx]
                heap_entry = (box_count, next(monotonic_counter), rgb_image.copy())
                if len(heap_for_bucket) < slots_per_bucket:
                    heapq.heappush(heap_for_bucket, heap_entry)
                else:
                    # heappushpop: thay slot có box_count nhỏ nhất nếu cần.
                    heapq.heappushpop(heap_for_bucket, heap_entry)

            batch_rgb_images.clear()
            batch_frame_indices.clear()
            batch_timestamps_sec.clear()

        try:
            for iteration_idx, sampled_frame in enumerate(
                self._frame_sampler.iter_frames(
                    metadata, roi=None, config=temp_sampling_config
                )
            ):
                if sampled_frame.is_duplicate or sampled_frame.is_error:
                    continue
                if iteration_idx > _MAX_ITERATIONS_FOR_AUTO_ROI:
                    logger.info(
                        "Đạt giới hạn {} frame mẫu — dừng để tránh chạy quá lâu.",
                        _MAX_ITERATIONS_FOR_AUTO_ROI,
                    )
                    break

                batch_rgb_images.append(sampled_frame.image_rgb)
                batch_frame_indices.append(sampled_frame.frame_index)
                batch_timestamps_sec.append(sampled_frame.timestamp_sec)

                _maybe_report_progress(iteration_idx)

                if len(batch_rgb_images) >= batch_size:
                    _flush_ocr_batch()

            _flush_ocr_batch()

            if progress_callback is not None:
                progress_callback(
                    total_estimated_frames,
                    total_estimated_frames,
                    "Đã trích xuất xong.",
                )

        except (VideoDecodeError, OSError, ValueError) as exc:
            raise VideoDecodeError(
                f"Quét Auto-ROI thất bại do lỗi đọc video: {exc}."
            ) from exc

        # Merge tất cả bucket → flat top-K, sort theo box_count desc.
        all_top_entries = [entry for heap in bucket_heaps for entry in heap]
        all_top_entries.sort(key=lambda entry: entry[0], reverse=True)
        # Giới hạn lại tối đa K frame cuối.
        top_rgb_images = [
            rgb_img for _, _, rgb_img in all_top_entries[:_COMPOSITE_FRAME_COUNT]
        ]
        return top_rgb_images, all_raw_bboxes

    def _build_composite_image(
        self, top_rgb_images: list[np.ndarray], width: int, height: int
    ) -> np.ndarray:
        """Tổng hợp ảnh BGR từ top-K frame có nhiều text nhất.

        v3.2 cải tiến:
            * Dùng :func:`numpy.median` thay :func:`numpy.mean` — robust hơn
              với outlier (vd 1 frame chuyển cảnh sáng lóa kéo mean lệch).
            * uint8 stack thay vì float32 — giảm 4× memory peak.
        """
        if not top_rgb_images:
            return np.zeros((height, width, 3), dtype=np.uint8)

        resized_bgr_frames: list[np.ndarray] = []
        for rgb_img in top_rgb_images:
            if rgb_img.shape[1] != width or rgb_img.shape[0] != height:
                rgb_img = cv2.resize(rgb_img, (width, height))
            # Convert màu duy nhất tại đây, chỉ tốn CPU cho đúng K frame.
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            resized_bgr_frames.append(bgr_img)

        # Stack thành 4D array (K, H, W, 3) uint8 → median per pixel.
        # Median robust hơn mean: 1 frame outlier không kéo composite lệch.
        # Trường hợp K=1: median = chính nó. Trường hợp K=2: median của 2
        # giá trị uint8 = (a+b)//2 (numpy default lower median cho even count).
        stacked_frames = np.stack(resized_bgr_frames, axis=0)
        return np.median(stacked_frames, axis=0).astype(np.uint8)


def _clusters_to_rois(clusters: list[ROICluster]) -> list[Roi]:
    return [
        Roi(
            x=int(cluster.coord_x_min),
            y=int(cluster.coord_y_min),
            width=max(1, int(cluster.coord_x_max - cluster.coord_x_min)),
            height=max(1, int(cluster.coord_y_max - cluster.coord_y_min)),
            alignment=cluster.alignment,
            orientation=cluster.orientation
        )
        for cluster in clusters if cluster.keep
    ]

__all__ = ["AnalysisResult", "OcrBasedAutoRoiDetector"]
