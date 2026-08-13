"""ViewModel cho trang Gỡ lỗi OCR (Raw Viewer).

BẢN CẬP NHẬT ĐỘT PHÁ (V3.26.1 - Hotfix):
    * [CRITICAL BUG FIX] Sửa lỗi AttributeError: `_SingleFrameReocrWorker` không gọi được
      `ocr_engine`. Cập nhật tham chiếu chính xác qua `ApplicationContainer`.
    * [CRITICAL BUG FIX] Smart I/O Decoder: Sử dụng đúng Backend Giải mã để
      tìm kiếm chính xác vị trí của Frame thông qua `cv2.CAP_PROP_POS_FRAMES`.
    * [CRITICAL BUG FIX] Unicode Bypass: Lệnh `cv2.VideoCapture` và `cv2.imdecode`
      được bảo vệ tuyệt đối để đọc mọi tên file, thư mục tiếng Trung, Nhật, Hàn, Việt.
    * [LỖI 7 LEAK THREAD FIX] Ngăn chặn tạo thêm Luồng Re-OCR nếu Luồng cũ vẫn đang chạy.
    * [LỖI 8 MAT ANH FIX] Thay đổi thứ tự Bắn Signal cập nhật Giao diện.
    * [SMART VRAM] Cấu hình chính xác Phân luồng Tiền xử lý CPU/GPU giúp đồng bộ Tọa độ.
    * [LỖI SAI LỆCH BBOX FIX] Tính toán Tỷ lệ Scale Động để vẽ Box chuẩn xác 100% khi Upscale.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig, PreprocessConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind, PrecisionMode
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment, TextOrientation
from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import RawOcrMeta, load_raw_ocr
from subtitles_extractor.presentation.utils.time_format import seconds_to_display
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig

from subtitles_extractor.infrastructure.ocr.preprocessing.image_filters import (
    upscale_to_min_height, apply_clahe, apply_sharpen, apply_contrast_boost
)

logger = logging.getLogger(__name__)


def _ndarray_to_qimage(arr: np.ndarray) -> QImage:
    if arr is None or arr.size == 0 or arr.ndim < 3:
        return QImage()

    h, w, c = arr.shape
    if w <= 0 or h <= 0:
        return QImage()

    bytes_per_line = c * w
    try:
        contiguous_arr = np.ascontiguousarray(arr, dtype=np.uint8)
        return QImage(contiguous_arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    except (ValueError, TypeError) as exc:
        logger.debug("Không dựng được QImage từ mảng: {}", exc)
        return QImage()


# ==============================================================================
# BẤT ĐỒNG BỘ OCR INFERENCE WORKER
# ==============================================================================
class _SingleFrameReocrWorker(QObject):
    success_result = Signal(object, object, object, str, str, bool)
    failed = Signal(str)

    def __init__(
        self, container: ApplicationContainer, old_frame, tweaks, base_img_dir, meta
    ):
        super().__init__()
        self.container = container
        self.settings = container.settings_service.current
        self.old_frame = old_frame
        self.tweaks = tweaks
        self.base_img_dir = base_img_dir
        self.meta = meta
        self._current_offset_x = 0
        self._current_offset_y = 0

    def run(self):
        try:
            full_frame_rgb = None
            is_already_cropped = False
            used_vram_preprocessing = False

            # 1. Trích xuất Full Frame từ Video Gốc bằng ĐÚNG BACKEND CÀI ĐẶT
            if self.base_img_dir and self.meta and hasattr(self.meta, 'video_name'):
                video_path = self.base_img_dir.parent / self.meta.video_name
                if video_path.exists():
                    try:
                        # Gọi thẳng Container để lấy Backend (PyNvVideoCodec / PyAV / OpenCV)
                        metadata = self.container.metadata_reader.read(video_path)
                        sampler = self.container.frame_sampler
                        target_ts = self.old_frame.timestamp_sec

                        # Lấy ROI từ tweaks hoặc meta để truyền trực tiếp vào sampler
                        target_roi = None
                        custom_roi = self.tweaks.get("custom_roi")
                        if custom_roi is not None:
                            target_roi = custom_roi
                        elif self.meta and self.meta.roi_xywh:
                            rx, ry, rw, rh = self.meta.roi_xywh
                            target_roi = Roi(x=rx, y=ry, width=rw, height=rh)

                        # Determine NVDEC Backend
                        from importlib.util import find_spec
                        has_cupy = find_spec("cupy") is not None
                        is_nvdec = self.settings.hardware.frame_decoder_backend == "pynvvideocodec" and has_cupy

                        # [SỬA LỖI TỌA ĐỘ VRAM]: Upscale và Border BẮT BUỘC chạy trên CPU
                        # để OCR Engine có thể tịnh tiến ngược tọa độ Box về kích thước gốc.
                        vram_upscale = False 
                        vram_sharpen = bool(self.tweaks.get("sharpen", False)) if is_nvdec else False
                        vram_contrast = bool(self.tweaks.get("contrast", False)) if is_nvdec else False
                        vram_contrast_factor = float(self.tweaks.get("contrast_factor", 1.2)) if vram_contrast else 1.0

                        # Cấu hình quét siêu hẹp quanh mục tiêu
                        cfg = FrameSamplingConfig(
                            sample_step_sec=0.01,
                            skip_intro_sec=max(0.0, target_ts - 1.0),
                            skip_outro_sec=max(0.0, metadata.duration_sec - target_ts - 1.0),
                            vram_upscale_small_text=vram_upscale,
                            vram_upscale_target_height_px=int(self.tweaks.get("upscale_h", 96)),
                            vram_sharpen=vram_sharpen,
                            vram_contrast_factor=vram_contrast_factor,
                        )

                        min_diff = float("inf")
                        for s_frame in sampler.iter_frames(metadata, target_roi, cfg):
                            diff = abs(s_frame.timestamp_sec - target_ts)
                            # Ưu tiên khớp chính xác bằng Timestamp (Do VFR, Frame Index bị reset)
                            if diff < min_diff:
                                min_diff = diff
                                full_frame_rgb = s_frame.image_rgb.copy()

                            if s_frame.timestamp_sec > target_ts + 0.5:
                                break
                        
                        if full_frame_rgb is not None and full_frame_rgb.size > 0:
                            is_already_cropped = True
                            if target_roi:
                                self._current_offset_x = target_roi.x
                                self._current_offset_y = target_roi.y
                            if is_nvdec:
                                used_vram_preprocessing = True

                    except Exception as e:
                        logger.debug("Lỗi đọc Frame bằng Backend: {}", e)

            # 2. Fallback: Đọc ảnh JPG Input nếu không tìm thấy/giải mã được Video gốc
            if (full_frame_rgb is None or full_frame_rgb.size == 0) and self.base_img_dir:
                fname = f"frame_{self.old_frame.frame_index:06d}_{int(self.old_frame.timestamp_sec * 1000)}ms.jpg"
                jpg_path = self.base_img_dir / "input" / fname
                if jpg_path.exists():
                    # Mở luồng Byte thay vì cv2.imread để né Bug Unicode Windows
                    with open(str(jpg_path), "rb") as f:
                        img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    full_frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if full_frame_bgr is not None:
                        full_frame_rgb = cv2.cvtColor(full_frame_bgr, cv2.COLOR_BGR2RGB)
                        is_already_cropped = True
                        self._current_offset_x = self.meta.roi_xywh[0] if self.meta and self.meta.roi_xywh else 0
                        self._current_offset_y = self.meta.roi_xywh[1] if self.meta and self.meta.roi_xywh else 0

            if full_frame_rgb is None or full_frame_rgb.size == 0:
                self.failed.emit("Không thể giải mã Video gốc hoặc không tìm thấy ảnh JPG Input lưu tạm.")
                return

            img_rgb = full_frame_rgb

            # 3. Tính toán Vùng Cắt (Crop)
            if is_already_cropped:
                cropped_rgb = img_rgb
            else:
                custom_roi = self.tweaks.get("custom_roi")
                if custom_roi is not None:
                    rx, ry, rw, rh = custom_roi.x, custom_roi.y, custom_roi.width, custom_roi.height
                elif self.meta and self.meta.roi_xywh:
                    rx, ry, rw, rh = self.meta.roi_xywh
                else:
                    rx, ry, rw, rh = 0, 0, img_rgb.shape[1], img_rgb.shape[0]

                rx, ry = max(0, rx), max(0, ry)
                rw = min(img_rgb.shape[1] - rx, rw)
                rh = min(img_rgb.shape[0] - ry, rh)

                if rw <= 0 or rh <= 0:
                    self.failed.emit("Vùng cắt ROI không hợp lệ. Vui lòng kiểm tra lại khung vẽ.")
                    return

                cropped_rgb = img_rgb[ry:ry+rh, rx:rx+rw]
                self._current_offset_x, self._current_offset_y = rx, ry

            # 4. Tiền xử lý Ảnh (Chỉ để hiển thị cho User xem)
            prep_rgb = cropped_rgb.copy()
            
            if not used_vram_preprocessing:
                if self.tweaks.get("upscale") and prep_rgb.shape[0] < self.tweaks.get("upscale_h", 96):
                    prep_rgb = upscale_to_min_height(prep_rgb, self.tweaks.get("upscale_h", 96))
                if self.tweaks.get("sharpen"):
                    prep_rgb = apply_sharpen(prep_rgb)
                if self.tweaks.get("contrast"):
                    prep_rgb = apply_contrast_boost(prep_rgb, self.tweaks.get("contrast_factor", 1.2))

            # CLAHE luôn chạy trên CPU (Vì VRAM chưa hỗ trợ CLAHE)
            if self.tweaks.get("clahe"):
                prep_rgb = apply_clahe(prep_rgb, self.tweaks.get("clahe_clip", 3.0), self.tweaks.get("clahe_tile", 8))

            qimg_input = _ndarray_to_qimage(prep_rgb)
            if qimg_input.isNull():
                self.failed.emit("Lỗi tạo ảnh Input (Kích thước rỗng).")
                return

            # 5. Cấu hình PaddleOCR Tạm thời
            from subtitles_extractor.application.services.ocr_model_resolver import resolve_ocr_model_names
            det_model, rec_model = resolve_ocr_model_names(self.settings.ocr)

            # VRAM xử lý cái gì thì tắt cái đó đi ở CPU để tránh bị xử lý 2 lần 
            temp_pre_cfg = PreprocessConfig(
                upscale_small_text=self.tweaks.get("upscale", False),
                upscale_target_height_px=self.tweaks.get("upscale_h", self.settings.preprocess.upscale_target_height_px),
                add_white_border=self.settings.preprocess.add_white_border,
                border_thickness_px=self.settings.preprocess.border_thickness_px,
                apply_clahe=self.tweaks.get("clahe", False),
                clahe_clip_limit=self.tweaks.get("clahe_clip", self.settings.preprocess.clahe_clip_limit),
                clahe_tile_size=self.tweaks.get("clahe_tile", self.settings.preprocess.clahe_tile_size),
                apply_sharpen=self.tweaks.get("sharpen", False) if not used_vram_preprocessing else False,
                apply_contrast_boost=self.tweaks.get("contrast", False) if not used_vram_preprocessing else False,
                contrast_factor=self.tweaks.get("contrast_factor", self.settings.preprocess.contrast_factor),
            )

            temp_ocr_cfg = OcrEngineConfig(
                device=DeviceKind(self.settings.hardware.device.value),
                detection_model_name=det_model,
                recognition_model_name=rec_model,
                language=self.settings.ocr.language,
                limit_side_len=self.tweaks.get("limit_side_len", self.settings.ocr.limit_side_len),
                det_thresh=self.tweaks.get("det_thresh", self.settings.ocr.det_thresh),
                det_unclip_ratio=self.tweaks.get("det_unclip_ratio", self.settings.ocr.det_unclip_ratio),
                det_box_thresh=self.tweaks.get("box_thresh", self.settings.ocr.det_box_thresh),
                score_threshold=self.settings.ocr.score_threshold,
                enable_mkldnn=self.settings.hardware.enable_mkldnn,
                use_tensorrt=self.settings.hardware.use_tensorrt,
                precision=PrecisionMode(self.settings.hardware.precision.value),
                preprocess=temp_pre_cfg,
            )

            ocr_engine = self.container.ocr_engine
            if not ocr_engine.is_initialized:
                ocr_engine.initialize()

            apply_cfg = getattr(ocr_engine, "apply_config", None) or getattr(ocr_engine, "update_preprocess_config", None)
            if callable(apply_cfg):
                apply_cfg(temp_ocr_cfg)

            # 6. Chạy Inference
            new_frame_result = ocr_engine.infer(
                cropped_rgb, self.old_frame.frame_index, self.old_frame.timestamp_sec
            )

            # 7. Vẽ Output Image bằng QPainter
            qimg_output = qimg_input.convertToFormat(QImage.Format.Format_RGB32)

            if qimg_output.height() > 0:
                painter = QPainter(qimg_output)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                # [LỖI SAI LỆCH BBOX FIX]: Dùng tỷ lệ nội suy động hoàn toàn thay vì hardcode 3.0
                scale_factor_x = qimg_output.width() / max(1, cropped_rgb.shape[1])
                scale_factor_y = qimg_output.height() / max(1, cropped_rgb.shape[0])

                box_pen = QPen(QColor(0, 255, 0), max(2, int(qimg_output.height() / 150)))
                text_fill_pen = QPen(QColor(255, 255, 0), 1)

                font = QFont()
                font.setPointSize(max(10, qimg_output.height() // 20))
                font.setBold(True)
                painter.setFont(font)

                for text_box in new_frame_result.text_boxes:
                    if not text_box.polygon: continue
                    # Nhân tỷ lệ từng trục độc lập để không bao giờ bị lệch Dù Crop/Scale thế nào
                    scaled_polygon = [(int(pt[0] * scale_factor_x), int(pt[1] * scale_factor_y)) for pt in text_box.polygon]

                    painter.setPen(box_pen)
                    num_pts = len(scaled_polygon)
                    for i in range(num_pts):
                        p1, p2 = scaled_polygon[i], scaled_polygon[(i+1)%num_pts]
                        painter.drawLine(p1[0], p1[1], p2[0], p2[1])

                    ax, ay = scaled_polygon[0]
                    painter.setPen(text_fill_pen)
                    painter.drawText(ax, max(0, ay - 5), text_box.text)

                painter.end()

            # 8. So Sánh Diff Thông Minh
            old_texts = [b.text for b in self.old_frame.text_boxes]
            new_texts = [b.text for b in new_frame_result.text_boxes]

            diff_msgs = []
            if len(old_texts) != len(new_texts):
                diff_msgs.append(self.container.translator.translate("debug.vm_diff_boxes").replace("{old}", str(len(old_texts))).replace("{new}", str(len(new_texts))))

            old_full = " ".join(old_texts)
            new_full = " ".join(new_texts)
            if old_full != new_full:
                diff_msgs.append(self.container.translator.translate("debug.vm_diff_content").replace("{old}", old_full).replace("{new}", new_full))

            is_noop = not diff_msgs
            if is_noop:
                diff_summary = self.container.translator.translate("debug.vm_diff_noop")
            else:
                diff_summary = "\n\n".join(diff_msgs)

            # 9. Tịnh tiến Offset (Quan trọng để nhúng lại Video MPV và Đồng bộ Hệ quy chiếu JSON)
            meta_rx = self.meta.roi_xywh[0] if self.meta and self.meta.roi_xywh else 0
            meta_ry = self.meta.roi_xywh[1] if self.meta and self.meta.roi_xywh else 0

            shifted_boxes = []
            for box in new_frame_result.text_boxes:
                shifted_poly = []
                for pt in box.polygon:
                    # Tọa độ Box Absolute so với Video Gốc = Tọa độ Box + Tọa độ Vùng Cắt
                    abs_x = pt[0] + self._current_offset_x
                    abs_y = pt[1] + self._current_offset_y
                    # Tọa độ Box Relative so với Meta ROI (Lưu vào JSON) = Absolute - Meta
                    rel_x = int(abs_x - meta_rx)
                    rel_y = int(abs_y - meta_ry)
                    shifted_poly.append((rel_x, rel_y))

                shifted_boxes.append(dataclasses.replace(box, polygon=shifted_poly))
            new_frame_result = dataclasses.replace(new_frame_result, text_boxes=shifted_boxes)

            # JSON Data
            boxes_data = [
                {
                    "Text": b.text,
                    "Confidence": round(float(b.confidence), 4),
                    "Polygon": b.polygon,
                }
                for b in new_frame_result.text_boxes
            ]
            json_txt = json.dumps({
                "Frame Index": new_frame_result.frame_index,
                "Time": seconds_to_display(new_frame_result.timestamp_sec),
                "Timestamp (s)": round(new_frame_result.timestamp_sec, 3),
                "Box Count": len(new_frame_result.text_boxes),
                "Detections": boxes_data
            }, ensure_ascii=False, indent=2)

            # Emit Final Results
            self.success_result.emit(new_frame_result, qimg_input, qimg_output, diff_summary, json_txt, is_noop)

        except Exception as e:
            logger.exception("Background Worker ReOCR Error")
            self.failed.emit(f"Lỗi hệ thống: {e}")


# ==============================================================================
# VIEW MODEL CHÍNH
# ==============================================================================
class DebugPageViewModel(QObject):
    file_loaded = Signal(bool, str)
    frame_changed = Signal(int, int, object, str, str, str)
    subtitles_built = Signal(object)
    action_error = Signal(str)
    action_success = Signal(str, str, bool)

    live_images_ready = Signal(QImage, QImage)
    is_busy = Signal(bool)

    def __init__(self, container: ApplicationContainer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._container = container
        self._frames: list[OcrFrameResult] = []
        self._meta: RawOcrMeta | None = None
        self._current_idx: int = -1
        self._base_img_dir: Path | None = None
        self._async_thread: QThread | None = None

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    def load_raw_file(self, file_path: Path) -> None:
        try:
            frames, meta = load_raw_ocr(file_path)
            self._frames = frames
            self._meta = meta
            self._current_idx = 0

            base_name = file_path.name
            for ext in [".gz", ".json", ".seraw"]:
                base_name = base_name.replace(ext, "")
            self._base_img_dir = file_path.parent / f"{base_name}_ocr_images"

            self.file_loaded.emit(True, f"Đã nạp thành công {len(frames)} frames từ {file_path.name}")
            self.jump_to_frame(0)

        except Exception as exc:
            logger.exception("Lỗi nạp file OCR Debug: %s", exc)
            self.file_loaded.emit(False, str(exc))
            self._frames = []
            self._meta = None
            self._current_idx = -1

    def jump_to_frame(self, index: int) -> None:
        if not self._frames or index < 0 or index >= len(self._frames):
            return

        self._current_idx = index
        frame = self._frames[index]

        input_img_path = ""
        output_img_path = ""

        if self._base_img_dir and self._base_img_dir.exists():
            fname = f"frame_{frame.frame_index:06d}_{int(frame.timestamp_sec * 1000)}ms.jpg"
            in_p = self._base_img_dir / "input" / fname
            out_p = self._base_img_dir / "output" / fname

            if in_p.exists(): input_img_path = str(in_p)
            if out_p.exists(): output_img_path = str(out_p)

        boxes_data = [
            {
                "Text": b.text,
                "Confidence": round(float(b.confidence), 4),
                "Polygon": b.polygon,
            }
            for b in frame.text_boxes
        ]

        json_txt = json.dumps({
            "Frame Index": frame.frame_index,
            "Time": seconds_to_display(frame.timestamp_sec),
            "Timestamp (s)": round(frame.timestamp_sec, 3),
            "Box Count": len(frame.text_boxes),
            "Detections": boxes_data
        }, ensure_ascii=False, indent=2)

        self.frame_changed.emit(
            self._current_idx,
            len(self._frames),
            frame,
            input_img_path,
            output_img_path,
            json_txt
        )

    def next_frame(self) -> None:
        self.jump_to_frame(self._current_idx + 1)

    def prev_frame(self) -> None:
        self.jump_to_frame(self._current_idx - 1)

    def save_current_frame_json(self, json_txt: str) -> bool:
        if not self._frames or self._current_idx < 0:
            return False
        try:
            data = json.loads(json_txt)
            new_boxes: list[OcrTextBox] = []

            for b in data.get("Detections",[]):
                text = str(b.get("Text", "")).strip()
                if not text: continue

                conf_val = float(b.get("Confidence", 1.0))
                normalized_conf = max(0.0, min(1.0, conf_val))
                poly_raw = b.get("Polygon", [])
                poly = [(int(p[0]), int(p[1])) for p in poly_raw if len(p) >= 2]

                new_boxes.append(OcrTextBox(
                    text=text,
                    confidence=Confidence(normalized_conf),
                    polygon=poly
                ))

            old_frame = self._frames[self._current_idx]
            new_frame = dataclasses.replace(old_frame, text_boxes=new_boxes)
            self._frames[self._current_idx] = new_frame
            return True

        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            self.action_error.emit(f"Cấu trúc JSON không hợp lệ: {exc}")
            return False

    def reocr_current_frame(self, tweaks: dict[str, Any]) -> None:
        if not self._frames or self._current_idx < 0:
            return

        # [LỖI 7 LEAK THREAD FIX] Chặn người dùng Spam bấm liên tục khi Thread cũ chưa xong
        if self._async_thread is not None and self._async_thread.isRunning():
            self.action_error.emit("Hệ thống vẫn đang phân tích khung hình trước đó. Vui lòng chờ giây lát...")
            return

        old_frame = self._frames[self._current_idx]

        self.is_busy.emit(True)

        self._async_thread = QThread()
        self._worker = _SingleFrameReocrWorker(
            self._container, old_frame, tweaks, self._base_img_dir, self._meta
        )
        self._worker.moveToThread(self._async_thread)

        self._async_thread.started.connect(self._worker.run)
        self._worker.success_result.connect(self._on_worker_success)
        self._worker.failed.connect(self._on_worker_failed)

        self._worker.success_result.connect(self._async_thread.quit)
        self._worker.failed.connect(self._async_thread.quit)
        self._async_thread.finished.connect(self._worker.deleteLater)
        self._async_thread.finished.connect(self._on_thread_finished_cleanup)
        self._async_thread.finished.connect(self._async_thread.deleteLater)

        self._async_thread.start()

    def _on_thread_finished_cleanup(self):
        """[LỖI 7 LEAK THREAD FIX] Dọn dẹp con trỏ luồng khi kết thúc"""
        self._async_thread = None
        self._worker = None

    def _on_worker_success(self, new_frame_result, qimg_in, qimg_out, diff_summary, json_txt, is_noop):
        self._container.apply_settings_changes()
        self._frames[self._current_idx] = new_frame_result

        # [LỖI 8 MẤT ẢNH FIX]: Bắn Signal Cập nhật Khung Text trống (frame_changed) TRƯỚC,
        # và Đổ Dữ liệu Ảnh Mới Sinh từ RAM (live_images_ready) SAU.
        self.frame_changed.emit(
            self._current_idx, len(self._frames), new_frame_result,
            "", "", json_txt
        )
        self.live_images_ready.emit(qimg_in, qimg_out)

        self.action_success.emit(self._container.translator.translate("debug.vm_title"), diff_summary, is_noop)
        self.is_busy.emit(False)

    def _on_worker_failed(self, error_msg):
        self._container.apply_settings_changes()
        self.action_error.emit(error_msg)
        self.is_busy.emit(False)

    def build_subtitles(self) -> None:
        if not self._frames:
            self.action_error.emit("Không có dữ liệu thô để xây dựng phụ đề.")
            return
        try:
            snapshot = self._container.settings_service.current
            post = snapshot.post_process
            threshold = snapshot.threshold

            builder_cfg = SubtitleBuilderConfig(
                similarity_threshold=min(post.similarity_threshold, threshold.text_similarity),
                min_duration_sec=post.min_duration_sec,
                max_duration_sec=post.max_duration_sec,
                merge_gap_sec=post.merge_gap_sec,
                min_confidence=threshold.ocr_min_confidence,
                use_viterbi=post.use_viterbi,
                viterbi_open_penalty=post.viterbi_open_penalty,
                min_text_chars=threshold.drop_short_text_chars,
                line_similarity_threshold=threshold.line_similarity,
                sample_step_sec=snapshot.frame.sample_step_sec,
                temporal_padding_sec=post.temporal_padding_sec,
                y_clustering_tolerance_ratio=post.y_clustering_tolerance_ratio,
                y_clustering_tolerance_min_px=post.y_clustering_tolerance_min_px,
                alignment_center_tolerance_ratio=post.alignment_center_tolerance_ratio,
                alignment_margin_tolerance_ratio=post.alignment_margin_tolerance_ratio,
                alignment_tolerance_min_px=post.alignment_tolerance_min_px
            )

            roi = None
            if self._meta and self._meta.roi_xywh:
                rx, ry, rw, rh = self._meta.roi_xywh
                saved_state = self._container.video_state_repository.get(self._meta.video_name)
                align = saved_state.roi.alignment if (saved_state and saved_state.roi) else TextAlignment.CENTER
                orient = saved_state.roi.orientation if (saved_state and saved_state.roi) else TextOrientation.HORIZONTAL
                roi = Roi(x=rx, y=ry, width=rw, height=rh, alignment=align, orientation=orient)

            builder = SubtitleBuilder(builder_cfg)
            events = builder.build(self._frames, roi=roi)

            self.subtitles_built.emit(events)

        except Exception as exc:
            logger.exception("Lỗi khi giả lập xây dựng phụ đề: %s", exc)
            self.action_error.emit(f"Quá trình xây dựng thất bại: {exc}")

__all__ = ["DebugPageViewModel"]
