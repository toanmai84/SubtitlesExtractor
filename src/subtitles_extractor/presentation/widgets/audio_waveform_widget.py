"""Widget hiển thị waveform audio bên dưới video player trong EditorPage.

BẢN CẬP NHẬT ĐỘT PHÁ (V3.44 - Dynamic Center Playhead):
    1. [UX BREAKTHROUGH] Tạm ngừng "Giữa Sóng Âm" thông minh: Khi người dùng lăn chuột
       hoặc kéo Viewport, tính năng Center Playhead sẽ tự động nhường quyền điều khiển. 
       Sau 2 giây không tương tác, Camera sẽ tự động trượt mượt mà (Smooth Pan) bám 
       lại vào Playhead. (Chuẩn NLE Adobe Premiere).
"""

from __future__ import annotations

import bisect
import enum
import functools
import hashlib
import logging
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import (
    QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, 
    Signal, QVariantAnimation, QEasingCurve
)
from PySide6.QtGui import (
    QColor, QFont, QImage, QKeyEvent, QMouseEvent, 
    QPainter, QPaintEvent, QPen, QPixmap, QResizeEvent, QWheelEvent, QPolygonF
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.presentation.utils.text_clean import (
    clean_subtitle_text_for_display,
)
from subtitles_extractor.presentation.utils.time_format import seconds_to_display

logger = logging.getLogger(__name__)

_MAX_PREALLOCATED_SAMPLES: int = 3_600_000
_MAGNETIC_SNAP_THRESHOLD_SEC: float = 0.05


@functools.lru_cache(maxsize=1024)
def _cached_clean_text(raw_text: str) -> str:
    return clean_subtitle_text_for_display(raw_text).replace('\n', ' ').strip()


class DragMode(enum.Enum):
    NONE = enum.auto()
    START = enum.auto()
    END = enum.auto()
    BODY = enum.auto()
    MOVE = enum.auto()
    SCRUB = enum.auto()


class WaveformRenderWorker(QThread):
    """Luồng Render Sóng âm ra QImage dưới nền, không chặn giao diện chính."""
    result_ready = Signal(QImage, object, int, int, tuple)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._render_request_queue: queue.Queue[
            tuple[int, int, int, int, float, tuple[float, float, int, int, float, int]]
        ] = queue.Queue(maxsize=2)

        self._worker_stop_event = threading.Event()
        self._audio_samples_reference: np.ndarray | None = None
        self._shared_audio_lock: threading.Lock | None = None

    def set_samples_reference(self, samples_array: np.ndarray | None, shared_lock: threading.Lock | None) -> None:
        self._shared_audio_lock = shared_lock
        self._audio_samples_reference = samples_array

    def request_render(
        self,
        start_idx: int,
        end_idx: int,
        canvas_width: int,
        canvas_height: int,
        zoom_level: float,
        cache_key: tuple[float, float, int, int, float, int]
    ) -> None:
        while not self._render_request_queue.empty():
            try:
                self._render_request_queue.get_nowait()
            except queue.Empty:
                break
        self._render_request_queue.put_nowait((start_idx, end_idx, canvas_width, canvas_height, zoom_level, cache_key))

    def run(self) -> None:
        while not self._worker_stop_event.is_set():
            try:
                payload = self._render_request_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            start_idx, end_idx, canvas_width, canvas_height, zoom_level, cache_key = payload

            if (
                self._audio_samples_reference is None
                or self._shared_audio_lock is None
                or canvas_width <= 0
                or canvas_height <= 0
            ):
                continue

            if end_idx <= start_idx:
                calculated_peaks_array = np.zeros(canvas_width, dtype=np.float32)
            else:
                with self._shared_audio_lock:
                    try:
                        if self._audio_samples_reference is not None and len(self._audio_samples_reference) > start_idx:
                            visible_slice_of_samples = self._audio_samples_reference[start_idx:end_idx].copy()
                        else:
                            visible_slice_of_samples = np.array([], dtype=np.float32)
                    except (IndexError, TypeError):
                        visible_slice_of_samples = np.array([], dtype=np.float32)

                visible_slice_length = len(visible_slice_of_samples)

                if visible_slice_length == 0:
                    calculated_peaks_array = np.zeros(canvas_width, dtype=np.float32)
                elif visible_slice_length > canvas_width:
                    downsample_step = visible_slice_length // canvas_width
                    if downsample_step > 1:
                        trimmed_slice_length = canvas_width * downsample_step
                        reshaped_2d_view = visible_slice_of_samples[:trimmed_slice_length].reshape(canvas_width, downsample_step)
                        calculated_peaks_array = reshaped_2d_view.max(axis=1)
                    else:
                        calculated_peaks_array = visible_slice_of_samples[:canvas_width]
                elif visible_slice_length < canvas_width:
                    if visible_slice_length == 1:
                        calculated_peaks_array = np.full(canvas_width, visible_slice_of_samples[0])
                    else:
                        original_linear_x_coords = np.linspace(0, canvas_width - 1, visible_slice_length)
                        target_linear_x_coords = np.arange(canvas_width)
                        calculated_peaks_array = np.interp(target_linear_x_coords, original_linear_x_coords, visible_slice_of_samples)
                else:
                    calculated_peaks_array = visible_slice_of_samples[:canvas_width]

            if len(calculated_peaks_array) < canvas_width:
                calculated_peaks_array = np.pad(calculated_peaks_array, (0, max(0, canvas_width - len(calculated_peaks_array))))

            calculated_peaks_array = calculated_peaks_array[:canvas_width].astype(np.float32)
            middle_y_position = canvas_height // 2
            half_canvas_height = max(1, canvas_height // 2 - 2)

            final_waveform_heights = np.clip(
                (calculated_peaks_array * half_canvas_height * zoom_level).astype(np.int32),
                0, half_canvas_height
            )

            rgba_pixel_buffer = np.full((canvas_height, canvas_width, 4), [18, 18, 22, 255], dtype=np.uint8)

            y_coordinate_grid = np.arange(canvas_height)[:, None]
            height_constraint_grid = final_waveform_heights[None, :]
            distance_from_middle_line = np.abs(y_coordinate_grid - middle_y_position)

            waveform_mask = distance_from_middle_line <= height_constraint_grid

            gradient_ratio = distance_from_middle_line.astype(np.float32) / half_canvas_height
            gradient_ratio = np.clip(gradient_ratio, 0.0, 1.0)

            r = np.ascontiguousarray((10 + gradient_ratio * 30).astype(np.uint8))
            g = np.ascontiguousarray((150 - gradient_ratio * 80).astype(np.uint8))
            b = np.ascontiguousarray((255 - gradient_ratio * 100).astype(np.uint8))

            r_full = np.broadcast_to(r, (canvas_height, canvas_width))
            g_full = np.broadcast_to(g, (canvas_height, canvas_width))
            b_full = np.broadcast_to(b, (canvas_height, canvas_width))

            rgba_pixel_buffer[waveform_mask, 0] = r_full[waveform_mask]
            rgba_pixel_buffer[waveform_mask, 1] = g_full[waveform_mask]
            rgba_pixel_buffer[waveform_mask, 2] = b_full[waveform_mask]
            rgba_pixel_buffer[waveform_mask, 3] = 255

            rendered_qimage = QImage(
                rgba_pixel_buffer.data,
                canvas_width,
                canvas_height,
                canvas_width * 4,
                QImage.Format.Format_RGBA8888
            ).copy()

            self.result_ready.emit(rendered_qimage, None, canvas_width, canvas_height, cache_key)

    def stop(self) -> None:
        self._worker_stop_event.set()
        self.set_samples_reference(None, None)


class _AudioExtractThread(QThread):
    """Trích xuất Audio bằng FFmpeg, phát từng phần ra UI."""
    chunk_ready = Signal(np.ndarray, int, int)
    finished_sig = Signal(np.ndarray, int)
    ffmpeg_error_sig = Signal(str)

    def __init__(self, target_video_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._source_video_path = target_video_path
        self._thread_stop_event = threading.Event()
        self._active_ffmpeg_process: subprocess.Popen | None = None

    def run(self) -> None:
        extraction_sample_rate = 16000
        downsampled_envelope_rate = 100
        bytes_allocated_per_sample = 2
        calculation_bin_size = max(1, extraction_sample_rate // downsampled_envelope_rate)
        calculation_bin_bytes = calculation_bin_size * bytes_allocated_per_sample
        subprocess_read_buffer_size = extraction_sample_rate * bytes_allocated_per_sample // 2

        try:
            hashed_video_path = hashlib.md5(os.path.abspath(self._source_video_path).encode('utf-8')).hexdigest()[:12]
            cache_file_destination = os.path.join(
                tempfile.gettempdir(),
                f"{os.path.basename(self._source_video_path)}_{hashed_video_path}.waveform.npy"
            )
            if os.path.exists(cache_file_destination):
                try:
                    loaded_cached_array = np.load(cache_file_destination)
                    self.finished_sig.emit(loaded_cached_array.astype(np.float32), downsampled_envelope_rate)
                    return
                except OSError:
                    pass
        except OSError:
            pass

        try:
            # [v3.23.297] Ưu tiên ffmpeg ĐÃ NHÚNG (bundle-first) — waveform chạy trên
            # máy standalone không có ffmpeg trên PATH.
            from subtitles_extractor.infrastructure.media import (
                find_ffmpeg,
                missing_ffmpeg_message,
            )

            ffmpeg_executable = find_ffmpeg()
            if not ffmpeg_executable:
                # [v3.23.306] Nêu rõ tính năng + cách sửa thay vì báo chung chung.
                logger.warning(
                    "%s", missing_ffmpeg_message(feature="Sóng âm trong Trình sửa")
                )
                return
            ffmpeg_subprocess_kwargs: dict[str, int | object] = {}
            if sys.platform == 'win32':
                ffmpeg_subprocess_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                ffmpeg_subprocess_kwargs['preexec_fn'] = os.setsid

            self._active_ffmpeg_process = subprocess.Popen([
                    ffmpeg_executable, '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', self._source_video_path,
                    # [Waveform Desync] Video livestream/cắt ghép thường có timebase
                    # lỗi → sóng âm lệch so với hình. aresample=async=1 ép nội suy âm
                    # thanh bám chặt trục thời gian thực của video.
                    '-af', 'aresample=async=1',
                    '-ar', str(extraction_sample_rate),
                    '-ac', '1',
                    '-f', 's16le', '-'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                # [v3.23.350] ``ffmpeg_subprocess_kwargs`` ĐÃ chứa ``creationflags``
                # (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) trên Windows và
                # ``preexec_fn=os.setsid`` trên POSIX. Trước đây còn nối thêm
                # ``**no_window_kwargs()`` (cũng đặt ``creationflags``) → Python báo
                # "got multiple values for keyword argument 'creationflags'". Giữ MỘT
                # nguồn duy nhất — bản đầy đủ có NEW_PROCESS_GROUP để _cleanup_process
                # kill được cả nhóm tiến trình.
                **ffmpeg_subprocess_kwargs,
            )

            accumulated_peaks_list: list[np.ndarray] = []
            leftover_remainder_bytes = b''
            current_chunk_index = 0

            while True:
                if self._thread_stop_event.is_set() or self._active_ffmpeg_process.poll() is not None:
                    break

                if self._active_ffmpeg_process.stdout is None:
                    break

                try:
                    if hasattr(self._active_ffmpeg_process.stdout, 'read1'):
                        read_raw_bytes = self._active_ffmpeg_process.stdout.read1(subprocess_read_buffer_size)
                    else:
                        read_raw_bytes = self._active_ffmpeg_process.stdout.read(subprocess_read_buffer_size)
                except (OSError, ValueError):
                    break

                if not read_raw_bytes:
                    break

                read_raw_bytes = leftover_remainder_bytes + read_raw_bytes
                number_of_complete_bins = len(read_raw_bytes) // calculation_bin_bytes

                if number_of_complete_bins > 0:
                    usable_bytes_threshold_length = number_of_complete_bins * calculation_bin_bytes
                    converted_audio_array = np.frombuffer(read_raw_bytes[:usable_bytes_threshold_length], dtype=np.int16).astype(np.float32)
                    np.divide(converted_audio_array, 32768.0, out=converted_audio_array)

                    calculated_peak_array = np.max(np.abs(converted_audio_array.reshape(number_of_complete_bins, calculation_bin_size)), axis=1)

                    accumulated_peaks_list.append(calculated_peak_array)
                    self.chunk_ready.emit(calculated_peak_array.copy(), downsampled_envelope_rate, current_chunk_index)

                    current_chunk_index += 1
                    leftover_remainder_bytes = read_raw_bytes[usable_bytes_threshold_length:]
                else:
                    leftover_remainder_bytes = read_raw_bytes

            if leftover_remainder_bytes and len(leftover_remainder_bytes) >= bytes_allocated_per_sample:
                usable_final_remainder_length = len(leftover_remainder_bytes) - (len(leftover_remainder_bytes) % bytes_allocated_per_sample)
                final_converted_array = np.frombuffer(leftover_remainder_bytes[:usable_final_remainder_length], dtype=np.int16).astype(np.float32)
                np.divide(final_converted_array, 32768.0, out=final_converted_array)
                if len(final_converted_array) > 0:
                    accumulated_peaks_list.append(np.array([np.max(np.abs(final_converted_array))], dtype=np.float32))

            if self._active_ffmpeg_process:
                try:
                    self._active_ffmpeg_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._cleanup_process(self._active_ffmpeg_process)

            complete_final_peaks_array = np.concatenate(accumulated_peaks_list) if accumulated_peaks_list else np.array([], dtype=np.float32)

            try:
                if len(complete_final_peaks_array) > 0 and not self._thread_stop_event.is_set():
                    unique_uuid_string = str(uuid.uuid4())[:8]
                    temporary_cache_file_destination = cache_file_destination + f".{unique_uuid_string}.tmp.npy"
                    with open(temporary_cache_file_destination, 'wb') as temp_file:
                        np.save(temp_file, complete_final_peaks_array)
                    os.replace(temporary_cache_file_destination, cache_file_destination)
            except OSError:
                pass

            self.finished_sig.emit(complete_final_peaks_array, downsampled_envelope_rate)

        except FileNotFoundError:
            logger.error("FFmpeg không được tìm thấy.")
            self.ffmpeg_error_sig.emit("Chưa cài đặt FFmpeg trên máy tính (Cần để trích xuất Sóng âm).")
            self.finished_sig.emit(np.array([], dtype=np.float32), downsampled_envelope_rate)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            logger.error("FFmpeg gặp lỗi: %s", exc)
            self.ffmpeg_error_sig.emit(f"Lỗi đọc Video: {exc}")
            self.finished_sig.emit(np.array([], dtype=np.float32), downsampled_envelope_rate)
        except Exception as exc:  # noqa: BLE001 — BIÊN THREAD: bắt buộc bao trọn.
            # [v3.23.350] Ngoại lệ KHÔNG lường trước (vd TypeError khi dựng lệnh
            # Popen) thoát khỏi QThread.run() sẽ bị Qt NUỐT; bản đóng gói
            # console=False còn có stderr=None nên KHÔNG bao giờ vào log → người
            # dùng thấy "lỗi mà nhật ký im lặng". Bao trọn tại đây, ghi kèm
            # traceback và báo UI để mọi lỗi tương lai đều lộ diện.
            logger.exception("Trích xuất sóng âm thất bại (lỗi không lường trước): %s", exc)
            self.ffmpeg_error_sig.emit(f"Lỗi trích xuất sóng âm: {exc}")
            self.finished_sig.emit(np.array([], dtype=np.float32), downsampled_envelope_rate)
        finally:
            self._cleanup_process(self._active_ffmpeg_process)
            self._active_ffmpeg_process = None

    def _cleanup_process(self, running_process: subprocess.Popen | None) -> None:
        if running_process is None:
            return

        try:
            if running_process.stdout:
                running_process.stdout.close()
            if running_process.stderr:
                running_process.stderr.close()
            if running_process.stdin:
                running_process.stdin.close()
        except (OSError, ValueError):
            # Pipe đã đóng/tiến trình đã thoát — vô hại khi dọn dẹp.
            pass

        if running_process.poll() is None:
            # [v3.20.3 #1] Dọn zombie FFmpeg theo trình tự "Soft → Wait → Force":
            # (1) terminate() mềm để FFmpeg tự đóng & nhả khoá file video;
            # (2) chờ tối đa 1.5s; (3) nếu vẫn sống → kill() cứng. Tránh để tiến
            # trình nền giữ khoá file khi người dùng chuyển video liên tục.
            try:
                running_process.terminate()
            except (OSError, ProcessLookupError):
                pass
            try:
                running_process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                try:
                    running_process.kill()
                    running_process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired, ProcessLookupError):
                    pass
            except (OSError, ProcessLookupError):
                pass

    def request_stop(self) -> None:
        self._thread_stop_event.set()
        self._cleanup_process(self._active_ffmpeg_process)


class AudioWaveformWidget(QWidget):
    """Widget Khung Sóng Âm (Waveform) - Trái tim của Editor Page."""

    seek_requested = Signal(float)
    subtitle_drag_done = Signal(int, float, float, float, float)
    subtitle_create_requested = Signal(float, float)
    play_toggle_requested = Signal()
    ffmpeg_error = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_variables()
        self._init_graphic_objects()

        self._scroll_anim = QVariantAnimation(self)
        self._scroll_anim.setDuration(200)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scroll_anim.valueChanged.connect(self._on_anim_scroll_value)

        self._edge_pan_timer = QTimer(self)
        self._edge_pan_timer.setInterval(16)
        self._edge_pan_timer.timeout.connect(self._process_edge_pan)

        self._active_render_worker = WaveformRenderWorker(self)
        self._active_render_worker.result_ready.connect(self._on_worker_render_result_ready)
        self._active_render_worker.start()

    def _init_variables(self) -> None:
        self._edge_interaction_hitbox_px = 10
        self._recorded_samples_array: np.ndarray | None = None
        self._audio_sample_rate = 100
        self._total_video_duration_sec = 0.0
        self._view_timeline_start_sec = 0.0
        self._view_timeline_duration_sec = 30.0
        self._loaded_subtitle_events: list[SubtitleEvent] = []
        self._cached_event_start_times: list[float] = []
        self._current_playback_time_sec = 0.0
        # [v3.20.3 #1] Tiết lưu lệnh seek xuống C-Core MPV ở 30Hz (~33ms): UI vẫn
        # cập nhật mượt mỗi mousemove nhưng lệnh seek gửi rải để không nghẽn MPV
        # (giật lag toàn hệ thống khi kéo trượt). QTimer flush giá trị CUỐI để
        # vị trí kết thúc kéo luôn chính xác.
        self._seek_throttle_min_interval_sec = 1.0 / 30.0
        self._seek_last_emit_time = 0.0
        self._seek_pending_value: float | None = None
        self._seek_flush_timer = QTimer(self)
        self._seek_flush_timer.setSingleShot(True)
        self._seek_flush_timer.timeout.connect(self._flush_pending_seek)
        self._shared_audio_data_lock = threading.Lock()

        self._mouse_drag_mode = DragMode.NONE
        self._mouse_drag_event_idx = -1
        self._cursor_current_x_px = 0.0
        self._is_user_drawing_new_sub = False

        self._highlighted_active_row_idx = -1
        self._hover_event_idx = -1
        self._hover_mode = DragMode.NONE

        self._vertical_y_zoom_factor = 1.0
        self._rendered_background_pixmap = None
        self._rendered_background_cache_key = None
        self._last_requested_render_key = None

        self._active_audio_extract_thread = None
        self._reocr_highlight_region_tuple = None
        self._loop_region_start_sec = None
        self._loop_region_end_sec = None
        
        # [V3.44 UX] Biến trạng thái quản lý việc tự động tạm ngưng "Giữa Sóng Âm"
        self._center_playhead_mode_enabled = False
        self._last_user_interaction_time = 0.0
        self._is_suspended_tracking = False
        
        self._is_extracting_audio = False

        self._mouse_drag_start_time = 0.0
        self._mouse_drag_original_start = 0.0
        self._mouse_drag_original_end = 0.0
        self._mouse_drag_preview_start = 0.0
        self._mouse_drag_preview_end = 0.0

        self._is_middle_dragging = False
        self._last_middle_drag_x = 0.0
        self._user_draw_start_time = 0.0
        self._user_draw_end_time = 0.0
        self._magnetic_snap_line_x = None

        self._loading_timer = QTimer(self)
        self._loading_timer.timeout.connect(self.update)
        self._loading_dots_count = 0

        self.setMinimumHeight(64)
        self.setMaximumHeight(128)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pre_allocated_samples_buffer = np.zeros(_MAX_PREALLOCATED_SAMPLES, dtype=np.float32)
        self._current_recorded_sample_count = 0
        self._retained_image_buffer = None

    def _init_graphic_objects(self) -> None:
        self._color_bg = QColor(18, 18, 22)

        self._color_active_box = QColor(100, 200, 255, 110)
        self._color_danger_box = QColor(255, 90, 90, 90)
        self._color_warn_box = QColor(255, 180, 50, 90)
        self._color_safe_box = QColor(0, 150, 230, 70)
        self._color_hover_glow = QColor(255, 255, 255, 30)

        self._pen_active_border = QPen(QColor(255, 255, 255, 255), 2)
        self._pen_inactive_border = QPen(QColor(255, 255, 255, 120), 1)
        self._pen_hover_border = QPen(QColor(100, 200, 255, 255), 2)

        self._color_text = QColor(255, 255, 255, 255)
        self._color_inactive_text = QColor(255, 255, 255, 180)
        self._color_text_shadow = QColor(0, 0, 0, 220)

        self._color_playhead = QColor('#FF2A2A')
        self._pen_playhead = QPen(self._color_playhead, 1)
        self._pen_playhead_shadow = QPen(QColor(255, 42, 42, 100), 3)
        self._color_playhead_tooltip_bg = QColor(20, 20, 25, 220)

        self._pen_magnetic_snap = QPen(QColor(0, 255, 255, 255), 1, Qt.PenStyle.DashLine)

        self._color_loop_bg = QColor(0, 150, 255, 25)
        self._pen_loop_start = QPen(QColor(80, 200, 255), 2)
        self._pen_loop_end = QPen(QColor(255, 165, 50), 2)

        self._color_reocr_bg = QColor(255, 165, 0, 40)
        self._pen_reocr_hatch = QPen(QColor(255, 165, 0, 90), 1)
        self._pen_reocr_text = QPen(QColor(255, 200, 50))

        self._color_draw_bg = QColor(0, 255, 128, 80)
        self._pen_draw_border = QPen(QColor(0, 255, 128, 200), 2)

        self._pen_ruler_major = QPen(QColor(255, 255, 255, 180), 1)
        self._pen_ruler_minor = QPen(QColor(255, 255, 255, 60), 1)

        self._font_text_active = QFont()
        self._font_text_active.setPixelSize(11)
        self._font_text_active.setBold(True)

        self._font_text_inactive = QFont()
        self._font_text_inactive.setPixelSize(11)
        self._font_text_inactive.setBold(False)

        self._font_tags = QFont()
        self._font_tags.setPixelSize(10)
        self._font_tags.setBold(True)

        self._font_ruler = QFont()
        self._font_ruler.setPixelSize(9)
        self._font_ruler.setBold(False)

    def _effective_duration_sec(self) -> float:
        if getattr(self, '_total_video_duration_sec', 0.0) > 0.0:
            return self._total_video_duration_sec
        if (self._recorded_samples_array is not None
            and self._audio_sample_rate > 0
            and len(self._recorded_samples_array) > 0):
            return float(len(self._recorded_samples_array)) / float(self._audio_sample_rate)
        return 60.0

    @property
    def _timeline_max_bounds(self) -> float:
        return self._effective_duration_sec()

    def _format_ruler_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)

        if self._total_video_duration_sec >= 3600 or h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _on_anim_scroll_value(self, val: Any) -> None:
        self._view_timeline_start_sec = float(val)
        self.update()

    def _smooth_scroll_to(self, target_sec: float, duration: int = 200) -> None:
        target_sec = max(0.0, min(target_sec, self._timeline_max_bounds - self._view_timeline_duration_sec))
        self._scroll_anim.stop()
        self._scroll_anim.setDuration(duration)
        self._scroll_anim.setStartValue(float(self._view_timeline_start_sec))
        self._scroll_anim.setEndValue(float(target_sec))
        self._scroll_anim.start()

    def _smooth_scroll_by(self, delta_sec: float, duration: int = 200) -> None:
        if self._scroll_anim.state() == QVariantAnimation.State.Running:
            current_target = float(self._scroll_anim.endValue())
        else:
            current_target = self._view_timeline_start_sec

        new_target = max(0.0, current_target + delta_sec)
        max_allowable_start = max(0.0, self._timeline_max_bounds - self._view_timeline_duration_sec)
        new_target = min(new_target, max_allowable_start)

        self._scroll_anim.stop()
        self._scroll_anim.setDuration(duration)
        self._scroll_anim.setStartValue(float(self._view_timeline_start_sec))
        self._scroll_anim.setEndValue(float(new_target))
        self._scroll_anim.start()

    def _process_edge_pan(self) -> None:
        if self._mouse_drag_mode == DragMode.NONE and not getattr(self, '_is_user_drawing_new_sub', False):
            self._edge_pan_timer.stop()
            return
            
        canvas_width = float(max(1, self.width()))
        margin = 40.0
        pan_speed = 0.0
        
        if self._cursor_current_x_px < margin:
            ratio = (margin - self._cursor_current_x_px) / margin
            pan_speed = - (self._view_timeline_duration_sec * 0.012) * ratio
        elif self._cursor_current_x_px > canvas_width - margin:
            ratio = (self._cursor_current_x_px - (canvas_width - margin)) / margin
            pan_speed = (self._view_timeline_duration_sec * 0.012) * ratio
            
        if pan_speed != 0.0:
            new_start = self._view_timeline_start_sec + pan_speed
            max_allowable = max(0.0, self._timeline_max_bounds - self._view_timeline_duration_sec)
            self._view_timeline_start_sec = max(0.0, min(max_allowable, new_start))
            
            self._update_drag_action()
            self.update()

    def _emit_seek_throttled(self, time_sec: float) -> None:
        """[v3.20.3 #1] Phát ``seek_requested`` giới hạn 30Hz, đảm bảo gửi giá trị cuối.

        Nếu đã quá ``_seek_throttle_min_interval_sec`` kể từ lần gửi trước → gửi
        ngay. Ngược lại lưu giá trị chờ và hẹn QTimer flush, để khi người dùng dừng
        kéo, vị trí CUỐI cùng vẫn được seek chính xác.
        """
        now = time.monotonic()
        if now - self._seek_last_emit_time >= self._seek_throttle_min_interval_sec:
            self._seek_last_emit_time = now
            self._seek_pending_value = None
            self.seek_requested.emit(time_sec)
        else:
            self._seek_pending_value = time_sec
            if not self._seek_flush_timer.isActive():
                remaining_ms = int(
                    (self._seek_throttle_min_interval_sec - (now - self._seek_last_emit_time)) * 1000
                )
                self._seek_flush_timer.start(max(1, remaining_ms))

    def _flush_pending_seek(self) -> None:
        if self._seek_pending_value is None:
            return
        self._seek_last_emit_time = time.monotonic()
        value = self._seek_pending_value
        self._seek_pending_value = None
        self.seek_requested.emit(value)

    def _update_drag_action(self) -> None:
        cursor_time = self._view_timeline_start_sec + (self._cursor_current_x_px / max(1, self.width())) * self._view_timeline_duration_sec
        
        if getattr(self, '_is_user_drawing_new_sub', False):
            self._user_draw_end_time = max(0.0, min(cursor_time, self._timeline_max_bounds))
            return
            
        if self._mouse_drag_mode == DragMode.SCRUB:
            self._emit_seek_throttled(max(0.0, min(cursor_time, self._timeline_max_bounds)))
            return
            
        if self._mouse_drag_mode != DragMode.NONE and 0 <= self._mouse_drag_event_idx < len(self._loaded_subtitle_events):
            time_delta = cursor_time - self._mouse_drag_start_time
            duration_length_sec = self._mouse_drag_original_end - self._mouse_drag_original_start
            
            if self._mouse_drag_mode == DragMode.MOVE:
                raw_start_time = max(0.0, self._mouse_drag_original_start + time_delta)
                new_start_time = self._snap_time_to_closest_event(raw_start_time, self._mouse_drag_event_idx)

                # [V3.43] Magnetic Playhead (Snap vào Playhead nếu gần sát)
                if abs(new_start_time - self._current_playback_time_sec) <= _MAGNETIC_SNAP_THRESHOLD_SEC:
                    new_start_time = self._current_playback_time_sec
                    self._magnetic_snap_line_x = new_start_time
                elif abs(new_start_time - raw_start_time) > 0.001:
                    self._magnetic_snap_line_x = new_start_time

                min_start = 0.0
                if self._mouse_drag_event_idx > 0:
                    min_start = self._loaded_subtitle_events[self._mouse_drag_event_idx - 1].end_sec + 0.001

                max_start = (self._timeline_max_bounds - duration_length_sec)
                if self._mouse_drag_event_idx + 1 < len(self._loaded_subtitle_events):
                    max_start = min(max_start, self._loaded_subtitle_events[self._mouse_drag_event_idx + 1].start_sec - duration_length_sec - 0.001)

                new_start_time = max(min_start, min(max_start, new_start_time))
                self._mouse_drag_preview_start = new_start_time
                self._mouse_drag_preview_end = new_start_time + duration_length_sec
                self.seek_requested.emit(new_start_time)

            elif self._mouse_drag_mode == DragMode.START:
                raw_start_time = self._mouse_drag_original_start + time_delta
                new_start_time = self._snap_time_to_closest_event(raw_start_time, self._mouse_drag_event_idx)

                # [V3.43] Magnetic Playhead (Snap vào Playhead nếu gần sát)
                if abs(new_start_time - self._current_playback_time_sec) <= _MAGNETIC_SNAP_THRESHOLD_SEC:
                    new_start_time = self._current_playback_time_sec
                    self._magnetic_snap_line_x = new_start_time
                elif abs(new_start_time - raw_start_time) > 0.001:
                    self._magnetic_snap_line_x = new_start_time

                min_start = 0.0
                if self._mouse_drag_event_idx > 0:
                    min_start = self._loaded_subtitle_events[self._mouse_drag_event_idx - 1].end_sec + 0.001

                maximum_allowable_start = max(min_start, self._mouse_drag_original_end - 0.05)
                maximum_allowable_start = min(maximum_allowable_start, max(0.0, self._timeline_max_bounds - 0.05))

                self._mouse_drag_preview_start = max(min_start, min(new_start_time, maximum_allowable_start))
                self._mouse_drag_preview_end = self._mouse_drag_original_end
                self.seek_requested.emit(self._mouse_drag_preview_start)

            elif self._mouse_drag_mode == DragMode.END:
                raw_end_time = self._mouse_drag_original_end + time_delta
                new_end_time = self._snap_time_to_closest_event(raw_end_time, self._mouse_drag_event_idx)

                # [V3.43] Magnetic Playhead (Snap vào Playhead nếu gần sát)
                if abs(new_end_time - self._current_playback_time_sec) <= _MAGNETIC_SNAP_THRESHOLD_SEC:
                    new_end_time = self._current_playback_time_sec
                    self._magnetic_snap_line_x = new_end_time
                elif abs(new_end_time - raw_end_time) > 0.001:
                    self._magnetic_snap_line_x = new_end_time

                minimum_allowable_end = self._mouse_drag_original_start + 0.05
                max_end = self._timeline_max_bounds
                if self._mouse_drag_event_idx + 1 < len(self._loaded_subtitle_events):
                    max_end = min(max_end, self._loaded_subtitle_events[self._mouse_drag_event_idx + 1].start_sec - 0.001)

                new_end_time = max(minimum_allowable_end, min(max_end, new_end_time))
                self._mouse_drag_preview_start = self._mouse_drag_original_start
                self._mouse_drag_preview_end = new_end_time
                self.seek_requested.emit(self._mouse_drag_preview_end)

    def set_edge_px(self, val: int) -> None:
        self._edge_interaction_hitbox_px = val

    def load_video(self, path: Path) -> None:
        self.clear()
        self._is_extracting_audio = True
        self._loading_dots_count = 0
        self._loading_timer.start(400)
        self.update()

        self._active_audio_extract_thread = _AudioExtractThread(str(path), self)
        self._active_audio_extract_thread.chunk_ready.connect(self._on_audio_chunk_data_received)
        self._active_audio_extract_thread.finished_sig.connect(self._on_audio_extract_thread_finished)
        self._active_audio_extract_thread.ffmpeg_error_sig.connect(self.ffmpeg_error)
        self._active_audio_extract_thread.start()

    def set_duration(self, duration_sec: float) -> None:
        if duration_sec > 0:
            self._total_video_duration_sec = duration_sec
            max_start = max(0.0, self._timeline_max_bounds - self._view_timeline_duration_sec)
            self._view_timeline_start_sec = min(self._view_timeline_start_sec, max_start)
            self.update()

    def set_events(self, events: list[SubtitleEvent]) -> None:
        self._loaded_subtitle_events = events
        self._cached_event_start_times = [event.start_sec for event in events]
        self.update()

    def set_active_row(self, idx: int) -> None:
        self._highlighted_active_row_idx = idx

        if 0 <= idx < len(self._loaded_subtitle_events):
            event = self._loaded_subtitle_events[idx]
            desired_view_start = event.start_sec - self._view_timeline_duration_sec / 2.0
            self._smooth_scroll_to(desired_view_start, duration=250)
        else:
            self.update()

    def set_center_playhead(self, enabled: bool) -> None:
        self._center_playhead_mode_enabled = enabled
        self.update()

    def set_y_zoom(self, factor: float) -> None:
        self._vertical_y_zoom_factor = max(0.1, min(10.0, float(factor)))
        self.update()

    def set_current_time(self, time_sec: float) -> None:
        if abs(self._current_playback_time_sec - time_sec) < 0.01:
            return

        self._current_playback_time_sec = time_sec
        safe_bounds = self._timeline_max_bounds

        if getattr(self, '_center_playhead_mode_enabled', False):
            # [V3.44 UX] Bỏ qua việc căn giữa nếu người dùng vừa mới tương tác với Waveform
            now = time.time()
            time_since_interaction = now - getattr(self, '_last_user_interaction_time', 0.0)
            
            if time_since_interaction < 2.0:
                self._is_suspended_tracking = True
                if time_sec > self._view_timeline_start_sec + self._view_timeline_duration_sec:
                    new_start = time_sec - (self._view_timeline_duration_sec * 0.1)
                    self._smooth_scroll_to(new_start, duration=150)
                elif time_sec < self._view_timeline_start_sec:
                    self._smooth_scroll_to(time_sec - (self._view_timeline_duration_sec * 0.9), duration=150)
                self.update()
                return

            new_calculated_view_start = time_sec - self._view_timeline_duration_sec / 2.0
            maximum_view_start_limit = max(0.0, safe_bounds - self._view_timeline_duration_sec)
            new_calculated_view_start = min(new_calculated_view_start, maximum_view_start_limit)
            
            # Khôi phục mượt mà (Smooth Snap) sau 2 giây ngưng tương tác
            if getattr(self, '_is_suspended_tracking', False):
                self._is_suspended_tracking = False
                self._smooth_scroll_to(max(0.0, new_calculated_view_start), duration=300)
            else:
                self._scroll_anim.stop() 
                if abs(self._view_timeline_start_sec - max(0.0, new_calculated_view_start)) > 0.01:
                    self._view_timeline_start_sec = max(0.0, new_calculated_view_start)
            self.update()
        else:
            if time_sec > self._view_timeline_start_sec + self._view_timeline_duration_sec:
                new_start = time_sec - (self._view_timeline_duration_sec * 0.1)
                self._smooth_scroll_to(new_start, duration=150)
            elif time_sec < self._view_timeline_start_sec:
                self._smooth_scroll_to(time_sec - (self._view_timeline_duration_sec * 0.9), duration=150)
            self.update()

    def set_loop_region(self, start: float | None, end: float | None) -> None:
        self._loop_region_start_sec = start
        self._loop_region_end_sec = end
        self.update()

    def clear_loop_region(self) -> None:
        self._loop_region_start_sec = None
        self._loop_region_end_sec = None
        self.update()

    def set_reocr_region(self, start: float, end: float) -> None:
        self._reocr_highlight_region_tuple = (start, end)
        self.update()

    def clear_reocr_region(self) -> None:
        self._reocr_highlight_region_tuple = None
        self.update()

    def _on_audio_extract_thread_finished(self, samples: np.ndarray, sample_rate: int) -> None:
        self._is_extracting_audio = False
        self._loading_timer.stop()
        with self._shared_audio_data_lock:
            if samples is None or len(samples) == 0:
                target_length = int(self._timeline_max_bounds * 100)
                self._recorded_samples_array = np.zeros(target_length, dtype=np.float32)
                self._audio_sample_rate = 100
            else:
                self._recorded_samples_array = samples.astype(np.float32, copy=False)
                self._audio_sample_rate = max(1, sample_rate)
            self._active_render_worker.set_samples_reference(self._recorded_samples_array, self._shared_audio_data_lock)

        max_start = max(0.0, self._timeline_max_bounds - self._view_timeline_duration_sec)
        self._view_timeline_start_sec = min(self._view_timeline_start_sec, max_start)
        self.update()

    def _on_audio_chunk_data_received(self, chunk: np.ndarray, sample_rate: int, chunk_index: int) -> None:
        with self._shared_audio_data_lock:
            self._audio_sample_rate = max(1, sample_rate)
            chunk_length_elements = len(chunk)

            if self._current_recorded_sample_count + chunk_length_elements > len(self._pre_allocated_samples_buffer):
                new_expanded_size = len(self._pre_allocated_samples_buffer) + _MAX_PREALLOCATED_SAMPLES
                new_buffer = np.zeros(new_expanded_size, dtype=np.float32)
                new_buffer[:self._current_recorded_sample_count] = self._pre_allocated_samples_buffer[:self._current_recorded_sample_count]
                self._pre_allocated_samples_buffer = new_buffer

            self._pre_allocated_samples_buffer[self._current_recorded_sample_count:self._current_recorded_sample_count + chunk_length_elements] = chunk
            self._current_recorded_sample_count += chunk_length_elements
            self._recorded_samples_array = self._pre_allocated_samples_buffer[:self._current_recorded_sample_count]

            self._active_render_worker.set_samples_reference(self._recorded_samples_array, self._shared_audio_data_lock)
        self.update()

    def clear(self) -> None:
        if self._active_audio_extract_thread and self._active_audio_extract_thread.isRunning():
            self._active_audio_extract_thread.request_stop()
            self._active_audio_extract_thread.quit()
            # [BUG FIX v2.9+]: Tăng từ 500ms lên 6000ms.
            # _cleanup_process() bên trong chờ FFmpeg tối đa 5s (2s wait + 3s kill),
            # nên 500ms không đủ → thread zombie + SIGPIPE khi process bị kill.
            self._active_audio_extract_thread.wait(6000)

        self._is_extracting_audio = False
        self._loading_timer.stop()
        
        with self._shared_audio_data_lock:
            self._recorded_samples_array = None
            self._current_recorded_sample_count = 0
            if self._active_render_worker is not None:
                self._active_render_worker.set_samples_reference(None, None)
            
        self._retained_image_buffer = None
        self._total_video_duration_sec = 0.0
        self._view_timeline_start_sec = 0.0
        self._current_playback_time_sec = 0.0
        
        self.update()

    def close_widget(self) -> None:
        self.clear()
        self._loaded_subtitle_events = []
        self._cached_event_start_times = []
        if self._active_render_worker is not None:
            self._active_render_worker.stop()
            self._active_render_worker.quit()
            # [BUG FIX v2.9+]: Tăng từ 500ms lên 2000ms để render worker kịp thoát
            # vòng lặp 50ms check + flush queue còn lại.
            self._active_render_worker.wait(2000)

    def _get_visible_event_range(self, time_start: float, time_end: float) -> tuple[int, int]:
        if not self._loaded_subtitle_events:
            return 0, 0
        start_time_boundaries = self._cached_event_start_times
        start_index = max(0, bisect.bisect_right(start_time_boundaries, time_start) - 2)
        end_index = min(len(self._loaded_subtitle_events), bisect.bisect_right(start_time_boundaries, time_end) + 1)
        return start_index, end_index

    def _snap_time_to_closest_event(self, raw_time_sec: float, ignore_index: int) -> float:
        if not self._loaded_subtitle_events:
            return raw_time_sec

        insert_bisect_idx = bisect.bisect_right(self._cached_event_start_times, raw_time_sec)
        scan_start_idx = max(0, insert_bisect_idx - 3)
        scan_end_idx = min(len(self._loaded_subtitle_events), insert_bisect_idx + 3)

        best_found_time = raw_time_sec
        best_calculated_distance = 0.15

        for scan_idx in range(scan_start_idx, scan_end_idx):
            if scan_idx == ignore_index:
                continue
            scan_event = self._loaded_subtitle_events[scan_idx]
            dist_to_event_start = abs(raw_time_sec - scan_event.start_sec)
            dist_to_event_end = abs(raw_time_sec - scan_event.end_sec)

            if dist_to_event_start < best_calculated_distance:
                best_calculated_distance = dist_to_event_start
                best_found_time = scan_event.start_sec
            if dist_to_event_end < best_calculated_distance:
                best_calculated_distance = dist_to_event_end
                best_found_time = scan_event.end_sec

        return best_found_time

    def _hit_test_cursor(self, cursor_x_position: float) -> tuple[int, DragMode] | None:
        current_widget_width = max(1, self.width())
        view_start_time = self._view_timeline_start_sec
        view_duration_time = self._view_timeline_duration_sec

        if view_duration_time <= 0:
            return None

        start_event_idx, end_event_idx = self._get_visible_event_range(view_start_time, view_start_time + view_duration_time)
        for check_idx in range(start_event_idx, end_event_idx):
            checked_event = self._loaded_subtitle_events[check_idx]
            if checked_event.end_sec < view_start_time or checked_event.start_sec > view_start_time + view_duration_time:
                continue

            x_pixel_start = ((checked_event.start_sec - view_start_time) / view_duration_time) * current_widget_width
            x_pixel_end = ((checked_event.end_sec - view_start_time) / view_duration_time) * current_widget_width

            if abs(cursor_x_position - x_pixel_start) <= self._edge_interaction_hitbox_px:
                return check_idx, DragMode.START
            if abs(cursor_x_position - x_pixel_end) <= self._edge_interaction_hitbox_px:
                return check_idx, DragMode.END
            if x_pixel_start <= cursor_x_position <= x_pixel_end:
                return check_idx, DragMode.BODY

        return None

    def _on_worker_render_result_ready(
        self, rendered_qimage: QImage, np_buffer_ref: object, image_width: int, image_height: int, generated_cache_key: tuple[float, float, int, int, float, int]
    ) -> None:
        self._retained_image_buffer = np_buffer_ref
        self._rendered_background_pixmap = QPixmap.fromImage(rendered_qimage)
        self._rendered_background_cache_key = generated_cache_key
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        # [Waveform Viewport Drift] Khi cửa sổ đổi kích thước, ảnh nền sóng âm đã
        # cache được vẽ theo kích thước CŨ → kéo giãn sai tỷ lệ / lộ mảng đen. Huỷ
        # cache và yêu cầu vẽ lại ngay theo kích thước mới.
        super().resizeEvent(event)
        self._rendered_background_cache_key = None
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        widget_rect = self.rect()
        canvas_width = float(widget_rect.width())
        canvas_height = float(widget_rect.height())
        
        painter.fillRect(widget_rect, self._color_bg)

        view_start_time = self._view_timeline_start_sec
        view_duration_time = self._view_timeline_duration_sec

        if view_duration_time <= 0:
            return

        with self._shared_audio_data_lock:
            current_samples_count = len(self._recorded_samples_array) if self._recorded_samples_array is not None else 0

        # --- 1. RENDER BACKROUND WAVEFORM (OVERSCAN BUFFERING) ---
        painter.save()
        if current_samples_count == 0 and getattr(self, '_is_extracting_audio', False):
            self._loading_dots_count += 1
            dots = "." * (self._loading_dots_count % 4)
            alpha = int(150 + 50 * math.sin(time.time() * 5))
            painter.setPen(QPen(QColor(255, 255, 255, alpha)))
            painter.setFont(self._font_ruler)
            painter.drawText(widget_rect, Qt.AlignmentFlag.AlignCenter, f"Đang phân tích dữ liệu Sóng âm{dots}")
        elif current_samples_count == 0 and not getattr(self, '_is_extracting_audio', False):
            painter.setPen(QPen(QColor(255, 255, 255, 100)))
            painter.setFont(self._font_ruler)
            painter.drawText(widget_rect, Qt.AlignmentFlag.AlignCenter, "🎵 Chưa có waveform — hãy chọn 1 video để bắt đầu")
        elif current_samples_count > 0:
            overscan_factor = 3.0
            buffer_zone_sec = view_duration_time * 0.5
            
            need_new_render = False
            if not self._rendered_background_cache_key:
                need_new_render = True
            else:
                c_start, c_dur, c_w, c_h, c_zoom, c_samples = self._rendered_background_cache_key
                if c_h != int(canvas_height) or c_zoom != self._vertical_y_zoom_factor or c_samples != current_samples_count:
                    need_new_render = True
                else:
                    if view_start_time < c_start + buffer_zone_sec and c_start > 0.001:
                        need_new_render = True
                    if (view_start_time + view_duration_time) > (c_start + c_dur - buffer_zone_sec) and (c_start + c_dur) < self._timeline_max_bounds - 0.001:
                        need_new_render = True

            if need_new_render:
                render_start = max(0.0, view_start_time - view_duration_time)
                render_dur = min(self._timeline_max_bounds - render_start, view_duration_time * overscan_factor)
                render_w = int(canvas_width * (render_dur / view_duration_time))
                
                if render_w > 0:
                    new_cache_key = (render_start, render_dur, render_w, int(canvas_height), self._vertical_y_zoom_factor, current_samples_count)
                    if self._last_requested_render_key != new_cache_key:
                        self._last_requested_render_key = new_cache_key
                        t_start_idx = int(render_start * self._audio_sample_rate)
                        t_end_idx = min(current_samples_count, int((render_start + render_dur) * self._audio_sample_rate))
                        self._active_render_worker.request_render(
                            t_start_idx, t_end_idx, render_w, int(canvas_height), self._vertical_y_zoom_factor, new_cache_key
                        )

            if self._rendered_background_pixmap and getattr(self, '_rendered_background_cache_key', None):
                old_c_start, old_c_dur, old_c_w, old_c_h, _, _ = self._rendered_background_cache_key
                
                time_offset_sec = old_c_start - view_start_time
                pixel_offset_x = (time_offset_sec / view_duration_time) * canvas_width
                rendered_pixel_width = (old_c_dur / view_duration_time) * canvas_width
                
                target_rect = QRectF(pixel_offset_x, 0, rendered_pixel_width, canvas_height)
                source_rect = QRectF(0, 0, self._rendered_background_pixmap.width(), self._rendered_background_pixmap.height())
                
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                painter.drawPixmap(target_rect, self._rendered_background_pixmap, source_rect)
        painter.restore()

        # --- 2. RENDER EOF TAG & RULER ---
        painter.save()
        video_dur = self._total_video_duration_sec
        if video_dur > 0.1:
            eof_x_pixel = ((video_dur - view_start_time) / view_duration_time) * canvas_width
            if -100 <= eof_x_pixel <= canvas_width + 100:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                painter.setPen(QPen(QColor(255, 60, 60, 200), 2, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(eof_x_pixel, 0.0), QPointF(eof_x_pixel, canvas_height))
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

                painter.setFont(self._font_tags)
                tag_text = "END OF VIDEO"
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(tag_text)

                if eof_x_pixel + tw + 16.0 > canvas_width:
                    tag_rect = QRectF(eof_x_pixel - tw - 12.0, 4.0, tw + 8.0, fm.height() + 4.0)
                else:
                    tag_rect = QRectF(eof_x_pixel + 4.0, 4.0, tw + 8.0, fm.height() + 4.0)

                painter.setBrush(QColor(20, 20, 25, 200))
                painter.setPen(QPen(QColor(255, 60, 60, 150), 1))
                painter.drawRoundedRect(tag_rect, 3.0, 3.0)

                painter.setPen(QColor(255, 100, 100))
                painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, tag_text)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setFont(self._font_ruler)

        if view_duration_time <= 2.0:
            major_tick, minor_tick = 0.5, 0.1
        elif view_duration_time <= 10.0:
            major_tick, minor_tick = 1.0, 0.5
        elif view_duration_time <= 30.0:
            major_tick, minor_tick = 5.0, 1.0
        elif view_duration_time <= 120.0:
            major_tick, minor_tick = 10.0, 5.0
        elif view_duration_time <= 300.0:
            major_tick, minor_tick = 30.0, 10.0
        elif view_duration_time <= 900.0:
            major_tick, minor_tick = 60.0, 30.0
        elif view_duration_time <= 3600.0:
            major_tick, minor_tick = 300.0, 60.0
        else:
            major_tick, minor_tick = 600.0, 300.0

        first_minor = math.floor(view_start_time / minor_tick) * minor_tick
        current_tick = first_minor

        while current_tick <= view_start_time + view_duration_time:
            if current_tick >= view_start_time:
                x_px = int(((current_tick - view_start_time) / view_duration_time) * canvas_width)
                is_major = abs((current_tick / major_tick) - round(current_tick / major_tick)) < 0.01

                if is_major:
                    painter.setPen(self._pen_ruler_major)
                    painter.drawLine(x_px, int(canvas_height), x_px, int(canvas_height - 10))

                    ruler_text = self._format_ruler_time(current_tick)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                    painter.setPen(self._color_text_shadow)
                    painter.drawText(x_px + 4, int(canvas_height - 2), ruler_text)
                    painter.setPen(self._color_inactive_text)
                    painter.drawText(x_px + 3, int(canvas_height - 3), ruler_text)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                else:
                    painter.setPen(self._pen_ruler_minor)
                    painter.drawLine(x_px, int(canvas_height), x_px, int(canvas_height - 4))

            current_tick += minor_tick
        painter.restore()

        # --- 3. RENDER SUBTITLE BLOCKS ---
        painter.save()
        start_visible_idx, end_visible_idx = self._get_visible_event_range(view_start_time, view_start_time + view_duration_time)

        for draw_idx in range(start_visible_idx, end_visible_idx):
            event_object = self._loaded_subtitle_events[draw_idx]

            if self._mouse_drag_mode != DragMode.NONE and draw_idx == self._mouse_drag_event_idx:
                event_actual_start = self._mouse_drag_preview_start
                event_actual_end = self._mouse_drag_preview_end
            else:
                event_actual_start = event_object.start_sec
                event_actual_end = event_object.end_sec

            if event_actual_end < view_start_time or event_actual_start > view_start_time + view_duration_time:
                continue

            x_pixel_start = max(0.0, ((event_actual_start - view_start_time) / view_duration_time) * canvas_width)
            x_pixel_end = min(canvas_width, ((event_actual_end - view_start_time) / view_duration_time) * canvas_width)
            box_pixel_width = max(1.0, x_pixel_end - x_pixel_start)

            calculated_cps_val = len(event_object.text) / (event_actual_end - event_actual_start) if event_actual_end - event_actual_start > 0 else 0

            is_currently_active_row = (draw_idx == self._highlighted_active_row_idx)
            is_hovering = (draw_idx == self._hover_event_idx)

            if is_currently_active_row:
                determined_box_color = self._color_active_box
            elif calculated_cps_val > 25:
                determined_box_color = self._color_danger_box
            elif calculated_cps_val > 18:
                determined_box_color = self._color_warn_box
            else:
                determined_box_color = self._color_safe_box

            margin_y = 6.0
            box_rect_f = QRectF(x_pixel_start, margin_y, box_pixel_width, canvas_height - margin_y * 2)

            if box_pixel_width <= 3.0:
                painter.fillRect(box_rect_f, determined_box_color)
                if is_currently_active_row:
                    painter.setPen(self._pen_active_border)
                    painter.drawLine(QPointF(x_pixel_start, margin_y), QPointF(x_pixel_start, canvas_height - margin_y))
                continue

            painter.setBrush(determined_box_color)
            painter.setPen(self._pen_active_border if is_currently_active_row else self._pen_inactive_border)
            painter.drawRoundedRect(box_rect_f, 4.0, 4.0)

            if is_hovering and self._hover_mode == DragMode.BODY:
                painter.setBrush(self._color_hover_glow)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(box_rect_f, 4.0, 4.0)

            if is_hovering and self._hover_mode == DragMode.START:
                painter.setPen(self._pen_hover_border)
                painter.drawLine(QPointF(x_pixel_start, margin_y), QPointF(x_pixel_start, canvas_height - margin_y))
            elif is_hovering and self._hover_mode == DragMode.END:
                painter.setPen(self._pen_hover_border)
                painter.drawLine(QPointF(x_pixel_end, margin_y), QPointF(x_pixel_end, canvas_height - margin_y))

            if box_pixel_width > 35.0:
                cleaned_display_text = _cached_clean_text(event_object.text)
                if cleaned_display_text:
                    painter.setFont(self._font_text_active if is_currently_active_row else self._font_text_inactive)
                    text_bounding_rect = QRectF(x_pixel_start + 4.0, margin_y, box_pixel_width - 8.0, canvas_height - margin_y * 2)
                    elided_truncated_text = painter.fontMetrics().elidedText(cleaned_display_text, Qt.TextElideMode.ElideRight, int(text_bounding_rect.width()))

                    painter.setPen(self._color_text_shadow)
                    painter.drawText(text_bounding_rect.translated(1.0, 1.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided_truncated_text)

                    painter.setPen(self._color_text if is_currently_active_row else self._color_inactive_text)
                    painter.drawText(text_bounding_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided_truncated_text)
        painter.restore()

        # --- 4. RENDER PLAYHEAD & DRAG SNAPS ---
        painter.save()
        if self._magnetic_snap_line_x is not None:
            snap_px = ((self._magnetic_snap_line_x - view_start_time) / view_duration_time) * canvas_width
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setPen(self._pen_magnetic_snap)
            painter.drawLine(QPointF(snap_px, 0.0), QPointF(snap_px, canvas_height))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if view_start_time <= self._current_playback_time_sec <= view_start_time + view_duration_time:
            px = ((self._current_playback_time_sec - view_start_time) / view_duration_time) * canvas_width

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setPen(self._pen_playhead_shadow)
            painter.drawLine(QPointF(px, 0.0), QPointF(px, canvas_height))

            painter.setPen(self._pen_playhead)
            painter.drawLine(QPointF(px, 0.0), QPointF(px, canvas_height))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            poly = QPolygonF([
                QPointF(px - 6.0, 0.0),
                QPointF(px + 6.0, 0.0),
                QPointF(px, 8.0)
            ])
            painter.setBrush(self._color_playhead)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(poly)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            playhead_time_str = seconds_to_display(self._current_playback_time_sec)[:-4]
            painter.setFont(self._font_tags)
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(playhead_time_str)

            tooltip_x = max(2.0, min(px + 8.0, canvas_width - text_width - 10.0))

            tooltip_rect = QRectF(tooltip_x, 2.0, text_width + 8, fm.height() + 2)
            painter.setBrush(self._color_playhead_tooltip_bg)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(tooltip_rect, 3, 3)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(tooltip_rect, Qt.AlignmentFlag.AlignCenter, playhead_time_str)

        if self._mouse_drag_mode != DragMode.NONE and 0 <= self._mouse_drag_event_idx < len(self._loaded_subtitle_events):
            drag_dur = self._mouse_drag_preview_end - self._mouse_drag_preview_start
            info_text = f"{seconds_to_display(self._mouse_drag_preview_start)[:-4]} (Δ {drag_dur:.2f}s)"

            painter.setFont(self._font_tags)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(info_text)

            tip_x = self._cursor_current_x_px + 12
            if tip_x + tw + 10 > canvas_width:
                tip_x = self._cursor_current_x_px - tw - 12

            tip_y = canvas_height / 2.0
            t_rect = QRectF(tip_x, tip_y - 10, tw + 10, fm.height() + 4)

            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.drawRoundedRect(t_rect, 4, 4)
            painter.setPen(QColor(200, 255, 200))
            painter.drawText(t_rect, Qt.AlignmentFlag.AlignCenter, info_text)
        painter.restore()

        # --- 5. RENDER LOOP & DRAWING REGIONS ---
        painter.save()
        active_loop_start = self._loop_region_start_sec
        active_loop_end = self._loop_region_end_sec
        if active_loop_start is not None and active_loop_end is not None and active_loop_end > active_loop_start:
            if active_loop_end > view_start_time and active_loop_start < view_start_time + view_duration_time:
                lx_pixel_start = max(0.0, ((active_loop_start - view_start_time) / view_duration_time) * canvas_width)
                lx_pixel_end = min(canvas_width, ((active_loop_end - view_start_time) / view_duration_time) * canvas_width)

                loop_rect_f = QRectF(lx_pixel_start, 0.0, lx_pixel_end - lx_pixel_start, canvas_height)
                painter.fillRect(loop_rect_f, self._color_loop_bg)

                painter.setFont(self._font_tags)
                painter.setPen(self._pen_loop_start)
                painter.drawLine(QPointF(lx_pixel_start, 0.0), QPointF(lx_pixel_start, canvas_height))
                painter.drawText(QPointF(lx_pixel_start + 2.0, 12.0), "I")

                painter.setPen(self._pen_loop_end)
                painter.drawLine(QPointF(lx_pixel_end, 0.0), QPointF(lx_pixel_end, canvas_height))
                painter.drawText(QPointF(max(0.0, lx_pixel_end - 10.0), 12.0), "O")

        if getattr(self, '_is_user_drawing_new_sub', False):
            draw_start_time = min(self._user_draw_start_time, self._user_draw_end_time)
            draw_end_time = max(self._user_draw_start_time, self._user_draw_end_time)

            if draw_end_time > view_start_time and draw_start_time < view_start_time + view_duration_time:
                rx_pixel_start = ((draw_start_time - view_start_time) / view_duration_time) * canvas_width
                r_pixel_width = ((draw_end_time - draw_start_time) / view_duration_time) * canvas_width

                draw_rect_f = QRectF(max(0.0, rx_pixel_start), 0.0, max(1.0, r_pixel_width), canvas_height)
                painter.fillRect(draw_rect_f, self._color_draw_bg)
                painter.setPen(self._pen_draw_border)
                painter.drawRect(draw_rect_f)
        painter.restore()

        painter.end()

    def leaveEvent(self, event: QEvent) -> None:
        if self._mouse_drag_mode == DragMode.NONE:
            self._hover_event_idx = -1
            self._hover_mode = DragMode.NONE
            self._magnetic_snap_line_x = None
            
        self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if hasattr(self, 'play_toggle_requested'):
                self.play_toggle_requested.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        if self._view_timeline_duration_sec <= 0:
            return

        cursor_x_position = event.position().x()
        self._cursor_current_x_px = cursor_x_position

        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_middle_dragging = True
            self._last_middle_drag_x = cursor_x_position
            self._last_user_interaction_time = time.time()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        hit_test_result = self._hit_test_cursor(cursor_x_position)
        time_at_cursor_click = self._view_timeline_start_sec + (cursor_x_position / max(1, self.width())) * self._view_timeline_duration_sec
        self._mouse_drag_start_time = time_at_cursor_click

        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self._is_user_drawing_new_sub = True
            self._user_draw_start_time = time_at_cursor_click
            self._user_draw_end_time = time_at_cursor_click
            self._mouse_drag_mode = DragMode.NONE
            self._edge_pan_timer.start()
            return

        if hit_test_result:
            event_idx, hit_zone_type = hit_test_result
            event_object = self._loaded_subtitle_events[event_idx]

            if hit_zone_type in (DragMode.START, DragMode.END):
                self._mouse_drag_mode = hit_zone_type
            else:
                self._mouse_drag_mode = DragMode.MOVE

            self._mouse_drag_event_idx = event_idx
            self._mouse_drag_original_start = event_object.start_sec
            self._mouse_drag_original_end = event_object.end_sec
            self._mouse_drag_preview_start = event_object.start_sec
            self._mouse_drag_preview_end = event_object.end_sec

            if hit_zone_type == DragMode.BODY:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self._mouse_drag_mode = DragMode.SCRUB
            self.seek_requested.emit(time_at_cursor_click)

        if self._mouse_drag_mode != DragMode.NONE:
            self._edge_pan_timer.start()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cursor_x_position = event.position().x()
        self._cursor_current_x_px = cursor_x_position
        self._magnetic_snap_line_x = None

        if getattr(self, '_is_middle_dragging', False):
            self._last_user_interaction_time = time.time()
            delta_px = cursor_x_position - getattr(self, '_last_middle_drag_x', cursor_x_position)
            delta_sec = (delta_px / max(1, self.width())) * self._view_timeline_duration_sec

            new_start = self._view_timeline_start_sec - delta_sec
            safe_bounds = self._timeline_max_bounds
            max_allowable_start = max(0.0, safe_bounds - self._view_timeline_duration_sec)

            self._view_timeline_start_sec = max(0.0, min(max_allowable_start, new_start))
            self._last_middle_drag_x = cursor_x_position
            self.update()
            return

        if self._mouse_drag_mode != DragMode.NONE or getattr(self, '_is_user_drawing_new_sub', False):
            self._last_user_interaction_time = time.time()
            self._update_drag_action()
            self.update()
            return

        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self.setCursor(Qt.CursorShape.IBeamCursor)
            if self._hover_event_idx != -1:
                self._hover_event_idx, self._hover_mode = -1, DragMode.NONE
                self.update()
        else:
            hit_test_result = self._hit_test_cursor(cursor_x_position)
            new_hover_idx = hit_test_result[0] if hit_test_result else -1
            new_hover_mode = hit_test_result[1] if hit_test_result else DragMode.NONE

            if new_hover_idx != self._hover_event_idx or new_hover_mode != self._hover_mode:
                self._hover_event_idx = new_hover_idx
                self._hover_mode = new_hover_mode
                self.update()

            if hit_test_result is None:
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif hit_test_result[1] in (DragMode.START, DragMode.END):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._edge_pan_timer.stop()

        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_middle_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if getattr(self, '_is_user_drawing_new_sub', False):
                self._is_user_drawing_new_sub = False
                draw_start_time = min(self._user_draw_start_time, self._user_draw_end_time)
                draw_end_time = max(self._user_draw_start_time, self._user_draw_end_time)

                if draw_end_time - draw_start_time >= 0.1:
                    self.subtitle_create_requested.emit(draw_start_time, draw_end_time)
                self.update()
                return

            if self._mouse_drag_mode == DragMode.SCRUB:
                self._mouse_drag_mode = DragMode.NONE
                return

            if self._mouse_drag_mode != DragMode.NONE and 0 <= self._mouse_drag_event_idx < len(self._loaded_subtitle_events):
                new_s = self._mouse_drag_preview_start
                new_e = self._mouse_drag_preview_end
                old_s = self._mouse_drag_original_start
                old_e = self._mouse_drag_original_end

                if abs(new_s - old_s) > 0.001 or abs(new_e - old_e) > 0.001:
                    self.subtitle_drag_done.emit(self._mouse_drag_event_idx, old_s, old_e, new_s, new_e)

                self._mouse_drag_mode = DragMode.NONE
                self._mouse_drag_event_idx = -1
                self._magnetic_snap_line_x = None
                self.update()

    def _zoom_at_center(self, factor: float, safe_bounds: float) -> None:
        center_time = self._view_timeline_start_sec + (self._view_timeline_duration_sec / 2.0)
        new_calculated_duration = self._view_timeline_duration_sec * factor
        new_calculated_duration = max(0.5, min(safe_bounds, new_calculated_duration))
        self._view_timeline_duration_sec = new_calculated_duration

        new_start = center_time - (new_calculated_duration / 2.0)
        max_allowable_start = max(0.0, safe_bounds - new_calculated_duration)
        
        self._scroll_anim.stop()
        self._view_timeline_start_sec = max(0.0, min(max_allowable_start, new_start))
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        k = event.key()
        mods = event.modifiers()
        safe_bounds = self._timeline_max_bounds

        if k == Qt.Key.Key_Left:
            self._smooth_scroll_by(- (10.0 / 100.0) * self._view_timeline_duration_sec)
        elif k == Qt.Key.Key_Right:
            self._smooth_scroll_by((10.0 / 100.0) * self._view_timeline_duration_sec)
        elif k == Qt.Key.Key_PageUp:
            self._smooth_scroll_by(-self._view_timeline_duration_sec)
        elif k == Qt.Key.Key_PageDown:
            self._smooth_scroll_by(self._view_timeline_duration_sec)
        elif k == Qt.Key.Key_Home:
            self._smooth_scroll_to(0.0)
        elif k == Qt.Key.Key_End:
            self._smooth_scroll_to(max(0.0, safe_bounds - self._view_timeline_duration_sec))
        elif k == Qt.Key.Key_0 and (mods & Qt.KeyboardModifier.ControlModifier):
            self._view_timeline_duration_sec = max(0.5, safe_bounds)
            self._smooth_scroll_to(0.0)
        elif k in (Qt.Key.Key_Equal, Qt.Key.Key_Plus) and (mods & Qt.KeyboardModifier.ControlModifier):
            self._zoom_at_center(1.0 / 1.5, safe_bounds)
        elif k == Qt.Key.Key_Minus and (mods & Qt.KeyboardModifier.ControlModifier):
            self._zoom_at_center(1.5, safe_bounds)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._last_user_interaction_time = time.time()
        delta = event.angleDelta().y() or event.angleDelta().x()
        safe_bounds = self._timeline_max_bounds

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            mouse_fraction_x = max(0.0, min(1.0, event.position().x() / max(1, self.width())))
            time_at_cursor_hover = self._view_timeline_start_sec + self._view_timeline_duration_sec * mouse_fraction_x

            zoom_factor = 1.15 if delta < 0 else (1.0 / 1.15)
            new_calculated_duration = self._view_timeline_duration_sec * zoom_factor

            new_calculated_duration = max(0.5, min(safe_bounds, new_calculated_duration))
            self._view_timeline_duration_sec = new_calculated_duration

            new_start = time_at_cursor_hover - new_calculated_duration * mouse_fraction_x
            max_allowable_start = max(0.0, safe_bounds - new_calculated_duration)

            self._scroll_anim.stop()
            self._view_timeline_start_sec = max(0.0, min(max_allowable_start, new_start))
            self.update()
        else:
            speed_multiplier = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            shift_amount_sec = -(delta / 120.0) * (self._view_timeline_duration_sec * 0.1) * speed_multiplier
            self._smooth_scroll_by(shift_amount_sec, duration=150)

__all__ = ["AudioWaveformWidget"]
