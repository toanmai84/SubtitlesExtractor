"""ViewModel cho trang "Chỉnh sửa phụ đề".

BẢN CẬP NHẬT ĐỘT PHÁ (V3.47 - The "Asynchronous" Polish):
    * [CRITICAL BUG FIX] Asynchronous Export: Sửa lỗi thi thoảng treo ứng dụng 
      (Not Responding) khi Lưu file phụ đề. Quá trình Ghi ổ cứng (I/O) nay được 
      đẩy hoàn toàn xuống Luồng chạy nền (Background Thread), đảm bảo UI luôn đạt 144 FPS.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, QTimer

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.device_kind import (
    DeviceKind,
    PrecisionMode,
    SubtitleFormat,
)
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.presentation.workers.qt_progress_reporter import (
    QtProgressReporter,
)

logger = logging.getLogger(__name__)

class _AutoSaveRunnable(QRunnable):
    def __init__(self, repo, path_str: str, events: list):
        super().__init__()
        self.repo = repo
        self.path_str = path_str
        self.events = events

    def run(self):
        try:
            self.repo.save_events(self.path_str, self.events)
        except (sqlite3.Error, OSError, ValueError) as exc:
            logger.warning("Lỗi auto-save Database: %s", exc)


# ---------------------------------------------------------------------------
# Export worker — PyQt6 object lifetime notes
# ---------------------------------------------------------------------------
# QUAN TRỌNG (v3.6 bugfix):
#   QRunnable không kế thừa QObject. Khi QThreadPool.start(worker) với
#   autoDelete=True (mặc định), sau khi run() kết thúc Qt xóa C++ wrapper.
#   PyQt6 giải phóng tham chiếu Python → refcount về 0 → Python GC chạy
#   → phá huỷ `self.signals` (_ExportSignals QObject).
#   Qt phát hiện sender QObject đã chết → huỷ toàn bộ queued connection
#   chưa được xử lý → _on_export_worker_success KHÔNG BAO GIỜ được gọi
#   → _is_busy kẹt True mãi mãi → **TREO GIAO DIỆN**.
#
# Giải pháp: EditorPageViewModel giữ strong Python reference (`_export_worker`)
# cho đến khi slot success/error được gọi, kết hợp setAutoDelete(False).
# ---------------------------------------------------------------------------


class _ExportSignals(QObject):
    """Tín hiệu cho luồng export bất đồng bộ."""

    success = Signal(Path)
    error = Signal(str)


class _ExportRunnable(QRunnable):
    """Runnable ghi file phụ đề trên background thread.

    autoDelete phải là False — ViewModel giữ strong reference để
    ngăn Python GC huỷ signals trước khi queued event được deliver.
    """

    def __init__(self, use_case, events: list, output_path: Path, output_format) -> None:
        super().__init__()
        self.setAutoDelete(False)   # ← CRITICAL: ViewModel quản lý lifetime
        self.use_case = use_case
        self.events = events
        self.output_path = output_path
        self.output_format = output_format
        self.signals = _ExportSignals()  # Tạo trên main thread

    def run(self) -> None:
        """Thực thi ghi file trên background thread.

        Xử lý exception chi tiết để user thấy thông báo lỗi rõ ràng thay vì
        chỉ thấy exception class name.
        """
        try:
            saved_path = self.use_case.execute(
                events=self.events,
                output_path=self.output_path,
                output_format=self.output_format,
            )
            self.signals.success.emit(saved_path)

        except PermissionError as exc:
            # Windows: file đang mở bởi app khác (VLC, Notepad, v.v.)
            self.signals.error.emit(
                f"Không có quyền ghi file — có thể file đang mở bởi ứng dụng khác.\n"
                f"Chi tiết: {exc}"
            )
        except FileNotFoundError as exc:
            # Thư mục đích bị xóa giữa chừng
            self.signals.error.emit(
                f"Thư mục đích không tồn tại. Vui lòng chọn lại đường dẫn.\n"
                f"Chi tiết: {exc}"
            )
        except OSError as exc:
            # Lỗi I/O chung: đĩa đầy, thiết bị offline, quyền truy cập, ...
            import errno as _errno
            if exc.errno == _errno.ENOSPC:
                self.signals.error.emit(
                    "Đĩa đầy — không thể ghi file. Vui lòng giải phóng không gian đĩa."
                )
            elif exc.errno == _errno.EROFS:
                self.signals.error.emit(
                    "Ổ đĩa ở chế độ chỉ đọc (read-only). Không thể ghi file."
                )
            else:
                self.signals.error.emit(
                    f"Lỗi I/O khi ghi file: {exc}\n"
                    f"Kiểm tra: quyền ghi, ổ đĩa còn chỗ, đường dẫn hợp lệ."
                )
        except Exception as exc:  # noqa: BLE001
            # Fallback cho các lỗi không lường trước
            self.signals.error.emit(str(exc))


class EditorPageViewModel(QObject):
    state_changed = Signal(object)
    error_occurred = Signal(str)
    export_finished = Signal(object)
    progress_changed = Signal(int, int, str)
    busy_changed = Signal(bool)

    # Phát khi Re-OCR hoàn tất (thành công hay thất bại) để View clear waveform region.
    reocr_region_should_clear = Signal()

    def __init__(self, container: ApplicationContainer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._container = container
        self._service: SubtitleEditorService = container.make_subtitle_editor_service()
        self._is_busy = False
        self._current_video_path: Path | None = None
        self._thread_pool = QThreadPool.globalInstance()
        # Strong reference giữ _ExportRunnable sống cho đến khi signal delivery xong.
        # Xem comment tại _ExportRunnable về lý do cần thiết.
        self._export_worker: _ExportRunnable | None = None
        # Safety watchdog: force-reset busy state nếu signal bị drop sau 60s.
        self._export_watchdog = QTimer(self)
        self._export_watchdog.setSingleShot(True)
        self._export_watchdog.setInterval(60_000)  # 60 giây
        self._export_watchdog.timeout.connect(self._on_export_watchdog_timeout)

    @property
    def current_events(self):
        return self._service.fast_snapshot().events

    def load_from_file(self, source_path: Path) -> None:
        """Nạp file phụ đề vào editor.

        [v3.6 bugfix LF-1]: Thêm UnicodeDecodeError và OSError vào bắt ngoại lệ.
        Trước đây chỉ bắt 3 loại → UnicodeDecodeError (file không phải UTF-8)
        và OSError (file bị khoá/xoá) lọt qua thành unhandled exception crash.

        [v3.6 bugfix KE-1]: KeyError.__str__() trả `"'message'"` (có dấu nháy
        đơn bao quanh) do Python repr. Dùng exc.args[0] để lấy chuỗi sạch.

        [v3.20.3 #2]: Tự hủy luồng Re-OCR cũ trước khi nạp phụ đề mới, chống
        nhiễm chéo (kết quả video cũ chép đè dữ liệu mới).
        """
        self._cancel_stale_reocr_on_new_data()
        try:
            use_case = self._container.make_import_subtitles_use_case()
            events = use_case.execute(source_path)
        except KeyError as exc:
            # KeyError.__str__() → "'message'" (có dấu nháy) — lấy args[0] cho sạch
            self.error_occurred.emit(exc.args[0] if exc.args else str(exc))
            return
        except (FileNotFoundError, ValueError) as exc:
            self.error_occurred.emit(str(exc))
            return
        except UnicodeDecodeError as exc:
            self.error_occurred.emit(
                f"File không phải UTF-8. Vui lòng chuyển encoding sang UTF-8 trước.\n"
                f"Chi tiết: {exc}"
            )
            return
        except OSError as exc:
            self.error_occurred.emit(
                f"Không thể đọc file: {exc}"
            )
            return
        self.state_changed.emit(self._service.load(events))

    def load_from_events(self, events) -> None:
        self._cancel_stale_reocr_on_new_data()
        self.state_changed.emit(self._service.load(events))

    def set_current_video(self, video_path: Path):
        # [v3.20.3 #2] Đổi video → hủy Re-OCR cũ để kết quả không chép nhầm sang
        # video mới (cross-video contamination).
        if getattr(self, "_current_video_path", None) != video_path:
            self._cancel_stale_reocr_on_new_data()
        self._current_video_path = video_path

    def _cancel_stale_reocr_on_new_data(self) -> None:
        """[v3.20.3 #2] Hủy luồng Re-OCR đang chạy khi người dùng nạp dữ liệu mới.

        Yêu cầu hủy mềm qua reporter; nếu thread còn sống thì ngắt tín hiệu để
        callback không bắn về widget/dữ liệu đã thay (tránh nhiễm chéo + segfault).
        """
        if getattr(self, "_reocr_reporter", None):
            self._reocr_reporter.request_cancel()
        thread = getattr(self, "_reocr_thread", None)
        if thread is not None and thread.isRunning():
            logger.info("Nạp dữ liệu mới → ngắt luồng Re-OCR cũ chống nhiễm chéo.")
            with contextlib.suppress(RuntimeError, TypeError):
                thread.requestInterruption()

    def update_text(self, index: int, new_text: str) -> None:
        self._dispatch(lambda: self._service.update_text(index, new_text))

    def batch_replace_text(self, replacements: dict[int, str]) -> None:
        self._dispatch(lambda: self._service.batch_update_text(replacements))

    def update_timing(self, index: int, start_sec: float, end_sec: float) -> None:
        self._dispatch(lambda: self._service.update_timing(index, start_sec, end_sec))

    def insert_after(self, index: int) -> None:
        self._dispatch(lambda: self._service.insert_after(index))

    def delete_event(self, index: int) -> None:
        self._dispatch(lambda: self._service.delete(index))

    def batch_delete_events(self, indices: list[int]) -> None:
        self._dispatch(lambda: self._service.batch_delete(indices))

    def split_event(self, index: int, at_sec: float) -> None:
        self._dispatch(lambda: self._service.split(index, at_sec))

    def merge_with_next(self, index: int) -> None:
        self._dispatch(lambda: self._service.merge_with_next(index))

    def shift_all(self, offset_sec: float) -> None:
        self._dispatch(lambda: self._service.shift_all(offset_sec))

    def strip_tags(self, index: int, text: str) -> None:
        from subtitles_extractor.application.services.subtitle_editor_service import (
            strip_formatting_tags,
        )

        self.update_text(index, strip_formatting_tags(text))

    def auto_fix_timeline(self) -> int:
        try:
            fixes = self._service.auto_fix_timeline()
            if fixes > 0:
                self.state_changed.emit(self._service.fast_snapshot())
            return fixes
        except (ConfigurationError, IndexError) as exc:
            self.error_occurred.emit(str(exc))
            return 0

    def find_similar_groups(self, max_gap_sec: float, min_sim: float) -> list[list[int]]:
        return self._service.find_similar_groups(max_gap_sec, min_sim)

    def apply_merge_groups(self, groups: list[list[int]]) -> int:
        try:
            applied = self._service.apply_merge_groups(groups)
            if applied > 0:
                self.state_changed.emit(self._service.fast_snapshot())
            return applied
        except (ConfigurationError, IndexError) as exc:
            self.error_occurred.emit(str(exc))
            return 0

    def undo(self) -> None:
        self.state_changed.emit(self._service.undo())

    def redo(self) -> None:
        self.state_changed.emit(self._service.redo())

    def save_autosave(self) -> None:
        events = self.current_events
        if not events or not self._current_video_path:
            return

        path_str = str(self._current_video_path.resolve())
        events_snapshot = list(events)

        worker = _AutoSaveRunnable(self._container.subtitle_repository, path_str, events_snapshot)
        self._thread_pool.start(worker)

    def export_to_file(self, output_path: Path, output_format: SubtitleFormat) -> None:
        """Ghi file phụ đề trên background thread — không block UI.

        [v3.6 bugfix] Lưu strong Python reference đến worker để ngăn PyQt6
        GC huỷ _ExportSignals trước khi queued signal được main-thread xử lý.

        Returns:
            True  — export đã được bắt đầu thành công.
            False — bị từ chối do hệ thống đang bận (_is_busy=True).
                    Caller có trách nhiệm khôi phục UI nếu cần.
        """
        if self._is_busy:
            # [v3.6 bugfix EXPORT-FIX-2]: Trả về False để caller biết export bị reject.
            # Trước đây trả về None (implicit) → caller không phân biệt được
            # "đã bắt đầu" vs "bị từ chối" → UI bị kẹt nếu caller đã disable nút.
            return False

        self._set_busy(True)
        events = self._service.fast_snapshot().events
        use_case = self._container.make_export_subtitles_use_case()

        worker = _ExportRunnable(use_case, events, output_path, output_format)
        worker.signals.success.connect(self._on_export_worker_success)
        worker.signals.error.connect(self._on_export_worker_error)

        # Giữ strong reference — QUAN TRỌNG: ngăn GC huỷ worker.signals.
        self._export_worker = worker
        # Khởi động watchdog 60s phòng signal drop không thể khôi phục.
        self._export_watchdog.start()
        self._thread_pool.start(worker)
        return True

    def _on_export_worker_success(self, saved_path: Path) -> None:
        """Xử lý export thành công.

        [v3.6 bugfix EXPORT-FIX-1]: _set_busy(False) phải được gọi TRƯỚC
        export_finished.emit(). Nguyên nhân treo:
          Cũ: state_changed.emit() [is_busy=True] → button disabled
              export_finished.emit() → _on_export_finished() → setEnabled(True)  ← SỚM!
              _set_busy(False) → on_reocr_busy → setEnabled(True)
        Vấn đề: Nếu _on_export_finished chạy trong khi _is_busy=True, nút được bật
        TRƯỚC KHI busy state được cập nhật. Trong window này, nếu user click, _is_busy
        vẫn True → export_to_file() trả về sớm → nút kẹt disable → TREO.
        Fix: đặt _set_busy(False) trước export_finished để trình tự luôn nhất quán.
        """
        self._export_watchdog.stop()
        self._export_worker = None
        self.save_autosave()
        self.state_changed.emit(self._service.mark_clean())
        # [v3.6 bugfix EXPORT-FIX-1]: Reset busy TRƯỚC khi emit export_finished.
        # Khi _on_export_finished gọi setEnabled(True), _is_busy đã là False → đúng.
        self._set_busy(False)
        self.export_finished.emit(saved_path)

    def _on_export_worker_error(self, err_msg: str) -> None:
        # Dừng watchdog và giải phóng reference.
        self._export_watchdog.stop()
        self._export_worker = None
        self.error_occurred.emit(f"Lỗi khi xuất tệp: {err_msg}")
        self._set_busy(False)

    def _on_export_watchdog_timeout(self) -> None:
        """Safety net: force-reset nếu signal export bị drop sau 60 giây.

        Tình huống: QRunnable bị GC → _ExportSignals bị huỷ → queued signal
        bị Qt huỷ → slot không bao giờ được gọi → UI bị treo vĩnh viễn.
        Watchdog phát hiện và phục hồi trạng thái.
        """
        if self._is_busy and self._export_worker is not None:
            logger.warning(
                "Export watchdog: signal không được deliver sau 60s — "
                "force-reset busy state để khôi phục UI."
            )
            self._export_worker = None
            self.error_occurred.emit(
                "Xuất file mất quá nhiều thời gian hoặc gặp lỗi nội bộ. "
                "Vui lòng thử lại."
            )
            self._set_busy(False)

    def start_reocr(
        self, video_path: Path, rows_to_replace: list[int], roi: Roi | None,
        tweaks: dict[str, object] | None = None, model_override: str | None = None,
        merge_window_sec: float = 1.0, persist_roi: bool = True,
        explicit_time_ranges: list | None = None,
    ) -> None:
        """Khởi động Re-OCR trên background thread.

        Args:
            video_path:            Đường dẫn video.
            rows_to_replace:       List row index của các event cần thay thế.
            roi:                   ROI quét. None = full frame.
            tweaks:                Cấu hình override tham số OCR/preprocess.
            model_override:        Tên model override ("PP-OCRv5_server", v.v.).
            merge_window_sec:      Khoảng merge time ranges gần nhau.
            persist_roi:           Có lưu ROI vào VideoState DB không.
            explicit_time_ranges:  [v3.6 bugfix R2/R4] Nếu được cung cấp, dùng
                                   trực tiếp làm time_ranges thay vì build từ
                                   event timing. Giúp loại bỏ việc phải gọi
                                   update_timing() trước Re-OCR (tránh orphaned
                                   undo entries và stale-ref revert bug).
        """
        if getattr(self, "_is_busy", False):
            return

        if not rows_to_replace:
            self.error_occurred.emit("Vui lòng chọn ít nhất một dòng để Re-OCR.")
            return

        events = self._service.snapshot_state().events
        rows_sorted = sorted(set(rows_to_replace))
        try:
            target_events = [events[row] for row in rows_sorted]
        except IndexError as exc:
            self.error_occurred.emit(f"Lỗi truy xuất dữ liệu dòng: {exc}")
            return

        replace_uids = [event.uid for event in target_events]

        # [v3.6 bugfix R2/R4]: Nếu explicit_time_ranges được cung cấp (từ
        # caller đã tính sẵn vùng quét), dùng trực tiếp. Cách này loại bỏ sự
        # cần thiết của update_timing() trước Re-OCR — tránh:
        #   • Bug R2: stale events ref khi single-row expand cả start lẫn end
        #   • Bug R4: orphaned timing changes khi Re-OCR không tìm thấy gì
        if explicit_time_ranges is not None:
            time_ranges = explicit_time_ranges
        else:
            time_ranges = self._build_time_ranges(target_events)

        if not time_ranges:
            self.error_occurred.emit("Thời lượng chọn không hợp lệ để phân tích OCR.")
            return

        # [v3.6 bugfix R3]: Đọc metadata TRƯỚC khi lưu ROI.
        # Trước đây lệnh ghi ROI vào DB được gọi trước metadata read →
        # nếu video lỗi/không đọc được, ROI đã bị persist sai vào DB.
        try:
            metadata = self._container.metadata_reader.read(video_path)
        except (OSError, ValueError, RuntimeError) as exc:
            self.error_occurred.emit(f"Lỗi đọc Video Metadata: {exc}")
            return

        try:
            request = self._build_reocr_request(
                video_path=video_path, time_ranges=time_ranges, replace_uids=replace_uids,
                roi=roi, tweaks=tweaks or {}, model_override=model_override, merge_window_sec=merge_window_sec,
            )
        except (ValueError, KeyError, TypeError) as exc:
            self.error_occurred.emit(f"Lỗi cấu hình thông số OCR: {exc}")
            return

        if request.total_duration_sec > metadata.duration_sec * 1.05:
            self.error_occurred.emit("Khoảng Re-OCR vượt quá thời lượng video cho phép.")
            return

        # Lưu ROI chỉ khi request đã được validate thành công.
        if persist_roi:
            self._save_roi_to_video_state(video_path, roi)

        self._set_busy(True)
        self._launch_reocr_thread(request)

    @staticmethod
    def _build_time_ranges(target_events) -> list:
        from subtitles_extractor.application.dtos.reocr_dto import TimeRange
        ranges = []
        for event in target_events:
            if event.duration_sec <= 0:
                continue
            with contextlib.suppress(ValueError):
                ranges.append(TimeRange(start_sec=event.start_sec, end_sec=event.end_sec))
        return ranges

    def _save_roi_to_video_state(self, video_path: Path, roi: Roi | None) -> None:
        import dataclasses
        try:
            from subtitles_extractor.domain.entities.video_state import VideoState
            path_str = str(video_path.resolve())
            existing = self._container.video_state_repository.get(path_str)
            if existing is not None:
                new_state = dataclasses.replace(existing, roi=roi)
            else:
                new_state = VideoState(video_path=path_str, roi=roi)
            self._container.video_state_repository.save(new_state)
        except (OSError, RuntimeError, ValueError, AttributeError) as exc:
            logger.warning("Không lưu được VideoState (ROI): %s.", exc)

    def _build_reocr_request(
        self, video_path: Path, time_ranges: list, replace_uids: list[str],
        roi: Roi | None, tweaks: dict[str, object], model_override: str | None, merge_window_sec: float,
    ):
        from subtitles_extractor.application.dtos.reocr_dto import ReOcrRequest
        snapshot = self._container.settings_service.current
        post = snapshot.post_process
        from subtitles_extractor.application.services.ocr_model_resolver import (
            default_models_for_version,
            resolve_ocr_model_names,
        )
        from subtitles_extractor.domain.ports.ocr_engine_port import PreprocessConfig

        # [v3.23.104] Khi Studio chỉ định version (model_override), map ĐÚNG qua nguồn sự
        # thật chung (hỗ trợ PP-OCRv6); không còn rơi về PP-OCRv5 như bug trước.
        if model_override:
            det_model, rec_model = default_models_for_version(model_override)
        else:
            det_model, rec_model = resolve_ocr_model_names(snapshot.ocr)

        from importlib.util import find_spec
        has_cupy = find_spec("cupy") is not None
        is_nvdec = snapshot.hardware.frame_decoder_backend == "pynvvideocodec" and has_cupy

        vram_upscale = bool(tweaks.get("upscale", snapshot.preprocess.upscale_small_text)) if is_nvdec else False
        vram_border = bool(tweaks.get("border", snapshot.preprocess.add_white_border)) if is_nvdec else False
        vram_sharpen = bool(tweaks.get("sharpen", snapshot.preprocess.apply_sharpen)) if is_nvdec else False
        vram_contrast_factor = float(tweaks.get("contrast_factor", snapshot.preprocess.contrast_factor)) if (is_nvdec and tweaks.get("contrast", snapshot.preprocess.apply_contrast_boost)) else 1.0

        cpu_upscale = bool(tweaks.get("upscale", snapshot.preprocess.upscale_small_text)) if not is_nvdec else False
        cpu_border = bool(tweaks.get("border", snapshot.preprocess.add_white_border)) if not is_nvdec else False
        cpu_sharpen = bool(tweaks.get("sharpen", snapshot.preprocess.apply_sharpen)) if not is_nvdec else False
        cpu_contrast = bool(tweaks.get("contrast", snapshot.preprocess.apply_contrast_boost)) if not is_nvdec else False

        sampling_cfg = FrameSamplingConfig(
            sample_step_sec=snapshot.frame.sample_step_sec, phash_distance_threshold=snapshot.frame.phash_distance,
            pixel_diff_threshold=snapshot.frame.pixel_diff_ratio, skip_intro_sec=snapshot.frame.skip_intro_sec, skip_outro_sec=snapshot.frame.skip_outro_sec,
            apply_median_blend=bool(tweaks.get("median_blend", snapshot.preprocess.apply_median_blend)),
            median_blend_frames=snapshot.preprocess.median_blend_frames,
            vram_upscale_small_text=vram_upscale, vram_upscale_target_height_px=int(tweaks.get("upscale_h", snapshot.preprocess.upscale_target_height_px)),
            vram_add_border=vram_border, vram_border_thickness_px=int(tweaks.get("border", snapshot.preprocess.border_thickness_px)),
            vram_sharpen=vram_sharpen, vram_contrast_factor=vram_contrast_factor,
        )

        min_conf = float(tweaks.get("min_conf", snapshot.threshold.ocr_min_confidence))
        pre_cfg = PreprocessConfig(
            upscale_small_text=cpu_upscale,
            upscale_target_height_px=int(tweaks.get("upscale_h", snapshot.preprocess.upscale_target_height_px)),
            add_white_border=cpu_border,
            border_thickness_px=int(tweaks.get("border", snapshot.preprocess.border_thickness_px)),
            apply_clahe=bool(tweaks.get("clahe", snapshot.preprocess.apply_clahe)),
            clahe_clip_limit=float(tweaks.get("clahe_clip", snapshot.preprocess.clahe_clip_limit)),
            clahe_tile_size=int(tweaks.get("clahe_tile", snapshot.preprocess.clahe_tile_size)),
            apply_sharpen=cpu_sharpen, apply_contrast_boost=cpu_contrast,
            contrast_factor=float(tweaks.get("contrast_factor", snapshot.preprocess.contrast_factor)),
        )

        ocr_cfg = OcrEngineConfig(
            device=DeviceKind(snapshot.hardware.device.value),
            detection_model_name=det_model,
            recognition_model_name=rec_model,
            language=snapshot.ocr.language,
            limit_side_len=int(tweaks.get("limit_side_len", snapshot.ocr.limit_side_len)),
            limit_type=snapshot.ocr.limit_type,
            det_thresh=float(tweaks.get("det_thresh", snapshot.ocr.det_thresh)),
            det_box_thresh=float(tweaks.get("box_thresh", snapshot.ocr.det_box_thresh)),
            det_unclip_ratio=float(tweaks.get("det_unclip_ratio", snapshot.ocr.det_unclip_ratio)),
            use_doc_orientation_classify=snapshot.ocr.use_doc_orientation_classify,
            use_doc_unwarping=snapshot.ocr.use_doc_unwarping,
            use_textline_orientation=snapshot.ocr.use_textline_orientation,
            score_threshold=min_conf,
            enable_mkldnn=snapshot.hardware.enable_mkldnn,
            use_tensorrt=snapshot.hardware.use_tensorrt,
            precision=PrecisionMode(snapshot.hardware.precision.value),
            batch_size=snapshot.hardware.batch_size_ocr,
            parallel_workers=snapshot.hardware.workers,
            preprocess=pre_cfg,
        )

        builder_cfg = SubtitleBuilderConfig(
            similarity_threshold=min(post.similarity_threshold, snapshot.threshold.text_similarity),
            min_duration_sec=post.min_duration_sec, max_duration_sec=post.max_duration_sec,
            merge_gap_sec=post.merge_gap_sec, min_confidence=min_conf,
            use_viterbi=post.use_viterbi, viterbi_open_penalty=post.viterbi_open_penalty,
            min_text_chars=snapshot.threshold.drop_short_text_chars, line_similarity_threshold=snapshot.threshold.line_similarity,
            sample_step_sec=snapshot.frame.sample_step_sec, temporal_padding_sec=post.temporal_padding_sec,
            y_clustering_tolerance_ratio=post.y_clustering_tolerance_ratio, y_clustering_tolerance_min_px=post.y_clustering_tolerance_min_px,
            alignment_center_tolerance_ratio=post.alignment_center_tolerance_ratio, alignment_margin_tolerance_ratio=post.alignment_margin_tolerance_ratio,
            alignment_tolerance_min_px=post.alignment_tolerance_min_px
        )

        return ReOcrRequest(
            video_path=video_path, time_ranges=time_ranges, replace_uids=replace_uids, roi=roi, sampling=sampling_cfg,
            ocr=ocr_cfg, builder=builder_cfg, auto_tune_batch=snapshot.hardware.auto_tune_batch, save_debug_frames=snapshot.advanced.save_debug_frames,
            debug_frames_dir=snapshot.advanced.debug_frames_dir, merge_window_sec=merge_window_sec,
        )

    def _launch_reocr_thread(self, request) -> None:
        from PySide6.QtCore import QThread
        from subtitles_extractor.presentation.workers.reocr_worker import ReOcrWorker

        reporter = QtProgressReporter()
        reporter.progress_changed.connect(self.progress_changed)
        # [BUG FIX v2.9+]: QThread(self) để ViewModel là parent — tránh memory leak,
        # đồng nhất với cách dùng trong ExtractPageViewModel.
        thread = QThread(self)
        worker = ReOcrWorker(use_case=self._container.make_reocr_use_case(), request=request, reporter=reporter)

        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        worker.finished.connect(self._on_reocr_finished)
        worker.failed.connect(self._on_reocr_failed)

        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)

        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_reocr_thread_finished)

        self._reocr_thread = thread
        self._reocr_worker = worker
        self._reocr_reporter = reporter
        thread.start()

    def _on_reocr_finished(self, response) -> None:
        # [Cross-Thread Contamination] Bỏ qua kết quả đến muộn từ worker Re-OCR cũ.
        if self.sender() is not self._reocr_worker:
            return
        # [v3.7 bugfix REOCR-CANCEL]: Nếu người dùng huỷ giữa chừng, KHÔNG áp dụng
        # thay thế. Lý do: replaced_uids chứa TẤT CẢ uid được chọn, nhưng new_events
        # chỉ có kết quả của các range đã quét xong. Áp dụng sẽ xóa phụ đề thuộc các
        # range CHƯA quét mà không có bản thay thế → mất dữ liệu gốc. Bỏ qua phần
        # việc dở dang là chấp nhận được; mất phụ đề gốc thì không.
        if getattr(response, "was_cancelled", False):
            self.progress_changed.emit(100, 100, "Đã huỷ Re-OCR — giữ nguyên phụ đề gốc.")
            return

        if not response.new_events:
            self.progress_changed.emit(100, 100, "Re-OCR hoàn tất: KHÔNG tìm thấy đoạn Text nào đủ chất lượng.")
            return

        try:
            state = self._service.replace_events_by_uid(
                uids_to_remove=response.replaced_uids, new_events=response.new_events,
                description=f"Re-OCR ({len(response.replaced_uids)} → {len(response.new_events)} câu)"
            )
        except (ValueError, KeyError, IndexError, RuntimeError, AttributeError) as exc:
            self.error_occurred.emit(f"Lỗi khi áp dụng kết quả Re-OCR vào Bảng dữ liệu: {exc}.")
            return

        self.state_changed.emit(state)
        self.progress_changed.emit(100, 100, f"Re-OCR hoàn tất trong {response.elapsed_seconds:.1f}s.")

    def _on_reocr_failed(self, message: str) -> None:
        self.error_occurred.emit(f"Tiến trình Re-OCR thất bại: {message}")

    def _on_reocr_thread_finished(self) -> None:
        # [Cross-Thread Contamination] Chỉ thread Re-OCR hiện tại mới được tắt busy.
        if self.sender() is not self._reocr_thread:
            return
        self._reocr_thread = None
        self._reocr_worker = None
        self._reocr_reporter = None
        self._set_busy(False)
        # [v3.6 bugfix R5]: Phát signal để View clear waveform reocr region.
        # Trước đây không có cơ chế này → region bị highlight mãi sau khi xong.
        self.reocr_region_should_clear.emit()
        # [v3.6 bugfix R1]: KHÔNG emit progress_changed(0, 0, "") ở đây.
        # Trước đây emit này xóa ngay thông báo "Re-OCR hoàn tất trong X.Xs"
        # mà _on_reocr_finished vừa set → user không bao giờ thấy message.
        # Thay vào đó: chỉ hide progress bar qua signal (0 total = hide bar)
        # nhưng GIỮ NGUYÊN status label text (msg="" sẽ được ignore trong View).
        self.progress_changed.emit(0, 0, "")

    def cancel_reocr(self) -> None:
        if getattr(self, "_reocr_reporter", None):
            self._reocr_reporter.request_cancel()

    def _set_busy(self, busy: bool) -> None:
        if self._is_busy != busy:
            self._is_busy = busy
            self.busy_changed.emit(busy)

    def _dispatch(self, action) -> None:
        try:
            action()
            fast_state = self._service.fast_snapshot()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.state_changed.emit(self._service.fast_snapshot())
            return
        self.state_changed.emit(fast_state)

__all__ = ["EditorPageViewModel"]
