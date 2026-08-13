"""ViewModel cho trang "Trích xuất phụ đề" — MVVM pattern.

TỐI ƯU HÓA ĐỘT PHÁ (V3.12 - Core Engine Polish):
    * [PERFORMANCE] Tránh ghi I/O Database rác: Chỉ lưu ROI khi thực sự có thay đổi.
    * [STABILITY] Quản lý Vòng đời Thread an toàn: Expose các biến quản lý luồng để
      View có thể ép dừng (Wait) khi tắt ứng dụng, chống tràn VRAM.
    * [CRITICAL BUG FIX] Sửa lỗi Tọa độ VRAM: Ép tính năng Upscale và Border chạy trên
      CPU để hệ thống có thể tịnh tiến ngược tọa độ Box về Kích thước Ảnh gốc, chống lệch Box.
"""

from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
    ExtractSubtitlesResponse,
    SubtitleBuilderConfig,
)
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.entities.video_state import VideoState
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.domain.ports.hardsub_detector_port import HardsubDetectionResult
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.device_kind import (
    DeviceKind,
    PrecisionMode,
    SubtitleFormat,
)
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.presentation.workers.detection_workers import (
    DetectAutoRoiWorker,
    DetectHardsubWorker,
)
from subtitles_extractor.presentation.workers.extract_subtitles_worker import (
    ExtractSubtitlesWorker,
)
from subtitles_extractor.presentation.workers.qt_progress_reporter import (
    QtProgressReporter,
)

logger = logging.getLogger(__name__)

class ExtractPageViewModel(QObject):
    video_loaded = Signal(object)
    progress_changed = Signal(int, int, str)
    extraction_finished = Signal(object)
    extraction_failed = Signal(str)
    busy_changed = Signal(bool)
    hardsub_detected = Signal(object)
    auto_roi_detected = Signal(object)
    detection_failed = Signal(str)
    roi_changed = Signal(object)
    detected_rois_changed = Signal(object)
    has_last_analysis = Signal(bool)
    cached_subtitles_found = Signal(bool)
    raw_export_finished = Signal(object, object)
    # [v3.21] Phụ đề nhúng: liệt kê track + kết quả trích.
    embedded_tracks_listed = Signal(object)   # list[EmbeddedSubtitleTrack]
    embedded_extract_finished = Signal(object)  # list[SubtitleEvent]
    embedded_failed = Signal(str)

    # [v3.23.168] Phụ đề rời cùng tên cạnh video (sidecar).
    sidecar_load_finished = Signal(object)  # list[SubtitleEvent]
    sidecar_load_failed = Signal(str)
    # [v3.21] STT WhisperX.
    transcribe_finished = Signal(object)  # list[SubtitleEvent]
    transcribe_failed = Signal(str)
    transcribe_raw_ready = Signal(object, str, str)  # raw_segments, lang, model

    def __init__(self, container: ApplicationContainer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._container = container
        self._video: VideoMetadata | None = None
        self._roi: Roi | None = None
        self._is_busy = False
        self._detected_rois: list[Roi] = []
        self._last_analysis_result = None
        # [Phần 6] Khi True: sau khi phân tích xong, tự chọn 1 ROI phụ đề đậm đặc
        # nhất (frame-presence) làm ROI duy nhất, BỎ QUA dialog kiểm duyệt.
        self._primary_roi_mode = False

        # [V3.12 STABILITY FIX] Expose Threads cho Safe Teardown
        self.active_extract_thread: QThread | None = None
        self.active_detect_thread: QThread | None = None

    @property
    def video(self) -> VideoMetadata | None:
        return self._video

    @property
    def roi(self) -> Roi | None:
        return self._roi

    @property
    def detected_rois(self) -> list[Roi]:
        return list(self._detected_rois)

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    @property
    def video_path(self) -> Path | None:
        return self._video.path if self._video else None

    @property
    def has_cached_subtitles(self) -> bool:
        if not self._video:
            return False
        return self._container.subtitle_repository.has_events(str(self._video.path.resolve()))

    def load_video(self, video_path: Path) -> None:
        """Nạp metadata video và lưu trạng thái.

        [v3.6 bugfix LV-1]: Thêm xử lý OSError, FileNotFoundError, ValueError.
        Trước đây chỉ bắt SubtitlesExtractorError → video lỗi/corrupt/không đọc được
        throw OSError/FileNotFoundError và crash silent (không thông báo cho user).
        """
        try:
            use_case = self._container.make_load_video_metadata_use_case()
            metadata = use_case.execute(video_path)
        except SubtitlesExtractorError as exc:
            self.extraction_failed.emit(str(exc))
            return
        except FileNotFoundError as exc:
            self.extraction_failed.emit(f"Không tìm thấy file video: {exc}")
            return
        except (OSError, ValueError) as exc:
            self.extraction_failed.emit(f"Không thể đọc video: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 — last-resort, prevents silent crash
            logger.warning("load_video: lỗi không mong đợi khi đọc metadata: %s", exc)
            self.extraction_failed.emit(f"Lỗi không xác định khi mở video: {exc}")
            return

        self._video = metadata
        self._detected_rois = []
        self._last_analysis_result = None
        self._roi = None

        if self._container.settings_service.current.roi.remember_last_roi:
            saved_state = self._container.video_state_repository.get(str(metadata.path.resolve()))
            if saved_state and saved_state.roi:
                self._roi = saved_state.roi

        self.cached_subtitles_found.emit(self.has_cached_subtitles)
        self.video_loaded.emit(metadata)
        self.roi_changed.emit(self._roi)
        self.detected_rois_changed.emit([])
        self.has_last_analysis.emit(False)

    def set_roi(self, roi: Roi | None, clear_detected: bool = True) -> None:
        """[V3.12 PERFORMANCE FIX] Chặn ghi I/O Database liên tục nếu ROI không đổi"""
        is_same_roi = False
        if self._roi is not None and roi is not None:
            is_same_roi = (
                self._roi.x == roi.x and self._roi.y == roi.y and
                self._roi.width == roi.width and self._roi.height == roi.height
            )
        elif self._roi is None and roi is None:
            is_same_roi = True

        if is_same_roi:
            return

        self._roi = roi
        if clear_detected:
            self._detected_rois = []

        self.roi_changed.emit(roi)

        if clear_detected:
            self.detected_rois_changed.emit([])

        # Chỉ ghi Database 1 lần khi ROI thực sự thay đổi
        if self._video and self._container.settings_service.current.roi.remember_last_roi:
            state = VideoState(video_path=str(self._video.path.resolve()), roi=roi)
            self._container.video_state_repository.save(state)

    def select_detected_roi(self, roi_index: int) -> None:
        if not self._detected_rois or not (0 <= roi_index < len(self._detected_rois)):
            return
        self.set_roi(self._detected_rois[roi_index], clear_detected=False)

    def detect_hardsub(self) -> None:
        if self._video is None or self._is_busy: return
        hw_settings = self._container.settings_service.current.hardware
        use_case = self._container.make_detect_hardsub_use_case()
        worker = DetectHardsubWorker(use_case=use_case, metadata=self._video, max_samples=hw_settings.batch_size_roi)
        worker.finished.connect(self._on_hardsub_detected)
        worker.failed.connect(self.detection_failed)
        self._start_detection_worker(worker)

    def _on_hardsub_detected(self, result: HardsubDetectionResult) -> None:
        self.hardsub_detected.emit(result)

    def detect_auto_roi(self) -> None:
        if self._video is None or self._is_busy:
            return
        settings = self._container.settings_service.current
        step_ms = getattr(settings.roi, "auto_detect_step_ms", 1000)

        is_nvdec = (getattr(settings.hardware, "frame_decoder_backend", "") == "pynvvideocodec")
        if is_nvdec:
            # [SỬA LỖI TỌA ĐỘ VRAM]: Tắt Upscale và Border trên VRAM để bảo vệ hệ tọa độ
            vram_flags = FrameSamplingConfig(
                vram_upscale_small_text=False,
                vram_upscale_target_height_px=settings.preprocess.upscale_target_height_px,
                vram_add_border=False,
                vram_border_thickness_px=settings.preprocess.border_thickness_px,
                vram_sharpen=settings.preprocess.apply_sharpen,
                vram_contrast_factor=settings.preprocess.contrast_factor if settings.preprocess.apply_contrast_boost else 1.0,
            )
        else:
            vram_flags = None

        worker = DetectAutoRoiWorker(
            use_case=self._container.make_detect_auto_roi_use_case(),
            metadata=self._video,
            step_ms=step_ms,
            batch_size=settings.hardware.batch_size_roi,
            vram_flags=vram_flags,
        )
        worker.progress.connect(self.progress_changed)
        worker.finished.connect(self._on_auto_roi_detected)
        worker.failed.connect(self.detection_failed)
        worker.analysis_ready.connect(self._on_analysis_ready_show_dialog, type=Qt.ConnectionType.QueuedConnection)
        self._start_detection_worker(worker)

    def detect_primary_subtitle_roi(self) -> None:
        """[Phần 6] Tự nhận diện rồi chọn DUY NHẤT vùng phụ đề chính (không review).

        Tái dùng worker phân tích của :meth:`detect_auto_roi`, nhưng bật cờ
        ``_primary_roi_mode`` để khi phân tích xong sẽ tự chọn cụm đậm đặc nhất
        thay vì mở dialog kiểm duyệt.
        """
        self._primary_roi_mode = True
        self.detect_auto_roi()

    def review_auto_roi_again(self) -> None:
        # [v3.6 bugfix RAR-1]: Kiểm tra _is_busy trước khi gọi dialog.
        # Trước đây không check → gọi giữa extraction → _set_busy(False) can thiệp sai.
        if self._last_analysis_result and not self._is_busy:
            self._on_analysis_ready_show_dialog(self._last_analysis_result)

    def _on_analysis_ready_show_dialog(self, result) -> None:
        self._last_analysis_result = result
        self.has_last_analysis.emit(True)

        # [Phần 6] Chế độ tự chọn 1 ROI phụ đề đậm đặc nhất — không mở dialog.
        if self._primary_roi_mode:
            self._primary_roi_mode = False
            from subtitles_extractor.infrastructure.video.roi_selection import (
                select_subtitle_roi_smart,
            )

            frame_width = self._video.width if self._video is not None else 0
            frame_height = self._video.height if self._video is not None else 0
            primary_roi = select_subtitle_roi_smart(
                result.clusters,
                getattr(result, "raw_bboxes", []),
                frame_width,
                frame_height,
            )
            if primary_roi is None:
                self.progress_changed.emit(0, 0, "Không tìm thấy vùng phụ đề.")
                self.auto_roi_detected.emit(None)
            else:
                self._on_auto_roi_detected(primary_roi)
                self.progress_changed.emit(0, 0, "Đã chọn vùng phụ đề chính.")
            return

        # [BUG FIX v2.9+]: Re-acquire busy=True trước khi hiện dialog.
        self._set_busy(True)

        from PySide6.QtWidgets import QApplication
        from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
            OcrBasedAutoRoiDetector,
            _clusters_to_rois,
        )

        # [v3.6 bugfix ARD-1]: try/finally đảm bảo _set_busy(False) luôn được gọi.
        # Trước đây thiếu guard → nếu _clusters_to_rois() hoặc _on_auto_roi_detected()
        # raise, _is_busy kẹt True mãi mãi → UI đơ vĩnh viễn.
        rois = None
        try:
            try:
                rois = OcrBasedAutoRoiDetector.review_clusters(result, parent_widget=QApplication.activeWindow(), translator=self._container.translator)
            except (RuntimeError, ImportError, ValueError):
                rois = _clusters_to_rois(result.clusters)

            if rois is None:
                self.progress_changed.emit(0, 0, "Đã hủy tự phát hiện ROI.")
                self.auto_roi_detected.emit("CANCELED")
                return

            self._on_auto_roi_detected(rois)
            self.progress_changed.emit(0, 0, "Sẵn sàng.")

        except Exception as exc:  # noqa: BLE001 — boundary slot, không để worker làm crash UI
            logger.warning("_on_analysis_ready_show_dialog: lỗi không mong đợi: %s", exc)
        finally:
            self._set_busy(False)

    def _on_auto_roi_detected(self, roi) -> None:
        if roi is not None:
            if isinstance(roi, list) and roi:
                primary = max(roi, key=lambda r: r.width * r.height)
                self.set_roi(primary, clear_detected=False)
                self._detected_rois = list(roi)
                self.detected_rois_changed.emit(list(roi))
            elif hasattr(roi, "x"):
                self.set_roi(roi, clear_detected=False)
                self._detected_rois = [roi]
                self.detected_rois_changed.emit([roi])
        else:
            self._detected_rois = []
            self.detected_rois_changed.emit([])
        self.auto_roi_detected.emit(roi)

    def _start_detection_worker(self, worker: QObject) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        done_signal = getattr(worker, "done", None)
        if done_signal is not None: done_signal.connect(thread.quit)
        else:
            worker.finished.connect(thread.quit)
            worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_detection_thread_finished)

        self.active_detect_thread = thread
        self._detect_worker = worker
        self._set_busy(True)
        thread.start()

    def _on_detection_thread_finished(self) -> None:
        # [Cross-Thread Contamination] Chỉ thread phát hiện hiện tại mới được tắt
        # busy — tín hiệu finished đến muộn từ thread cũ (đã huỷ) bị bỏ qua.
        if self.sender() is not self.active_detect_thread:
            return
        self.active_detect_thread = None
        self._detect_worker = None
        self._set_busy(False)

    def list_embedded_tracks(self) -> None:
        """[v3.21] Liệt kê track phụ đề nhúng của video hiện tại (chạy nền)."""
        if not self._video:
            self.embedded_failed.emit("Chưa chọn video.")
            return
        from subtitles_extractor.presentation.workers.embedded_extract_worker import (
            ListEmbeddedTracksWorker,
        )

        use_case = self._container.make_extract_embedded_use_case()
        worker = ListEmbeddedTracksWorker(use_case, Path(self._video.path))
        worker.finished.connect(self.embedded_tracks_listed.emit)
        worker.failed.connect(self.embedded_failed.emit)
        self._start_detection_worker(worker)

    def start_embedded_extraction(self, track: object, ocr_language: str = "") -> None:
        """[v3.21] Trích một track phụ đề nhúng (OCR nếu bitmap), chạy nền.

        Args:
            track: Track phụ đề nhúng cần trích.
            ocr_language: Mã ngôn ngữ PaddleOCR cho OCR ảnh. Rỗng = TỰ ĐỘNG suy từ ngôn
                ngữ track (tránh nhiễu □ do dùng sai model ngôn ngữ).
        """
        if not self._video:
            self.embedded_failed.emit("Chưa chọn video.")
            return
        from subtitles_extractor.application.services.embedded_ocr_language import (
            describe_paddle_lang,
            resolve_paddle_lang,
        )
        from subtitles_extractor.presentation.workers.embedded_extract_worker import (
            ExtractEmbeddedWorker,
        )

        paddle_lang = ocr_language
        if not paddle_lang:  # tự động: suy từ ngôn ngữ track
            track_language = getattr(track, "language", "") or ""
            paddle_lang = resolve_paddle_lang(track_language) or ""
        logger.info(
            "OCR phụ đề nhúng dùng ngôn ngữ: {} (track lang='{}').",
            describe_paddle_lang(paddle_lang),
            getattr(track, "language", ""),
        )

        use_case = self._container.make_extract_embedded_use_case(paddle_lang or None)
        worker = ExtractEmbeddedWorker(use_case, Path(self._video.path), track)
        worker.progress.connect(self.progress_changed.emit)
        worker.finished.connect(self._on_embedded_extract_finished)
        worker.failed.connect(self.embedded_failed.emit)
        self._embedded_worker = worker
        self._start_detection_worker(worker)

    def _on_embedded_extract_finished(self, events: object) -> None:
        self._embedded_worker = None
        self.embedded_extract_finished.emit(events)

    def find_sidecar_subtitles(self) -> list:
        """[v3.23.168] Liệt kê file phụ đề rời cùng tên cạnh video hiện tại.

        Returns:
            Danh sách ``SidecarSubtitle`` (rỗng nếu chưa mở video hoặc không có file).
        """
        if self._video is None:
            return []
        use_case = self._container.make_load_sidecar_subtitles_use_case()
        return use_case.find(Path(self._video.path))

    def load_sidecar_subtitle(self, subtitle_path: Path) -> None:
        """[v3.23.168] Nạp một file phụ đề rời thành event và phát tín hiệu hoàn tất.

        Phát ``sidecar_load_finished`` khi thành công, ``sidecar_load_failed`` kèm thông
        điệp lỗi tiếng Việt khi thất bại (không ném ra ngoài để UI không sập).

        Args:
            subtitle_path: Đường dẫn file phụ đề rời cần nạp.
        """
        use_case = self._container.make_load_sidecar_subtitles_use_case()
        try:
            events = use_case.load(subtitle_path)
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            logger.warning("Nạp phụ đề rời '{}' thất bại: {}", subtitle_path, exc)
            self.sidecar_load_failed.emit(
                f"Không nạp được phụ đề '{subtitle_path.name}': {exc}"
            )
            return
        logger.info("Đã nạp {} câu từ phụ đề rời '{}'.", len(events), subtitle_path.name)
        self.sidecar_load_finished.emit(events)

    def is_stt_available(self) -> bool:
        """[v3.21] WhisperX có sẵn sàng không (để UI ẩn/hiện nhóm STT).

        Cache kết quả: probe ``import whisperx`` chỉ chạy lần đầu (tốn kém, không
        đổi trong một phiên chạy).
        """
        cached = getattr(self, "_stt_available_cache", None)
        # [v3.23.335] CHỈ cache kết quả DƯƠNG. Trước đây cache cả kết quả âm, nên sau
        # khi người dùng tạo môi trường `whisperx_env` thì vẫn bị báo "chưa cài" cho tới
        # khi khởi động lại ứng dụng — gây hiểu nhầm là cài hỏng.
        # Phép kiểm khi CHƯA có chỉ là vài lần thử đường dẫn nên chạy lại rất rẻ.
        if cached:
            return True
        try:
            available = self._container.make_transcribe_speech_use_case().is_available()
        except (ImportError, RuntimeError, OSError):
            available = False
        if available:
            self._stt_available_cache = True
        return available

    def get_stt_diagnosis_hint(self) -> str:
        """[v3.22.4] Chẩn đoán vì sao STT không khả dụng + cách khắc phục.

        Dùng Dependency Doctor phân biệt 'chưa cài' vs 'cài rồi nhưng lỗi DLL'.
        """
        try:
            from subtitles_extractor.infrastructure.diagnostics.dependency_doctor import (
                check_whisperx,
            )

            report = check_whisperx()
            if report.is_ok:
                return "WhisperX đã sẵn sàng."
            return f"⚠️ {report.detail}\n💡 {report.install_hint}"
        except (ImportError, OSError, RuntimeError) as exc:
            return f"⚠️ Không kiểm tra được WhisperX: {exc}"

    def start_transcription(self, config: object) -> None:
        """[v3.21] Phiên âm giọng nói video hiện tại bằng WhisperX (chạy nền)."""
        if not self._video:
            self.transcribe_failed.emit("Chưa chọn video.")
            return
        from subtitles_extractor.presentation.workers.transcribe_worker import (
            TranscribeSpeechWorker,
        )

        use_case = self._container.make_transcribe_speech_use_case()
        worker = TranscribeSpeechWorker(use_case, Path(self._video.path), config)
        worker.progress.connect(self.progress_changed.emit)
        worker.finished.connect(self._on_transcribe_finished)
        worker.raw_ready.connect(self.transcribe_raw_ready.emit)
        worker.failed.connect(self.transcribe_failed.emit)
        self._transcribe_worker = worker
        self._start_detection_worker(worker)

    def _on_transcribe_finished(self, events: object) -> None:
        self._transcribe_worker = None
        self.transcribe_finished.emit(events)

    def load_cached_subtitles(self) -> None:
        if not self._video: return
        events = self._container.subtitle_repository.load_events(str(self._video.path.resolve()))
        if events:
            resp = ExtractSubtitlesResponse(events=events, output_path=self._video.path, elapsed_seconds=0.0, frames_processed=0)
            self.extraction_finished.emit(resp)

    def start_extraction(self) -> None:
        if self._video is None or self._is_busy: return
        requests = self._build_requests()
        # [v3.23.322] Tự lưu cache OCR thô cạnh video để sau này dựng lại phụ đề với
        # tham số khác mà KHÔNG phải chạy lại OCR (khâu tốn hàng chục phút).
        requests = self._attach_raw_ocr_cache(requests)
        self._launch_extract_worker(requests, is_raw_export=False)

    def _attach_raw_ocr_cache(self, requests: list) -> list:
        """Gắn đường dẫn lưu cache OCR thô vào các yêu cầu chưa có.

        Args:
            requests: Danh sách yêu cầu trích xuất.

        Returns:
            Danh sách mới, mỗi yêu cầu đều có ``raw_ocr_output_path``.
        """
        import dataclasses

        from subtitles_extractor.domain.value_objects.output_naming import (
            raw_ocr_cache_path,
        )

        if self._video is None:
            return requests
        cache_path = raw_ocr_cache_path(self._video.path)
        return [
            request
            if request.raw_ocr_output_path is not None
            else dataclasses.replace(request, raw_ocr_output_path=cache_path)
            for request in requests
        ]

    def has_cached_ocr(self) -> bool:
        """``True`` nếu video hiện tại đã có cache OCR để dựng lại."""
        from subtitles_extractor.domain.value_objects.output_naming import (
            raw_ocr_cache_path,
        )

        if self._video is None:
            return False
        return raw_ocr_cache_path(self._video.path).is_file()

    def rebuild_from_cached_ocr(self) -> None:
        """Dựng lại phụ đề từ cache OCR với THAM SỐ HIỆN TẠI — không chạy lại OCR.

        Đây là vòng lặp tinh chỉnh nhanh: đổi ngưỡng gộp câu/độ tương đồng rồi dựng lại
        trong vài giây, thay vì chạy lại OCR cả tập.

        Phát ``extraction_finished`` khi xong, hoặc ``extraction_failed`` nếu lỗi.
        """
        from subtitles_extractor.application.dtos.extract_subtitles_dto import (
            ExtractSubtitlesResponse,
        )
        from subtitles_extractor.domain.value_objects.output_naming import (
            raw_ocr_cache_path,
        )
        from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
            load_raw_ocr,
        )

        if self._video is None or self._is_busy:
            return
        cache_path = raw_ocr_cache_path(self._video.path)
        if not cache_path.is_file():
            self.extraction_failed.emit(
                "Chưa có cache OCR cho video này. Hãy chạy trích xuất một lần trước."
            )
            return

        try:
            frames, _meta = load_raw_ocr(cache_path)
        except (OSError, ValueError) as exc:
            logger.warning("Không đọc được cache OCR %s: %s", cache_path, exc)
            self.extraction_failed.emit(f"Không đọc được cache OCR: {exc}")
            return

        try:
            from subtitles_extractor.application.services.subtitle_builder import (
                SubtitleBuilder,
            )

            request = self._build_requests()[0]
            # Dùng CHÍNH cấu hình dựng câu đang đặt trên giao diện -> đổi tham số rồi
            # bấm dựng lại là thấy kết quả mới ngay.
            builder = SubtitleBuilder(config=request.builder)
            events = builder.build(frames, roi=request.roi)
        except (IndexError, AttributeError, ValueError, RuntimeError) as exc:
            logger.exception("Dựng lại phụ đề từ cache thất bại.")
            self.extraction_failed.emit(f"Dựng lại thất bại: {exc}")
            return

        logger.info(
            "Đã dựng lại %d câu từ cache OCR (%d khung) — không chạy lại OCR.",
            len(events), len(frames),
        )
        self.extraction_finished.emit(
            ExtractSubtitlesResponse(
                events=events, output_path=self._video.path,
                elapsed_seconds=0.0, frames_processed=len(frames),
            )
        )



    @property
    def ocr_language_override(self) -> str:
        """Ngôn ngữ OCR do người dùng chọn ngay trên trang Trích xuất.

        Rỗng = dùng giá trị trong Cài đặt.
        """
        return getattr(self, "_ocr_language_override", "")

    def set_ocr_language_override(self, language: str) -> None:
        """Đặt ngôn ngữ OCR cho phiên làm việc hiện tại.

        [v3.23.321] Phim bộ CJK phải đổi Trung/Nhật/Hàn theo từng bộ; trước đây chỉ đổi
        được trong trang Cài đặt nên mỗi lần phải rời trang Trích xuất.

        Args:
            language: Mã ngôn ngữ PaddleOCR (vd ``"ch"``, ``"japan"``); rỗng = theo Cài đặt.
        """
        self._ocr_language_override = (language or "").strip()
        if self._ocr_language_override:
            logger.info("Ngôn ngữ OCR (phiên này): %s.", self._ocr_language_override)

    def _effective_ocr_language(self, snapshot: object) -> str:
        """Ngôn ngữ OCR thực dùng: ưu tiên lựa chọn trên trang, sau đó tới Cài đặt."""
        override = self.ocr_language_override
        if override:
            return override
        return snapshot.ocr.language

    def start_probe_extraction(
        self, probe_seconds: float = 60.0, center_ratio: float = 0.5
    ) -> None:
        """[v3.23.320] Chạy THỬ NHANH trong một cửa sổ ngắn giữa phim.

        Dùng đúng pipeline như chạy thật, chỉ bó phạm vi lấy mẫu bằng
        ``skip_intro_sec``/``skip_outro_sec`` — nên kết quả thử phản ánh trung thực kết
        quả thật trên đoạn đó. Không ghi tệp ra đĩa (``skip_export=True``).

        Args:
            probe_seconds: Độ dài cửa sổ thử (giây).
            center_ratio: Vị trí tâm cửa sổ theo tỉ lệ 0.0–1.0 (0.5 = giữa phim).
        """
        import dataclasses

        from subtitles_extractor.application.services.ocr_probe_window import (
            compute_probe_window,
        )

        if self._video is None or self._is_busy:
            return

        window = compute_probe_window(
            self._video.duration_sec,
            probe_seconds=probe_seconds,
            center_ratio=center_ratio,
        )
        self.last_probe_window = window

        probed_requests = [
            dataclasses.replace(
                request,
                sampling=dataclasses.replace(
                    request.sampling,
                    skip_intro_sec=window.skip_intro_sec,
                    skip_outro_sec=window.skip_outro_sec,
                ),
                skip_export=True,
            )
            for request in self._build_requests()
        ]
        logger.info(
            "Thử nhanh OCR: %s (bỏ %.1fs đầu, %.1fs cuối).",
            window.label_vi, window.skip_intro_sec, window.skip_outro_sec,
        )
        self._launch_extract_worker(probed_requests, is_raw_export=False)

    def start_extraction_with_raw_export(self, raw_ocr_output_path: Path, export_images: bool = False) -> None:
        if self._video is None or self._is_busy: return
        requests = self._build_requests(raw_ocr_output_path=raw_ocr_output_path, export_images=export_images)
        self._launch_extract_worker(requests, is_raw_export=True)

    def _launch_extract_worker(self, requests: list[ExtractSubtitlesRequest], is_raw_export: bool = False) -> None:
        use_case = self._container.make_extract_subtitles_use_case()
        reporter = QtProgressReporter()
        reporter.progress_changed.connect(self.progress_changed)

        # [BUG FIX v2.9+]: Dùng QThread(self) để ViewModel là parent — tránh memory
        # leak khi widget bị destroy trước thread (đồng nhất với _start_detection_worker).
        thread = QThread(self)
        worker = ExtractSubtitlesWorker(use_case=use_case, requests=requests, reporter=reporter, export_use_case=None)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        if is_raw_export:
            worker.finished.connect(self._on_raw_export_finished)
        else:
            worker.finished.connect(self._on_extract_finished)

        worker.failed.connect(self._on_extract_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_extract_thread_finished)

        self.active_extract_thread = thread
        self._extract_worker = worker
        self._extract_reporter = reporter
        self._set_busy(True)
        thread.start()

    def cancel_extraction(self) -> None:
        if getattr(self, "_extract_reporter", None) is not None:
            self._extract_reporter.request_cancel()
        # [v3.22] Hủy cả worker nguồn mới (embedded OCR bitmap / STT WhisperX) — chúng
        # hỗ trợ dừng mềm giữa chừng qua request_cancel.
        for worker_attr in ("_embedded_worker", "_transcribe_worker"):
            worker = getattr(self, worker_attr, None)
            if worker is not None and hasattr(worker, "request_cancel"):
                worker.request_cancel()

    def _on_raw_export_finished(self, response: ExtractSubtitlesResponse) -> None:
        raw_ocr_path = getattr(response, "raw_ocr_path", None)
        self.raw_export_finished.emit(raw_ocr_path, response)

    def _on_extract_finished(self, response: ExtractSubtitlesResponse) -> None:
        if self._video and response.events:
            events_copy = copy.deepcopy(response.events)
            video_path_str = str(self._video.path.resolve())

            def _bg_save() -> None:
                try:
                    self._container.subtitle_repository.save_events(
                        video_path_str, events_copy,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Lưu phụ đề vào SQLite thất bại — bỏ qua: %s.", exc,
                    )

            threading.Thread(target=_bg_save, daemon=True).start()
        self.extraction_finished.emit(response)

    def _on_extract_failed(self, message: str) -> None:
        self.extraction_failed.emit(message)

    def _on_extract_thread_finished(self) -> None:
        self.active_extract_thread = None
        self._extract_worker = None
        self._extract_reporter = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        """Hàm cài đặt trạng thái bận và phát tín hiệu cho giao diện."""
        if self._is_busy != busy:
            self._is_busy = busy
            self.busy_changed.emit(busy)

    def _build_requests(self, raw_ocr_output_path: Path | None = None, export_images: bool = False) -> list[ExtractSubtitlesRequest]:
        if self._video is None:
            raise RuntimeError("VideoMetadata phải được load trước khi build request.")

        rois_to_extract: list[Roi | None]
        if len(self._detected_rois) >= 2:
            rois_to_extract = list(self._detected_rois)
        elif len(self._detected_rois) == 1:
            rois_to_extract = [self._detected_rois[0]]
        else:
            rois_to_extract = [self._roi]

        requests = []
        for i, roi in enumerate(rois_to_extract):
            out_path = raw_ocr_output_path

            if out_path and len(rois_to_extract) > 1:
                name = out_path.name
                if name.endswith(".seraw.json"): new_name = name[:-11] + f"_roi{i+1}.seraw.json"
                elif name.endswith(".seraw.json.gz"): new_name = name[:-14] + f"_roi{i+1}.seraw.json.gz"
                else: new_name = f"{out_path.stem}_roi{i+1}{out_path.suffix}"
                out_path = out_path.with_name(new_name)

            requests.append(self._build_single_request(roi, raw_ocr_output_path=out_path, export_images=export_images))

        return requests

    def _build_single_request(self, roi: Roi | None, raw_ocr_output_path: Path | None = None, export_images: bool = False) -> ExtractSubtitlesRequest:
        snapshot = self._container.settings_service.current
        post = snapshot.post_process
        from subtitles_extractor.application.services.ocr_model_resolver import (
            resolve_ocr_model_names,
        )
        from subtitles_extractor.domain.ports.ocr_engine_port import PreprocessConfig
        det_model, rec_model = resolve_ocr_model_names(snapshot.ocr)

        from importlib.util import find_spec
        has_cupy = find_spec("cupy") is not None

        is_nvdec = snapshot.hardware.frame_decoder_backend == "pynvvideocodec" and has_cupy

        # [GIẢI PHÓNG VRAM]: Cấp toàn quyền Tiền xử lý cho VRAM để tốc độ đạt đỉnh,
        # Trừ hao tọa độ sẽ được lớp ExtractSubtitlesUseCase tự động nội suy.
        vram_upscale = snapshot.preprocess.upscale_small_text if is_nvdec else False 
        vram_border = snapshot.preprocess.add_white_border if is_nvdec else False
        vram_sharpen = snapshot.preprocess.apply_sharpen if is_nvdec else False
        vram_contrast_factor = snapshot.preprocess.contrast_factor if (is_nvdec and snapshot.preprocess.apply_contrast_boost) else 1.0

        cpu_upscale = snapshot.preprocess.upscale_small_text if not is_nvdec else False
        cpu_border = snapshot.preprocess.add_white_border if not is_nvdec else False
        cpu_sharpen = snapshot.preprocess.apply_sharpen if not is_nvdec else False
        cpu_contrast = snapshot.preprocess.apply_contrast_boost if not is_nvdec else False

        return ExtractSubtitlesRequest(
            video_path=self._video.path, output_path=Path("Database"), output_format=SubtitleFormat.SRT, roi=roi,
            sampling=FrameSamplingConfig(
                sample_step_sec=snapshot.frame.sample_step_sec, phash_distance_threshold=snapshot.frame.phash_distance,
                pixel_diff_threshold=snapshot.frame.pixel_diff_ratio, skip_intro_sec=snapshot.frame.skip_intro_sec, skip_outro_sec=snapshot.frame.skip_outro_sec,
                apply_median_blend=snapshot.preprocess.apply_median_blend,
                median_blend_frames=snapshot.preprocess.median_blend_frames,
                vram_upscale_small_text=vram_upscale, vram_upscale_target_height_px=snapshot.preprocess.upscale_target_height_px,
                vram_add_border=vram_border, vram_border_thickness_px=snapshot.preprocess.border_thickness_px,
                vram_sharpen=vram_sharpen, vram_contrast_factor=vram_contrast_factor,
            ),
            ocr=OcrEngineConfig(
                device=DeviceKind(snapshot.hardware.device.value),
                detection_model_name=det_model,
                recognition_model_name=rec_model,
                language=self._effective_ocr_language(snapshot),
                limit_side_len=snapshot.ocr.limit_side_len if snapshot.ocr.limit_side_len > 0 else 0,
                limit_type=snapshot.ocr.limit_type,
                det_thresh=snapshot.ocr.det_thresh,
                det_box_thresh=snapshot.ocr.det_box_thresh,
                det_unclip_ratio=snapshot.ocr.det_unclip_ratio,
                use_doc_orientation_classify=snapshot.ocr.use_doc_orientation_classify,
                use_doc_unwarping=snapshot.ocr.use_doc_unwarping,
                use_textline_orientation=snapshot.ocr.use_textline_orientation,
                score_threshold=snapshot.ocr.score_threshold,
                enable_mkldnn=snapshot.hardware.enable_mkldnn,
                use_tensorrt=snapshot.hardware.use_tensorrt,
                precision=PrecisionMode(snapshot.hardware.precision.value),
                batch_size=snapshot.hardware.batch_size_ocr,
                parallel_workers=snapshot.hardware.workers,
                preprocess=PreprocessConfig(
                    upscale_small_text=cpu_upscale, upscale_target_height_px=snapshot.preprocess.upscale_target_height_px,
                    add_white_border=cpu_border, border_thickness_px=snapshot.preprocess.border_thickness_px,
                    apply_clahe=snapshot.preprocess.apply_clahe, clahe_clip_limit=snapshot.preprocess.clahe_clip_limit, clahe_tile_size=snapshot.preprocess.clahe_tile_size,
                    apply_sharpen=cpu_sharpen, apply_contrast_boost=cpu_contrast, contrast_factor=snapshot.preprocess.contrast_factor,
                ),
            ),
            builder=SubtitleBuilderConfig(
                similarity_threshold=min(post.similarity_threshold, snapshot.threshold.text_similarity),
                min_duration_sec=post.min_duration_sec, max_duration_sec=post.max_duration_sec,
                merge_gap_sec=post.merge_gap_sec, min_confidence=snapshot.threshold.ocr_min_confidence,
                use_viterbi=post.use_viterbi, viterbi_open_penalty=post.viterbi_open_penalty,
                min_text_chars=snapshot.threshold.drop_short_text_chars, line_similarity_threshold=snapshot.threshold.line_similarity,
                sample_step_sec=snapshot.frame.sample_step_sec, temporal_padding_sec=post.temporal_padding_sec,
                y_clustering_tolerance_ratio=post.y_clustering_tolerance_ratio, y_clustering_tolerance_min_px=post.y_clustering_tolerance_min_px,
                alignment_center_tolerance_ratio=post.alignment_center_tolerance_ratio, alignment_margin_tolerance_ratio=post.alignment_margin_tolerance_ratio,
                alignment_tolerance_min_px=post.alignment_tolerance_min_px
            ),
            auto_tune_batch=snapshot.hardware.auto_tune_batch, save_debug_frames=snapshot.advanced.save_debug_frames,
            debug_frames_dir=snapshot.advanced.debug_frames_dir, keep_temp_files=snapshot.advanced.keep_temp_files,
            skip_export=True, raw_ocr_output_path=raw_ocr_output_path, export_raw_images=export_images,
        )

__all__ = ["ExtractPageViewModel"]
