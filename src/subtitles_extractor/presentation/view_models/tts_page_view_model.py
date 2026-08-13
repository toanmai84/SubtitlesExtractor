"""ViewModel cho trang TTS phụ đề."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest, TTSSegmentResult

if TYPE_CHECKING:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent

logger = logging.getLogger(__name__)


class TTSPageViewModel(QObject):
    """ViewModel trang TTS."""

    source_changed = Signal(int)
    busy_changed = Signal(bool)
    progress_changed = Signal(float, str)
    result_ready = Signal(object)
    error_occurred = Signal(str)
    status_message = Signal(str)
    tts_cancelled = Signal()
    engine_available = Signal(str, bool)

    def __init__(self, container: "ApplicationContainer") -> None:
        super().__init__()
        self._container = container
        self._source_events: list["SubtitleEvent"] = []
        self._last_results: list[TTSSegmentResult] = []
        self._last_output_path: Path | None = None
        self._last_request: "TTSRequest | None" = None
        self._last_engine_name: str = ""
        self._worker = None
        self._is_busy = False

    @property
    def source_events(self) -> list["SubtitleEvent"]:
        return self._source_events

    @property
    def has_source(self) -> bool:
        return len(self._source_events) > 0

    @property
    def has_result(self) -> bool:
        return bool(self._last_results)

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    @property
    def last_output_path(self) -> Path | None:
        return self._last_output_path

    @property
    def last_results(self) -> list[TTSSegmentResult]:
        return self._last_results

    @property
    def last_request(self) -> "TTSRequest | None":
        return self._last_request

    @property
    def last_engine_name(self) -> str:
        return self._last_engine_name

    def set_source_events(self, events: list["SubtitleEvent"]) -> None:
        self._source_events = list(events)
        self.source_changed.emit(len(self._source_events))

    def load_source_from_file(self, source_path: Path) -> None:
        try:
            use_case = self._container.make_import_subtitles_use_case()
            events = use_case.execute(source_path)
            self.set_source_events(events)
            self.status_message.emit(f"Đã nạp {len(events)} dòng từ {source_path.name}")
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Không đọc được tệp: {exc}")

    def check_engines(self) -> None:
        edge   = self._container.make_edge_tts_adapter()
        gemini = self._container.make_gemini_tts_adapter()
        vieneu = self._container.make_vieneu_tts_adapter()
        self.engine_available.emit(edge.get_engine_name(),   edge.is_available())
        self.engine_available.emit(gemini.get_engine_name(), gemini.is_available())
        self.engine_available.emit(vieneu.get_engine_name(), vieneu.is_available())

    def start_tts(
        self,
        engine: str,
        language: str = "",
        speaker: str = "",
        base_speed: float = 1.0,
        max_speed: float = 3.0,
        device: str = "auto",
        normalize: bool = True,
        output_path: Path | None = None,
        retry_count: int = 3,
        retry_delay_s: float = 1.0,
        # Timing controls
        clean_tags: bool = True,
        dialog_pause_ms: int = 300,
        max_overlap_ms: int = 500,
        skip_overlap_ms: int = 0,
        double_pass: bool = True,
        elastic_timing: bool = True,
        max_drift_s: float = 2.5,
        lead_in_s: float = 0.3,
        anchor_gap_s: float = 0.7,
        max_segment_s: float = 10.0,
        comfort_speed_ratio: float = 1.25,
        min_pause_ratio: float = 0.35,
        max_intra_gap_s: float = 0.5,
        timing_strategy: str = "lipsync",
        allow_audio_overlap: bool = True,
        min_stretch_ratio: float = 0.75,
        edge_concurrency: int = 16,
        last_line_max_extend_s: float = 0.0,
        skip_paren: bool = True,
        skip_square: bool = True,
        skip_curly: bool = True,
        skip_music_pair: bool = True,
        skip_music_line: bool = True,
        target_lufs: float = -16.0,
        voice_clarity: bool = True,
        high_quality: bool = True,
        output_format: str = "wav",
        output_bitrate_kbps: int = 320,
        wav_subtype: str = "PCM_16",
        # Gemini / VieNeu (voice cloning)
        api_key: str = "",
        ref_audio_path: str = "",
        ref_text: str = "",
        style_prompt: str = "",
        affective_dialog: bool = True,
        gemini_temperature: float | None = None,
        # VieNeu
        vieneu_mode: str = "standard",
        vieneu_emotion: str = "natural",
        vieneu_force_cpu: bool = True,
        media_duration_s: float | None = None,
    ) -> bool:
        if self._is_busy:
            self.status_message.emit("Đang bận, vui lòng chờ.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề nguồn để TTS.")
            return False
        if output_path is None:
            self.error_occurred.emit("Chưa chọn đường dẫn file đầu ra.")
            return False
        # Đồng bộ đuôi file với định dạng đã chọn (để hiển thị/mở đúng file thực).
        fmt = (output_format or "wav").lower()
        if output_path.suffix.lower() != f".{fmt}":
            output_path = output_path.with_suffix(f".{fmt}")

        request = TTSRequest(
            events=self._source_events,
            language=language, speaker=speaker,
            base_speed=base_speed, max_speed=max_speed,
            device=device, normalize=normalize,
            retry_count=retry_count, retry_delay_s=retry_delay_s,
            clean_tags=clean_tags, dialog_pause_ms=dialog_pause_ms,
            max_overlap_ms=max_overlap_ms, skip_overlap_ms=skip_overlap_ms,
            double_pass=double_pass,
            elastic_timing=elastic_timing, max_drift_s=max_drift_s, lead_in_s=lead_in_s,
            anchor_gap_s=anchor_gap_s, max_segment_s=max_segment_s,
            comfort_speed_ratio=comfort_speed_ratio, min_pause_ratio=min_pause_ratio,
            max_intra_gap_s=max_intra_gap_s,
            allow_audio_overlap=allow_audio_overlap, min_stretch_ratio=min_stretch_ratio,
            timing_strategy=timing_strategy,
            edge_concurrency=edge_concurrency,
            last_line_max_extend_s=last_line_max_extend_s,
            skip_paren=skip_paren, skip_square=skip_square, skip_curly=skip_curly,
            skip_music_pair=skip_music_pair, skip_music_line=skip_music_line,
            target_lufs=target_lufs, voice_clarity=voice_clarity,
            high_quality=high_quality,
            output_format=fmt, output_bitrate_kbps=output_bitrate_kbps,
            wav_subtype=wav_subtype,
            ref_audio_path=ref_audio_path, ref_text=ref_text,
            style_prompt=style_prompt, affective_dialog=affective_dialog,
            gemini_temperature=gemini_temperature,
            media_duration_s=media_duration_s,
        )

        if engine == "gemini":
            adapter = self._container.make_gemini_tts_adapter(api_key=api_key)
        elif engine == "vieneu":
            adapter = self._container.make_vieneu_tts_adapter(
                mode=vieneu_mode, emotion=vieneu_emotion, force_cpu=vieneu_force_cpu
            )
        else:
            adapter = self._container.make_edge_tts_adapter()

        if not adapter.is_available():
            install_map = {
                "edge": "edge-tts soundfile",
                "gemini": "google-genai soundfile",
                "vieneu": "vieneu soundfile",
            }
            self.error_occurred.emit(
                f"{adapter.get_engine_name()} chưa sẵn sàng.\n"
                f"Cài đặt: pip install {install_map.get(engine, 'soundfile')}"
            )
            return False

        use_case = self._container.make_generate_tts_use_case(adapter)
        from subtitles_extractor.presentation.workers.tts_worker import TTSWorker
        self._worker = TTSWorker(use_case, request, output_path)
        self._last_request = request  # snapshot cấu hình để xuất debug
        self._last_engine_name = adapter.get_engine_name()
        self._worker.progress_changed.connect(self.progress_changed.emit)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.finished.connect(self._on_thread_finished)
        self._set_busy(True)
        self._last_output_path = output_path
        self._worker.start()
        self.status_message.emit(f"Đang TTS {len(self._source_events)} dòng bằng {adapter.get_engine_name()}...")
        return True

    def cancel_tts(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self.status_message.emit("Đang huỷ TTS...")

    def _on_worker_finished(self, results: object) -> None:
        self._last_results = list(results)  # type: ignore[arg-type]
        self.result_ready.emit(self._last_results)

    def _on_worker_failed(self, message: str) -> None:
        self.error_occurred.emit(message)

    def _on_worker_cancelled(self) -> None:
        self.tts_cancelled.emit()

    def _on_thread_finished(self) -> None:
        # [1.2] Chống race: chỉ luồng HIỆN HÀNH (đang phát tín hiệu) mới được đổi
        # trạng thái UI. Nếu người dùng Huỷ rồi Bắt đầu ngay, tín hiệu 'finished' trễ
        # của luồng cũ không được phép gỡ busy của luồng mới.
        sender = self.sender()
        if sender is not None and sender is not self._worker and self._worker is not None:
            # Luồng cũ kết thúc muộn: chỉ dọn chính nó, không đụng trạng thái hiện tại.
            sender.deleteLater()
            return
        self._set_busy(False)
        worker = self._worker
        self._worker = None
        if worker is not None:
            # [1.1] Tín hiệu 'finished' chỉ phát SAU khi run() kết thúc, nên luồng đã
            # dừng hẳn — KHÔNG cần worker.wait() (vốn chặn luồng giao diện tới 5s).
            # Dùng deleteLater() chuẩn của Qt để dọn an toàn ngoài vòng sự kiện.
            worker.deleteLater()
            logger.debug("Đã dọn luồng TTS.")

    def _set_busy(self, busy: bool) -> None:
        if busy != self._is_busy:
            self._is_busy = busy
            self.busy_changed.emit(busy)

    def cleanup(self) -> None:
        """Dừng worker an toàn khi đóng/chuyển trang.

        Ngắt mọi kết nối signal TRƯỚC để slot không chạy sau khi VM/page đã bị
        huỷ (tránh crash do truy cập widget đã xoá). Terminate nếu không dừng kịp
        (tránh treo ứng dụng vô hạn khi API chậm).
        """
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        # Ngắt tất cả signal — chặn slot chạy sau khi đối tượng đã bị huỷ.
        for signal in (
            worker.progress_changed, worker.finished_ok,
            worker.failed, worker.cancelled, worker.finished,
        ):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass  # chưa kết nối hoặc đã ngắt — bỏ qua
        if worker.isRunning():
            worker.request_cancel()
            if not worker.wait(8000):
                logger.warning("TTSWorker không dừng kịp trong 8s — buộc dừng.")
                worker.terminate()
                worker.wait(2000)
        worker.deleteLater()


__all__ = ["TTSPageViewModel"]
