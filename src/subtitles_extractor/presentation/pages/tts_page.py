"""Trang TTS — chuyển phụ đề đã dịch thành file WAV.

3 engine TTS:
- **Edge TTS** (online):  Tiếng Việt, EN, ZH, JP... — khuyến nghị cho TV/phim dịch.
- **Gemini TTS** (online): 30 voices, 40+ ngôn ngữ, chất lượng neural cao nhất.
- **VieNeu-TTS** (offline): TTS tiếng Việt on-device, voice cloning tức thì.

Luồng: Lấy phụ đề → Chọn engine/giọng → Tổng hợp → WAV khớp timing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QMessageBox, QWidget,
)

from subtitles_extractor.application.services.dubbing_cast import (
    DubbingSource,
)
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSSegmentResult
from subtitles_extractor.presentation.qt_compat import is_deleted
from subtitles_extractor.presentation.view_models.tts_page_view_model import TTSPageViewModel
from subtitles_extractor.presentation.utils import safe_dialogs as _safe_dialogs
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets
from subtitles_extractor.presentation.theme import feedback as _feedback
from subtitles_extractor.infrastructure.process.hidden_process import (
    no_window_kwargs,
)

from subtitles_extractor.presentation.utils.task_notifier import notify_task_finished

logger = logging.getLogger(__name__)



from subtitles_extractor.infrastructure.tts.gemini_tts_schema import uses_verbatim_tts_schema  # noqa: E402
from subtitles_extractor.presentation.pages.tts_constants import (  # noqa: E402,F401 — [v3.23.692/695] tách khỏi trang; giữ tên cũ
    _EDGE_ONLY_FIELDS,
    _ENGINE_EDGE,
    _ENGINE_GEMINI,
    _ENGINE_VIENEU,
    _DEFAULT_MAX_SPEED,
    _OVERLAP_WARN_S,
    _EDGE_ONLY_HINT,
    _ENGINES,
    _EDGE_LANGUAGES,
    _EDGE_LANGUAGE_DISPLAY,
)
from subtitles_extractor.presentation.pages.tts_batch import TTSBatchMixin  # noqa: E402
from subtitles_extractor.presentation.pages.tts_engines import TTSEnginesMixin  # noqa: E402
from subtitles_extractor.presentation.pages.tts_results import TTSResultsMixin  # noqa: E402
from subtitles_extractor.presentation.pages.tts_review import TTSReviewMixin  # noqa: E402
from subtitles_extractor.presentation.pages.tts_settings import TTSSettingsMixin  # noqa: E402
from subtitles_extractor.presentation.pages.tts_ui import TTSUiMixin  # noqa: E402
from subtitles_extractor.presentation.pages.tts_voice_loader import _VieNeuVoiceLoader  # noqa: E402,F401 — giữ tên cũ
from subtitles_extractor.presentation.pages.tts_export_details import TTSExportDetailsMixin  # noqa: E402
from subtitles_extractor.presentation.pages.tts_gpu_install import TTSGpuInstallMixin  # noqa: E402



class TTSPage(
    TTSUiMixin, TTSEnginesMixin, TTSResultsMixin, TTSReviewMixin, TTSSettingsMixin,
    TTSBatchMixin, TTSGpuInstallMixin, TTSExportDetailsMixin, QWidget,
):
    """Trang TTS phụ đề → WAV."""

    # Phát khi tạo TTS xong, mang đường dẫn file WAV — để liên thông lưu dự án.
    tts_completed = Signal(str)
    #: [v3.23.682] Số dòng cần nghe lại / duyệt (thanh điều hướng hiện "● n" cạnh tên trang).
    review_count_changed = Signal(int)
    # [v3.23.652] Kết quả dò GPU (chạy ``pip list`` ở luồng nền) gửi về luồng giao diện.
    _gpu_plan_ready = Signal(object)
    # [v3.23.653] Người nói chọn tay quy về câu gốc ``{id cue: vai}`` — cửa sổ chính ghi vào Visual Cues trang Dịch.
    #: [v3.23.658] Dây chuyền phim bộ: khâu TTS chạy xong (``StageRunResult``).
    series_batch_finished = Signal(object)
    speaker_corrections_ready = Signal(object)  # object: Signal(dict) của Qt làm mất khoá số

    def __init__(self, container: ApplicationContainer, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ttsPage")
        self._container = container
        # [v3.23.372] Translator để externalize chuỗi UI (đa ngôn ngữ).
        self._translator = container.translator
        self._settings = QSettings("SubtitlesExtractorPaddleOCR", "TTSPage_v2")
        self._view_model = TTSPageViewModel(container)
        self._view_model.set_gemini_key_provider(self._gemini_api_keys)  # [v3.23.698]
        # [v3.23.331] Hàng đợi tổng hợp cả bộ.
        self._batch_items: list = []
        self._batch_index: int = 0
        self._batch_failures: list[str] = []
        self._batch_cancelled: bool = False
        self._batch_pending_generate: bool = False
        self._gpu_install_thread = None
        self._gpu_install_worker = None
        self._editor_event_provider: Callable[[], list[SubtitleEvent]] | None = None
        self._translate_event_provider: Callable[[], list[SubtitleEvent]] | None = None
        # [v3.23.593] Giọng VieNeu kèm giới tính theo chế độ (nạp ở luồng nền).
        self._vieneu_catalog: dict[str, list] = {}

        self._build_ui()
        self._connect_signals()
        self._restore_settings()
        # [v3.23.195] Đồng bộ panel cấu hình theo engine ĐANG CHỌN sau restore. Trước
        # đây gọi cứng _on_engine_changed(0) -> panel bị ép về Edge dù combo đang là
        # engine khác (bug "engine một đàng, cấu hình một nẻo" khi mở ứng dụng).
        self._on_engine_changed(self._engine_combo.currentIndex())
        self._update_action_states()
        self._view_model.check_engines()
        protect_scroll_widgets(self)
        self._last_results: list = []

    # ── Public API ────────────────────────────────────────────────────────────

    def set_editor_event_provider(self, p: Callable[[], list[SubtitleEvent]]) -> None:
        self._editor_event_provider = p

    def set_translate_event_provider(self, p: Callable[[], list[SubtitleEvent]]) -> None:
        self._translate_event_provider = p

    def set_dubbing_source_provider(self, p: Callable[[], DubbingSource]) -> None:
        """[v3.23.593] Nguồn phân vai lồng tiếng (nhân vật + Visual Cues ở trang Dịch)."""
        self._dubbing.set_source_provider(p)

    def suggest_output_path(self, video_path: str) -> None:
        """Cập nhật đường dẫn WAV theo tên video (``<tên>.wav``) khi nạp video.

        [Sticky Output Path Fix] LUÔN cập nhật theo video hiện tại (không còn chỉ
        điền khi ô trống), để mở video mới không bị kẹt đường dẫn của video cũ.
        """
        if not video_path:
            return
        try:
            from subtitles_extractor.domain.value_objects.output_naming import (
                tts_audio_path,
            )

            suggested = Path(str(tts_audio_path(video_path)))
            fmt = self._fmt.currentData() if hasattr(self, "_fmt") else "wav"
            self._output.setText(str(suggested.with_suffix(f".{fmt}")))
            # [v3.23.207] Đọc thời lượng video (PyAV, ~ms) để file TTS xuất ra dài
            # ĐÚNG bằng video -> mux không lệch (người dùng báo file dài hơn 3.5s).
            self._media_duration_s = None
            try:
                metadata = self._view_model._container.metadata_reader.read(
                    Path(video_path)
                )
                if metadata.duration_sec > 0:
                    self._media_duration_s = float(metadata.duration_sec)
            except (FileNotFoundError, OSError, ValueError, RuntimeError):
                logger.debug("Không đọc được thời lượng video cho TTS (bỏ qua).")
        except (ValueError, OSError):
            pass

    def load_events(self, events: list[SubtitleEvent]) -> None:
        self._view_model.set_source_events(events)

    def cleanup(self) -> None:
        self._save_settings()
        self._view_model.cleanup()

    # ── UI Build ──────────────────────────────────────────────────────────────

    # ── Lồng tiếng ────────────────────────────────────────────────────────────

    def _dubbing_main_voice(self) -> str:
        """[v3.23.597] Giọng chung của engine hiện tại (mặc định cho chế độ 2 giọng)."""
        engine_id = self._engine_combo.currentData() or _ENGINE_EDGE
        if engine_id == _ENGINE_VIENEU:
            return str(self._vieneu_voice.currentData() or "")
        if engine_id == _ENGINE_GEMINI:
            return self._gemini_voice.currentText()
        return self._edge_voice.currentText()

    def _dubbing_catalog(self) -> tuple[str, list]:
        """(khoá engine, bộ giọng) của engine đang chọn cho bảng phân vai."""
        engine_id = self._engine_combo.currentData() or _ENGINE_EDGE
        if engine_id == _ENGINE_VIENEU:
            mode = self._vieneu_mode.currentData() or "standard"
            return f"vieneu_{mode}", list(self._vieneu_catalog.get(mode, []))
        if engine_id == _ENGINE_GEMINI:
            return _ENGINE_GEMINI, self._view_model.voice_catalog(_ENGINE_GEMINI, "")
        language = self._edge_lang.currentData() or "vi-VN"
        return f"edge_{language}", self._view_model.voice_catalog(_ENGINE_EDGE, language)

    # ── Source ────────────────────────────────────────────────────────────────

    # ── Engine selector ───────────────────────────────────────────────────────

    # ── Engine-specific config panels ─────────────────────────────────────────

    def _gemini_api_keys(self) -> str:
        """[v3.23.677] Mọi key Gemini (Cài đặt → Khoá API và AI), nối xuống dòng — adapter tự xoay key."""
        return "\n".join(self._ai_store.gemini_keys())

    def set_open_settings_callback(self, callback: "Callable[[], None] | None") -> None:
        """[v3.23.677] MainWindow tiêm hàm mở Cài đặt → "Khoá API và AI"."""
        self._gemini_key_summary.set_open_settings_callback(callback)

    def hideEvent(self, event) -> None:  # noqa: N802 — rời trang / đóng app: ghi sửa tay đang chờ
        line_edits = getattr(self, "_line_edits", None)
        if line_edits is not None:
            line_edits.flush_pending_save()
        super().hideEvent(event)

    def _make_edits_store(self):
        """[v3.23.654] Kho tệp sửa tay TTS trong thư mục dữ liệu người dùng (``None`` nếu không xác định được)."""
        from subtitles_extractor.infrastructure.tts.line_edits_store import LineEditsFileStore

        base = getattr(self._container, "user_data_dir", None)
        return LineEditsFileStore(base / "dubbing_edits") if base is not None else None

    def _emit_speaker_corrections(self) -> None:
        corrections = self._view_model.cue_corrections()  # [v3.23.654] người nói + "nói với ai"
        fingerprint = self._view_model.subtitle_fingerprint
        payload = self._with_auto_releases(fingerprint, corrections)
        key = (fingerprint, payload)
        if not payload or key == self._last_speaker_corrections:
            return
        self._last_speaker_corrections = key
        self._pending_cue_sync = (fingerprint, corrections)
        self.speaker_corrections_ready.emit(payload)

    def _with_auto_releases(self, fingerprint: str, corrections: dict) -> dict:
        """[v3.23.655] Thêm "chọn lại Tự động" cho phần (người nói / người nghe) đã đồng bộ lần trước mà nay bỏ sửa tay.

        Trước đây chọn lại "Tự động" chỉ bỏ sửa tay ở trang TTS; cue người dùng đã ghi sang trang Dịch vẫn ở đó (lần dịch
        sau vẫn theo người nói cũ, phân tích lại cũng không được đè).
        """
        from subtitles_extractor.application.services.visual_cue_serializer import CUE_AUTO

        synced = getattr(self, "_cue_sync_base", {}).get(fingerprint, {})
        payload = dict(corrections)
        for line_no, (old_speaker, old_addressee) in synced.items():
            speaker, addressee = corrections.get(line_no, (None, None))
            released = (
                CUE_AUTO if old_speaker is not None and speaker is None else speaker,
                CUE_AUTO if old_addressee is not None and addressee is None else addressee,
            )
            if released != (speaker, addressee):
                payload[line_no] = released
        return payload

    def show_speaker_sync_result(self, changed: int) -> None:
        """[v3.23.653] Báo kết quả đồng bộ người nói về trang Dịch (``changed`` < 0: chưa ghi được, sẽ thử lại)."""
        pending = getattr(self, "_pending_cue_sync", None)
        if changed >= 0 and pending is not None:
            # [v3.23.655] Đã ghi sang trang Dịch: mốc để nhận ra lần sau người dùng chọn lại "Tự động".
            if not hasattr(self, "_cue_sync_base"):
                self._cue_sync_base = {}
            self._cue_sync_base[pending[0]] = dict(pending[1])
            self._pending_cue_sync = None
        if changed > 0:
            text = self._translator.translate("tts.speaker_sync_done").replace("{count}", str(changed))
        elif changed == -1:
            text = self._translator.translate("tts.speaker_sync_bad_json")
        elif changed == -2:
            text = self._translator.translate("tts.speaker_sync_busy")
        elif changed == -3:
            text = self._translator.translate("tts.speaker_sync_other_film")
        else:
            return
        if changed < 0:
            self._last_speaker_corrections = ("", {})  # lần sửa sau gửi lại
        self._stage_lbl.setText(text)

    # ── Signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        vm = self._view_model
        vm.source_changed.connect(self._on_source_changed)
        # [v3.23.598] SAU khi trang xoá kết quả cũ: khôi phục sửa tay + hiện các dòng mới.
        vm.source_changed.connect(self._line_edits.on_source_changed)
        vm.busy_changed.connect(self._on_busy_changed)
        vm.progress_changed.connect(lambda r, l: (
            self._progress.setValue(int(r * 100)),
            self._stage_lbl.setText(f"{l}  ({int(r*100)}%)")
        ))
        vm.result_ready.connect(self._on_result_ready)
        # [v3.23.331] Điều phối hàng loạt: nạp xong tập nào thì tổng hợp ngay tập đó.
        vm.source_changed.connect(self._on_source_changed_for_batch)
        vm.error_occurred.connect(self._on_error_for_batch)
        vm.status_message.connect(self._stage_lbl.setText)
        vm.tts_cancelled.connect(self._on_tts_cancelled)
        vm.engine_available.connect(self._on_engine_available)

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _on_tts_cancelled(self) -> None:
        self._progress.setValue(0)
        self._stage_lbl.setText(self._translator.translate("tts.status_cancelled"))
        # [v3.23.650] Huỷ khi đang chạy CẢ BỘ: dọn hàng đợi. Trước đây `_batch_items` còn nguyên → lần TTS thường kế
        # tiếp bị coi là một tập của hàng đợi (không báo thành công, nuốt lỗi, hiện hộp "Đã huỷ…" cũ).
        if self._batch_items:
            self._batch_cancelled = True
            self._finish_batch_tts()
            return
        self._show_info(self._translator.translate("tts.toast_cancelled_t"), self._translator.translate("tts.toast_cancelled_b"))

    def _clear_results(self) -> None:
        """[1.5] Quét sạch bảng + tóm tắt kết quả TTS cũ khi đổi nguồn dữ liệu."""
        self._last_results = []
        # [v3.23.452] Trạng thái phát audio per-line (khởi tạo lười; phát KHI media nạp xong).
        self._audio_player = None
        self._audio_out = None
        self._pending_play = False
        self._table.clearContents()
        self._table.setRowCount(0)
        self._lbl_summary.setText(self._translator.translate("tts.status_no_result"))
        if hasattr(self, "_lbl_filter_count"):
            self._lbl_filter_count.setText("")
        for btn in (self._btn_export, self._btn_open, self._btn_open_file):
            btn.setEnabled(False)

    def _on_load_clicked(self) -> None:
        p = _safe_dialogs.open_file(
            self, "Chọn file phụ đề", "", "Phụ đề (*.srt *.ass *.ssa);;Tất cả (*)"
        )
        if p:
            self._clear_results()
            self._view_model.load_source_from_file(Path(p))

    def _on_pull_editor(self) -> None:
        if self._editor_event_provider:
            ev = self._editor_event_provider()
            if ev:
                self._clear_results()
                self._view_model.set_source_events(ev)
            else:
                self._show_warning(self._translator.translate("tts.toast_empty_t"), self._translator.translate("tts.toast_editor_empty"))
        else:
            self._show_warning(self._translator.translate("tts.toast_nolink_t"), self._translator.translate("tts.toast_no_editor_link"))

    def _on_pull_translate(self) -> None:
        if self._translate_event_provider:
            ev = self._translate_event_provider()
            if ev:
                self._clear_results()
                self._view_model.set_source_events(ev)
                self._show_info(self._translator.translate("common.ok"), self._translator.translate("tts.toast_got_lines").replace("{n}", str(len(ev))))
            else:
                self._show_warning(self._translator.translate("tts.toast_empty_t"), self._translator.translate("tts.toast_translate_empty"))
        else:
            self._show_warning(self._translator.translate("tts.toast_nolink_t"), self._translator.translate("tts.toast_no_translate_link"))

    def _on_base_speed_changed(self, value: float) -> None:
        """Đảm bảo max_speed ≥ tốc độ cơ bản + cảnh báo nếu base quá cao."""
        self._max_speed.setMinimum(value)
        if self._max_speed.value() < value:
            self._max_speed.setValue(value)
        if value >= 1.4:
            self._show_warning(
                self._translator.translate("tts.toast_highspeed_t"),
                self._translator.translate("tts.toast_highspeed_b").replace(
                    "{value}", f"{value:.1f}"
                ),
            )

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Trang TTS hiện ra: nạp bù danh sách giọng VieNeu nếu đang chọn engine này."""
        super().showEvent(event)
        engine_id = self._engine_combo.currentData() or _ENGINE_EDGE
        if engine_id == _ENGINE_VIENEU and not getattr(
            self, "_vieneu_voices_loaded", False
        ):
            self._load_vieneu_voices()

    def _on_browse_output(self) -> None:
        fmt = self._fmt.currentData() if hasattr(self, "_fmt") else "wav"
        fmt_names = {
            "wav": "WAV Audio", "flac": "FLAC Audio", "mp3": "MP3 Audio",
            "opus": "Opus Audio", "ogg": "OGG Vorbis", "m4a": "M4A/AAC Audio",
        }
        cur = self._output.text().strip()
        default = cur if cur else f"output_tts.{fmt}"
        p = _safe_dialogs.save_file(
            self, "Lưu file âm thanh", default,
            f"{fmt_names.get(fmt, 'Âm thanh')} (*.{fmt})",
        )
        if p:
            from pathlib import Path as _P
            self._output.setText(str(_P(p).with_suffix(f".{fmt}")))

    def _on_generate(self) -> None:
        # [v3.23.598] Ô lời đang mở sửa phải vào dữ liệu trước khi đọc phụ đề để tổng hợp.
        self._commit_active_edit()
        out = self._output.text().strip()
        if not out:
            self._show_warning(self._translator.translate("tts.toast_nopath_t"), self._translator.translate("tts.toast_no_wav_path"))
            return

        idx = self._engine_combo.currentIndex()
        engine_id = self._engine_combo.itemData(idx) or _ENGINE_EDGE

        # [v3.23.115] Kiểm engine đã cài chưa TRƯỚC khi chạy -> báo cách cài ngay, tránh
        # để người dùng chờ rồi mới nhận lỗi import giữa chừng.
        if getattr(self, "_engine_avail", {}).get(engine_id) is False:
            _install = {
                _ENGINE_GEMINI: "pip install google-genai",
                _ENGINE_EDGE: "pip install edge-tts",
                _ENGINE_VIENEU: "pip install vieneu soundfile",
            }.get(engine_id, "")
            self._show_warning(
                self._translator.translate("tts.toast_engine_notready_t"),
                self._translator.translate("tts.toast_engine_notready_install").replace(
                    "{install}", _install
                )
                if _install
                else self._translator.translate("tts.toast_engine_notready_b"),
            )
            return

        # [fix B4] Validate theo engine trước khi start worker
        if engine_id == _ENGINE_GEMINI:
            if not self._gemini_api_keys():
                self._show_warning(
                    self._translator.translate("tts.toast_noapikey_t"),
                    self._translator.translate("tts.toast_need_gemini_key"),
                )
                return
        elif engine_id == _ENGINE_VIENEU:
            # [v3.23.252] Đang nạp giọng VieNeu (thread nền) -> combo chưa sẵn
            # sàng, bấm chạy sẽ lỗi. Chặn sớm với thông báo rõ. Loader đang chạy
            # = giọng chưa nạp xong.
            if self._voice_loader_running():
                self._show_warning(
                    self._translator.translate("tts.toast_loading_voices_t"),
                    self._translator.translate("tts.toast_loading_voices_b"),
                )
                return
            # [v3.23.195] Ref audio VieNeu là TUỲ CHỌN (có preset), nhưng nếu đã nhập
            # đường dẫn thì phải tồn tại — báo sớm thay vì lỗi giữa phiên TTS.
            vieneu_ref = self._vieneu_ref_path.text().strip()
            if vieneu_ref and not Path(vieneu_ref).exists():
                self._show_warning(
                    self._translator.translate("tts.toast_file_notfound_t"),
                    self._translator.translate("tts.toast_file_notfound_b").replace(
                        "{ref}", str(vieneu_ref)
                    ),
                )
                return
        self._save_settings()
        kwargs = dict(
            engine=engine_id,
            base_speed=self._speed.value(),
            max_speed=self._max_speed.value(),
            device=self._device.currentText(),
            normalize=self._normalize.isChecked(),
            voice_clarity=self._voice_clarity.isChecked(),
            high_quality=self._high_quality.isChecked(),
            target_lufs=self._lufs.currentData(),
            output_format=self._fmt.currentData(),
            output_bitrate_kbps=self._bitrate.currentData(),
            wav_subtype=self._wav_subtype.currentData(),
            retry_count=self._retry.value(),
            retry_delay_s=self._retry_delay.value(),
            output_path=Path(out),
            clean_tags=self._clean_tags.isChecked(),
            dialog_pause_ms=self._dialog_pause.value(),
            max_overlap_ms=self._max_overlap.value(),
            skip_overlap_ms=self._skip_overlap.value(),
            double_pass=self._double_pass.isChecked(),
            elastic_timing=self._elastic_timing.isChecked(),
            max_drift_s=self._max_drift.value() / 1000.0,
            lead_in_s=self._lead_in.value() / 1000.0,
            anchor_gap_s=self._anchor_gap.value(),
            max_segment_s=self._max_segment.value(),
            comfort_speed_ratio=self._comfort_ratio.value(),
            min_pause_ratio=self._min_pause.value() / 100.0,
            max_intra_gap_s=self._max_intra_gap.value(),
            timing_strategy=self._strategy.currentData(),
            allow_audio_overlap=self._allow_overlap.isChecked(),
            min_stretch_ratio=self._min_stretch.value(),
            last_line_max_extend_s=self._last_extend.value() / 1000.0,
            skip_paren=self._skip_paren.isChecked(),
            skip_square=self._skip_square.isChecked(),
            skip_curly=self._skip_curly.isChecked(),
            skip_music_pair=self._skip_music_pair.isChecked(),
            skip_music_line=self._skip_music_line.isChecked(),
        )
        if engine_id == _ENGINE_EDGE:
            kwargs.update(language=self._edge_lang.currentData() or "vi-VN",
                          speaker=self._edge_voice.currentText(),
                          edge_concurrency=self._edge_concurrency.value())
        elif engine_id == _ENGINE_GEMINI:
            # [v3.23.248] "Tự động" (value < 0) -> None = dùng mặc định của model.
            temp_val = self._gemini_temperature.value()
            gemini_temp = temp_val if temp_val >= 0.0 else None
            kwargs.update(language=self._gemini_model.currentText(),
                          speaker=self._gemini_voice.currentText(),
                          api_key=self._gemini_api_keys(),
                          style_prompt=self._gemini_style.toPlainText().strip(),
                          affective_dialog=self._gemini_affective.isChecked(),
                          gemini_temperature=gemini_temp)
            # [v3.23.696] Chỉ model TTS 3.8+ đọc chỉ dẫn cách đọc riêng (speech_metadata.style); model cũ bỏ qua nó mà
            # chỉ dẫn vẫn vào chữ ký cache → bật/tắt ô này làm TTS lại cả phim vô ích.
            if self._gemini_line_emotion.isChecked() and uses_verbatim_tts_schema(self._gemini_model.currentText()):
                kwargs["line_styles"] = self._view_model.dubbing_line_styles()
        else:  # VieNeu
            # [v3.23.195] Danh sách giọng nạp LAZY: nếu chưa nạp (combo chỉ có mặc định)
            # thì dùng giọng đã LƯU từ phiên trước — tránh mất lựa chọn giọng; tên giọng
            # từ chế độ khác đã có khớp mềm (v194) xử lý an toàn phía adapter.
            vieneu_speaker = self._vieneu_voice.currentData() or ""
            if not getattr(self, "_vieneu_voices_loaded", False):
                vieneu_speaker = vieneu_speaker or getattr(
                    self, "_vieneu_saved_voice", ""
                )
            kwargs.update(language="vi-VN",
                          speaker=vieneu_speaker,
                          ref_audio_path=self._vieneu_ref_path.text().strip(),
                          ref_text="",
                          vieneu_mode=self._vieneu_mode.currentData() or "standard",
                          # [v3.23.595] v3 Turbo bỏ qua style → luôn "natural" để dùng
                          # chung engine đã nạp (khoá cache gồm sắc thái).
                          vieneu_emotion=(
                              "natural"
                              if (self._vieneu_mode.currentData() or "") == "v3turbo"
                              else self._vieneu_emotion.currentData() or "natural"
                          ),
                          vieneu_precision=self._vieneu_precision(),
                          vieneu_emotion_cues=self._vieneu_emotion_cues.isChecked(),
                          vieneu_force_cpu=self._vieneu_force_cpu.isChecked(),
                          vieneu_batch_size=self._vieneu_batch.currentData())
        kwargs["media_duration_s"] = getattr(self, "_media_duration_s", None)
        kwargs["incremental"] = self._incremental.isChecked()
        # [v3.23.593] Lồng tiếng: giọng riêng từng dòng theo bảng phân vai (rỗng = một
        # giọng).
        kwargs["line_voices"] = self._dubbing.line_voices()
        # [v3.23.594] Nhớ bộ giọng (engine + ngôn ngữ/chế độ) của lần chạy này: TTS
        # lại vài dòng chỉ lấy bảng phân vai hiện tại khi vẫn cùng bộ giọng.
        self._last_dubbing_key = self._dubbing_catalog()[0]
        self._view_model.start_tts(**kwargs)

    @staticmethod
    def _reveal_in_os(target: Path) -> None:
        """Mở đường dẫn bằng trình quản lý file mặc định trên Windows/macOS/Linux."""
        import os
        import subprocess
        import sys

        path_str = str(target)
        if sys.platform == "darwin":
            subprocess.Popen(["open", path_str], **no_window_kwargs())
        elif os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(path_str)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path_str], **no_window_kwargs())

    def _on_open_folder(self) -> None:
        out = self._view_model.last_output_path
        if out and out.parent.exists():
            self._reveal_in_os(out.parent)

    def _on_open_file(self) -> None:
        """Mở trực tiếp file âm thanh bằng ứng dụng mặc định."""
        out = self._view_model.last_output_path
        if out and out.exists():
            self._reveal_in_os(out)

    def _on_cancel_clicked(self) -> None:
        """Huỷ: khi đang chạy cả bộ thì dừng HẲN hàng đợi, không chỉ tập hiện tại."""
        if self._batch_items:
            self._batch_cancelled = True
            remaining = len(self._batch_items) - self._batch_index - 1
            self._stage_lbl.setText(
                f"Đang huỷ… (bỏ {max(0, remaining)} tập còn lại)"
            )
        self._view_model.cancel_tts()

    # ── Tổng hợp cả bộ ───────────────────────────────────────────────────────
    def _on_source_changed(self, n: int) -> None:
        # [v3.23.138] Nhất quán với trang Dịch: nạp nguồn mới -> xoá kết quả audio cũ
        # (bảng + tóm tắt + nút xuất). Trước đây chỉ đường "Nạp từ file" mới xoá; khi
        # phụ đề được ĐẨY sang (từ Editor) thì bảng vẫn giữ kết quả của nguồn CŨ.
        self._clear_results()
        self._refresh_line_audio_usage()  # [v3.23.657] kết quả cũ đã bỏ → nút Dọn không còn nhắm vào lượt trước
        if n > 0:
            self._lbl_source.setText(self._translator.translate("tts.res_loaded").replace("{n}", str(n)))
            self._stage_lbl.setText(self._translator.translate("tts.status_ready"))
        else:
            self._lbl_source.setText(self._translator.translate("tts.no_subtitle"))
        self._lbl_source.setStyleSheet("font-weight:bold;")
        self._update_action_states()

    def _on_output_text_changed(self, text: str) -> None:
        """[3.3] Khi người dùng gõ đuôi file (.mp3/.flac…), đồng bộ combo Định dạng.

        Bỏ qua nếu thay đổi do chính combo gây ra (cờ ``_syncing_fmt``) để tránh vòng
        lặp. Chỉ đổi combo khi đuôi nằm trong các định dạng được hỗ trợ.
        """
        if getattr(self, "_syncing_fmt", False):
            return
        from pathlib import Path as _P
        suffix = _P(text.strip()).suffix.lower().lstrip(".")
        if not suffix or not hasattr(self, "_fmt"):
            return
        idx = self._fmt.findData(suffix)
        if idx >= 0 and idx != self._fmt.currentIndex():
            self._syncing_fmt = True
            try:
                self._fmt.setCurrentIndex(idx)
            finally:
                self._syncing_fmt = False

    def _on_busy_changed(self, busy: bool) -> None:
        if busy:
            self._progress.setValue(0)
            self._stage_lbl.setText(self._translator.translate("tts.status_starting"))
            # [3.4] Khoá các nút thao tác file khi đang gen/ghi để tránh click đúp
            # tạo nhiều file lỗi; sẽ được bật lại đúng trạng thái khi có kết quả.
            self._btn_export.setEnabled(False)
            self._btn_open.setEnabled(False)
            self._btn_open_file.setEnabled(False)
        self._btn_gen.setEnabled(not busy)
        self._btn_can.setEnabled(busy)
        # [v3.23.331] Khoá nút "cả bộ" khi đang chạy để không xếp chồng hai hàng đợi.
        self._btn_batch.setEnabled(not busy)
        # [v3.23.252] Vừa xong TTS mà ĐANG chọn VieNeu và giọng VẪN đang nạp (vd
        # đổi chế độ ngay sau khi xong) -> giữ nút khoá, tránh bật khi chưa sẵn sàng.
        if not busy and (
            self._engine_combo.currentData() or _ENGINE_EDGE
        ) == _ENGINE_VIENEU:
            if self._voice_loader_running():
                self._btn_gen.setEnabled(False)
        # Disable nguồn + output khi đang bận
        for btn in (self._btn_load, self._btn_editor, self._btn_trans, self._btn_out):
            btn.setEnabled(not busy)
        if not busy:
            self._update_action_states()
        # [v3.23.657] Không dọn audio dòng khi đang chạy; xong lượt (kể cả huỷ/TTS lại) → đo lại.
        if busy and hasattr(self, "_btn_clean_line_audio"):
            self._btn_clean_line_audio.setEnabled(False)
        elif not busy:
            self._refresh_line_audio_usage()

    #: [v3.23.455] Im lặng chèn trước tiếng nói khi nghe thử (giây) — cho thiết bị âm thanh
    #: khởi động xong, tránh nuốt đầu câu / rè "tẹt tẹt".
    _PREVIEW_SILENCE_S = 0.8

    def set_shorten_provider(self, provider: Callable[[], tuple[str, str, str]]) -> None:
        """[v3.23.629] Nguồn khoá API / model / ngôn ngữ cho bước rút gọn bằng AI (trang Dịch)."""
        self._shorten_provider = provider

    def _start_shorten(self, fits: list) -> None:
        """Gửi các dòng không vừa khung cho AI rút gọn (luồng nền) rồi hiện xem trước."""
        # [v3.23.650] Bấm "Rút gọn" lần hai khi lần đầu còn chạy: gán đè ``_shorten_worker`` làm QThread (không parent)
        # đang chạy bị Python giải phóng → "Destroyed while thread is still running" → abort().
        previous = getattr(self, "_shorten_worker", None)
        if previous is not None and not is_deleted(previous) and previous.isRunning():
            return
        provider = getattr(self, "_shorten_provider", None)
        keys, model, lang = provider() if provider is not None else ("", "", "")
        if not keys.strip():
            self._stage_lbl.setText(self._translator.translate("tts.shorten_no_key"))
            self._show_warning(
                self._translator.translate("tts.btn_shorten"),
                self._translator.translate("tts.shorten_no_key"),
            )
            return
        from subtitles_extractor.application.use_cases.shorten_for_speech import (
            ShortenForSpeechUseCase,
        )
        from subtitles_extractor.presentation.workers.shorten_worker import ShortenWorker

        vm = self._view_model
        texts = [str(getattr(e, "text", "") or "").replace("\n", " ") for e in vm.source_events]
        contexts = [vm.line_context(p) for p in range(len(texts))]
        translator = self._container.make_subtitle_translator(keys, retry_count=3)
        self._shorten_worker = ShortenWorker(
            ShortenForSpeechUseCase(translator), fits, texts, lang or "Vietnamese",
            model or "gemini-3.1-flash-lite",
            source_texts=[c[0] for c in contexts], speakers=[c[1] for c in contexts],
            addressees=[c[2] for c in contexts],
        )
        self._shorten_worker.finished_ok.connect(self._on_shortened)
        self._shorten_worker.failed.connect(self._on_shorten_failed)
        self._shorten_worker.cancelled.connect(self._on_shorten_cancelled)
        self._shorten_worker.finished.connect(self._shorten_worker.deleteLater)
        self._stage_lbl.setText(
            self._translator.translate("tts.shorten_running").replace("{count}", str(len(fits)))
        )
        self._shorten_worker.start()

    def _on_shorten_cancelled(self) -> None:
        # [v3.23.650] Slot của trang (chạy ở luồng GUI) thay cho lambda (chạy ở luồng worker → QLabel.setText sai luồng).
        self._stage_lbl.setText("…")

    def _on_shortened(self, lines: list) -> None:
        self._line_edits.apply_shortened(list(lines))

    def _on_shorten_failed(self, message: str) -> None:
        text = self._translator.translate("tts.shorten_failed").replace("{error}", message)
        self._stage_lbl.setText(text)
        self._show_warning(self._translator.translate("tts.btn_shorten"), text)

    def _rebuild_after_structure(self) -> None:
        """[v3.23.625] Vẽ lại bảng dòng sau khi thêm/xoá dòng."""
        results = list(self._view_model.last_results)
        if results:
            self._last_results = results
            self._populate_table(results)
            self._on_result_selection_changed()
        else:
            self._last_results = []
            self._line_edits.show_source_rows()

    def _resynthesize_positions(self, positions: set[int]) -> None:
        """[v3.23.625] TTS lại các dòng ``positions`` (dòng vừa thêm) rồi ghép lại master."""
        if not self._view_model.resynthesize_lines(
            set(positions), line_voices=self._resynthesis_line_voices()
        ):
            self._stage_lbl.setText(self._translator.translate("tts.line_added_retts"))

    def _resynthesis_line_voices(self) -> tuple[str, ...] | None:
        """[v3.23.594] Giọng từng dòng cho TTS lại: theo bảng phân vai HIỆN TẠI.

        Bản 593 luôn dùng giọng của lần TTS trước: đổi giọng một vai rồi TTS lại các dòng
        của vai đó vẫn ra giọng cũ. Chỉ lấy bảng hiện tại khi lồng tiếng đang bật và vẫn
        cùng bộ giọng với lần chạy trước (adapter TTS lại là adapter cũ — giọng engine
        khác nó không đọc được).

        Returns:
            Giọng theo vị trí dòng, hoặc ``None`` để giữ giọng của lần TTS trước.
        """
        if not self._dubbing_panel.is_dubbing_enabled():
            return None
        if self._dubbing_catalog()[0] != getattr(self, "_last_dubbing_key", None):
            return None
        return self._dubbing.line_voices() or None

    def _on_resynthesize_selected(self) -> None:
        """TTS lại các dòng đang chọn (kể cả nội dung vừa sửa)."""
        sel = self._selected_event_indices()  # capture TRƯỚC khi commit (commit có thể đổi ô)
        if not sel:
            return
        self._commit_active_edit()  # đảm bảo nội dung vừa sửa đã vào model + source_events
        ok = self._view_model.resynthesize_lines(
            set(sel), line_voices=self._resynthesis_line_voices()
        )
        if not ok:
            self._show_warning(
                "Không TTS lại được",
                self._translator.translate("tts.tip_resynth_unsupported")
                if self._view_model.has_result and not self._view_model.can_resynthesize
                else "Cần đã chạy TTS một lần và còn file audio gốc.",
            )

    def _refresh_line_audio_usage(self) -> None:
        """[v3.23.657] Đo lại dung lượng audio từng dòng (gọi sau mỗi lượt chạy/dọn, không mỗi lần chọn dòng)."""
        if not hasattr(self, "_btn_clean_line_audio"):
            return
        try:
            self._line_audio_bytes = int(self._view_model.line_audio_usage())
        except (AttributeError, OSError, TypeError, ValueError):
            self._line_audio_bytes = 0
        key = "tts.btn_clean_line_audio_size" if self._line_audio_bytes else "tts.btn_clean_line_audio"
        self._btn_clean_line_audio.setText(self._translator.translate(
            key, size=f"{self._line_audio_bytes / 1e6:.1f}"
        ))
        busy = bool(getattr(self._view_model, "is_busy", False))
        self._btn_clean_line_audio.setEnabled(self._line_audio_bytes > 0 and not busy)

    def _on_clean_line_audio(self) -> None:
        """[v3.23.657] Người dùng bấm "Dọn audio từng dòng" — xác nhận rồi xoá (master + SRT giữ nguyên)."""
        if getattr(self._view_model, "is_busy", False):
            return
        size = f"{self._line_audio_bytes / 1e6:.1f}"
        answer = QMessageBox.question(
            self, self._translator.translate("tts.dlg_clean_line_audio_title"),
            self._translator.translate("tts.dlg_clean_line_audio_body", size=size),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        freed = self._view_model.clean_line_audio()
        if freed < 0:
            return
        self._last_results = list(self._view_model.last_results)
        self._refresh_line_audio_usage()
        self._on_result_selection_changed()
        self._stage_lbl.setText(self._translator.translate(
            "tts.ss_clean_line_audio_done", size=f"{freed / 1e6:.1f}"
        ))

    def _on_result_ready(self, results) -> None:
        rs: list[TTSSegmentResult] = list(results)
        logger.info("UI: bắt đầu hiển thị kết quả TTS (%d câu).", len(rs))
        self._last_results = rs  # lưu để xuất chi tiết debug
        self._populate_table(rs)
        if not getattr(self, "_batch_items", None):  # [v3.23.682] cả bộ: báo một lần khi xong hàng đợi
            notify_task_finished(
                self, self._translator.translate("common.task_done_tts"),
                self._translator.translate("common.task_lines", count=len(rs)),
            )
        logger.debug("UI: đã đổ bảng kết quả TTS.")
        n_ok    = sum(1 for r in rs if not r.was_skipped)
        n_trunc = sum(1 for r in rs if r.was_truncated)
        n_skip  = sum(1 for r in rs if r.was_skipped)
        out = self._view_model.last_output_path
        size_txt = ""
        if out and out.exists():
            try:
                size_txt = f" · {out.stat().st_size / 1e6:.1f} MB · {out.suffix.lstrip('.').upper()}"
            except OSError:
                pass
        # Hiển thị breakdown API+TS nếu có
        try:
            from subtitles_extractor.infrastructure.tts.edge_tts_adapter import _EDGE_API_SPEED_MAX
            n_ts = sum(1 for r in rs if not r.was_skipped and r.speed_used > _EDGE_API_SPEED_MAX + 0.02)
            ts_info = f"  🔀 {n_ts} API+TS" if n_ts > 0 else ""
        except ImportError:
            ts_info = ""
        self._lbl_summary.setText(
            self._translator.translate("tts.res_summary").replace("{ok}", str(n_ok)).replace("{ts}", ts_info).replace("{trunc}", str(n_trunc)).replace("{skip}", str(n_skip))
            + (f"  →  {out.name}{size_txt}" if out else "")
            # [v3.23.459] Nhắc số dòng dài bất thường (nếu có) để người dùng biết cần soi lại.
            + (f"  ·  ⚠ {len(self._abnormal_event_indices())} dòng dài bất thường"
               if self._abnormal_event_indices() else "")
            + self._incremental_note()
        )
        self._btn_open.setEnabled(bool(out and out.parent.exists()))
        self._btn_open_file.setEnabled(bool(out and out.exists()))
        self._btn_export.setEnabled(bool(rs))
        self._refresh_line_audio_usage()
        # [v3.23.450] Cập nhật trạng thái nút nghe/TTS-lại theo kết quả mới (connect 1 lần ở
        # _build_result_group để tránh nối trùng).
        self._on_result_selection_changed()

        # Cảnh báo chất lượng: nếu nhiều câu phải nén nhanh hoặc lệch đồng bộ
        # lớn, gợi ý người dùng dịch súc tích hơn (gốc rễ là bản dịch quá dài).
        spoken = [r for r in rs if not r.was_skipped]
        if spoken:
            n_fast = sum(1 for r in spoken if r.speed_used >= 1.8)
            n_drift = sum(
                1 for r in rs
                if r.adjusted_start_sec >= 0 and (r.adjusted_start_sec - r.start_sec) > 1.5
            )
            pct_fast = n_fast / len(spoken) * 100
            if pct_fast >= 25 or n_drift >= max(5, len(rs) * 0.02):
                drift_frag = (
                    self._translator.translate("tts.toast_sync_drift").replace(
                        "{drift}", str(n_drift)
                    )
                    if n_drift
                    else ""
                )
                self._show_warning(
                    self._translator.translate("tts.toast_sync_t"),
                    self._translator.translate("tts.toast_sync_b")
                    .replace("{pct}", f"{pct_fast:.0f}")
                    .replace("{drift}", drift_frag),
                )
        # [v3.23.331] Đang chạy cả bộ -> vẫn lưu dự án nhưng KHÔNG hiện hộp thoại từng
        # tập (84 tập sẽ là 84 hộp thoại), rồi chuyển sang tập kế.
        if self._batch_items:
            # [v3.23.659] KHÔNG phát tts_completed: cửa sổ chính lưu nó vào dự án ĐANG MỞ — 86 tập cả bộ từng ghi đè
            # đường dẫn giọng đọc của dự án đó (và Xuất bản "Lấy từ app" lấy giọng của tập cuối cả bộ).
            self._batch_done = getattr(self, "_batch_done", 0) + 1
            self._advance_batch_tts()
            return

        self._show_success(
            self._translator.translate("tts.toast_complete_t"),
            self._translator.translate("tts.toast_complete_b")
            .replace("{ok}", str(n_ok))
            .replace("{total}", str(len(rs)))
            .replace("{name}", out.name if out else "WAV"),
        )
        # Liên thông: báo cho MainWindow lưu kết quả TTS vào dự án.
        if out and Path(out).exists():
            logger.debug("UI: phát tín hiệu tts_completed để lưu dự án.")
            self.tts_completed.emit(str(out))
        logger.info("UI: hoàn tất hiển thị kết quả TTS.")

    def _incremental_note(self) -> str:
        """[v3.23.630] Đoạn cuối dòng tóm tắt: lượt này dùng lại bao nhiêu dòng cũ."""
        summary = self._view_model.reuse_summary
        if summary is None:
            return ""
        reused, total, changed = summary
        return "  ·  " + (
            self._translator.translate("tts.status_incremental_note")
            .replace("{reused}", str(reused)).replace("{total}", str(total))
            .replace("{changed}", str(changed))
        )

    #: [v3.23.674] Lệnh cài từng engine (chỉ hiện cho engine CHƯA cài).
    _INSTALL_COMMANDS = (
        (_ENGINE_EDGE, "Edge TTS", "pip install edge-tts soundfile"),
        (_ENGINE_GEMINI, "Gemini TTS", "pip install google-genai soundfile"),
        (_ENGINE_VIENEU, "VieNeu-TTS", "pip install vieneu soundfile"),
    )

    # ── Settings ──────────────────────────────────────────────────────────────

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_action_states(self) -> None:
        has = self._view_model.has_source and not self._view_model.is_busy
        # [v3.23.252] Nếu ĐANG chọn VieNeu và giọng đang nạp, giữ nút khoá — giọng
        # chưa sẵn sàng thì chạy sẽ lỗi. Engine khác không phụ thuộc loader này.
        if (self._engine_combo.currentData() or _ENGINE_EDGE) == _ENGINE_VIENEU:
            if self._voice_loader_running():
                has = False
        self._btn_gen.setEnabled(has)

    def _show_success(self, t, c): _feedback.show_success(self, t, c)
    def _show_info(self, t, c):    _feedback.show_info(self, t, c)
    def _show_warning(self, t, c): _feedback.show_warning(self, t, c)


__all__ = ["TTSPage"]
