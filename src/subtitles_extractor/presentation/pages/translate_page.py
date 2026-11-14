"""Trang "Dịch phụ đề" — dịch phụ đề đa giai đoạn bằng AI (Gemini).

Quy trình: nạp nguồn (từ tệp SRT/ASS hoặc từ trang Biên tập) → cấu hình API key,
ngôn ngữ đích và 4 giai đoạn (tiền xử lý → dịch thô → phong cách → bản địa hoá)
→ chạy nền → xem bảng so sánh gốc/đích → xuất tệp.

Tầng View chỉ dựng widget và uỷ thác mọi logic cho ``TranslatePageViewModel``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QTableWidgetItem,
    QWidget,
)

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.presentation.theme import feedback as _feedback
from subtitles_extractor.presentation.utils.table_selection import select_rows_bulk
from subtitles_extractor.presentation.utils.time_format import seconds_to_display
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets
from subtitles_extractor.presentation.view_models.translate_page_view_model import (
    TranslatePageViewModel,
)

from subtitles_extractor.presentation.utils.task_notifier import notify_task_finished
from subtitles_extractor.presentation.pages.translate_batch import TranslateBatchMixin
from subtitles_extractor.presentation.pages.translate_constants import (  # noqa: F401 — giữ tên cũ trên module này
    _BATCH_STOP_MARKERS,
    _LANGUAGE_CODES,
    DEFAULT_ANALYSIS_MODEL,
    _TARGET_LANGUAGES,
    _DEFAULT_MODELS,
    NO_FALLBACK_CHOICE,
    DEFAULT_REFINE_MODEL,
    _STYLE_PRESETS,
    _SETTINGS_ORG,
    _SETTINGS_APP,
    _RESIZE_ROWS_THRESHOLD,
    _DEFAULT_ROW_HEIGHT_PX,
    EDITS_ONLY_DEFAULT,
)
from subtitles_extractor.presentation.pages.translate_cues import TranslateCuesMixin
from subtitles_extractor.presentation.pages.translate_analysis import TranslateAnalysisMixin
from subtitles_extractor.presentation.pages.translate_keys import TranslateKeysMixin
from subtitles_extractor.presentation.pages.translate_models import TranslateModelsMixin
from subtitles_extractor.presentation.pages.translate_table import TranslateTableMixin
from subtitles_extractor.presentation.pages.translate_tools import TranslateToolsMixin
from subtitles_extractor.presentation.pages.translate_ui import TranslateUiMixin
from subtitles_extractor.presentation.pages.translate_widgets import (  # noqa: F401 — giữ tên cũ trên module này
    _StagePanel,
    parse_quota_rows,
    replace_model_items,
)
from subtitles_extractor.presentation.pages.translate_export import TranslateExportMixin

logger = logging.getLogger(__name__)



class TranslatePage(
    TranslateUiMixin, TranslateKeysMixin, TranslateModelsMixin, TranslateAnalysisMixin, TranslateTableMixin,
    TranslateToolsMixin, TranslateBatchMixin, TranslateExportMixin, TranslateCuesMixin, QWidget,
):
    """Trang dịch phụ đề đa giai đoạn bằng AI."""

    #: [v3.23.658] Dây chuyền phim bộ: khâu Dịch chạy xong (``StageRunResult``).
    series_batch_finished = Signal(object)

    def __init__(
        self, container: ApplicationContainer, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("translatePage")
        self._container = container
        # [v3.23.365] Truy cập translator để externalize dần chuỗi UI (đa ngôn ngữ).
        self._translator = container.translator
        self._view_model = TranslatePageViewModel(container)
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

        self._build_ui()
        protect_scroll_widgets(self)
        self._connect_signals()
        self._restore_persisted_inputs()
        self._update_action_states()
        # Provider được MainWindow tiêm vào để lấy events từ trang Biên tập.
        self._editor_event_provider: "Callable[[], list[SubtitleEvent]] | None" = None
        # [#11] Nguồn đường dẫn video đang mở ở trang Biên tập (cho Auto-Attach).
        self._editor_video_provider: "Callable[[], str | None] | None" = None
        # Nguồn video hiện hành (để đặt tên file xuất theo quy ước <tên>.<lang>.<fmt>).
        self._source_video_path: str = ""

    def suggest_source_path(self, video_path: str | Path) -> None:
        """Ghi nhớ nguồn video hiện hành để đặt tên file bản dịch theo quy ước.

        Args:
            video_path: Đường dẫn video, dạng ``str`` hoặc ``pathlib.Path``. Luôn được
                chuẩn hoá về ``str`` để các hàm tầng TM (vốn khai báo ``video_path: str``)
                không nhận nhầm ``WindowsPath`` rồi gọi ``.strip()`` gây lỗi.
        """
        self._flush_context_save()  # [v3.23.652] lưu sửa tay ngữ cảnh của video CŨ trước khi đổi
        self._source_video_path = str(video_path) if video_path else ""

    def set_editor_event_provider(
        self, provider: "Callable[[], list[SubtitleEvent]]"
    ) -> None:
        """Tiêm hàm cung cấp danh sách events hiện tại của trang Biên tập."""
        self._editor_event_provider = provider

    def set_editor_video_provider(
        self, provider: "Callable[[], str | None]"
    ) -> None:
        """[#11] Tiêm hàm cung cấp đường dẫn video đang mở ở trang Biên tập."""
        self._editor_video_provider = provider

    # ── Dựng giao diện ───────────────────────────────────────────────────
    def _on_emotion_cues_toggled(self, checked: bool) -> None:
        """[v3.23.448] Lưu lựa chọn sinh tag cảm xúc vào Cài đặt để container đọc khi dịch."""
        try:
            self._container.settings_service.update(
                translation={"translation_emotion_cues": bool(checked)}
            )
        except (AttributeError, KeyError, ValueError) as exc:
            logger.warning("Không lưu được cài đặt tag cảm xúc: %s", exc)

    def _on_attach_video_toggled(self, checked: bool) -> None:
        self._set_video_controls_enabled(checked)
        if not checked:
            self._emotion_cues_check.setChecked(False)
        # [v3.23.120] Mặc định BẬT phân tích hình ảnh khi đính video (theo yêu cầu Toan);
        # người dùng vẫn có thể bỏ chọn nếu muốn tiết kiệm token.
        self._visual_cues_check.setChecked(checked)

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        """Bật/tắt các ô chỉ có nghĩa khi có video ngữ cảnh."""
        self._btn_pick_video.setEnabled(enabled)
        for cb in (self._video_stage_literal, self._video_stage_style, self._video_stage_localize):
            cb.setEnabled(enabled)
        clip_check = getattr(self, "_clip_video_check", None)
        if clip_check is not None:
            clip_check.setEnabled(enabled)
        # Visual Cues cũng cần video → chỉ cho bật khi đã bật ngữ cảnh video.
        self._visual_cues_check.setEnabled(enabled)
        # [v3.23.448] Tag cảm xúc cũng cần video ngữ cảnh (hình + âm thanh).
        self._emotion_cues_check.setEnabled(enabled)

    def _session_video_path(self) -> str:
        """[v3.23.609] Video mà phụ đề đang nạp thuộc về (khoá phiên, đặt tên file xuất).

        Khác video ĐÍNH KÈM gửi Gemini (:meth:`_selected_video_path`, chỉ có khi bật ô
        đính video): phiên dịch, Bộ nhớ dịch phim bộ và tên file xuất cần biết phim là
        phim nào kể cả khi không đính video.
        """
        return self._source_video_path or self._video_path or ""

    def _on_pick_video_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video làm ngữ cảnh", "",
            "Video (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.ts);;Tất cả (*.*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._video_path = path
            from pathlib import Path as _P
            self._video_path_label.setText(_P(path).name)

    def _selected_video_path(self) -> str | None:
        if getattr(self, "_attach_video_check", None) is None:
            return None
        if self._attach_video_check.isChecked() and self._video_path:
            return self._video_path
        return None

    def _selected_attach_video_stages(self) -> frozenset:
        if not (getattr(self, "_attach_video_check", None) and self._attach_video_check.isChecked()):
            return frozenset()
        kinds = set()
        if self._video_stage_literal.isChecked():
            kinds.add(TranslationStageKind.LITERAL)
        if self._video_stage_style.isChecked():
            kinds.add(TranslationStageKind.STYLE)
        if self._video_stage_localize.isChecked():
            kinds.add(TranslationStageKind.LOCALIZE)
        return frozenset(kinds)

    # ── Kết nối signal ───────────────────────────────────────────────────
    def _connect_signals(self) -> None:
        vm = self._view_model
        vm.source_changed.connect(self._on_source_changed)
        # [v3.23.332] Điều phối dịch cả bộ: nạp xong tập nào thì dịch ngay tập đó.
        vm.source_changed.connect(self._on_source_changed_for_batch)
        vm.result_ready.connect(self._on_result_ready)
        vm.progress_changed.connect(self._on_progress_changed)
        vm.busy_changed.connect(self._on_busy_changed)
        vm.error_occurred.connect(self._on_error)
        vm.analyze_error.connect(self._on_analyze_error)   # [fix B6] riêng biệt
        vm.status_message.connect(self._on_status_message)
        vm.translation_cancelled.connect(self._on_cancelled)
        vm.translation_finished.connect(self._on_translation_finished)
        vm.resume_detected.connect(self._on_resume_detected)   # [Q3]
        vm.analyze_context_ready.connect(self._on_analyze_context_ready)
        vm.analysis_restored.connect(self._on_analysis_restored)  # [v3.23.45] nạp lại
        vm.visual_cues_topped_up.connect(self._on_visual_cues_topped_up)  # [v3.23.559]
        # [v3.23.601] Kết quả nút "Bổ sung người nói".
        vm.speaker_completion_done.connect(self._on_speaker_completion_done)
        vm.context_names_corrected.connect(self._fill_context_fields)  # [v3.23.567]
        vm.models_ready.connect(self._on_models_ready)
        vm.error_occurred.connect(self._on_fetch_error_restore_btn)

    def _on_fetch_error_restore_btn(self, _msg: str) -> None:
        """Khôi phục trạng thái nút Tải model nếu fetch thất bại."""
        if not self._btn_refresh_models.isEnabled():
            self._btn_refresh_models.setEnabled(True)
            self._btn_refresh_models.setText(self._translator.translate("translate.btn_load_models"))

    def set_push_to_editor_callback(
        self, callback: "Callable[[list[SubtitleEvent]], None] | None"
    ) -> None:
        """Tiêm callback để gửi bản dịch sang trang Biên tập."""
        self._view_model.set_push_to_editor_callback(callback)

    def cleanup(self) -> None:
        """Dọn worker dịch khi đóng app/trang (gọi từ MainWindow.closeEvent)."""
        self._view_model.cleanup()

    def _restore_persisted_inputs(self) -> None:
        s = self._settings
        # [v3.23.677] Khoá API + nhà cung cấp đọc từ kho chung (Cài đặt → Khoá API và AI).
        self._sync_custom_provider()
        self._ai_store.changed.connect(self._sync_custom_provider)
        saved_lang = s.value("target_lang", "Vietnamese", type=str)
        if self._lang_combo.findText(saved_lang) >= 0:
            self._lang_combo.setCurrentText(saved_lang)
        # [v3.7 U3] Khôi phục đầy đủ: ngữ cảnh, tuỳ chọn, cấu hình từng giai đoạn. [v3.23.677] Số lần thử lại nay ở
        # Cài đặt → "Dịch và video ngữ cảnh" (giá trị cũ của trang được chuyển sang một lần lúc khởi động).
        self._source_lang_edit.setText(s.value("source_lang", "", type=str))
        saved_analyze_model = s.value("analyze_model", DEFAULT_ANALYSIS_MODEL, type=str)
        self._analyze_model_combo.setCurrentText(saved_analyze_model)
        saved_fallback = s.value("fallback_model", "", type=str)
        # [v3.23.667] Bản cũ lưu "-" (= Không chuyển model) trong ô chọn — nay là ô "Bật" bỏ tick.
        enabled = s.value("fallback_enabled", saved_fallback != NO_FALLBACK_CHOICE, type=bool)
        self._fallback_enabled_check.blockSignals(True)
        self._fallback_enabled_check.setChecked(bool(enabled))
        self._fallback_enabled_check.blockSignals(False)
        self._fallback_model_combo.setEnabled(bool(enabled))
        self._fill_fallback_combo(
            [self._analyze_model_combo.itemText(i) for i in range(self._analyze_model_combo.count())],
            "" if saved_fallback == NO_FALLBACK_CHOICE else saved_fallback,
        )
        self._fill_audit_combo(
            [self._analyze_model_combo.itemText(i) for i in range(self._analyze_model_combo.count())],
            s.value("audit_model", "", type=str),
        )
        if s.contains("enable_tags"):
            self._tags_check.setChecked(s.value("enable_tags", False, type=bool))
        if s.contains("include_desc"):
            self._desc_check.setChecked(s.value("include_desc", True, type=bool))
        if s.contains("text_cues"):
            self._text_cues_check.setChecked(s.value("text_cues", True, type=bool))
        if s.contains("voice_roster"):
            self._voice_roster_check.setChecked(s.value("voice_roster", True, type=bool))
        self._stage_preprocess.restore_state(s, "stage_preprocess")
        self._stage_literal.restore_state(s, "stage_literal")
        self._stage_style.restore_state(s, "stage_style")
        self._stage_localize.restore_state(s, "stage_localize")

    def _persist_inputs(self) -> None:
        s = self._settings
        s.setValue("target_lang", self._lang_combo.currentText())
        s.setValue("source_lang", self._source_lang_edit.text().strip())
        s.setValue("analyze_model", self._analyze_model_combo.currentText())
        s.setValue("fallback_model", self._fallback_selection())
        s.setValue("fallback_enabled", self._fallback_enabled_check.isChecked())
        s.setValue("audit_model", self._audit_choice())
        s.setValue("enable_tags", self._tags_check.isChecked())
        s.setValue("include_desc", self._desc_check.isChecked())
        s.setValue("text_cues", self._text_cues_check.isChecked())
        s.setValue("voice_roster", self._voice_roster_check.isChecked())
        self._stage_preprocess.save_state(s, "stage_preprocess")
        self._stage_literal.save_state(s, "stage_literal")
        self._stage_style.save_state(s, "stage_style")
        self._stage_localize.save_state(s, "stage_localize")
        s.sync()

    # ── Public API cho MainWindow ────────────────────────────────────────
    def load_events(
        self, events: list[SubtitleEvent], video_path: str | Path | None = None
    ) -> None:
        """Nạp phụ đề từ bên ngoài (vd trang Biên tập đẩy sang).

        Args:
            events: Phụ đề nguồn.
            video_path: [v3.23.609] Video của phụ đề (khoá phiên, tên file xuất). Trước
                đây video của dự án mở TRƯỚC vẫn được coi là video của phụ đề mới.
        """
        # [#2] Nạp phụ đề mới = dự án mới → tẩy sạch TOÀN BỘ ngữ cảnh phim cũ.
        self._flush_context_save()  # [v3.23.652] lưu sửa tay ngữ cảnh của video CŨ trước khi đổi
        self._reset_context_and_video()
        self._source_video_path = str(video_path) if video_path else ""
        self._view_model.set_source_events(events)

    def _reset_context_and_video(self) -> None:
        """[#2 v3.18] "Tẩy não" toàn bộ ngữ cảnh khi chuyển sang phụ đề/dự án mới.

        Chống rò rỉ ngữ cảnh (context leakage) khiến AI dịch bị ảo giác: xoá Bảng
        nhân vật, Tóm tắt cốt truyện, Ngôn ngữ gốc, và gỡ Video ngữ cảnh (đường dẫn
        + nhãn + bỏ tick ô đính video). Gọi mỗi khi nạp phụ đề mới (từ File, từ trang
        Biên tập, hay khi đổi dự án).
        """
        self._clear_context_inputs()
        # Gỡ video ngữ cảnh của phim cũ.
        self._video_path = ""
        if getattr(self, "_video_path_label", None) is not None:
            self._video_path_label.setText(self._translator.translate("translate.no_video"))
        if getattr(self, "_attach_video_check", None) is not None:
            # Chặn tín hiệu để KHÔNG ghi đè lựa chọn "tag cảm xúc" đã lưu, nhưng vẫn phải
            # đưa các ô phụ thuộc video về đúng trạng thái. [v3.23.609] Trước đây chỉ chặn
            # tín hiệu: ô Visual Cues còn bật + đánh dấu trong khi ô đính video đã tắt và
            # video đã gỡ → bấm Dịch bị chặn với "Chưa chọn video ngữ cảnh".
            self._attach_video_before_custom = False   # [v3.23.700] phim mới: không bật lại tick của phim cũ
            self._attach_video_check.blockSignals(True)
            self._attach_video_check.setChecked(False)
            self._attach_video_check.blockSignals(False)
            self._set_video_controls_enabled(False)
            self._visual_cues_check.setChecked(False)

    def _clear_context_inputs(self) -> None:
        """Xoá các ô ngữ cảnh: nhân vật, tóm tắt, thuật ngữ, cues, ngôn ngữ gốc."""
        self._characters_edit.clear()
        self._overview_edit.clear()
        # [v3.23.138] Nhất quán với _clear_analysis/clear_context_fields: xoá LUÔN bảng
        # thuật ngữ và Visual Cues của phim cũ (trước đây bỏ sót -> rò rỉ ngữ cảnh dù hàm
        # này có mục đích "tẩy não" toàn bộ ngữ cảnh phim cũ).
        if getattr(self, "_glossary_edit", None) is not None:
            self._glossary_edit.clear()
        if getattr(self, "_visual_cues_edit", None) is not None:
            self._visual_cues_edit.clear()
        if hasattr(self, "_source_lang_edit"):
            self._source_lang_edit.clear()

    def _clear_results(self) -> None:
        """[v3.23.138] Xoá kết quả dịch cũ khỏi giao diện khi nạp nguồn/dự án mới:
        bảng kết quả, hai ô chi tiết (gốc/dịch) và thanh tiến trình. Tránh hiển thị
        bản dịch của file CŨ khi vừa nạp phụ đề mới (dễ hiểu nhầm là đã dịch)."""
        if getattr(self, "_table", None) is not None:
            self._table.setRowCount(0)
        if getattr(self, "_detail_source", None) is not None:
            self._detail_source.clear()
        if getattr(self, "_detail_translation", None) is not None:
            self._detail_translation.clear()
        if getattr(self, "_progress", None) is not None:
            self._progress.setValue(0)

    def _clear_analysis(self) -> None:
        """[1.5] Xoá bảng nhân vật + tóm tắt cũ khi đổi nguồn phụ đề (tránh dữ liệu
        của phim cũ dính sang phim mới)."""
        self._flush_context_save()  # [v3.23.652] lưu sửa tay ngữ cảnh của video CŨ trước khi đổi
        self._characters_edit.clear()
        self._overview_edit.clear()
        self._glossary_edit.clear()
        self._visual_cues_edit.document().setModified(False)  # [v3.23.652] cues của phim cũ không được lưu sang phim mới

    # ── Slot xử lý sự kiện UI ─────────────────────────────────────────────
    def clear_context_fields(self) -> None:
        """Xoá trắng các ô Ngữ cảnh (Bảng nhân vật + Tóm tắt bối cảnh).

        Dùng khi chuyển sang dự án/phụ đề khác để chống rò rỉ ngữ cảnh giữa các phim.
        """
        self._flush_context_save()  # [v3.23.652] lưu sửa tay ngữ cảnh của video CŨ trước khi đổi
        self._characters_edit.clear()
        self._overview_edit.clear()
        self._glossary_edit.clear()
        self._visual_cues_edit.document().setModified(False)  # [v3.23.652] cues của phim cũ không được lưu sang phim mới

    def set_context_fields(
        self, characters: str, overview: str, glossary: str = "", visual_cues: str = ""
    ) -> None:
        """Nạp ngữ cảnh (vd khi khôi phục dự án từ Thư viện)."""
        self._characters_edit.setPlainText(characters or "")
        self._overview_edit.setPlainText(overview or "")
        self._glossary_edit.setPlainText(glossary or "")
        self._visual_cues_edit.setPlainText(visual_cues or "")

    def get_context_fields(self) -> tuple[str, str]:
        """Trả về (bảng nhân vật, tóm tắt) hiện tại để lưu vào dự án."""
        return (
            self._characters_edit.toPlainText().strip(),
            self._overview_edit.toPlainText().strip(),
        )

    def get_speaker_context(self) -> tuple[str, str]:
        """[v3.23.593] (bảng nhân vật, Visual Cues JSON) cho phân vai lồng tiếng."""
        return (
            self._characters_edit.toPlainText().strip(),
            self._visual_cues_edit.toPlainText().strip(),
        )

    def _on_load_file_clicked(self) -> None:
        self._flush_context_save()  # [v3.23.652] lưu sửa tay ngữ cảnh của video CŨ trước khi đổi
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Chọn tệp phụ đề nguồn", "", "Phụ đề (*.srt *.ass)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path_str:
            return
        from subtitles_extractor.application.services.sibling_video_finder import (
            find_sibling_video,
        )

        sub_path = Path(path_str)
        # [v3.23.609] Đọc tệp TRƯỚC: đọc hỏng thì phụ đề cũ và ngữ cảnh của nó còn nguyên.
        events = self._view_model.read_source_file(sub_path)
        if events is None:
            return
        # [#2] Nạp phụ đề mới = dự án mới → tẩy sạch toàn bộ ngữ cảnh phim trước
        # (Bảng nhân vật / Tóm tắt / Ngôn ngữ gốc / Video) để không lẫn sang phim
        # hiện tại.
        self._reset_context_and_video()
        # [Smart Context Auto-Detect] Tự dò video cùng tên nằm cạnh tệp phụ đề để nạp
        # thẳng làm Ngữ cảnh Video. [v3.23.609] Làm TRƯỚC khi nạp phụ đề và đặt luôn video
        # của phụ đề: trước đây video của dự án mở trước đó vẫn được coi là video của
        # phụ đề mới → nạp phân tích/bản dịch của phim cũ, ghi phiên vào video cũ, đặt tên
        # file xuất theo video cũ.
        sibling_video = find_sibling_video(sub_path)
        self._source_video_path = str(sibling_video) if sibling_video is not None else ""
        if sibling_video is not None:
            self._video_path = str(sibling_video)
            self._video_path_label.setText(sibling_video.name)
            # [v3.23.666] Nhà cung cấp tuỳ chỉnh không xem video: không tự tick ô (đang bị khoá).
            if getattr(self, "_attach_video_check", None) is not None and not self._custom_mode():
                self._attach_video_check.setChecked(True)
        self._view_model.set_source_events(events)

    def _on_pull_editor_clicked(self) -> None:
        self._flush_context_save()  # [v3.23.652] lưu sửa tay ngữ cảnh của video CŨ trước khi đổi
        if self._editor_event_provider is None:
            self._show_info(self._translator.translate("translate.toast_hint_title"), self._translator.translate("translate.toast_no_editor_source"))
            return
        events = self._editor_event_provider()
        if not events:
            self._show_warning(
                "Trang Biên tập trống", "Chưa có phụ đề nào ở trang Biên tập để lấy sang."
            )
            return
        # [#2] Phụ đề mới = dự án mới → tẩy sạch toàn bộ ngữ cảnh phim cũ trước.
        self._reset_context_and_video()

        # [#11 v3.18] Auto-Attach: tự bốc luôn VIDEO đang mở ở trang Biên tập sang
        # trang Dịch (đường dẫn + nhãn + tự tick ô đính video làm ngữ cảnh).
        # [v3.23.609] Làm TRƯỚC khi nạp phụ đề, và video của trang Biên tập là video của
        # phụ đề: trước đây nạp phụ đề xong mới gắn video nên bước nạp lại phân tích/bản
        # dịch đã lưu chạy với video cũ (hoặc không có video).
        attached_name = ""
        editor_video = (
            self._editor_video_provider() if self._editor_video_provider is not None
            else None
        )
        self._source_video_path = str(editor_video) if editor_video else ""
        if editor_video:
            self._video_path = str(editor_video)
            attached_name = Path(editor_video).name
            if getattr(self, "_video_path_label", None) is not None:
                self._video_path_label.setText(attached_name)
            if getattr(self, "_attach_video_check", None) is not None and not self._custom_mode():
                self._attach_video_check.setChecked(True)
        self._view_model.set_source_events(events)

        message = f"Đã lấy {len(events)} dòng từ trang Biên tập."
        if attached_name:
            message += f"\nĐã tự đính video ngữ cảnh: {attached_name}"
        self._show_info(self._translator.translate("common.success"), message)

    def _on_run_clicked(self) -> None:
        # [v3.23.609] Nút Dịch / phím Enter ở ô API key: không chen vào hàng đợi dịch cả
        # bộ (kết quả sẽ bị ghi thành bản dịch của tập đang chạy).
        if self._batch_active():
            return
        self._start_translation_from_ui()

    def _batch_active(self) -> bool:
        """Đang chạy hàng đợi "Dịch cả bộ"."""
        return bool(getattr(self, "_batch_items", None))

    def _start_translation_from_ui(self) -> bool:
        """Kiểm cấu hình trên trang rồi bắt đầu dịch.

        Returns:
            True khi lượt dịch đã bắt đầu. [v3.23.609] Dịch cả bộ cần biết điều này:
            trước đây cảnh báo cấu hình (thiếu key, chưa chọn video…) làm cả hàng đợi
            đứng im mà không báo gì.
        """
        # [v3.23.44] Validate sớm tại UI để báo lỗi rõ ràng (thay vì lẫn lộn ở tầng dưới).
        # [v3.23.121] Gộp tất cả API key (chính + dự phòng) -> adapter tự xoay khi hết quota.
        if not self._visual_cues_valid():
            return False
        if self._warn_custom_incomplete():
            return False
        api_key = self._combined_keys_text()
        if not api_key:
            self._show_warning(
                "Thiếu API Key",
                self._translator.translate("ai.warn_no_key_body"),
            )
            self._open_ai_settings()
            return False
        stages = self._collect_enabled_stages()
        if not stages:
            self._show_warning(
                "Chưa bật giai đoạn dịch nào",
                "Hãy bật ít nhất một giai đoạn (vd 'Giai đoạn 2: Dịch thô sát nghĩa') "
                "bằng ô 'Bật giai đoạn này'.",
            )
            return False

        self._persist_inputs()

        # [v3.23.44] Cảnh báo nếu bật ngữ cảnh video / Visual Cues nhưng CHƯA chọn file
        # video — nếu không, các tính năng đó bị bỏ qua ngầm mà người dùng không biết.
        wants_video = (
            self._selected_attach_video_stages() or self._visual_cues_check.isChecked()
        )
        attach_on = getattr(self, "_attach_video_check", None) and self._attach_video_check.isChecked()
        if (wants_video or attach_on) and not self._video_path:
            self._show_warning(
                "Chưa chọn video ngữ cảnh",
                "Bạn đã bật tính năng cần video (đính video / phân tích hình ảnh) nhưng "
                "chưa chọn tệp video. Hãy bấm 'Chọn video' hoặc tắt các tuỳ chọn đó.",
            )
            return False

        self._sync_text_cues()
        return self._view_model.start_translation(
            api_key=api_key,
            retry_count=None,  # [v3.23.677] lấy từ Cài đặt
            stages=stages,
            context=self._current_translation_context(),
            video_path=self._selected_video_path(),
            attach_video_stages=self._selected_attach_video_stages(),
            enable_visual_cues=self._visual_cues_check.isChecked(),
            # [v3.23.602] Hỏi lại người nói bằng model phân tích (thị giác mạnh hơn).
            speaker_model=self._analyze_model_combo.currentText().strip(),
            session_path=self._session_video_path(),
        )

    def _current_translation_context(self) -> TranslationContext:
        """Ngữ cảnh dịch đang nhập trên trang (dùng chung cho Dịch / bộ nhớ đệm)."""
        return TranslationContext(
            target_lang=self._lang_combo.currentText().strip() or "Vietnamese",
            source_lang=self._source_lang_edit.text().strip(),
            overview=self._overview_edit.toPlainText().strip(),
            characters=self._characters_edit.toPlainText().strip(),
            glossary=self._glossary_edit.toPlainText().strip(),
            visual_cues=self._visual_cues_edit.toPlainText().strip(),
            enable_tags=self._tags_check.isChecked(),
            include_desc=self._desc_check.isChecked(),
        )

    def _stage_cache_settings(self) -> dict[str, object]:
        """[v3.23.608] Cấu hình xác định bộ nhớ đệm giai đoạn — ĐÚNG như lúc bấm Dịch.

        Trước bản này nút "Xoá bộ nhớ đệm dịch" chỉ gửi ngôn ngữ đích (thiếu ngôn ngữ gốc,
        Visual Cues) nên tính ra khoá khác khoá lúc dịch và không xoá được gì.
        """
        self._sync_text_cues()
        return {
            "stages": self._collect_enabled_stages(),
            "context": self._current_translation_context(),
            "video_path": self._selected_video_path(),
            "enable_visual_cues": self._visual_cues_check.isChecked(),
            "attach_video_stages": self._selected_attach_video_stages(),
            "session_path": self._session_video_path(),
        }

    def _collect_enabled_stages(self) -> list[TranslationStageConfig]:
        panels = (
            self._stage_preprocess,
            self._stage_literal,
            self._stage_style,
            self._stage_localize,
        )
        return [panel.build_config() for panel in panels if panel.is_enabled()]

    # ── [v3.23.666/677] Nhà cung cấp AI (cấu hình ở Cài đặt → Khoá API và AI) ──────────
    def set_open_settings_callback(self, callback: "Callable[[], None] | None") -> None:
        """[v3.23.677] MainWindow tiêm hàm mở Cài đặt → "Khoá API và AI"."""
        self._open_settings_cb = callback
        self._ai_summary.set_open_settings_callback(callback)

    def _open_ai_settings(self) -> None:
        callback = getattr(self, "_open_settings_cb", None)
        if callback is not None:
            callback()

    # ── Nhiều API key ────────────────────────────────────────────────────
    # ── Sửa/thêm hạn mức quota model ─────────────────────────────────────
    def _on_media_resolution_changed(self) -> None:
        """[v3.23.129] Lưu mức phân giải video phân tích (áp cho lần phân tích kế tiếp)."""
        level = self._media_res_combo.currentData() or "medium"
        try:
            self._view_model.set_analysis_media_resolution(level)
            self._show_info(
                self._translator.translate("translate.toast_saved_res_title"),
                self._translator.translate("translate.toast_saved_analysis_body"),
            )
        except (AttributeError, RuntimeError, ValueError):
            pass

    def _on_analysis_thinking_changed(self) -> None:
        """[v3.23.140] Lưu mức Thinking phân tích (áp cho lần phân tích ngữ cảnh kế tiếp)."""
        level = self._analysis_think_combo.currentData() or "medium"
        try:
            self._view_model.set_analysis_thinking_level(level)
            self._show_info(
                self._translator.translate("translate.toast_saved_thinking_title"),
                self._translator.translate("translate.toast_saved_analysis_body"),
            )
        except (AttributeError, RuntimeError, ValueError):
            pass

    def _on_clip_video_toggled(self, checked: bool) -> None:
        """[v3.23.648] Lưu tuỳ chọn cắt video theo lô (áp dụng cho lượt dịch kế tiếp)."""
        try:
            self._view_model.set_clip_video_to_batch(bool(checked))
        except (AttributeError, KeyError, ValueError) as exc:
            logger.warning("Không lưu được tuỳ chọn cắt video theo lô: %s", exc)

    def _on_fit_for_speech_toggled(self, checked: bool) -> None:
        """[v3.23.703] Lưu tuỳ chọn khớp giọng đọc (áp dụng cho lượt dịch kế tiếp)."""
        try:
            self._view_model.set_fit_for_speech(bool(checked))
        except (AttributeError, KeyError, ValueError) as exc:
            logger.warning("Không lưu được tuỳ chọn khớp giọng đọc: %s", exc)

    def _on_native_address_toggled(self, checked: bool) -> None:
        """[v3.23.612] Lưu tuỳ chọn xưng hô thuần Việt; bật thì đổi ngay ngữ cảnh và
        bản dịch đang có."""
        try:
            self._view_model.set_native_address(bool(checked))
        except (AttributeError, KeyError, ValueError) as exc:
            logger.warning("Không lưu được tuỳ chọn xưng hô thuần Việt: %s", exc)
            return
        body_key = "translate.toast_native_off_body"
        if checked:
            self._view_model.apply_native_address(
                self._current_translation_context(), self._session_video_path()
            )
            body_key = "translate.toast_native_on_body"
        self._show_info(
            self._translator.translate("translate.toast_native_title"),
            self._translator.translate(body_key),
        )

    def _on_parallel_changed(self) -> None:
        """[v3.23.149] Lưu mức dịch song song (áp dụng cho phiên dịch kế tiếp)."""
        count = self._parallel_combo.currentData() or 1
        try:
            self._view_model.set_translation_parallel_batches(int(count))
            self._show_info(
                self._translator.translate("translate.toast_saved_parallel_title"),
                self._translator.translate("translate.toast_saved_parallel_body"),
            )
        except (AttributeError, RuntimeError, ValueError):
            pass

    # ── Slot phản hồi ViewModel ──────────────────────────────────────────
    def _on_source_changed(self, count: int) -> None:
        # [v3.23.138] Nạp nguồn mới -> XOÁ kết quả dịch cũ (bảng + chi tiết + tiến trình).
        # Trước đây chỉ đổi nhãn trạng thái, còn bảng vẫn hiện bản dịch của FILE CŨ đến khi
        # dịch xong lần mới -> gây hiểu nhầm là đã dịch.
        self._clear_results()
        if count > 0:
            self._lbl_source.setText(self._translator.translate("translate.tr_loaded").replace("{count}", str(count)))
            self._lbl_source.setStyleSheet("font-weight: bold;")
            # Reset trạng thái thực thi để không hiện kết quả của file cũ
            self._stage_label.setText(self._translator.translate("translate.status_ready2"))
            # [v3.23.45] Tự nạp lại phân tích ngữ cảnh đã lưu cho video này (nếu có) →
            # người dùng không phải phân tích lại khi mở lại dự án đã làm trước đó.
            video_path = self._session_video_path()
            self._source_analysis_restored = False
            if video_path:
                target_lang = self._lang_combo.currentText().strip() or "Vietnamese"
                self._source_analysis_restored = self._view_model.try_restore_saved_analysis(
                    video_path, target_lang
                )
                if self._source_analysis_restored:
                    note = getattr(self._view_model, "last_restore_note", "") or ""
                    self._stage_label.setText(
                        "Đã nạp lại phân tích ngữ cảnh đã lưu cho video này."
                        + (f" {note}" if note else "")
                    )
                    if note:
                        # [v3.23.668] Phụ đề đã đổi sau ReOCR: báo rõ phần giữ lại / cần hỏi lại.
                        self._view_model.status_message.emit(note)
                else:
                    # [v3.23.56] Video chưa phân tích, nhưng phim bộ (cùng thư mục) có thể
                    # đã có ngữ cảnh chung từ các tập trước → đề xuất điền vào ô đang trống.
                    self._apply_series_context(video_path)
        else:
            self._lbl_source.setText(self._translator.translate("translate.status_no_source"))
            self._lbl_source.setStyleSheet("font-weight: bold;")
        self._update_action_states()

    def _apply_series_context(self, video_path: str) -> None:
        """[v3.23.56] Điền glossary/roster/tóm tắt chung của phim bộ vào ô đang trống.

        Chỉ điền vào ô trống để không ghi đè nội dung người dùng đang nhập. Hữu ích khi
        chuyển sang tập mới của phim bộ đã dịch tập trước.
        """
        ctx = self._view_model.restore_series_context(video_path)
        if ctx is None:
            return
        filled = []
        if ctx.glossary and not self._glossary_edit.toPlainText().strip():
            self._glossary_edit.setPlainText(ctx.glossary)
            filled.append("bảng thuật ngữ")
        if ctx.characters and not self._characters_edit.toPlainText().strip():
            self._characters_edit.setPlainText(ctx.characters)
            filled.append("nhân vật")
        if ctx.overview and not self._overview_edit.toPlainText().strip():
            # [v3.23.669] Tóm tắt phim bộ nay gồm mọi tập: chỉ điền diễn biến các tập GẦN NHẤT (không gồm
            # chính tập này) — đủ ngữ cảnh mà không phình mọi lời gọi dịch tới 12 000 ký tự.
            from subtitles_extractor.application.services.translation_memory import (
                recent_series_overview,
            )

            story = recent_series_overview(
                ctx.overview, exclude_episode=Path(video_path).stem, max_chars=3000,
            )
            if story:
                self._overview_edit.setPlainText(story)
                filled.append("tóm tắt")
        if filled:
            self._show_info(
                self._translator.translate("translate.toast_series_ctx_title"),
                self._translator.translate("translate.toast_series_ctx_body").replace(
                    "{filled}", ", ".join(filled)
                ),
            )

    def _on_result_ready(
        self, source_events: list[SubtitleEvent], translated_events: list[SubtitleEvent]
    ) -> None:
        """Vẽ bảng kết quả (dịch xong, nạp lại phiên, dịch lại dòng, dùng một giai đoạn).

        [v3.23.609] Chỉ vẽ bảng. Việc của "dịch xong" (dịch cả bộ ghi tệp rồi sang tập
        kế, thông báo hoàn tất) chuyển sang :meth:`_on_translation_finished`: tín hiệu này
        còn phát khi nạp lại phiên đã lưu, nên dịch cả bộ từng ghi bản dịch CŨ nạp lại
        thành bản dịch của tập vừa nạp rồi nhảy tập, và thông báo "Dịch hoàn tất" hiện cả
        khi chỉ mở lại phiên hay dịch lại hai dòng.
        """
        selected_rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        self._table.setUpdatesEnabled(False)
        # [v3.23.650] Xoá hết hàng TRƯỚC khi vẽ lại: điền lại bảng đã có sẵn 3649 hàng (header ResizeToContents) tốn
        # O(hàng) cho MỖI setItem → ~100 s, Windows báo "Not Responding". Bảng rỗng điền lần đầu chỉ ~0,2 s.
        self._table.setRowCount(0)
        self._table.setRowCount(len(translated_events))
        edited_rows = self._view_model.final_edit_positions()
        for row_idx, translated in enumerate(translated_events):
            original = source_events[row_idx] if row_idx < len(source_events) else None
            time_text = seconds_to_display(translated.start_sec)
            original_text = original.text if original is not None else ""
            cells = [
                str(row_idx + 1),
                time_text,
                original_text.replace("\\N", "\n"),
                translated.text.replace("\\N", "\n"),
            ]
            for col_idx, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if col_idx <= 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
                self._table.setItem(row_idx, col_idx, item)
            self._mark_edited_row(row_idx, row_idx in edited_rows)
        # [v3.7 U5] resizeRowsToContents() trên hàng nghìn dòng word-wrap rất chậm
        # (O(n) layout, có thể đứng UI vài giây). Chỉ làm khít từng dòng cho bảng
        # nhỏ; bảng lớn dùng chiều cao mặc định 2 dòng để giữ UI mượt.
        if len(translated_events) <= _RESIZE_ROWS_THRESHOLD:
            self._table.resizeRowsToContents()
        else:
            self._table.verticalHeader().setDefaultSectionSize(_DEFAULT_ROW_HEIGHT_PX)
        if selected_rows:
            select_rows_bulk(self._table, [r for r in selected_rows if r < len(translated_events)])
        self._table.setUpdatesEnabled(True)
        # Đảm bảo bảng chiếm phần lớn splitter sau mỗi lần dịch xong
        if hasattr(self, "_splitter"):
            self._splitter.setSizes([420, 160])
        self._update_action_states()

    def _on_translation_finished(self) -> None:
        """[v3.23.609] Dịch xong: cả bộ thì ghi tệp và sang tập kế, không thì báo xong."""
        if getattr(self, "_batch_items", None):
            # [v3.23.366] Ghi .translate.<lang>.srt cạnh video TRƯỚC khi sang tập kế, để
            # khâu TTS → Xuất bản hàng loạt tìm thấy.
            # [v3.23.650] Ghi tệp thất bại: `export_translation` phát error_occurred → `_on_error_for_batch` từng tự sang
            # tập kế, rồi đây lại sang lần nữa ⇒ chỉ số nhảy 2, mất một tập (đã tốn hạn mức mà không ghi tệp) và tập
            # lỗi bị đếm vừa xong vừa lỗi. Nay: chặn xử lý lỗi lồng nhau, chỉ sang tập kế MỘT lần.
            if self._persist_batch_translation():
                self._batch_done += 1
                QTimer.singleShot(0, self._advance_batch_translate)
            else:
                items = self._batch_items
                name = items[min(self._batch_index, len(items) - 1)].video_path.name
                QTimer.singleShot(0, lambda n=name: self._advance_batch_translate(failed=n))
            return
        elapsed = self._view_model.last_elapsed_sec
        elapsed_str = (
            f" ({elapsed:.0f}s)" if elapsed < 60
            else f" ({int(elapsed // 60)}p{int(elapsed % 60):02d}s)"
        )
        tokens = self._view_model.last_token_report
        notify_task_finished(  # [v3.23.682]
            self, self._translator.translate("common.task_done_translate"),
            self._translator.translate("common.task_lines", count=len(self._view_model.translated_events)),
        )
        self._show_success(
            "Dịch hoàn tất",
            f"Đã dịch {len(self._view_model.translated_events)} dòng qua "
            f"{self._view_model.last_stage_count} giai đoạn{elapsed_str}."
            + (f"\n{tokens}" if tokens else ""),
        )


    def _persist_batch_translation(self) -> bool:
        """[v3.23.366] Ghi ``.translate.<lang>.srt`` cho tập vừa dịch (chế độ hàng loạt).

        Dùng đúng quy ước tên như khi xuất tay (``_LANGUAGE_CODES`` + ``translated_
        subtitle_path``) để khâu TTS hàng loạt tìm thấy. Không ném lỗi ra ngoài.
        """
        items = getattr(self, "_batch_items", None)
        if not items:
            return False
        index = min(self._batch_index, len(items) - 1)
        video_path = items[index].video_path
        self._persisting_batch_translation = True
        try:
            from subtitles_extractor.domain.value_objects.output_naming import (
                SubtitleFormat as NamingFormat,
            )
            from subtitles_extractor.domain.value_objects.output_naming import (
                translated_subtitle_path,
            )

            lang_code = _LANGUAGE_CODES.get(self._lang_combo.currentText(), "vi")
            out_path = translated_subtitle_path(
                video_path, lang_code, NamingFormat.SRT
            )
            if self._view_model.export_translation(out_path, SubtitleFormat.SRT):
                logger.info("Hàng loạt: đã ghi bản dịch → %s", out_path.name)
                return True
            return False
        except (ValueError, OSError, KeyError) as exc:
            logger.warning(
                "Hàng loạt: không ghi được bản dịch cho %s: %s", video_path.name, exc
            )
            return False
        finally:
            self._persisting_batch_translation = False

    def _on_error(self, message: str) -> None:
        # [v3.23.332] Đang chạy cả bộ -> xử lý riêng (hết hạn mức thì dừng hàng đợi).
        if self._on_error_for_batch(message):
            return
        from subtitles_extractor.presentation.utils.error_humanizer import (
            humanize_gemini_error,
        )
        self._show_warning(self._translator.translate("translate.warn_translate_error_title"), humanize_gemini_error(message))
        self._stage_label.setText(self._translator.translate("translate.status_stopped_err"))

    # ── Dịch cả bộ ───────────────────────────────────────────────────────────
    def _on_cancel_clicked(self) -> None:
        """Huỷ: đang chạy cả bộ thì dừng HẲN hàng đợi, không chỉ tập hiện tại."""
        if self._batch_active():
            self._batch_cancelled = True
            remaining = len(self._batch_items) - self._batch_index - 1
            self._stage_label.setText(self._translator.translate("translate.tr_cancelling").replace("{n}", str(max(0, remaining))))
            if not self._view_model.is_busy:
                # [v3.23.609] Bấm Huỷ đúng lúc chuyển tập (không có lượt nào đang chạy để
                # báo "đã huỷ"): kết thúc hàng đợi ngay.
                self._finish_batch_translate()
                return
        self._view_model.cancel_translation()

    def _backfill_original_from_db(self, videos: list[Path]) -> int:
        """[v3.23.365] Ghi bù ``.original.srt`` từ CSDL cho các video đang xét.

        Trả số tệp đã bù. Không ném lỗi ra ngoài — chỉ log; nếu bù được thì thông báo nhẹ.
        """
        try:
            from subtitles_extractor.application.services.backfill_extracted import (
                backfill_original_subtitles,
            )
            from subtitles_extractor.infrastructure.subtitle.atomic_save import (
                atomic_write_text,
            )

            records = self._container.project_repository.list_all()
        except (AttributeError, OSError) as exc:
            logger.warning("Không đọc được CSDL để bù .original.srt: %s", exc)
            return 0

        result = backfill_original_subtitles(
            records,
            file_exists=lambda path: path.is_file(),
            # BOM (utf-8-sig) giúp VLC/Aegisub nhận đúng tiếng Việt — đồng bộ SrtExporter.
            write_text=lambda path, text: atomic_write_text(
                path, text, encoding="utf-8-sig"
            ),
            only_video_names={video.name for video in videos},
        )
        if result.written:
            logger.info("Dịch cả bộ: %s", result.summary_vi())
            self._show_info_toast(result.summary_vi())
        return len(result.written)

    def _show_info_toast(self, message: str) -> None:
        """Hiển thị thông báo nhẹ (an toàn nếu thiếu InfoBar)."""
        try:
            from qfluentwidgets import InfoBar, InfoBarPosition

            InfoBar.success(
                title="Bù phụ đề gốc", content=message, parent=self,
                position=InfoBarPosition.TOP_RIGHT, duration=4000,
            )
        except (ImportError, TypeError, AttributeError):
            logger.info(message)

    def _on_cancelled(self) -> None:
        self._progress.setValue(0)
        self._stage_label.setText(self._translator.translate("translate.status_cancelled"))
        # [v3.23.609] Huỷ khi đang dịch cả bộ: kết thúc hàng đợi (tổng kết). Trước đây
        # hàng đợi treo lại, và lần dịch tay kế tiếp ghi kết quả vào tệp của tập dở.
        if getattr(self, "_batch_items", None):
            self._batch_cancelled = True
            self._finish_batch_translate()
            return
        self._show_info(self._translator.translate("translate.toast_cancelled_title"), self._translator.translate("translate.toast_cancelled_body"))

    def _on_progress_changed(self, percent: int, stage_label: str) -> None:
        self._progress.setValue(percent)
        self._stage_label.setText(f"{stage_label} — {percent}%")

    def _on_busy_changed(self, busy: bool) -> None:
        if busy:
            # [v3.23.655] Bug #9: chụp ô cues lúc lượt chạy bắt đầu — kết quả quét bổ sung được gộp 3 chiều với sửa tay
            # trong lúc chạy thay vì ghi đè.
            self._cues_run_base = self._visual_cues_edit.toPlainText()
            # Reset tiến độ ngay khi bắt đầu để không hiện 100% của lần dịch cũ
            self._progress.setValue(0)
            self._stage_label.setText(self._translator.translate("translate.status_starting"))
        self._apply_run_locks(busy)
        if not busy and getattr(self, "_batch_analyzing_index", -1) >= 0:
            # [v3.23.669] Phân tích của hàng loạt kết thúc mà không có kết quả/lỗi (vd bị huỷ):
            # đi tiếp thay vì để hàng đợi treo.
            QTimer.singleShot(0, lambda: self._batch_analysis_done(False, "phân tích dừng"))
        if busy:
            self._btn_complete_speakers.setEnabled(False)
            self._btn_cue_recheck.setEnabled(False)  # [v3.23.655]
        else:
            self.refresh_cue_issues()
        self._btn_clear_cache.setEnabled(not busy)
        if hasattr(self, "_btn_stage_cache") and busy:
            self._btn_stage_cache.setEnabled(False)
        # [v3.23.58] Khoá quản lý bộ nhớ phim bộ khi đang dịch (worker đang ghi TM) để
        # tránh tranh chấp đọc/ghi SQLite.
        if hasattr(self, "_btn_manage_tm"):
            self._btn_manage_tm.setEnabled(not busy)
        # [U2] Disable refresh model khi đang bận (tránh fetch đồng thời với dịch)
        if not busy and self._btn_refresh_models.text() == self._translator.translate("translate.btn_load_models"):
            self._btn_refresh_models.setEnabled(True)
        elif busy:
            self._btn_refresh_models.setEnabled(False)
        if not busy:
            self._update_action_states()

    def _sync_text_cues(self) -> None:
        """[v3.23.623] Báo view-model: có gán người nói từ phụ đề không, và video nào cạnh phụ
        đề để tách giọng (chỉ lấy âm thanh, không tải video lên Gemini)."""
        video = self._source_video_path or self._video_path
        voice = video if video and Path(video).is_file() else ""
        self._view_model.set_text_cues(self._text_cues_check.isChecked(), voice)
        set_voice_roster = getattr(self._view_model, "set_voice_roster", None)
        if callable(set_voice_roster):
            set_voice_roster(self._voice_roster_check.isChecked(), voice)

    def _apply_run_locks(self, busy: bool) -> None:
        """Các nút chạy / nạp nguồn / huỷ theo trạng thái bận và hàng đợi dịch cả bộ.

        [v3.23.609] Giữa hai tập của "Dịch cả bộ" không có luồng nào chạy, nên trước đây
        nút Dịch, Nạp phụ đề, Lấy từ Biên tập lại bật (kết quả của lượt bấm tay bị ghi
        thành bản dịch của tập đang chạy) còn nút Huỷ thì bị khoá.
        """
        has_source = self._view_model.has_source
        locked = busy or self._batch_active()
        self._btn_run.setEnabled(has_source and not locked)
        self._btn_analyze.setEnabled(has_source and not locked)
        # [v3.23.609] Bắt đầu "Dịch cả bộ" giữa lúc đang dịch làm hàng đợi nhận kết quả
        # của lượt đang chạy là bản dịch tập 1 và ghi ra tệp của tập đó.
        self._btn_batch.setEnabled(not locked)
        self._btn_cancel.setEnabled(locked)
        self._btn_load_file.setEnabled(not locked)
        self._btn_pull_editor.setEnabled(not locked)

    def _update_action_states(self) -> None:
        has_source = self._view_model.has_source
        has_result = self._view_model.has_result
        busy = self._view_model.is_busy
        has_sel = bool(self._table.selectedItems())
        self._apply_run_locks(busy)
        self._btn_export_srt.setEnabled(has_result and not busy)
        self._btn_export_ass.setEnabled(has_result and not busy)
        self._btn_export_bilingual.setEnabled(has_result and not busy)
        self._btn_copy_selection.setEnabled(has_result and has_sel and not busy)
        self._btn_send_to_editor.setEnabled(has_result and not busy)
        # [v3.23.47] Chỉ bật So sánh khi có ≥2 giai đoạn được lưu.
        has_stages = self._view_model.stage_output_count >= 2
        self._btn_compare_stages.setEnabled(has_stages and not busy)
        self._btn_stage_cache.setEnabled(has_source and not busy)
        # [v3.23.54] Kiểm tra thuật ngữ: bật khi có bản dịch + có glossary.
        has_glossary = bool(self._glossary_edit.toPlainText().strip())
        self._btn_check_glossary.setEnabled(has_result and has_glossary and not busy)
        self._update_speaker_button()
        refine = getattr(self, "_btn_refine_analysis", None)
        if refine is not None:
            has_analysis = bool(
                self._characters_edit.toPlainText().strip()
                or self._overview_edit.toPlainText().strip()
            )
            refine.setEnabled(has_source and has_analysis and not busy)

    def _on_status_message(self, message: str) -> None:
        self._stage_label.setText(message)

    def _on_resume_detected(self, detail: str) -> None:
        """[v3.23.608] Báo phần lượt dịch vừa rồi lấy từ bộ nhớ đệm giai đoạn."""
        self._show_info(
            self._translator.translate("translate.toast_resume_title"),
            self._translator.translate("translate.toast_resume_body").replace(
                "{detail}", detail
            ),
        )

    def _show_success(self, title: str, content: str) -> None:
        _feedback.show_success(self, title, content)

    def _show_info(self, title: str, content: str) -> None:
        _feedback.show_info(self, title, content)

    def _on_context_text_changed(self) -> None:
        # Nhớ video ở LÚC GÕ (không phải lúc hẹn giờ chạy).
        if not self._context_save_timer.isActive():
            self._context_edit_path = self._session_video_path()
        self._context_save_timer.start()

    def _flush_context_save(self) -> None:
        """[v3.23.652] Lưu ngay sửa tay ngữ cảnh đang chờ (đổi video / rời trang / đóng app)."""
        timer = getattr(self, "_context_save_timer", None)
        if timer is None:
            return
        timer.stop()
        self._save_context_edits(force=True)

    def hideEvent(self, event) -> None:  # noqa: N802 — rời trang hoặc đóng cửa sổ: không mất 1,5 s gõ cuối
        self._flush_context_save()
        super().hideEvent(event)

    def _save_context_edits(self, *, force: bool = False) -> None:
        video_path = self._context_edit_path or self._session_video_path()
        if not video_path:
            return
        from subtitles_extractor.application.services.visual_cue_serializer import (
            visual_cues_syntax_error,
        )

        edits = (self._characters_edit, self._overview_edit, self._glossary_edit, self._visual_cues_edit)
        # Chỉ lưu khi NGƯỜI DÙNG đã sửa (setPlainText lúc nạp phiên/đổi video xoá cờ modified) — không ghi ngược
        # trạng thái trung gian lúc đang nạp đè lên phiên của video khác.
        if not any(edit.document().isModified() for edit in edits):
            self._context_edit_path = ""
            return
        if self._view_model.is_busy and not force:
            # Lượt dịch đang chạy có thể tự lưu thêm cues vào phiên: đợi xong mới ghi (tránh ghi bản UI cũ đè lên).
            self._context_save_timer.start()
            return
        cues = self._visual_cues_edit.toPlainText().strip()
        if visual_cues_syntax_error(cues) is not None:
            return  # đang gõ dở JSON: chờ lần lưu sau, không ghi bản hỏng đè bản tốt
        for edit in edits:
            edit.document().setModified(False)
        self._context_edit_path = ""
        self.refresh_cue_issues()  # [v3.23.654] sửa tay JSON cues → cập nhật số dòng nghi
        self._view_model.save_context_edits(
            video_path,
            characters=self._characters_edit.toPlainText().strip(),
            overview=self._overview_edit.toPlainText().strip(),
            glossary=self._glossary_edit.toPlainText().strip(),
            visual_cues=cues,
        )

    def _show_warning(self, title: str, content: str) -> None:
        _feedback.show_warning(self, title, content)


__all__ = ["TranslatePage"]
