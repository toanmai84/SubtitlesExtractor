"""Cửa sổ chính của ứng dụng — :class:`FluentWindow` của qfluentwidgets.

CẢI TIẾN:
    1. Lắng nghe Signal `subtitles_built` từ DebugPage để nạp thẳng vào EditorPage.
"""

from __future__ import annotations

from subtitles_extractor.presentation.theme import metrics as _m

import contextlib
import logging
import sqlite3
from pathlib import Path

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QCloseEvent
from subtitles_extractor.presentation.fluent_compat import FluentIcon, FluentWindow, NavigationItemPosition

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.project_record import WorkflowStage
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.presentation.pages.debug_page import DebugPage
from subtitles_extractor.presentation.pages.editor_page import EditorPage
from subtitles_extractor.presentation.pages.extract_page import ExtractPage
from subtitles_extractor.presentation.pages.log_page import LogPage
from subtitles_extractor.presentation.pages.projects_page import ProjectsPage
from subtitles_extractor.presentation.pages.publish_page import PublishPage
from subtitles_extractor.presentation.widgets.workflow_bar import WorkflowBar
from subtitles_extractor.presentation.pages.settings_page import SettingsPage
from subtitles_extractor.presentation.pages.translate_page import TranslatePage
from subtitles_extractor.presentation.pages.tts_page import TTSPage

logger = logging.getLogger(__name__)


def _is_page_busy(page: object) -> bool:
    """Cho biết một trang có view-model đang bận (chạy tác vụ dài) hay không."""
    vm = getattr(page, "_view_model", None)
    if vm is None:
        return False
    busy = getattr(vm, "is_busy", None)
    if busy is None:
        busy = getattr(vm, "_is_busy", False)
    return bool(busy)


def _build_close_warnings(
    editor_page: object,
    extract_page: object,
    translate_page: object,
    tts_page: object,
    translator=None,
) -> list[str]:
    """[v3.23.113] Dựng danh sách cảnh báo khi thoát (hàm thuần để dễ kiểm thử).

    Trả về các dòng mô tả việc sẽ mất nếu thoát ngay: thay đổi chưa lưu ở Biên tập, hoặc
    các tác vụ dài đang chạy (Trích xuất / Dịch / TTS).
    """
    warnings: list[str] = []
    if editor_page is not None and getattr(editor_page, "has_unsaved_changes", None):
        try:
            if editor_page.has_unsaved_changes():
                warnings.append("• " + (translator.translate("mw.warn_unsaved") if translator else "Trang Biên tập có thay đổi CHƯA LƯU"))
        except (AttributeError, RuntimeError):
            pass
    _labels = (
        (extract_page, translator.translate("nav.extract") if translator else "Trích xuất"),
        (translate_page, translator.translate("nav.translate") if translator else "Dịch"),
        (tts_page, translator.translate("mw.tts_full") if translator else "Lồng tiếng (TTS)"),
    )
    for page, label in _labels:
        if _is_page_busy(page):
            _tpl = translator.translate("mw.warn_busy") if translator else "Tác vụ {label} đang chạy (sẽ bị huỷ)"
            warnings.append("• " + _tpl.replace("{label}", label))
    return warnings


class MainWindow(FluentWindow):
    """Cửa sổ chính quản lý và điều hướng các trang chức năng thông qua sidebar."""

    _extraction_completed = Signal(object)

    def __init__(self, container: ApplicationContainer) -> None:
        super().__init__()
        self._container = container
        self._translator = container.translator

        self._ocr_preloaded = False
        self._current_video_hash = ""

        self._setup_window()

        self._extract_page = ExtractPage(container, parent=self)
        self._editor_page = EditorPage(container, parent=self)
        self._translate_page = TranslatePage(container, parent=self)
        self._tts_page = TTSPage(container, parent=self)
        self._publish_page = PublishPage(container, parent=self)
        self._debug_page = DebugPage(container, parent=self)
        self._projects_page = ProjectsPage(container.project_repository, container.translator, parent=self)
        self._log_page = LogPage(container.log_bridge, container.translator, parent=self)
        self._settings_page = SettingsPage(container, parent=self)

        # Mở lại dự án từ Thư viện → nạp vào quy trình các trang.
        self._projects_page.set_open_callback(self._open_project)

        # Cho trang Dịch lấy phụ đề hiện tại từ trang Biên tập theo yêu cầu.
        self._translate_page.set_editor_event_provider(self._collect_editor_events)
        # [#11] Cho trang Dịch tự đính video đang mở ở trang Biên tập (Auto-Attach).
        self._translate_page.set_editor_video_provider(self._collect_editor_video_path)
        # Cho trang Dịch gửi bản dịch ngược lại sang trang Biên tập.
        self._translate_page.set_push_to_editor_callback(self._push_translated_to_editor)
        # Cho trang TTS lấy phụ đề từ Biên tập và bản dịch từ trang Dịch.
        self._tts_page.set_editor_event_provider(self._collect_editor_events)
        self._tts_page.set_translate_event_provider(self._collect_translated_events)
        # Liên thông: TTS xong → lưu đường dẫn WAV + cài đặt vào dự án.
        self._tts_page.tts_completed.connect(self._persist_tts_project)
        # [v3.23.315] Trang Xuất bản lấy dữ liệu từ 3 trang trước.
        self._last_tts_wav_path: str | None = None
        self._tts_page.tts_completed.connect(self._remember_tts_wav_path)
        self._publish_page.set_video_provider(self._collect_editor_video_path)
        self._publish_page.set_subtitle_provider(self._export_translated_subtitle_temp)
        self._publish_page.set_audio_provider(lambda: self._last_tts_wav_path)
        self._publish_page.publish_completed.connect(self._persist_published_project)

        # Signal Trích xuất
        self._extract_page.extraction_completed = self._extraction_completed
        self._extraction_completed.connect(self._on_extraction_completed)

        # [FEATURE]: Signal Gỡ lỗi (Debugger -> Editor)
        self._debug_page._view_model.subtitles_built.connect(self._on_debug_subtitles_built)

        self._settings_page.view_model.ui_changed.connect(
            self._editor_page.apply_ui_settings
        )

        # [v3.23.316] Theo dõi trang trước để biết lúc nào người dùng RỜI một trang.
        self._previous_page: object | None = None

        self._register_pages()
        # [v3.23.317] Thanh tiến độ quy trình — luôn hiện, cho biết đang ở khâu nào.
        self._workflow_bar: WorkflowBar | None = None
        self._install_workflow_bar()
        logger.info("MainWindow đã khởi tạo xong.")

        # [v3.23.388] Nhắc tải lõi OCR nếu bản build nhỏ chưa kèm/chưa tải paddle. Hoãn
        # bằng singleShot để cửa sổ hiện ra trước, tránh chặn khởi tạo.
        from PySide6.QtCore import QTimer

        QTimer.singleShot(800, self._maybe_prompt_paddle_download)

    def _install_workflow_bar(self) -> None:
        """Chèn thanh tiến độ quy trình lên PHÍA TRÊN vùng nội dung.

        Cách làm: bọc ``stackedWidget`` vào một widget dọc chứa [thanh, nội dung] rồi
        đặt lại đúng vị trí cũ trong layout. Cách này KHÔNG phụ thuộc cấu trúc cụ thể
        của lớp cửa sổ nên chạy được với cả ``fluent_compat`` lẫn ``qfluentwidgets``.

        Nếu bố cục khác dự kiến (phiên bản qfluentwidgets đổi cấu trúc), hàm bỏ qua
        trong im lặng — ứng dụng vẫn chạy, chỉ là không có thanh.
        """
        try:
            from PySide6.QtWidgets import QVBoxLayout, QWidget

            stack = self.stackedWidget
            parent_layout = stack.parentWidget().layout()
            if parent_layout is None:
                return
            index = parent_layout.indexOf(stack)
            if index < 0:
                return
            stretch = getattr(parent_layout, "stretch", lambda _i: 1)(index)

            bar = WorkflowBar(self._translator)
            bar.go_to_page.connect(self._switch_to_page_key)

            wrapper = QWidget(stack.parentWidget())
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            wrapper_layout.setSpacing(0)
            parent_layout.removeWidget(stack)
            wrapper_layout.addWidget(bar)
            wrapper_layout.addWidget(stack, 1)
            parent_layout.insertWidget(index, wrapper, stretch)

            self._workflow_bar = bar
            self._refresh_workflow_bar()
        except Exception as exc:  # noqa: BLE001 — không có thanh vẫn dùng app được
            logger.warning("Không gắn được thanh tiến độ quy trình: %s", exc)

    def _maybe_prompt_paddle_download(self) -> None:
        """[v3.23.388] Nhắc tải lõi OCR (paddle) nếu bản build chưa kèm và chưa tải.

        Chỉ nhắc khi: (1) đang chạy bản ĐÓNG GÓI (frozen) và (2) KHÔNG import được ``paddle``
        (bootstrap đã thêm models/paddle_runtime vào sys.path trước đó, nên nếu đã tải thì
        import được → không nhắc). Bấm "Có" → mở trang Cài đặt để tải. Chạy từ nguồn (dev) bỏ
        qua vì paddle nằm trong môi trường phát triển.
        """
        import sys

        if not getattr(sys, "frozen", False):
            return
        import importlib.util

        try:
            if importlib.util.find_spec("paddle") is not None:
                return  # đã có paddle (nhúng sẵn hoặc đã tải trước đó)
        except (ImportError, ValueError):
            pass

        from PySide6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            self,
            self._translator.translate("mw.paddle_setup_title"),
            self._translator.translate("mw.paddle_setup_body"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.switchTo(self._settings_page)

    def _switch_to_page_key(self, object_name: str) -> None:
        """Chuyển tới trang theo ``objectName`` (dùng cho nút “Đi tới”).

        Args:
            object_name: Tên đối tượng của trang, vd ``"publishPage"``.
        """
        for index in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(index)
            if widget is not None and widget.objectName() == object_name:
                self.switchTo(widget)
                return
        logger.debug("Không tìm thấy trang có objectName='%s'.", object_name)

    def _current_stage(self) -> WorkflowStage:
        """Khâu hiện tại của dự án đang mở (``NEW`` nếu chưa có dự án)."""
        if not self._current_video_hash:
            return WorkflowStage.NEW
        try:
            record = self._container.project_repository.get(self._current_video_hash)
        except (sqlite3.Error, AttributeError, OSError) as exc:
            logger.debug("Không đọc được khâu dự án: %s", exc)
            return WorkflowStage.NEW
        return record.stage if record is not None else WorkflowStage.NEW

    def _refresh_workflow_bar(self) -> None:
        """Đồng bộ thanh tiến độ với khâu hiện tại của dự án."""
        if self._workflow_bar is not None:
            self._workflow_bar.set_stage(self._current_stage())

    def _setup_window(self) -> None:
        self.setWindowTitle(self._translator.translate("app.title"))
        # [v3.23.370] Icon ứng dụng (Subtitle Studio). Dùng _resolve_data_dir() — CÁCH
        # RESOLVE DUY NHẤT ĐÁNG TIN trong bản đóng gói PyInstaller (giống hệt nơi nạp
        # strings_*.json). Đường dẫn theo __file__ không đáng tin khi module nằm trong PYZ.
        from subtitles_extractor.composition.bootstrap import _resolve_data_dir

        icon_path = _resolve_data_dir() / "app.ico"
        if icon_path.is_file():
            from PySide6.QtGui import QIcon

            app_icon = QIcon(str(icon_path))
            self.setWindowIcon(app_icon)
            from PySide6.QtWidgets import QApplication

            app_instance = QApplication.instance()
            if app_instance is not None:
                app_instance.setWindowIcon(app_icon)
        else:
            logger.warning("Không tìm thấy icon ứng dụng tại: %s", icon_path)
        self.resize(QSize(1280, 820))
        self.setMinimumSize(QSize(960, 640))
        self.showMaximized()

    def _register_pages(self) -> None:
        self.addSubInterface(
            self._extract_page,
            FluentIcon.VIDEO,
            self._translator.translate("nav.extract"),
        )
        self.addSubInterface(
            self._editor_page,
            FluentIcon.EDIT,
            self._translator.translate("nav.editor"),
        )
        self.addSubInterface(
            self._translate_page,
            FluentIcon.LANGUAGE,
            self._translator.translate("nav.translate"),
        )
        self.addSubInterface(
            self._tts_page,
            FluentIcon.MICROPHONE,
            self._translator.translate("nav.tts"),
        )
        self.addSubInterface(
            self._publish_page,
            FluentIcon.VIDEO,
            self._translator.translate("nav.publish"),
        )
        self.addSubInterface(
            self._debug_page,
            FluentIcon.CODE,
            self._translator.translate("nav.debug"),
        )
        self.addSubInterface(
            self._projects_page,
            FluentIcon.FOLDER,
            self._translator.translate("nav.library"),
        )
        self.addSubInterface(
            self._log_page,
            FluentIcon.HISTORY,
            self._translator.translate("nav.log"),
            position=NavigationItemPosition.BOTTOM,
        )
        self.addSubInterface(
            self._settings_page,
            FluentIcon.SETTING,
            self._translator.translate("nav.settings"),
            position=NavigationItemPosition.BOTTOM,
        )

        self.stackedWidget.currentChanged.connect(self._on_page_changed)

    def _on_page_changed(self, index: int) -> None:
        current_widget = self.stackedWidget.widget(index)

        if not self._ocr_preloaded:
            if current_widget is self._editor_page or current_widget is self._extract_page:
                self._container.preload_ocr_engine_async()
                self._ocr_preloaded = True

        # [v3.23.316] RỜI trang Biên tập mà đang có phụ đề -> ghi khâu "Đã chỉnh sửa".
        # Trước đây WorkflowStage.EDITED có trong enum nhưng KHÔNG chỗ nào ghi, nên
        # Thư viện không bao giờ hiển thị khâu này.
        if self._previous_page is self._editor_page and current_widget is not self._editor_page:
            self._persist_edited_project()

        # [v3.23.316] VÀO trang Xuất bản -> tự điền sẵn 3 nguồn dữ liệu, người dùng
        # không phải bấm "Lấy từ app" ba lần.
        if current_widget is self._publish_page:
            self._publish_page.autofill_from_app()

        self._previous_page = current_widget
        self._refresh_workflow_bar()

    def _on_extraction_completed(self, events: list[SubtitleEvent]) -> None:
        if not events:
            return

        loaded = self._editor_page.load_events(events)

        try:
            current_video_metadata = self._extract_page.current_video
            if current_video_metadata is not None:
                if loaded:
                    self._editor_page.load_video(current_video_metadata.path)
                # Luôn lưu vào dự án dù người dùng giữ bản đang sửa -> kết quả không mất.
                self._persist_extracted_project(current_video_metadata.path, events)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không thể tự động đồng bộ Video Preview: %s.", exc)

        self._suggest_next_step(WorkflowStage.EXTRACTED)
        self._refresh_workflow_bar()

        if not loaded:
            logger.info("Giữ bản đang sửa; kết quả trích xuất đã lưu vào dự án.")
            return

        self.switchTo(self._editor_page)
        logger.info(
            "Trích xuất hoàn tất. Đã nạp %d câu và chuyển Tab sang Editor.", len(events)
        )

    def _persist_extracted_project(
        self, video_path: str, events: list[SubtitleEvent]
    ) -> None:
        """Lưu phụ đề gốc vào CSDL theo hash video để có thể mở lại sau này."""
        try:
            from subtitles_extractor.domain.entities.project_record import (
                ProjectRecord,
                WorkflowStage,
            )
            from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import (
                SrtExporter,
            )
            from subtitles_extractor.infrastructure.video.video_hasher import (
                compute_video_hash,
            )
            from pathlib import Path

            video_hash = compute_video_hash(video_path)
            srt_text = SrtExporter._build_content(events)
            repo = self._container.project_repository
            record = repo.get(video_hash) or ProjectRecord(video_hash=video_hash)
            record.video_path = str(video_path)
            record.video_name = Path(video_path).name
            record.original_subtitle = srt_text
            record.subtitle_format = "srt"
            # Lưu cài đặt OCR đang dùng (nếu đọc được) để mở lại tái lập.
            try:
                import json

                roi = self._container.video_state_repository.get(video_path)
                if roi is not None and roi.roi is not None:
                    record.ocr_settings_json = json.dumps({
                        "roi": {
                            "x": roi.roi.x, "y": roi.roi.y,
                            "width": roi.roi.width, "height": roi.roi.height,
                            "alignment": roi.roi.alignment.name,
                            "orientation": roi.roi.orientation.name,
                        }
                    }, ensure_ascii=False)
            except (AttributeError, OSError, ValueError):
                logger.debug("Không đọc được cài đặt OCR (ROI) để lưu dự án.")
            if record.stage < WorkflowStage.EXTRACTED:
                record.stage = WorkflowStage.EXTRACTED
            repo.save(record)
            self._current_video_hash = video_hash
            self._tts_page.suggest_output_path(video_path)
            self._translate_page.suggest_source_path(video_path)
            self._projects_page.refresh()
            logger.info(
                "Đã lưu dự án '%s' (hash %s) vào thư viện.",
                record.video_name, video_hash,
            )
        except (FileNotFoundError, OSError, ValueError, sqlite3.Error) as exc:
            logger.warning("Không lưu được dự án sau trích xuất: %s.", exc)

    def _open_project(self, record: object) -> None:
        """Mở lại một dự án từ Thư viện — nạp phụ đề gốc/bản dịch vào quy trình."""
        from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import (
            _parse_srt,
        )

        loaded = False
        if getattr(record, "original_subtitle", ""):
            try:
                events = list(_parse_srt(record.original_subtitle))
                if events:
                    loaded = self._editor_page.load_events(events)
            except (ValueError, AttributeError) as exc:
                logger.warning("Không nạp được phụ đề gốc của dự án: %s.", exc)

        video_path = getattr(record, "video_path", "")
        if video_path:
            try:
                self._editor_page.load_video(video_path)
            except Exception:  # noqa: BLE001
                logger.debug("Bỏ qua nạp video preview cho dự án mở lại.")
            self._tts_page.suggest_output_path(video_path)
            self._translate_page.suggest_source_path(video_path)

        self._current_video_hash = getattr(record, "video_hash", "")
        if loaded:
            self.switchTo(self._editor_page)
        logger.info("Đã mở lại dự án %s.", getattr(record, "video_name", ""))

    def _persist_tts_project(self, wav_path: str) -> None:
        """Lưu kết quả TTS (đường dẫn WAV + cài đặt) vào dự án hiện hành."""
        if not self._current_video_hash:
            logger.debug("Không có dự án hiện hành để lưu kết quả TTS — bỏ qua.")
            return
        try:
            import json

            from subtitles_extractor.domain.entities.project_record import (
                WorkflowStage,
            )

            repo = self._container.project_repository
            record = repo.get(self._current_video_hash)
            if record is None:
                return
            record.tts_audio_path = wav_path
            # Lưu kèm cài đặt TTS chính (đọc từ trang TTS) để mở lại tái lập.
            try:
                tts = self._tts_page
                record.tts_settings_json = json.dumps({
                    "base_speed": tts._speed.value(),
                    "max_speed": tts._max_speed.value(),
                    "max_drift_ms": tts._max_drift.value(),
                    "lead_in_ms": tts._lead_in.value(),
                    "anchor_gap_s": tts._anchor_gap.value(),
                    "max_segment_s": tts._max_segment.value(),
                    "comfort_ratio": tts._comfort_ratio.value(),
                    "min_pause_pct": tts._min_pause.value(),
                    "max_intra_gap_s": tts._max_intra_gap.value(),
                    "timing_strategy": tts._strategy.currentData(),
                    "high_quality": tts._high_quality.isChecked(),
                    "voice_clarity": tts._voice_clarity.isChecked(),
                }, ensure_ascii=False)
            except AttributeError:
                logger.debug("Không đọc được cài đặt TTS để lưu — bỏ qua phần này.")
            record.stage = WorkflowStage.TTS_DONE
            self._suggest_next_step(WorkflowStage.TTS_DONE)
            self._refresh_workflow_bar()
            repo.save(record)
            self._projects_page.refresh()
            logger.info("Đã lưu kết quả TTS vào dự án %s → %s.",
                        self._current_video_hash, wav_path)
        except (OSError, ValueError, sqlite3.Error) as exc:
            logger.warning("Không lưu được kết quả TTS vào dự án: %s.", exc)

    def _collect_editor_events(self) -> list[SubtitleEvent]:
        """Cung cấp phụ đề hiện tại của trang Biên tập cho trang Dịch/TTS."""
        return self._editor_page.get_current_events()

    def _collect_editor_video_path(self) -> str | None:
        """[#11] Cung cấp đường dẫn video đang mở ở trang Biên tập cho trang Dịch."""
        return self._editor_page.get_current_video_path()

    def _persist_edited_project(self) -> None:
        """Ghi khâu "Đã chỉnh sửa" khi người dùng rời trang Biên tập có phụ đề."""
        if not self._current_video_hash:
            return
        events = self._collect_editor_events()
        if not events:
            return
        try:
            repo = self._container.project_repository
            record = repo.get(self._current_video_hash)
            if record is None or record.stage >= WorkflowStage.EDITED:
                return
            record.stage = WorkflowStage.EDITED
            repo.save(record)
            logger.debug("Dự án %s -> khâu Đã chỉnh sửa.", self._current_video_hash)
        except (sqlite3.Error, AttributeError, OSError) as exc:
            logger.warning("Không ghi được khâu Đã chỉnh sửa: %s", exc)

    def _persist_published_project(self, output_path: object) -> None:
        """Ghi khâu "Đã xuất bản" + đường dẫn tệp hoàn chỉnh, rồi chúc mừng."""
        if self._current_video_hash:
            try:
                repo = self._container.project_repository
                record = repo.get(self._current_video_hash)
                if record is not None:
                    record.stage = WorkflowStage.PUBLISHED
                    record.published_video_path = str(output_path)
                    repo.save(record)
            except (sqlite3.Error, AttributeError, OSError) as exc:
                logger.warning("Không ghi được khâu Đã xuất bản: %s", exc)
        self._suggest_next_step(WorkflowStage.PUBLISHED)
        self._refresh_workflow_bar()

    def _suggest_next_step(self, completed_stage: WorkflowStage) -> None:
        """Gợi ý việc cần làm tiếp sau khi hoàn thành một khâu.

        Trước v3.23.316 ứng dụng KHÔNG hề gợi ý bước tiếp theo ở bất kỳ đâu — người dùng
        phải tự biết thứ tự Trích xuất → Biên tập → Dịch → TTS → Xuất bản.

        Args:
            completed_stage: Khâu vừa hoàn thành.
        """
        try:
            from subtitles_extractor.presentation.fluent_compat import (
                InfoBar,
                InfoBarPosition,
            )

            if completed_stage.is_complete:
                InfoBar.success(
                    title=self._translator.translate("mw.done_t"),
                    content=self._translator.translate("mw.done_c"),
                    parent=self, position=InfoBarPosition.TOP, duration=6000,
                )
                return
            InfoBar.info(
                title=f"Xong: {completed_stage.label_vi}",
                content=self._translator.translate("mw.next_step").replace("{action}", str(completed_stage.next_action_vi)),
                parent=self, position=InfoBarPosition.TOP, duration=6000,
            )
        except Exception as exc:  # noqa: BLE001 — gợi ý hỏng không được phá luồng chính
            logger.debug("Không hiển thị được gợi ý bước tiếp theo: %s", exc)

    def _remember_tts_wav_path(self, wav_path: str) -> None:
        """[v3.23.315] Ghi nhớ tệp giọng đọc mới nhất cho trang Xuất bản."""
        self._last_tts_wav_path = wav_path or None

    def _export_translated_subtitle_temp(self) -> str | None:
        """Trả về tệp phụ đề ĐÚNG NHẤT cho trang Xuất bản.

        [v3.23.318] SỬA LỖI LỆCH ĐỒNG BỘ: khâu TTS không chỉ tạo giọng đọc mà còn
        **chỉnh lại mốc thời gian** phụ đề cho khớp lời thoại đã tổng hợp, rồi ghi ra
        ``<tên>.tts.<lang>.srt``. Trước đây hàm này luôn lấy phụ đề từ trang Dịch (mốc
        GỐC), nên phim xuất ra có phụ đề LỆCH so với giọng thuyết minh.

        Thứ tự ưu tiên: phụ đề đồng bộ TTS → bản dịch → bản đang sửa ở Biên tập.

        Returns:
            Đường dẫn tệp phụ đề, hoặc ``None`` nếu chưa có phụ đề nào.
        """
        from subtitles_extractor.application.services.publish_subtitle_selector import (
            SubtitleSource,
            choose_publish_subtitle,
        )

        tts_audio = Path(self._last_tts_wav_path) if self._last_tts_wav_path else None
        translated = self._write_translated_subtitle_temp()

        choice = choose_publish_subtitle(
            tts_audio_path=tts_audio,
            translated_subtitle_path=translated,
            target_language=self._current_target_language(),
            audio_will_be_used=tts_audio is not None,
        )
        if choice.warning:
            logger.warning("Xuất bản: %s", choice.warning)
        if choice.source is SubtitleSource.TTS_SYNCED:
            logger.info("Xuất bản dùng phụ đề ĐỒNG BỘ TTS: %s", choice.path)
        return str(choice.path) if choice.path is not None else None

    def _current_target_language(self) -> str:
        """Mã ngôn ngữ đích của dự án hiện hành (rỗng nếu chưa biết)."""
        if not self._current_video_hash:
            return ""
        try:
            record = self._container.project_repository.get(self._current_video_hash)
        except (sqlite3.Error, AttributeError, OSError):
            return ""
        return (record.target_lang or "") if record is not None else ""

    def _write_translated_subtitle_temp(self) -> Path | None:
        """Ghi phụ đề bản dịch ra tệp SRT tạm (ffmpeg cần TỆP, không nhận danh sách).

        Returns:
            Đường dẫn tệp tạm, hoặc ``None`` nếu chưa có phụ đề nào.
        """
        events = self._collect_translated_events() or self._collect_editor_events()
        if not events:
            return None
        try:
            import tempfile

            from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import (
                SrtExporter,
            )

            temp_dir = Path(tempfile.gettempdir()) / "SubtitlesExtractor_publish"
            temp_dir.mkdir(parents=True, exist_ok=True)
            return Path(SrtExporter().export(events, temp_dir / "phu_de_da_dich.srt"))
        except (OSError, SubtitlesExtractorError) as exc:
            logger.warning("Không xuất được phụ đề tạm cho Xuất bản: %s", exc)
            return None

    def _collect_translated_events(self) -> list[SubtitleEvent]:
        """Cung cấp bản dịch từ trang Dịch cho trang TTS."""
        return self._translate_page._view_model.translated_events

    def _push_translated_to_editor(self, events: list[SubtitleEvent]) -> None:
        """Nhận bản dịch từ trang Dịch, nạp vào trang Biên tập.

        Gọi qua callback được tiêm khi khởi tạo. MainWindow tự chuyển sang
        trang Biên tập để người dùng thấy kết quả ngay.
        """
        loaded = self._editor_page.load_events(events)
        self._persist_translated_project(events)
        if not loaded:
            logger.info("Người dùng giữ bản đang sửa; bản dịch đã lưu vào dự án.")
            return
        # Điều hướng sang trang Biên tập
        try:
            self.switchTo(self._editor_page)
        except (AttributeError, RuntimeError, TypeError) as exc:
            # API điều hướng khác/đối tượng đã huỷ → bỏ qua, kết quả đã được nạp.
            logger.debug("Không điều hướng được sang trang Biên tập: %s", exc)

    def _persist_translated_project(self, events: list[SubtitleEvent]) -> None:
        """Lưu bản dịch vào CSDL nếu đang gắn với một dự án (theo hash video)."""
        if not self._current_video_hash or not events:
            logger.debug("Không có dự án hiện hành để lưu bản dịch — bỏ qua.")
            return
        try:
            from subtitles_extractor.domain.entities.project_record import (
                WorkflowStage,
            )
            from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import (
                SrtExporter,
            )

            repo = self._container.project_repository
            record = repo.get(self._current_video_hash)
            if record is None:
                return
            record.translated_subtitle = SrtExporter._build_content(events)
            if record.stage < WorkflowStage.TRANSLATED:
                record.stage = WorkflowStage.TRANSLATED
                self._suggest_next_step(WorkflowStage.TRANSLATED)
                self._refresh_workflow_bar()
            repo.save(record)
            self._projects_page.refresh()
            logger.info("Đã lưu bản dịch (%d câu) vào dự án %s.",
                        len(events), self._current_video_hash)
        except (OSError, ValueError) as exc:
            logger.warning("Không lưu được bản dịch vào dự án: %s.", exc)

    def _on_debug_subtitles_built(self, events: list[SubtitleEvent]) -> None:
        """Hứng Signal từ trang Gỡ lỗi (Tương tự như Trích xuất)"""
        if not events:
            return

        from subtitles_extractor.presentation.fluent_compat import InfoBar

        if not self._editor_page.load_events(events):
            logger.info("Người dùng giữ bản đang sửa; bỏ qua nạp dữ liệu gỡ lỗi.")
            return
        self.switchTo(self._editor_page)

        InfoBar.success(
            self._translator.translate("mw.build_ok_t"),
            self._translator.translate("mw.build_ok_c").replace("{n}", str(len(events))),
            parent=self._editor_page,
            duration=3000
        )
        logger.info("Debugger: Đã nạp %d câu phụ đề vào Editor.", len(events))

    def _collect_close_warnings(self) -> list[str]:
        """[v3.23.113] Liệt kê các việc sẽ mất nếu thoát ngay (để hỏi xác nhận)."""
        return _build_close_warnings(
            getattr(self, "_editor_page", None),
            getattr(self, "_extract_page", None),
            getattr(self, "_translate_page", None),
            getattr(self, "_tts_page", None),
            self._translator,
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        # [v3.23.113] Hỏi xác nhận nếu đang chạy tác vụ dài hoặc có sửa dở chưa lưu,
        # tránh người dùng bấm nhầm nút X làm mất công việc.
        risk = self._collect_close_warnings()
        if risk:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, self._translator.translate("mw.exit_t"),
                self._translator.translate("mw.exit_b1") + "\n\n"
                + "\n".join(risk)
                + "\n\n" + self._translator.translate("mw.exit_b2"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        logger.info("Người dùng đóng ứng dụng. Đang dọn dẹp Worker...")

        # [v3.20.3 #1] Graceful shutdown: làm mờ UI + hiện "Đang đóng an toàn" +
        # bơm processEvents() để Windows KHÔNG báo "Not Responding" trong lúc chờ
        # các luồng AI nhả VRAM (tối đa 8s ở Bước 2).
        try:
            from PySide6.QtWidgets import QApplication, QLabel
            from PySide6.QtCore import Qt as _Qt

            self.setEnabled(False)
            self._shutdown_overlay = QLabel(self._translator.translate("mw.closing"), self)
            self._shutdown_overlay.setAlignment(_Qt.AlignmentFlag.AlignCenter)
            self._shutdown_overlay.setStyleSheet(
                f"background-color: rgba(0,0,0,170); color: white; font-size: {_m.FONT_SIZE_HEADING}px;"
            )
            self._shutdown_overlay.setGeometry(self.rect())
            self._shutdown_overlay.show()
            self._shutdown_overlay.raise_()
            QApplication.processEvents()
        except (ImportError, RuntimeError, AttributeError) as exc:
            logger.debug("Không dựng được overlay shutdown (bỏ qua): %s", exc)

        try:
            # ── Bước 1: Yêu cầu hủy tất cả tác vụ đang chạy ──
            if hasattr(self._extract_page, "cancel_extraction"):
                self._extract_page.cancel_extraction()

            if hasattr(self._editor_page, "cancel_reocr"):
                self._editor_page.cancel_reocr()

            # [v3.7 fix Q2] Dọn worker dịch (huỷ + chờ) để tránh crash
            # "QThread: Destroyed while thread is still running" khi đang dịch.
            if hasattr(self._translate_page, "cleanup"):
                self._translate_page.cleanup()

            # Dọn worker TTS
            if hasattr(self._tts_page, "cleanup"):
                self._tts_page.cleanup()

            # [BUG FIX v2.9+]: Dừng AudioWaveformWidget trước khi chờ thread.
            waveform = getattr(self._editor_page, "_waveform_widget", None)
            if waveform is not None and hasattr(waveform, "close_widget"):
                try:
                    waveform.close_widget()
                except (RuntimeError, AttributeError) as exc:
                    logger.debug("Lỗi khi đóng waveform widget: %s.", exc)

            # ── Bước 2: CHỜ threads kết thúc TRƯỚC khi shutdown container ──
            # CRITICAL: container.shutdown() giải phóng GPU model. Nếu gọi khi
            # thread OCR vẫn đang infer → crash do model bị xoá giữa chừng.
            _THREAD_WAIT_MS = 8_000

            # [v3.20.3 #1] Bơm processEvents trước các wait dài để overlay
            # "Đang đóng an toàn" kịp vẽ và Windows không cờ "Not Responding".
            def _pump_ui() -> None:
                with contextlib.suppress(ImportError, RuntimeError):
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents()

            vm = getattr(self._extract_page, "_view_model", None)
            if vm is not None:
                extract_thread = getattr(vm, "active_extract_thread", None)
                if extract_thread is not None and extract_thread.isRunning():
                    logger.info("Chờ Extract thread kết thúc (tối đa %ds)...", _THREAD_WAIT_MS // 1000)
                    _pump_ui()
                    if not extract_thread.wait(_THREAD_WAIT_MS):
                        logger.warning("Extract thread không kết thúc đúng hạn — buộc terminate.")
                        extract_thread.terminate()
                        extract_thread.wait(2000)

                detect_thread = getattr(vm, "active_detect_thread", None)
                if detect_thread is not None and detect_thread.isRunning():
                    logger.info("Chờ Detect thread kết thúc...")
                    _pump_ui()
                    if not detect_thread.wait(4000):
                        detect_thread.terminate()
                        detect_thread.wait(1000)

            evm = getattr(self._editor_page, "_view_model", None)
            if evm is not None:
                reocr_thread = getattr(evm, "_reocr_thread", None)
                if reocr_thread is not None and reocr_thread.isRunning():
                    logger.info("Chờ ReOCR thread kết thúc...")
                    _pump_ui()
                    if not reocr_thread.wait(_THREAD_WAIT_MS):
                        logger.warning("ReOCR thread không kết thúc đúng hạn — buộc terminate.")
                        reocr_thread.terminate()
                        reocr_thread.wait(2000)

            # [v3.6 bugfix MW-2]: Dọn dẹp tường minh SeekWorker và PersistentVideoReader
            # của extract_page. closeEvent() của child widget KHÔNG tự động được gọi
            # khi parent window đóng → SeekWorker và file handle bị leak.
            seek_thread = getattr(self._extract_page, "_seek_thread", None)
            if seek_thread is not None and seek_thread.isRunning():
                logger.debug("Dọn dẹp extract_page SeekWorker...")
                seek_thread.requestInterruption()
                seek_thread.quit()
                seek_thread.wait(2000)

            video_reader = getattr(self._extract_page, "_video_reader", None)
            if video_reader is not None and hasattr(video_reader, "close"):
                try:
                    video_reader.close()
                except (RuntimeError, AttributeError) as exc:
                    logger.debug("Lỗi khi đóng video reader: %s.", exc)

            # [v3.6 bugfix MW-1]: Chờ QThreadPool hoàn tất AutoSave/Export runnables.
            # Trước đây thiếu bước này → _AutoSaveRunnable có thể bị kill giữa
            # transaction SQLite, _ExportRunnable có thể bị kill giữa atomic write.
            from PySide6.QtCore import QThreadPool
            thread_pool = QThreadPool.globalInstance()
            if not thread_pool.waitForDone(5000):
                logger.warning(
                    "QThreadPool: %d active thread(s) không hoàn tất trong 5s — "
                    "tiếp tục shutdown.",
                    thread_pool.activeThreadCount(),
                )

            # ── Bước 3: Shutdown container (giải phóng GPU, SQLite, settings) ──
            self._container.shutdown()
            logger.info("Shutdown hệ thống an toàn.")

        except Exception as exc:
            logger.warning("Quá trình đóng ứng dụng gặp lỗi nhẹ: %s.", exc)
        finally:
            super().closeEvent(event)

__all__ = ["MainWindow"]
