"""Trang "Chỉnh sửa phụ đề" — chuyên nghiệp với video player.

BẢN CẬP NHẬT ĐỘT PHÁ (V3.47 - The "Asynchronous" Polish):
    * [STABILITY FIX] Tự động nán lại vài giây khi đóng App nếu ổ cứng chưa ghi xong.
    * [UX FIX] Vô hiệu hóa nút Xuất (Export) tạm thời khi hệ thống đang ghi ngầm 
      vào ổ cứng, chống lỗi đúp file.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import time
from pathlib import Path

from loguru import logger
from subtitles_extractor.presentation.qt_compat import is_valid
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, QItemSelection, QItemSelectionModel, Signal
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QKeyEvent, QShowEvent, QHideEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QListWidget, QMenu, QMessageBox, QStackedLayout, QTableView, QVBoxLayout, QWidget,
)
from subtitles_extractor.presentation.fluent_compat import (
    CaptionLabel, ComboBox, DoubleSpinBox, FluentIcon, InfoBar,
    InfoBarPosition, LineEdit, PrimaryPushButton, ProgressBar, PushButton, Slider, StrongBodyLabel, TextEdit, ToolButton,
)

from subtitles_extractor.application.services.subtitle_editor_service import EditorState
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.presentation.view_models.editor_page_view_model import EditorPageViewModel
from subtitles_extractor.presentation.widgets import create_video_widget
from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme.styles import mono_label_style
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name
from subtitles_extractor.presentation.widgets.audio_waveform_widget import AudioWaveformWidget
from subtitles_extractor.presentation.widgets.editor_widgets import CpsGauge, SubtitleTagHighlighter, TimeSpinBox
from subtitles_extractor.presentation.widgets.mpv_video_widget import MpvVideoWidget as RoiMpvVideoWidget  # noqa: F401
from subtitles_extractor.presentation.widgets.subtitle_table_model import (
    COL_DUR, COL_END, COL_GAP, COL_INDEX, COL_SCORE, COL_START, COL_TEXT, COL_WARN,
    SubtitleFilterProxyModel, SubtitleTableModel,
)

from subtitles_extractor.presentation.pages.editor_constants import (  # noqa: F401 — giữ tên cũ trong editor_page
    _MAX_LINE_CHARS,
    _DEBOUNCED_SEEK_MS,
    _PREVIEW_DEBOUNCE_MS,
    _FILTER_DEBOUNCE_MS,
    _SYNC_HIGHLIGHT_MS,
    _AUTOSAVE_INTERVAL_MS,
    _AUTO_SCROLL_PAUSE_AFTER_USER_SEC,
    _MIN_PLAYBACK_SPEED,
    _MAX_PLAYBACK_SPEED,
    _SLIDER_RESOLUTION,
    _LOW_CONFIDENCE_THRESHOLD,
    _TOO_FAST_CPS_THRESHOLD,
    _TOO_SHORT_DURATION_SEC,
)
from subtitles_extractor.presentation.pages.editor_export import EditorExportMixin
from subtitles_extractor.presentation.pages.editor_find_replace import EditorFindReplaceMixin
from subtitles_extractor.presentation.pages.editor_reocr import EditorReOcrMixin
from subtitles_extractor.presentation.pages.editor_dialogs import (  # noqa: F401 — giữ tên cũ trong editor_page
    AdvancedReOcrDialog,
    EditorSettingsDialog,
    _DetachedVideoWindow,
    _fmt_mmss,
)

# [v3.23.111] Bảng phím tắt trình chỉnh sửa — gom nhóm để người dùng dễ khám phá.
# (key = nhãn nhóm, value = danh sách (phím, mô tả)). Dùng để dựng hộp thoại trợ giúp.
_EDITOR_SHORTCUTS: dict[str, list[tuple[str, str]]] = {
    "sc_grp_playback": [
        ("Space / K", "sc_playpause"),
        ("Alt+Space", "sc_play_selected"),
        ("J / L", "sc_seek2"),
        ("← / →", "sc_seek5"),
        ("[ / ]", "sc_speed"),
        ("C", "sc_center_wave"),
        ("M", "sc_mute"),
    ],
    "sc_grp_navigate": [
        ("Ctrl+↑ / ↓", "sc_select_updown"),
        ("Ctrl+Shift+← / →", "sc_jump_edge"),
        ("F8 / Shift+F8", "sc_next_suspect"),
    ],
    "sc_grp_timing": [
        ("Alt+← / →", "sc_shift_start"),
        ("Alt+Shift+← / →", "sc_shift_end"),
        ("Ctrl+[ / Ctrl+]", "sc_set_at_playhead"),
    ],
    "sc_grp_edit": [
        ("Ctrl+B / Ctrl+I", "sc_bold_italic"),
        ("Alt+Insert", "sc_insert"),
        ("Delete", "sc_delete"),
        ("Ctrl+M", "sc_merge"),
        ("Ctrl+T", "sc_split"),
    ],
    "sc_grp_file": [
        ("Ctrl+Z / Ctrl+Y", "sc_undo_redo"),
        ("Ctrl+S", "sc_save"),
        ("Ctrl+F", "sc_find"),
        ("F3 / Shift+F3", "sc_find_next"),
        ("Ctrl+O", "sc_open_sub"),
        ("Ctrl+Shift+O", "sc_open_video"),
        ("F11 / ESC", "sc_fullscreen"),
        ("F1", "sc_help"),
    ],
}


def next_suspect_row(suspect_rows: list[int] | set[int], current: int, forward: bool = True) -> int | None:
    """[v3.23.682] Hàng nghi ngờ kế tiếp (vòng lại từ đầu/cuối); ``None`` nếu không còn dòng nghi."""
    rows = sorted(suspect_rows)
    if not rows:
        return None
    if forward:
        return next((r for r in rows if r > current), rows[0])
    return next((r for r in reversed(rows) if r < current), rows[-1])


def editor_shortcuts_html(translator) -> str:
    """Dựng nội dung HTML bảng phím tắt (tách riêng để test được, không cần GUI).

    Args:
        translator: Bộ dịch (``JsonTranslator``) để dịch nhãn nhóm + mô tả.
    """
    def tr(key: str) -> str:
        return translator.translate(f"editor.{key}")

    parts = [f"<h3>{tr('sc_title')}</h3>"]
    for group_key, rows in _EDITOR_SHORTCUTS.items():
        parts.append(f"<p><b>{tr(group_key)}</b></p><table cellspacing='4'>")
        for key, desc_key in rows:
            parts.append(
                f"<tr><td><code>{key}</code></td>"
                f"<td>&nbsp;&nbsp;{tr(desc_key)}</td></tr>"
            )
        parts.append("</table>")
    return "".join(parts)


class EditorPage(EditorReOcrMixin, EditorFindReplaceMixin, EditorExportMixin, QWidget):
    #: [v3.23.656] Lượt Re-OCR đang chạy (trang Trích xuất khoá bắt đầu trích xuất theo tín hiệu này).
    reocr_running_changed = Signal(bool)
    #: [v3.23.682] Số dòng nghi ngờ OCR (thanh điều hướng hiện "● n" cạnh tên trang).
    review_count_changed = Signal(int)

    _VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts", ".m4v"})
    _SUBTITLE_EXTENSIONS: frozenset[str] = frozenset({".srt", ".ass", ".ssa", ".vtt"})

    def __init__(self, container: ApplicationContainer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("editorPage")
        self._container = container
        self._translator = container.translator
        self._view_model = EditorPageViewModel(container, parent=self)

        self._current_video_path: Path | None = None
        self._detached_window = None

        self._loop_active = False
        self._loop_start_time: float | None = None
        self._loop_end_time: float | None = None
        self._loop_single_play: bool = False

        self._last_active_row: int = -1
        self._need_snap_back: bool = False

        self._pending_seek_row = -1
        self._last_user_scroll_time = 0.0
        self._last_sync_highlight_time: float = 0.0
        
        self._is_syncing_from_waveform: bool = False
        self._last_waveform_seek_time: float = 0.0

        self._updating_slider_from_playback: bool = False
        self._is_closing = False
        self._was_playing_before_seek = False
        self._is_scrubbing_slider = False

        self._preview_idx = 0
        uid = id(self)
        self._preview_files = [
            os.path.join(tempfile.gettempdir(), f"se_preview_A_{uid}.ass"),
            os.path.join(tempfile.gettempdir(), f"se_preview_B_{uid}.ass"),
        ]

        self._debounced_seek_timer = QTimer(self)
        self._debounced_seek_timer.setSingleShot(True)
        self._debounced_seek_timer.setInterval(_DEBOUNCED_SEEK_MS)
        self._debounced_seek_timer.timeout.connect(self._execute_debounced_seek)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._generate_ass_preview)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(_AUTOSAVE_INTERVAL_MS)
        self._autosave_timer.timeout.connect(self._view_model.save_autosave)

        self._filter_debounce_timer = QTimer(self)
        self._filter_debounce_timer.setSingleShot(True)
        self._filter_debounce_timer.setInterval(_FILTER_DEBOUNCE_MS)
        self._filter_debounce_timer.timeout.connect(self._apply_filters)
        
        self._html_tag_regex = re.compile(r"<[^>]+>")
        self._ass_override_regex = re.compile(r"\{[^}]*\}")

        self.setAcceptDrops(True)

        self._build_ui()
        self._connect_signals()
        
        self._remove_focus_from_buttons()

    def _remove_focus_from_buttons(self) -> None:
        # [v3.23.267] PySide6 findChildren nhận MỘT type. PushButton/PrimaryPushButton đều
        # kế thừa QPushButton, ToolButton là QToolButton -> tìm 2 lớp cơ sở là đủ.
        from PySide6.QtWidgets import QPushButton, QToolButton

        for btn in (*self.findChildren(QPushButton), *self.findChildren(QToolButton)):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            # [v3.23.659] Nút không lấy focus → ô sửa chữ không mất focus → chữ đang gõ chưa được ghi. Ghi TRƯỚC khi
            # nút làm việc (pressed phát trước clicked): Gộp / Áp thời gian / Xoá / Chèn… từng làm mất chữ vừa gõ.
            btn.pressed.connect(self._commit_pending_text)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._container.ocr_engine.is_initialized:
            InfoBar.info(
                title=self._translator.translate("editor.ib_loading_t"), content=self._translator.translate("editor.ib_loading_c"),
                parent=self, position=InfoBarPosition.TOP_RIGHT, duration=3000
            )

        self._container.preload_ocr_engine_async()

        if self._view_model.current_events:
            self._autosave_timer.start()

        if not getattr(self, '_layout_restored', False):
            self._layout_restored = True
            settings_repo = self._container.settings_service._repo

            saved_v_split = settings_repo.load("editor_v_splitter", None)
            saved_h_split = settings_repo.load("editor_h_splitter", None)

            if saved_v_split:
                with contextlib.suppress(ValueError, RuntimeError, TypeError):
                    self._v_splitter.restoreState(bytes.fromhex(saved_v_split))
            if saved_h_split:
                with contextlib.suppress(ValueError, RuntimeError, TypeError):
                    self._h_splitter.restoreState(bytes.fromhex(saved_h_split))

            h_sizes = self._h_splitter.sizes()
            if not h_sizes or sum(h_sizes) < 100 or h_sizes[0] < 50 or h_sizes[1] < 50:
                total_w = self.width() if self.width() > 100 else 1280
                self._h_splitter.setSizes([total_w // 2, total_w // 2])

            v_sizes = self._v_splitter.sizes()
            if not v_sizes or sum(v_sizes) < 100 or v_sizes[0] < 100:
                total_h = self.height() if self.height() > 100 else 720
                wf_height = max(150, int(total_h * 0.25))
                self._v_splitter.setSizes([total_h - wf_height, wf_height])

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        try:
            self._autosave_timer.stop()
            self._debounced_seek_timer.stop()
            self._preview_timer.stop()
            if hasattr(self, "_filter_debounce_timer"):
                self._filter_debounce_timer.stop()
        except RuntimeError:
            pass
        
        if hasattr(self, '_video_widget') and self._video_widget.is_playing:
            self._video_widget.pause()

    def has_unsaved_changes(self) -> bool:
        """[v3.23.113] Có thay đổi chưa lưu/xuất không (dùng cho cảnh báo khi thoát app)."""
        try:
            return bool(self._table_model.events) and (
                self._view_model._service.snapshot_state().is_dirty
            )
        except (AttributeError, RuntimeError):
            return False

    def _confirm_discard_unsaved(self) -> bool:
        if not self._table_model.events or not self._view_model._service.snapshot_state().is_dirty:
            return True
            
        reply = QMessageBox.question(
            self, self._translator.translate("editor.dlg_unsaved_t"),
            self._translator.translate("editor.dlg_unsaved_b"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        return reply == QMessageBox.StandardButton.Yes

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if not event.mimeData().hasUrls():
            event.ignore(); return
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if not local_path: continue
            if Path(local_path).suffix.lower() in self._VIDEO_EXTENSIONS | self._SUBTITLE_EXTENSIONS:
                event.acceptProposedAction(); return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasUrls(): return

        video_path: Path | None = None
        subtitle_path: Path | None = None
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if not local_path: continue
            ext = Path(local_path).suffix.lower()
            if ext in self._VIDEO_EXTENSIONS and video_path is None: video_path = Path(local_path)
            elif ext in self._SUBTITLE_EXTENSIONS and subtitle_path is None: subtitle_path = Path(local_path)

        if video_path is not None or subtitle_path is not None:
            if not self._confirm_discard_unsaved():
                event.ignore()
                return

        if video_path is not None:
            self.load_video(video_path)
            InfoBar.success(self._translator.translate("editor.ib_opened_video"), video_path.name, parent=self, duration=2000)

            if subtitle_path is None:
                has_cached = self._container.subtitle_repository.has_events(str(video_path.resolve()))
                if has_cached:
                    cached_events = self._container.subtitle_repository.load_events(str(video_path.resolve()))
                    if cached_events:
                        self.load_events(cached_events, confirm=False)
                        InfoBar.success(self._translator.translate("editor.ib_db"), self._translator.translate("editor.ib_db_loaded"), parent=self, duration=2500)

        if subtitle_path is not None:
            self._view_model.load_from_file(subtitle_path)
            InfoBar.success(self._translator.translate("editor.ib_opened_sub"), subtitle_path.name, parent=self, duration=2000)

        event.acceptProposedAction()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # [v3.23.674] Hai thanh công cụ tự xuống dòng (trước đây bị tràn/cắt chữ ở màn hình 1366×768); nút OCR lại
        # chuyển lên nhóm sửa câu để hàng tìm/thay đủ chỗ.
        root.addWidget(self._build_file_toolbar())
        root.addWidget(self._build_edit_toolbar())
        root.addLayout(self._build_search_toolbar())

        from PySide6.QtWidgets import QSplitter
        self._v_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._v_splitter.setChildrenCollapsible(False); self._v_splitter.setOpaqueResize(False)
        self._h_splitter = QSplitter(Qt.Orientation.Horizontal, self._v_splitter)
        self._h_splitter.setChildrenCollapsible(False); self._h_splitter.setOpaqueResize(False)

        top_left_panel = self._build_top_left_panel()
        top_left_panel.setMinimumWidth(350)
        self._h_splitter.addWidget(top_left_panel)

        top_right_panel = self._build_top_right_panel()
        top_right_panel.setMinimumWidth(400)
        self._h_splitter.addWidget(top_right_panel)

        self._h_splitter.setSizes([600, 600])
        self._h_splitter.setStretchFactor(0, 1)
        self._h_splitter.setStretchFactor(1, 1)

        bottom_panel = self._build_bottom_waveform_panel()
        bottom_panel.setMinimumHeight(80)
        self._v_splitter.addWidget(bottom_panel)

        self._v_splitter.setSizes([800, 150])
        self._v_splitter.setStretchFactor(0, 1)
        self._v_splitter.setStretchFactor(1, 0)

        root.addWidget(self._v_splitter, stretch=1)
        root.addLayout(self._build_statusbar())

    def _build_file_toolbar(self) -> QWidget:
        from subtitles_extractor.presentation.widgets.flow_layout import flow_row

        holder, layout = flow_row(self)
        self._open_video_button = PushButton(self._translator.translate("editor.ed_btn_open_video"))
        self._open_button = PushButton("📂 " + self._translator.translate("editor.btn_open"))
        self._export_button = PrimaryPushButton("💾 " + self._translator.translate("editor.btn_export"))
        self._export_button.setEnabled(False)
        self._undo_button = PushButton(self._translator.translate("editor.ed_btn_undo"))
        self._undo_button.setEnabled(False)
        self._redo_button = PushButton(self._translator.translate("editor.ed_btn_redo"))
        self._redo_button.setEnabled(False)
        self._history_button = PushButton(self._translator.translate("editor.ed_btn_history"))
        self._btn_shortcuts = PushButton(self._translator.translate("editor.ed_btn_shortcuts"))
        self._btn_shortcuts.setToolTip(self._translator.translate("editor.ed_tt_shortcuts"))

        layout.addWidget(self._open_video_button)
        layout.addWidget(self._open_button)
        layout.addWidget(self._export_button)
        layout.addWidget(self._undo_button)
        layout.addWidget(self._redo_button)
        layout.addWidget(self._history_button)
        layout.addWidget(self._btn_shortcuts)
        return holder

    def _build_edit_toolbar(self) -> QWidget:
        from subtitles_extractor.presentation.widgets.flow_layout import flow_row

        holder, layout = flow_row(self)
        self._insert_button = PushButton(self._translator.translate("editor.ed_btn_insert"))
        self._insert_button.setToolTip(self._translator.translate("editor.ed_tt_insert"))
        
        self._delete_button = PushButton(self._translator.translate("editor.ed_btn_delete"))
        self._delete_button.setToolTip(self._translator.translate("editor.ed_tt_delete"))
        
        self._split_button = PushButton(self._translator.translate("editor.ed_btn_split"))
        self._split_button.setToolTip(self._translator.translate("editor.ed_tt_split"))
        
        self._merge_button = PushButton(self._translator.translate("editor.ed_btn_merge_down"))
        self._merge_button.setToolTip(self._translator.translate("editor.ed_tt_merge"))
        
        self._btn_autofix = PushButton(self._translator.translate("editor.ed_btn_autofix"))
        self._btn_autofix.setStyleSheet(f"color: {_c.warning()}; font-weight: bold;")
        self._btn_autofix.setToolTip(self._translator.translate("editor.ed_tt_autofix"))
        
        self._btn_merge_similar = PushButton(self._translator.translate("editor.ed_btn_merge_dup"))
        self._btn_merge_similar.setToolTip(self._translator.translate("editor.ed_tt_merge_dup"))

        for btn in (self._insert_button, self._delete_button, self._split_button, self._merge_button, self._btn_autofix, self._btn_merge_similar):
            btn.setEnabled(False)

        layout.addWidget(self._insert_button); layout.addWidget(self._delete_button)
        layout.addWidget(self._split_button); layout.addWidget(self._merge_button)
        layout.addWidget(self._btn_autofix); layout.addWidget(self._btn_merge_similar)

        self._btn_fast_reocr = PushButton(self._translator.translate("editor.ed_btn_reocr_fast"))
        self._btn_fast_reocr.setStyleSheet(f"color: {_c.success()}; font-weight: bold;")
        self._btn_fast_reocr.setToolTip(
            self._translator.translate("editor.ed_tt_reocr_fast")
        )
        layout.addWidget(self._btn_fast_reocr)
        self._btn_reocr = PushButton(self._translator.translate("editor.ed_btn_reocr_adv"))
        self._btn_reocr.setToolTip(self._translator.translate("editor.ed_tt_reocr_adv"))
        layout.addWidget(self._btn_reocr)

        # Dời giờ cả bộ: nhãn + ô số + nút đi liền nhau (không bị tách khi xuống dòng).
        shift_box = QWidget(holder)
        shift_lay = QHBoxLayout(shift_box)
        shift_lay.setContentsMargins(0, 0, 0, 0)
        shift_lay.addWidget(StrongBodyLabel(self._translator.translate("editor.shift_label")))
        self._shift_spin = DoubleSpinBox()
        self._shift_spin.setRange(-3600.0, 3600.0); self._shift_spin.setSingleStep(0.1)
        self._shift_spin.setDecimals(3); self._shift_spin.setSuffix(" s")
        shift_lay.addWidget(self._shift_spin)
        self._shift_button = PushButton(self._translator.translate("editor.btn_shift_apply"))
        self._shift_button.setEnabled(False)
        shift_lay.addWidget(self._shift_button)
        layout.addWidget(shift_box)
        return holder

    def _build_search_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(StrongBodyLabel(self._translator.translate("editor.ed_lbl_find")))
        self._find_edit = LineEdit()
        self._find_edit.setPlaceholderText(self._translator.translate("editor.ed_find_ph")); self._find_edit.setFixedWidth(200)

        self._btn_find_prev = PushButton("▲")
        self._btn_find_prev.setFixedWidth(32)
        self._btn_find_next = PushButton("▼")
        self._btn_find_next.setFixedWidth(32)
        self._lbl_find_count = CaptionLabel("")

        layout.addWidget(self._find_edit); layout.addWidget(self._btn_find_prev); layout.addWidget(self._btn_find_next)

        self._filter_combo = ComboBox()
        self._filter_combo.addItems([self._translator.translate("editor.ed_flt_all"), self._translator.translate("editor.ed_flt_ocr_err"), self._translator.translate("editor.ed_flt_warn"), self._translator.translate("editor.ed_flt_suspect")])
        # [v3.23.654] "Nghi ngờ OCR": theo hình thái kết quả (ít khung, hiện dần, biến thể liền kề, chữ Latin lẫn).
        self._filter_combo.setToolTip(self._translator.translate("editor.ed_flt_suspect_tip"))
        self._filter_combo.setFixedWidth(140)
        layout.addWidget(self._filter_combo); layout.addWidget(self._lbl_find_count)
        layout.addSpacing(10); layout.addWidget(StrongBodyLabel(self._translator.translate("editor.ed_lbl_replace")))

        self._replace_edit = LineEdit()
        layout.addWidget(self._replace_edit, stretch=1)
        self._replace_one_button = PushButton("Thay 1"); self._replace_all_button = PushButton(self._translator.translate("editor.ed_btn_replace_all"))
        self._replace_one_button.setEnabled(False); self._replace_all_button.setEnabled(False)
        layout.addWidget(self._replace_one_button); layout.addWidget(self._replace_all_button)

        return layout

    def _build_top_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._table = QTableView(parent=self)
        self._table_model = SubtitleTableModel(self)
        self._proxy_model = SubtitleFilterProxyModel(self)
        # [v3.23.657] Lý do "Nghi ngờ OCR" hiện bằng chữ (cột cảnh báo + tooltip).
        self._proxy_model.set_reason_labeler(
            lambda reason: self._translator.translate(f"editor.ed_suspect_{reason}")
        )
        self._proxy_model.setSourceModel(self._table_model)
        self._table.setModel(self._proxy_model)
        # [v3.23.682] Đếm lại dòng nghi sau mỗi lần sửa/nạp (gom các thay đổi liền nhau thành một lần tính).
        self._review_timer = QTimer(self)
        self._review_timer.setSingleShot(True)
        self._review_timer.setInterval(500)
        self._review_timer.timeout.connect(self._emit_review_count)
        for signal in (self._table_model.modelReset, self._table_model.rowsInserted, self._table_model.rowsRemoved):
            signal.connect(self._review_timer.start)
        self._table_model.dataChanged.connect(self._on_model_data_changed_for_review)

        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.setWordWrap(True)

        self._table.setSortingEnabled(False)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(COL_TEXT, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(COL_INDEX, 40); self._table.setColumnWidth(COL_START, 90)
        self._table.setColumnWidth(COL_END, 90); self._table.setColumnWidth(COL_DUR, 70)
        self._table.setColumnWidth(COL_SCORE, 50); self._table.setColumnWidth(COL_WARN, 40)
        self._table.setColumnWidth(COL_GAP, 60)

        from subtitles_extractor.presentation.widgets.rich_text_delegate import RichTextSubtitleDelegate
        self._text_delegate = RichTextSubtitleDelegate(text_column=COL_TEXT, parent=self)
        self._table.setItemDelegateForColumn(COL_TEXT, self._text_delegate)

        v_header = self._table.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        v_header.setDefaultSectionSize(65)

        layout.addWidget(self._table, stretch=1)

        edit_area = QFrame()
        edit_area.setObjectName("textEditArea")
        edit_area.setStyleSheet(f"QFrame#textEditArea {{ border-top: 1px solid {_c.border()}; margin-top: 4px; }}")
        edit_area.setMinimumHeight(140)

        edit_lay = QHBoxLayout(edit_area)
        edit_lay.setContentsMargins(0, 6, 0, 0)
        edit_lay.setSpacing(10)

        time_lay = QVBoxLayout()
        time_lay.setSpacing(4)
        row_s = QHBoxLayout(); row_s.addWidget(StrongBodyLabel(self._translator.translate("editor.ed_lbl_show"))); self._edit_start = TimeSpinBox(); row_s.addWidget(self._edit_start)
        row_e = QHBoxLayout(); row_e.addWidget(StrongBodyLabel(self._translator.translate("editor.ed_lbl_hide"))); self._edit_end = TimeSpinBox(); row_e.addWidget(self._edit_end)
        time_lay.addLayout(row_s); time_lay.addLayout(row_e)

        self._edit_apply_timing = PushButton(self._translator.translate("editor.ed_btn_apply_time"))
        self._edit_apply_timing.setEnabled(False)
        time_lay.addWidget(self._edit_apply_timing); time_lay.addStretch(1)
        edit_lay.addLayout(time_lay)

        text_lay = QVBoxLayout(); text_lay.setSpacing(2)
        info_row = QHBoxLayout()
        info_row.addWidget(StrongBodyLabel(self._translator.translate("editor.ed_lbl_content"))); info_row.addStretch(1)
        self._cps_gauge = CpsGauge(); self._cps_gauge.setFixedWidth(80)
        info_row.addWidget(self._cps_gauge)
        self._cps_label = CaptionLabel("CPS: 0.0")
        info_row.addWidget(self._cps_label); info_row.addSpacing(10)
        self._lbl_char_limit = CaptionLabel(self._translator.translate("editor.ed_charlen").replace("{cur}", "0").replace("{max}", str(_MAX_LINE_CHARS)))
        info_row.addWidget(self._lbl_char_limit)
        text_lay.addLayout(info_row)

        self._edit_text = TextEdit()
        self._edit_text.setMaximumHeight(85)
        self._tag_highlighter = SubtitleTagHighlighter(self._edit_text.document())
        text_lay.addWidget(self._edit_text)
        edit_lay.addLayout(text_lay, stretch=1)

        format_lay = QVBoxLayout(); format_lay.setSpacing(4)
        self._btn_bold = PushButton("B"); self._btn_bold.setFixedSize(32, 32)
        font_b = self._btn_bold.font(); font_b.setBold(True); self._btn_bold.setFont(font_b)
        self._btn_italic = PushButton("I"); self._btn_italic.setFixedSize(32, 32)
        font_i = self._btn_italic.font(); font_i.setItalic(True); self._btn_italic.setFont(font_i)
        self._btn_strip_tags = PushButton("T"); self._btn_strip_tags.setFixedSize(32, 32)
        self._btn_strip_tags.setToolTip(self._translator.translate("editor.ed_tt_clear_fmt"))

        for b in (self._btn_bold, self._btn_italic, self._btn_strip_tags):
            b.setEnabled(False); format_lay.addWidget(b)

        format_lay.addStretch(1); edit_lay.addLayout(format_lay)
        layout.addWidget(edit_area)

        self._table.viewport().installEventFilter(self)
        self._table.installEventFilter(self)
        return panel

    def _build_top_right_panel(self) -> QWidget:
        self._player_panel = QWidget()
        layout = QVBoxLayout(self._player_panel)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(4)

        self._video_stack = QStackedLayout()
        self._video_placeholder = QLabel(self._translator.translate("editor.ed_lbl_video_placeholder"))
        self._video_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_placeholder.setStyleSheet(f"background-color: {_c.mono_bg()}; color: {_c.on_surface_muted()}; font-size: {_m.FONT_SIZE_BODY}px; border-radius: 4px;")

        self._video_widget = create_video_widget(mpv_options=self._container.build_mpv_player_kwargs(), parent=self)
        self._video_widget.enable_roi_drawing(False)

        self._video_widget.position_changed.connect(self._on_canvas_position_changed)
        self._video_widget.state_changed.connect(self._on_canvas_state_changed)
        self._video_widget.seek_fallback_requested.connect(self._on_seek_fallback_requested)

        self._video_stack.addWidget(self._video_placeholder)
        self._video_stack.addWidget(self._video_widget)
        layout.addLayout(self._video_stack, stretch=1)

        self._controls_container = QWidget(); self._controls_container.setFixedHeight(45)
        controls = QHBoxLayout(self._controls_container)
        controls.setContentsMargins(0, 5, 0, 5); controls.setSpacing(8)

        self._btn_prev_frame = PushButton("◄|"); self._btn_prev_frame.setFixedWidth(40)
        self._btn_step_back = ToolButton(FluentIcon.CARE_LEFT_SOLID)
        self._btn_play = ToolButton(FluentIcon.PLAY)
        self._btn_step_forward = ToolButton(FluentIcon.CARE_RIGHT_SOLID)
        self._btn_next_frame = PushButton("|►"); self._btn_next_frame.setFixedWidth(40)
        set_accessible_name(self._btn_prev_frame, self._translator.translate("editor.ed_acc_prev_frame"))
        set_accessible_name(self._btn_step_back, self._translator.translate("editor.ed_acc_prev_line"))
        set_accessible_name(self._btn_play, self._translator.translate("editor.ed_acc_playpause"))
        set_accessible_name(self._btn_step_forward, self._translator.translate("editor.ed_acc_next_line"))
        set_accessible_name(self._btn_next_frame, self._translator.translate("editor.ed_acc_next_frame"))

        media_box = QHBoxLayout()
        media_box.addWidget(self._btn_prev_frame); media_box.addWidget(self._btn_step_back)
        media_box.addWidget(self._btn_play); media_box.addWidget(self._btn_step_forward)
        media_box.addWidget(self._btn_next_frame)

        left_box = QHBoxLayout()
        left_box.addWidget(CaptionLabel("🔊"))
        self._vol = Slider(Qt.Orientation.Horizontal)
        set_accessible_name(self._vol, self._translator.translate("editor.ed_acc_volume"), set_tooltip=False)
        self._vol.setRange(0, 100); self._vol.setValue(80); self._vol.setFixedWidth(80)

        self._speed_combo = ComboBox()
        self._speed_combo.addItems(["0.5x", "1.0x", "1.25x", "1.5x", "2.0x"]); self._speed_combo.setCurrentText("1.0x")
        self._speed_combo.setFixedWidth(80)

        left_box.addWidget(self._vol); left_box.addWidget(self._speed_combo)

        right_box = QHBoxLayout()
        self._lbl_time = CaptionLabel("00:00:00.000 / 00:00:00.000")
        self._lbl_time.setStyleSheet(mono_label_style())

        self._btn_detach = PushButton(self._translator.translate("editor.ed_btn_detach")); self._btn_detach.setFixedWidth(100)
        right_box.addWidget(self._lbl_time); right_box.addSpacing(10); right_box.addWidget(self._btn_detach)

        controls.addLayout(left_box); controls.addStretch()
        controls.addLayout(media_box); controls.addStretch()
        controls.addLayout(right_box)

        self._position_slider = Slider(Qt.Orientation.Horizontal)
        set_accessible_name(self._position_slider, self._translator.translate("editor.ed_acc_timeline"), set_tooltip=False)
        self._position_slider.setRange(0, _SLIDER_RESOLUTION)
        self._position_slider.valueChanged.connect(self._on_slider_value_changed)
        self._position_slider.sliderPressed.connect(self._on_slider_pressed)
        self._position_slider.sliderReleased.connect(self._on_slider_released)

        layout.addWidget(self._position_slider, stretch=0)
        layout.addWidget(self._controls_container, stretch=0)
        return self._player_panel

    def _build_bottom_waveform_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(4)

        wf_container = QHBoxLayout(); wf_container.setSpacing(4)
        self._waveform_widget = AudioWaveformWidget(parent=self)
        show_waveform = self._container.settings_service.current.ui.show_waveform
        self._waveform_widget.setVisible(show_waveform)

        if hasattr(self._waveform_widget, 'ffmpeg_error'):
            self._waveform_widget.ffmpeg_error.connect(self._on_ffmpeg_error)

        self._waveform_y_zoom = Slider(Qt.Orientation.Vertical, self)
        set_accessible_name(self._waveform_y_zoom, self._translator.translate("editor.ed_acc_wave_zoom"), set_tooltip=False)
        self._waveform_y_zoom.setRange(10, 500); self._waveform_y_zoom.setValue(100); self._waveform_y_zoom.setFixedWidth(20)
        self._waveform_y_zoom.setSingleStep(10); self._waveform_y_zoom.setPageStep(50)
        self._waveform_y_zoom.setVisible(show_waveform)

        wf_container.addWidget(self._waveform_widget, stretch=1); wf_container.addWidget(self._waveform_y_zoom)
        layout.addLayout(wf_container, stretch=1)

        wf_ctrl = QHBoxLayout(); wf_ctrl.setSpacing(10)
        self._btn_jump_to_event = PushButton(self._translator.translate("editor.ed_btn_goto_line"))
        self._btn_auto_scroll = PushButton(self._translator.translate("editor.ed_btn_follow_scroll")); self._btn_auto_scroll.setCheckable(True); self._btn_auto_scroll.setChecked(True)
        self._btn_center_wf = PushButton(self._translator.translate("editor.ed_btn_center_wave")); self._btn_center_wf.setCheckable(True)
        self._btn_loop = PushButton(self._translator.translate("editor.ed_btn_loop")); self._btn_loop.setCheckable(True)

        wf_ctrl.addWidget(self._btn_jump_to_event); wf_ctrl.addWidget(self._btn_auto_scroll)
        wf_ctrl.addWidget(self._btn_center_wf); wf_ctrl.addWidget(self._btn_loop); wf_ctrl.addStretch(1)
        layout.addLayout(wf_ctrl)
        return panel

    def _build_statusbar(self) -> QHBoxLayout:
        bottom = QHBoxLayout()
        self._btn_settings = PushButton(FluentIcon.SETTING, self._translator.translate("editor.ed_btn_video_settings"), self)
        bottom.addWidget(self._btn_settings)

        bottom.addSpacing(20)
        self._progress_bar = ProgressBar()
        self._progress_bar.setRange(0, 100); self._progress_bar.setVisible(False); self._progress_bar.setFixedWidth(200)
        bottom.addWidget(self._progress_bar); bottom.addSpacing(10)

        self._status_label = CaptionLabel("")
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bottom.addWidget(self._status_label, stretch=1)
        return bottom

    def _connect_signals(self) -> None:
        self._open_video_button.clicked.connect(self._on_open_video_clicked)
        self._open_button.clicked.connect(self._on_open_clicked)
        self._export_button.clicked.connect(self._on_export_clicked)
        self._undo_button.clicked.connect(self._view_model.undo)
        self._redo_button.clicked.connect(self._view_model.redo)
        self._history_button.clicked.connect(self._show_undo_history)
        self._btn_shortcuts.clicked.connect(self._on_show_shortcuts)
        self._btn_settings.clicked.connect(self._on_show_settings)

        self._insert_button.clicked.connect(self._on_insert_clicked)
        self._delete_button.clicked.connect(self._on_delete_clicked)
        self._split_button.clicked.connect(self._on_split_clicked)
        self._merge_button.clicked.connect(self._on_merge_clicked)
        self._btn_autofix.clicked.connect(self._on_auto_fix_timeline)
        self._btn_merge_similar.clicked.connect(self._on_merge_similar_dialog)
        self._shift_button.clicked.connect(self._on_shift_clicked)

        self._find_edit.textChanged.connect(self._filter_debounce_timer.start)
        self._filter_combo.currentIndexChanged.connect(self._apply_filters)
        self._find_edit.returnPressed.connect(self._find_next)
        self._btn_find_next.clicked.connect(self._find_next)
        self._btn_find_prev.clicked.connect(self._find_prev)
        self._replace_one_button.clicked.connect(self._replace_one)
        self._replace_all_button.clicked.connect(self._replace_all)

        self._btn_play.clicked.connect(self._video_widget.toggle_play_pause)
        self._btn_step_back.clicked.connect(lambda: self._seek_delta(-5.0))
        self._btn_step_forward.clicked.connect(lambda: self._seek_delta(5.0))
        self._btn_prev_frame.clicked.connect(lambda: self._video_widget.send_mpv_command("frame-back-step"))
        self._btn_next_frame.clicked.connect(lambda: self._video_widget.send_mpv_command("frame-step"))
        self._vol.valueChanged.connect(self._on_volume_changed)
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)

        self._btn_jump_to_event.clicked.connect(self._jump_to_current_event)
        self._btn_loop.clicked.connect(self._toggle_loop)
        self._btn_detach.clicked.connect(self._toggle_detach)
        
        self._btn_auto_scroll.toggled.connect(self._on_auto_scroll_toggled)

        self._btn_strip_tags.clicked.connect(self._on_strip_tags_clicked)
        self._btn_bold.clicked.connect(lambda: self._insert_tag("<b>", "</b>"))
        self._btn_italic.clicked.connect(lambda: self._insert_tag("<i>", "</i>"))

        self._edit_text.textChanged.connect(self._on_text_editing)
        self._edit_text.focusOutEvent = self._on_text_focus_out

        # [v3.23.650] Sửa xong là model bị reset (set_events); reset ngay trong setData/commitData làm index proxy mà
        # view/delegate đang cầm thành con trỏ treo (SIGSEGV). Slot trực tiếp CHỈ ghi nhận uid dòng rồi hoãn việc áp dụng
        # sang vòng lặp sự kiện kế tiếp; khi áp dụng, tra lại chỉ số theo uid (dòng đã dời/xoá giữa chừng → bỏ qua).
        self._table_model.text_edit_requested.connect(self._on_table_text_edit)
        self._table_model.time_edit_requested.connect(self._on_table_time_edit)
        self._table_model.timing_error_occurred.connect(self._show_error)

        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._table.customContextMenuRequested.connect(self._show_context_menu)

        self._edit_apply_timing.clicked.connect(self._on_apply_timing_clicked)

        self._waveform_widget.seek_requested.connect(self._on_waveform_seek)
        self._waveform_widget.subtitle_drag_done.connect(self._on_waveform_drag)
        self._waveform_widget.subtitle_create_requested.connect(self._on_waveform_create_sub)
        
        self._btn_center_wf.toggled.connect(self._on_center_waveform_toggled)
        self._waveform_y_zoom.valueChanged.connect(lambda v: self._waveform_widget.set_y_zoom(v / 100.0))
        
        if hasattr(self._waveform_widget, 'play_toggle_requested'):
            self._waveform_widget.play_toggle_requested.connect(self._video_widget.toggle_play_pause)

        self._video_widget.video_clicked.connect(self._on_video_clicked)
        self._video_widget.video_double_clicked.connect(self._on_video_double_clicked)

        self._btn_reocr.clicked.connect(self._on_reocr_clicked)
        self._btn_fast_reocr.clicked.connect(self._on_fast_reocr_clicked)

        self._view_model.state_changed.connect(self._refresh_table)
        self._view_model.error_occurred.connect(self._show_error)
        self._view_model.export_finished.connect(self._on_export_finished)
        self._view_model.progress_changed.connect(self._on_reocr_progress)
        self._view_model.busy_changed.connect(self._on_reocr_busy)
        # [v3.6 bugfix R5]: Xóa waveform reocr region khi Re-OCR hoàn tất.
        self._view_model.reocr_region_should_clear.connect(
            self._waveform_widget.clear_reocr_region
        )

        self._edit_text.installEventFilter(self)
        self.installEventFilter(self)

    def _on_auto_scroll_toggled(self, checked: bool) -> None:
        if checked and self._last_active_row >= 0 and self._last_active_row < self._table_model.rowCount():
            source_index = self._table_model.index(self._last_active_row, 0)
            proxy_index = self._proxy_model.mapFromSource(source_index)
            if proxy_index.isValid():
                self._table.scrollTo(proxy_index, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _on_center_waveform_toggled(self, checked: bool) -> None:
        self._waveform_widget.set_center_playhead(checked)
        if checked:
            self._btn_center_wf.setStyleSheet(f"color: {_c.success()}; font-weight: bold;")
        else:
            self._btn_center_wf.setStyleSheet("")

    def _break_loop_if_outside(self, ts: float) -> None:
        if self._loop_active and self._loop_start_time is not None and self._loop_end_time is not None:
            if ts < self._loop_start_time - 0.5 or ts > self._loop_end_time + 0.5:
                self._btn_loop.setChecked(False)
                self._toggle_loop()

    def _on_canvas_state_changed(self, is_playing: bool) -> None:
        if hasattr(self, '_btn_play'):
            self._btn_play.setIcon(
                (FluentIcon.PAUSE if is_playing else FluentIcon.PLAY).icon()
            )

    def _on_canvas_position_changed(self, pos: float) -> None:
        if not hasattr(self, '_video_widget') or self._video_widget is None:
            return
        dur = self._video_widget.duration_sec
        if dur > 0:
            val = int((pos / dur) * _SLIDER_RESOLUTION)
            self._updating_slider_from_playback = True
            self._position_slider.setValue(val)
            self._updating_slider_from_playback = False
            self._lbl_time.setText(f"{_fmt_mmss(pos)} / {_fmt_mmss(dur)}")

        self._waveform_widget.set_current_time(pos)

        if self._loop_active and self._loop_start_time is not None and self._loop_end_time is not None:
            if pos >= self._loop_end_time:
                if getattr(self, '_loop_single_play', False):
                    self._loop_active = False
                    self._loop_single_play = False
                    self._video_widget.pause()
                else:
                    self._video_widget.seek(self._loop_start_time)

        now = time.monotonic()
        if (now - self._last_sync_highlight_time) * 1000 < _SYNC_HIGHLIGHT_MS:
            return
        self._last_sync_highlight_time = now

        if hasattr(self, '_view_model') and hasattr(self._view_model, '_service'):
            new_source_row = self._view_model._service.find_event_index_at_time(pos)

            # [Active Row OOB] Sau khi xoá câu đang phát, chỉ số có thể vượt quá số
            # dòng hiện tại → IndexError khi highlight. Bỏ qua nếu ngoài phạm vi.
            if new_source_row != -1 and new_source_row >= self._table_model.rowCount():
                new_source_row = -1

            if new_source_row != -1:
                row_changed = False
                if new_source_row != getattr(self, '_last_active_row', -1):
                    self._last_active_row = new_source_row
                    row_changed = True
                    
                    self._table_model.set_active_row(new_source_row)
                    if hasattr(self, '_waveform_widget'):
                        self._waveform_widget.set_active_row(new_source_row)

                # [v3.20.3 #4] Tôn trọng thao tác người dùng: ngừng auto-scroll khi
                # đang gõ text/chỉnh thời gian HOẶC chuột đang lơ lửng trên bảng
                # (người dùng có thể đang ngắm/chuẩn bị click) — chống "cướp chuột".
                is_user_editing = (
                    self._edit_text.hasFocus()
                    or self._edit_start.hasFocus()
                    or self._edit_end.hasFocus()
                )
                is_hovering_table = (
                    self._table.underMouse()
                    or self._table.viewport().underMouse()
                )

                if (
                    hasattr(self, '_btn_auto_scroll')
                    and self._btn_auto_scroll.isChecked()
                    and not is_user_editing
                    and not is_hovering_table
                ):
                    time_since_user_scrolled = time.time() - self._last_user_scroll_time
                    need_snap_back = getattr(self, '_need_snap_back', False)
                    
                    if time_since_user_scrolled > _AUTO_SCROLL_PAUSE_AFTER_USER_SEC:
                        if row_changed or need_snap_back:
                            self._need_snap_back = False
                            source_index = self._table_model.index(new_source_row, 0)
                            proxy_index = self._proxy_model.mapFromSource(source_index)
                            if proxy_index.isValid():
                                visual_rect = self._table.visualRect(proxy_index)
                                viewport_rect = self._table.viewport().rect()

                                safe_margin = viewport_rect.height() * 0.2
                                if visual_rect.top() < safe_margin or visual_rect.bottom() > viewport_rect.height() - safe_margin:
                                    self._table.scrollTo(proxy_index, QAbstractItemView.ScrollHint.PositionAtCenter)
                    else:
                        self._need_snap_back = True

    def _on_slider_pressed(self) -> None:
        if self._video_widget.is_playing:
            self._was_playing_before_seek = True
            self._video_widget.pause()
        else:
            self._was_playing_before_seek = False

    def _on_slider_released(self) -> None:
        if getattr(self, '_is_scrubbing_slider', False):
            self._is_scrubbing_slider = False
        if self._was_playing_before_seek:
            self._video_widget.play()

    def _on_slider_value_changed(self, value: int) -> None:
        if self._updating_slider_from_playback: return
        dur = self._video_widget.duration_sec
        if dur > 0:
            ts = (value / float(_SLIDER_RESOLUTION)) * dur
            self._break_loop_if_outside(ts)
            
            self._is_scrubbing_slider = True
            self._video_widget.seek(ts)

    def _seek_delta(self, delta: float) -> None:
        new_ts = max(0.0, self._video_widget.position_sec + delta)
        self._break_loop_if_outside(new_ts)
        self._video_widget.seek(new_ts)

    def _on_seek_fallback_requested(self, ts: float) -> None:
        # EditorPage không có PreviewSeekService (CPU frame decode) fallback.
        # Khi MPV chưa load, seek không thực hiện được — hủy pending row-seek
        # để tránh timer fire sau đó nhảy đến sự kiện sai.
        # [v3.6 bugfix SF-1]: Bỏ self._debounced_seek_timer.start() vô ích.
        # Timer start trước đây chỉ làm timer fire rồi không làm gì vì
        # _pending_seek_row = -1, gây CPU waste không cần thiết.
        self._pending_seek_row = -1

    def _execute_debounced_seek(self) -> None:
        if getattr(self, '_pending_seek_row', -1) >= 0:
            self._jump_to_event_at(self._pending_seek_row)

    def _on_ffmpeg_error(self, message: str) -> None:
        InfoBar.error(
            title=self._translator.translate("editor.ib_wave_err_t"),
            content=self._translator.translate("editor.ib_wave_err_c").replace("{msg}", str(message)),
            parent=self, position=InfoBarPosition.TOP_RIGHT, duration=5000
        )

    def _current_row(self) -> int:
        """Hàng NGUỒN đang là dòng hiện tại (-1 nếu chưa chọn).

        [v3.23.650] Không dùng ``selectedRows()``: nó duyệt từng ô qua ``flags()`` của proxy (O(dòng×cột), 150–400 ms
        với 3649 dòng đang chọn hết) và thứ tự trả về không xác định khi chọn nhiều dòng → khung sửa và thao tác
        sửa/tách/gộp có thể rơi vào hai dòng khác nhau. Dùng ``currentIndex`` (khớp khung sửa), nếu nó không nằm
        trong vùng chọn thì lấy dòng đầu của vùng chọn đầu tiên.
        """
        selection_model = self._table.selectionModel()
        selection = selection_model.selection()
        if selection.isEmpty():
            return -1
        current = selection_model.currentIndex()
        if current.isValid() and selection_model.isRowSelected(current.row()):
            proxy_index = current
        else:
            proxy_index = self._proxy_model.index(selection[0].top(), 0)
        if not proxy_index.isValid():
            return -1
        return self._proxy_model.mapToSource(proxy_index).row()

    def _jump_to_current_event(self) -> None:
        idx = self._current_row()
        if idx >= 0: self._jump_to_event_at(idx)

    def _jump_to_event_at(self, row_idx: int) -> None:
        if not (0 <= row_idx < len(self._table_model.events)):
            return
        start_sec = self._table_model.events[row_idx].start_sec

        if hasattr(self, '_waveform_widget'):
            self._waveform_widget.set_current_time(start_sec)
            self._waveform_widget.set_active_row(row_idx)

        mpv_player = self._video_widget.peek_player()
        if mpv_player:
            mpv_player.seek(start_sec)
        else:
            self._video_widget.seek(start_sec)

    def _toggle_loop(self) -> None:
        self._loop_active = self._btn_loop.isChecked()
        if self._loop_active:
            idx = self._current_row()
            if idx >= 0:
                ev = self._table_model.events[idx]
                self._loop_start_time = ev.start_sec
                self._loop_end_time = ev.end_sec
                self._waveform_widget.set_loop_region(ev.start_sec, ev.end_sec)
                self._jump_to_event_at(idx)

                if not self._video_widget.is_playing:
                    self._video_widget.play()
            else:
                # [v3.6 bugfix TL-1]: Khi không có row được chọn, phải dọn sạch
                # loop region trên waveform và reset các giá trị thời gian.
                # Trước đây chỉ reset flag + button mà không clear waveform →
                # vùng loop cũ vẫn hiển thị dù loop đã tắt.
                self._loop_active = False
                self._btn_loop.setChecked(False)
                self._waveform_widget.clear_loop_region()
                self._loop_start_time = None
                self._loop_end_time = None
        else:
            self._waveform_widget.clear_loop_region()
            self._loop_start_time = None
            self._loop_end_time = None

    def _on_show_settings(self) -> None:
        dlg = EditorSettingsDialog(self._container.settings_service._repo, self._translator, self)
        # [v3.6 bugfix DLG-2]: deleteLater để tránh tích lũy dialog con của EditorPage.
        try:
            if dlg.exec() == QDialog.DialogCode.Accepted:
                dlg.save_to_repo()
                edge_px = int(self._container.settings_service._repo.load("editor_waveform_edge_px", 10))
                self._waveform_widget.set_edge_px(edge_px)
                self._schedule_preview()
        finally:
            dlg.deleteLater()

    def _insert_tag(self, open_tag: str, close_tag: str) -> None:
        cursor = self._edit_text.textCursor()
        if cursor.hasSelection():
            cursor.insertText(f"{open_tag}{cursor.selectedText()}{close_tag}")
        else:
            pos = cursor.position()
            cursor.insertText(f"{open_tag}{close_tag}")
            # [v3.23.353] Kẹp trong phạm vi document: về lý thuyết pos+len(open_tag) luôn
            # hợp lệ sau insertText, nhưng nếu có tín hiệu textChanged đổi nội dung đồng bộ
            # thì vị trí có thể vượt biên → Qt cảnh báo "QTextCursor::setPosition out of
            # range". characterCount()-1 là vị trí con trỏ tối đa cho document hiện tại.
            max_pos = self._edit_text.document().characterCount() - 1
            cursor.setPosition(min(pos + len(open_tag), max_pos))
            self._edit_text.setTextCursor(cursor)
        self._edit_text.setFocus()
        self._on_text_focus_out(None)

    def _commit_pending_text(self) -> None:
        """[v3.23.659] Ghi chữ đang gõ trong ô sửa vào ĐÚNG dòng đang sửa (tra theo uid), nếu có thay đổi."""
        uid: str | None = getattr(self, "_pending_edit_uid", None)
        if not uid:
            return
        self._pending_edit_uid: str | None = None
        text = self._edit_text.toPlainText()
        for index, event in enumerate(self._table_model.events):
            if event.uid == uid:
                if event.text != text:
                    self._view_model.update_text(index, text)
                return

    def _on_text_editing(self) -> None:
        idx = self._current_row()
        if 0 <= idx < len(self._table_model.events):
            self._pending_edit_uid = self._table_model.events[idx].uid  # [v3.23.659] dòng có chữ đang gõ dở
        if self._video_widget.is_playing:
            self._video_widget.pause()

        text = self._edit_text.toPlainText()

        # [v3.6 bugfix]: KHÔNG mutate SubtitleEvent.text in-place ở đây.
        # Lỗi cũ: `self._view_model.current_events[idx].text = text`
        # Hậu quả:
        #   1. `_on_text_focus_out` so sánh thấy text đã bằng nhau → KHÔNG gọi
        #      `update_text()` → không push undo, không set _is_dirty.
        #   2. Undo history bị corrupt vì snapshot trong undo stack giữ reference
        #      đến cùng object → sau khi mutate, snapshot cũng có text mới.
        #   3. Tất cả text edits bị "vô hình" với undo system.
        # Fix: chỉ schedule preview và cập nhật char counter.
        # Preview đã đọc `_edit_text.toPlainText()` trực tiếp khi `is_editing`,
        # không cần mutation event.
        idx = self._current_row()
        if idx >= 0:
            self._schedule_preview()

        text_clean = self._html_tag_regex.sub("", text).replace("\\N", "")
        lines = text_clean.split("\n")
        max_len = max((len(line) for line in lines), default=0)
        self._lbl_char_limit.setText(self._translator.translate("editor.ed_charlen").replace("{cur}", str(max_len)).replace("{max}", str(_MAX_LINE_CHARS)))
        self._lbl_char_limit.setStyleSheet(f"color: {_c.danger()}; font-weight: bold;" if max_len > _MAX_LINE_CHARS else "")

    def _on_text_focus_out(self, ev: QEvent | None) -> None:
        if ev:
            TextEdit.focusOutEvent(self._edit_text, ev)
        if getattr(self, "_pending_edit_uid", None):
            self._commit_pending_text()  # [v3.23.659] ghi vào đúng dòng đã gõ, kể cả khi lựa chọn vừa đổi
            return
        idx = self._current_row()
        if idx < 0 or idx >= len(self._table_model.events):
            return
        # [v3.6 bugfix FO-1]: Dùng _table_model.events[idx] thay vì
        # current_events[idx] — tránh tạo list copy O(n) chỉ để lấy 1 event.
        # Sau bugfix mutation, _table_model luôn sync với service qua state_changed.
        stored_text = self._table_model.events[idx].text
        new_text = self._edit_text.toPlainText()
        if stored_text != new_text:
            self._view_model.update_text(idx, new_text)

    def _on_table_text_edit(self, row: int, new_text: str) -> None:
        events = self._table_model.events
        if not (0 <= row < len(events)):
            return
        uid = events[row].uid
        QTimer.singleShot(0, lambda: self._apply_table_edit(uid, lambda i: self._view_model.update_text(i, new_text)))

    def _on_table_time_edit(self, row: int, start_sec: float, end_sec: float) -> None:
        events = self._table_model.events
        if not (0 <= row < len(events)):
            return
        uid = events[row].uid
        QTimer.singleShot(
            0, lambda: self._apply_table_edit(uid, lambda i: self._view_model.update_timing(i, start_sec, end_sec)))

    def _apply_table_edit(self, uid: str, apply) -> None:
        for index, event in enumerate(self._table_model.events):
            if event.uid == uid:
                apply(index)
                return

    def _event_uids(self) -> list[str]:
        return [event.uid for event in self._table_model.events]

    def _step_selected_row(self, delta: int) -> None:
        """Ctrl+↑/↓: chọn dòng liền trước/sau (theo thứ tự đang hiển thị) và tua video tới đó.

        [v3.23.650] Hàng của proxy phải đổi sang hàng NGUỒN trước khi tua: khi đang lọc, hai số này khác nhau nên
        video từng tua tới sai câu.
        """
        selection_model = self._table.selectionModel()
        if selection_model.selection().isEmpty():
            return
        current = selection_model.currentIndex()
        if not (current.isValid() and selection_model.isRowSelected(current.row())):
            current = self._proxy_model.index(selection_model.selection()[0].top(), 0)
        last = self._proxy_model.rowCount() - 1
        if not current.isValid() or last < 0:
            return
        proxy_row = max(0, min(last, current.row() + delta))
        self._table.selectRow(proxy_row)
        if not self._video_widget.is_playing:
            source_row = self._proxy_model.mapToSource(self._proxy_model.index(proxy_row, 0)).row()
            self._jump_to_event_at(source_row)

    def _show_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        idx = self._current_row()
        if idx >= 0:
            menu.addAction(self._translator.translate("editor.ed_ctx_insert_above"), lambda: self._insert_event_at(idx - 1))
            menu.addAction(self._translator.translate("editor.ed_ctx_insert_below"), lambda: self._insert_event_at(idx))
            menu.addAction(self._translator.translate("editor.ed_ctx_delete"), self._on_delete_clicked)
            menu.addSeparator()
            if idx < self._table_model.rowCount() - 1:
                menu.addAction(self._translator.translate("editor.ed_ctx_merge"), self._on_merge_clicked)
            menu.addAction(self._translator.translate("editor.ed_ctx_split"), self._on_split_clicked)
            menu.exec(self._table.viewport().mapToGlobal(pos))
        # [v3.23.650] Menu có parent=self nên không tự biến mất: mỗi lần mở là thêm một đối tượng con (rò rỉ).
        menu.deleteLater()

    def _insert_event_at(self, target_idx: int) -> None:
        self._view_model.insert_after(target_idx)
        
        def _focus_new_row():
            new_row_idx = target_idx + 1
            if new_row_idx < self._proxy_model.rowCount():
                source_index = self._table_model.index(new_row_idx, 0)
                proxy_index = self._proxy_model.mapFromSource(source_index)
                if proxy_index.isValid():
                    self._table.selectRow(proxy_index.row())
                    self._table.scrollTo(proxy_index, QAbstractItemView.ScrollHint.PositionAtCenter)
                    self._edit_text.setFocus()
                    self._edit_text.selectAll()
        
        QTimer.singleShot(50, _focus_new_row)

    def _on_show_shortcuts(self) -> None:
        """Mở hộp thoại liệt kê đầy đủ phím tắt (giúp người dùng khám phá tính năng)."""
        from PySide6.QtWidgets import QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle(self._translator.translate("editor.ed_win_shortcuts"))
        dlg.resize(440, 560)
        lay = QVBoxLayout(dlg)
        viewer = QTextBrowser(dlg)
        viewer.setHtml(editor_shortcuts_html(self._translator))
        viewer.setOpenExternalLinks(False)
        lay.addWidget(viewer)
        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bbox.rejected.connect(dlg.reject)
        bbox.accepted.connect(dlg.accept)
        lay.addWidget(bbox)
        dlg.exec()
        dlg.deleteLater()  # [v3.23.650] parent=self → không tự giải phóng

    def _show_undo_history(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._translator.translate("editor.ed_win_history"))
        dlg.resize(300, 400)
        lay = QVBoxLayout(dlg)
        lw = QListWidget(dlg)

        service = self._view_model._service
        undo_list = list(service._undo_stack)
        redo_list = list(service._redo_stack)

        for i, snap in enumerate(undo_list):
            lw.addItem(f"[{i+1}] {snap.description}")

        current_item = self._translator.translate("editor.ed_current_hist").replace("{n}", str(len(self._table_model.events)))
        lw.addItem(current_item)

        for snap in reversed(redo_list):
            lw.addItem(f"~[Redo] {snap.description}")

        lw.setCurrentRow(len(undo_list))
        lay.addWidget(lw)
        # [v3.6 bugfix DLG-3]: deleteLater tránh tích lũy dialog con.
        try:
            dlg.exec()
        finally:
            dlg.deleteLater()

    def _on_waveform_drag(self, idx: int, old_s: float, old_e: float, new_s: float, new_e: float) -> None:
        # [v3.6 bugfix WD-1]: Dùng _table_model.events thay vì current_events
        # cho bounds check — tránh tạo list copy per drag event (hot path).
        if idx < 0 or idx >= len(self._table_model.events):
            return
        if abs(old_s - new_s) > 0.001 or abs(old_e - new_e) > 0.001:
            self._last_user_scroll_time = time.time()
            self._view_model.update_timing(idx, new_s, new_e)

    def _on_waveform_create_sub(self, start_t: float, end_t: float) -> None:
        # [v3.6 bugfix WCS-1]: Dùng _table_model.events thay vì current_events
        # để tránh tạo 2 list copy liên tiếp (current_events + list comprehension bisect).
        model_events = self._table_model.events
        import bisect
        insert_idx = bisect.bisect_right([e.start_sec for e in model_events], start_t)
        self._view_model.insert_after(insert_idx - 1)

        # Sau insert_after → state_changed emitted → _table_model đã được cập nhật.
        if insert_idx < len(self._table_model.events):
            self._view_model.update_timing(insert_idx, start_t, end_t)

            source_index = self._table_model.index(insert_idx, 0)
            proxy_index = self._proxy_model.mapFromSource(source_index)
            if proxy_index.isValid():
                self._table.selectRow(proxy_index.row())

            self._edit_text.setFocus()

    def _on_video_clicked(self, button: Qt.MouseButton) -> None:
        if button == Qt.MouseButton.LeftButton:
            self._video_widget.toggle_play_pause()

    def _on_video_double_clicked(self, button: Qt.MouseButton) -> None:
        if button == Qt.MouseButton.LeftButton:
            if self._detached_window and is_valid(self._detached_window) and self._detached_window.isVisible():
                if self._detached_window.isFullScreen(): self._detached_window.showNormal()
                else: self._detached_window.showFullScreen()
            else:
                self._toggle_detach()
                if self._detached_window and is_valid(self._detached_window):
                    self._detached_window.showFullScreen()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._table.viewport() and event.type() in (QEvent.Type.Wheel, QEvent.Type.TouchBegin):
            self._last_user_scroll_time = time.time()

        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            k = event.key()
            mods = event.modifiers()

            if obj is self._table and k == Qt.Key.Key_Space:
                self._video_widget.toggle_play_pause()
                return True

            if obj is self._edit_text:
                if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    if mods & Qt.KeyboardModifier.ShiftModifier:
                        return False
                    self._edit_text.clearFocus()

                    indexes = self._table.selectionModel().selectedRows()
                    if indexes:
                        r = indexes[0].row()
                        if r >= 0 and r + 1 < self._proxy_model.rowCount():
                            self._table.selectRow(r + 1)
                            self._edit_text.setFocus()
                    return True
                if k == Qt.Key.Key_Escape:
                    self._edit_text.clearFocus()
                    self._table.setFocus()
                    return True

            if self._handle_global_keys(k, mods):
                return True

        return super().eventFilter(obj, event)

    def _handle_global_keys(self, k: int, mods: Qt.KeyboardModifier) -> bool:
        if getattr(self, '_table', None) and self._table.state() == QAbstractItemView.State.EditingState: return False

        focus_widget = QApplication.focusWidget()
        if focus_widget:
            cls_name = focus_widget.metaObject().className()
            input_classes = ["LineEdit", "TextEdit", "QTextEdit", "QLineEdit", "SpinBox", "TimeSpinBox", "QSpinBox", "QDoubleSpinBox", "ComboBox"]
            if any(name in cls_name for name in input_classes):
                ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
                if ctrl:
                    if k == Qt.Key.Key_B: self._btn_bold.click(); return True
                    if k == Qt.Key.Key_I: self._btn_italic.click(); return True
                    if k == Qt.Key.Key_S: self._on_export_clicked(); return True
                    if k == Qt.Key.Key_Z and not (mods & Qt.KeyboardModifier.ShiftModifier):
                        if "TextEdit" in cls_name or "LineEdit" in cls_name: return False
                        self._view_model.undo(); return True
                    if k == Qt.Key.Key_Y:
                        if "TextEdit" in cls_name or "LineEdit" in cls_name: return False
                        self._view_model.redo(); return True
                    if k == Qt.Key.Key_F:
                        if focus_widget is self._find_edit: self._find_edit.selectAll(); return True
                        self._find_edit.setFocus(); self._find_edit.selectAll(); return True
                    if k == Qt.Key.Key_Space:
                        self._video_widget.toggle_play_pause()
                        return True

                if k == Qt.Key.Key_Escape:
                    focus_widget.clearFocus()
                    self._table.setFocus()
                    return True
                return False

        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if k == Qt.Key.Key_B and ctrl: self._btn_bold.click(); return True
        if k == Qt.Key.Key_I and ctrl: self._btn_italic.click(); return True
        if k in (Qt.Key.Key_Space, Qt.Key.Key_K) and not ctrl and not alt: self._video_widget.toggle_play_pause(); return True
        if k == Qt.Key.Key_Space and alt and not ctrl: self._play_current_line(); return True
        if k == Qt.Key.Key_J and not ctrl and not alt: self._seek_delta(-2.0); return True
        if k == Qt.Key.Key_L and not ctrl and not alt: self._seek_delta(2.0); return True
        if k == Qt.Key.Key_Left and not (ctrl or alt or shift): self._seek_delta(-5.0); return True
        if k == Qt.Key.Key_Right and not (ctrl or alt or shift): self._seek_delta(5.0); return True
        if k == Qt.Key.Key_BracketLeft and not ctrl: self._change_speed(-0.25); return True
        if k == Qt.Key.Key_BracketRight and not ctrl: self._change_speed(0.25); return True
        if k == Qt.Key.Key_Left and alt and not shift: self._nudge_time(is_start=True, delta=-0.1); return True
        if k == Qt.Key.Key_Right and alt and not shift: self._nudge_time(is_start=True, delta=0.1); return True
        if k == Qt.Key.Key_Left and alt and shift: self._nudge_time(is_start=False, delta=-0.1); return True
        if k == Qt.Key.Key_Right and alt and shift: self._nudge_time(is_start=False, delta=0.1); return True
        
        if k == Qt.Key.Key_C and not (ctrl or alt or shift): 
            self._btn_center_wf.setChecked(not self._btn_center_wf.isChecked())
            return True
            
        if k == Qt.Key.Key_M and not (ctrl or alt or shift):
            player = self._video_widget.peek_player()
            if player:
                is_muted = getattr(self, '_is_muted', False)
                self._is_muted = not is_muted
                player.set_mute(self._is_muted)
                InfoBar.info(self._translator.translate("editor.ib_audio"), self._translator.translate("editor.ib_muted") if self._is_muted else self._translator.translate("editor.ib_unmuted"), parent=self, duration=1500)
            return True
            
        if k == Qt.Key.Key_Left and ctrl and shift:
            r = self._current_row()
            if r >= 0 and r < len(self._table_model.events):
                # [v3.6 bugfix PP-1]: Dùng _table_model.events tránh tạo list copy
                ts = self._table_model.events[r].start_sec
                self._break_loop_if_outside(ts)
                self._video_widget.seek(ts)
            return True
        if k == Qt.Key.Key_Right and ctrl and shift:
            r = self._current_row()
            if r >= 0 and r < len(self._table_model.events):
                ts = self._table_model.events[r].end_sec - 0.1
                self._break_loop_if_outside(ts)
                self._video_widget.seek(ts)
            return True
        if k == Qt.Key.Key_Up and ctrl:
            self._step_selected_row(-1)
            return True
        if k == Qt.Key.Key_Down and ctrl:
            self._step_selected_row(+1)
            return True

        if k == Qt.Key.Key_S and ctrl: self._on_export_clicked(); return True
        if k == Qt.Key.Key_Z and ctrl: self._view_model.undo(); return True
        if k == Qt.Key.Key_Y and ctrl: self._view_model.redo(); return True
        if k == Qt.Key.Key_Delete: self._on_delete_clicked(); return True
        if k == Qt.Key.Key_M and ctrl: self._on_merge_clicked(); return True
        if k == Qt.Key.Key_T and ctrl: self._on_split_clicked(); return True
        if k == Qt.Key.Key_Insert and alt: self._insert_event_at(self._current_row()); return True
        if k == Qt.Key.Key_F and ctrl: self._find_edit.setFocus(); self._find_edit.selectAll(); return True
        if k == Qt.Key.Key_F3: self._find(forward=not shift); return True
        # [v3.23.682] Duyệt nhanh dòng nghi ngờ OCR, đặt mốc theo vị trí phát, bảng phím tắt.
        if k == Qt.Key.Key_F8 and not (ctrl or alt): self._jump_to_suspect(forward=not shift); return True
        if k == Qt.Key.Key_BracketLeft and ctrl and not alt: self._set_time_at_playhead(is_start=True); return True
        if k == Qt.Key.Key_BracketRight and ctrl and not alt: self._set_time_at_playhead(is_start=False); return True
        if k == Qt.Key.Key_F1 and not (ctrl or alt or shift): self._on_show_shortcuts(); return True
        if k == Qt.Key.Key_O and ctrl and shift: self._on_open_video_clicked(); return True
        if k == Qt.Key.Key_O and ctrl: self._on_open_clicked(); return True

        return False

    def _play_current_line(self) -> None:
        row = self._current_row()
        if row < 0:
            return
        if row >= len(self._table_model.events):
            return
        # [v3.6 bugfix PP-1]: Dùng _table_model.events[row] thay vì
        # current_events[row] — tránh tạo list copy O(n) chỉ để đọc 1 phần tử.
        ev = self._table_model.events[row]
        self._video_widget.seek(ev.start_sec)
        self._loop_active = True
        self._loop_start_time = ev.start_sec
        self._loop_end_time = ev.end_sec
        self._loop_single_play = True
        if not self._video_widget.is_playing:
            self._video_widget.play()
        InfoBar.info(self._translator.translate("editor.ib_play_current_t"), self._translator.translate("editor.ib_play_current_c").replace("{n}", str(row + 1)).replace("{dur}", f"{ev.duration_sec:.2f}"), parent=self, duration=1500)

    def review_count(self) -> int:
        """[v3.23.682] Số dòng nghi ngờ OCR hiện có."""
        return len(self._proxy_model.suspects())

    def _emit_review_count(self) -> None:
        self.review_count_changed.emit(self.review_count())

    def _on_model_data_changed_for_review(self, _tl=None, _br=None, roles=()) -> None:
        # Tô hàng đang phát đổi mỗi nhịp phát: không đụng chữ thì không đếm lại.
        if roles and Qt.ItemDataRole.DisplayRole not in roles and Qt.ItemDataRole.EditRole not in roles:
            return
        self._review_timer.start()

    def _jump_to_suspect(self, forward: bool = True) -> None:
        """[v3.23.682] F8 / Shift+F8: chọn dòng nghi ngờ OCR kế tiếp / trước đó và tua video tới đó.

        Dùng cùng tiêu chí với bộ lọc "Nghi ngờ OCR"; dòng đang bị bộ lọc khác ẩn thì bỏ lọc để thấy được.
        """
        suspects = self._proxy_model.suspects()
        target = next_suspect_row(suspects.keys(), self._current_row(), forward)
        if target is None:
            InfoBar.info(
                self._translator.translate("editor.ib_suspect"), self._translator.translate("editor.ib_no_suspect"),
                parent=self, duration=1500,
            )
            return
        proxy_index = self._proxy_model.mapFromSource(self._table_model.index(target, 0))
        if not proxy_index.isValid():
            self._filter_combo.setCurrentIndex(0)
            proxy_index = self._proxy_model.mapFromSource(self._table_model.index(target, 0))
        if proxy_index.isValid():
            self._table.selectRow(proxy_index.row())
            self._table.scrollTo(proxy_index)
        self._jump_to_event_at(target)
        reasons = self._proxy_model.suspect_reasons_text(target)
        if reasons:
            InfoBar.info(self._translator.translate("editor.ib_suspect"), reasons, parent=self, duration=2500)

    def _set_time_at_playhead(self, is_start: bool) -> None:
        """[v3.23.682] Ctrl+[ / Ctrl+]: đặt mốc bắt đầu / kết thúc dòng đang chọn bằng vị trí đang phát (hoàn tác được)."""
        idx = self._current_row()
        if idx < 0 or idx >= len(self._table_model.events):
            return
        pos = max(0.0, float(getattr(self._video_widget, "position_sec", 0.0) or 0.0))
        ev = self._table_model.events[idx]
        start, end = (pos, ev.end_sec) if is_start else (ev.start_sec, pos)
        if end - start < 0.1:          # mốc ngược / quá ngắn → bỏ, không làm hỏng dòng
            InfoBar.warning(
                self._translator.translate("editor.ib_timing"), self._translator.translate("editor.ib_bad_playhead"),
                parent=self, duration=2000,
            )
            return
        self._view_model.update_timing(idx, start, end)

    def _nudge_time(self, is_start: bool, delta: float) -> None:
        idx = self._current_row()
        if idx < 0 or idx >= len(self._table_model.events):
            return
        # [v3.6 bugfix PP-1]: Dùng _table_model.events thay vì current_events.
        ev = self._table_model.events[idx]
        if is_start:
            self._edit_start.setValue(int(max(0.0, ev.start_sec + delta) * 1000))
            self._on_apply_timing_clicked()
        else:
            self._edit_end.setValue(int(max(ev.start_sec + 0.1, ev.end_sec + delta) * 1000))
            self._on_apply_timing_clicked()

    def _on_volume_changed(self, value: int) -> None:
        player = self._video_widget.player()
        if player: player.set_volume(value)

    def _on_speed_changed(self, text: str) -> None:
        player = self._video_widget.player()
        if player:
            with contextlib.suppress(ValueError): player.set_speed(float(text.replace("x", "")))

    def _change_speed(self, delta_rate: float) -> None:
        player = self._video_widget.peek_player()
        if not player or not player.is_loaded: return

        try: current_speed = float(self._speed_combo.currentText().replace("x", ""))
        except (ValueError, AttributeError): current_speed = 1.0

        new_speed = max(_MIN_PLAYBACK_SPEED, min(_MAX_PLAYBACK_SPEED, current_speed + delta_rate))
        new_speed = round(new_speed * 4) / 4.0
        
        player.set_speed(new_speed)

        formatted_speed = f"{new_speed:.2f}x".replace(".00x", ".0x")
        self._speed_combo.blockSignals(True)
        try:
            existing = [self._speed_combo.itemText(i) for i in range(self._speed_combo.count())]
            if formatted_speed not in existing: self._speed_combo.addItem(formatted_speed)
            self._speed_combo.setCurrentText(formatted_speed)
        finally:
            self._speed_combo.blockSignals(False)

        InfoBar.info(self._translator.translate("editor.ib_speed_t"), self._translator.translate("editor.ib_speed_c").replace("{speed}", str(formatted_speed)), parent=self, duration=1500)

    def _on_apply_timing_clicked(self) -> None:
        idx = self._current_row()
        if idx >= 0: 
            self._view_model.update_timing(idx, self._edit_start.value() / 1000.0, self._edit_end.value() / 1000.0)
            self._jump_to_event_at(idx)

    def _toggle_detach(self) -> None:
        if self._detached_window is None:
            self._detached_window = _DetachedVideoWindow(self)

        if self._detached_window.isVisible():
            self._on_detach_reattach()
        else:
            self._player_panel.layout().removeWidget(self._video_widget)
            self._detached_window.attach_widget(self._video_widget)
            self._detached_window.show()
            self._video_widget.rebind_if_window_changed()  # [v3.23.650] HWND có thể đã đổi sau reparent

    def dispose_detached_window(self) -> None:
        """[v3.23.619] Gắn video về trang rồi đóng HẲN cửa sổ video tách (khi thoát ứng dụng)."""
        window = self._detached_window
        if window is None or not is_valid(window):
            self._detached_window = None
            return
        if window.isVisible():
            self._on_detach_reattach()
        window.closing_for_good = True
        window.close()
        window.deleteLater()
        self._detached_window = None

    def _on_detach_reattach(self) -> None:
        if not is_valid(self._video_widget):
            return
        if self._detached_window is not None and is_valid(self._detached_window):
            self._detached_window.lay.removeWidget(self._video_widget)
            self._video_stack.insertWidget(1, self._video_widget)
            self._video_stack.setCurrentIndex(1)
            self._video_widget.show()
            self._detached_window.hide()
            self._video_widget.rebind_if_window_changed()  # [v3.23.650] HWND có thể đã đổi sau reparent
            QTimer.singleShot(50, self._video_widget.schedule_overlay_refresh)  # [v3.23.619] API chung

    # [v3.7 bugfix EXPORT-HANG]: Tránh deadlock native file dialog với MPV/GPU.
    # QFileDialog mặc định dùng shell dialog Windows (qua COM/DWM). Top-level
    # window của EditorPage chứa widget MPV vo=gpu-next (native child window giữ
    # swapchain GPU). Sau nhiều lần Re-OCR/seek tạo–hủy hàng loạt GPU context,
    # khi shell dialog khởi tạo (tạo surface GPU cho preview + tương tác DWM
    # compositor) → deadlock với context MPV đang giữ device → TREO ngay trước
    # khi cửa sổ kịp hiện. Ép dùng dialog Qt thuần (DontUseNativeDialog) loại bỏ
    # hoàn toàn đường shell/COM/DWM nên không còn tranh chấp.
    _FILE_DIALOG_OPTIONS = QFileDialog.Option.DontUseNativeDialog

    def _prepare_for_file_dialog(self) -> None:
        """Tạm dừng video trước khi mở file dialog để giảm tranh chấp GPU.

        MPV (vo=gpu-next) render trên thread nền và giữ GPU device. Tạm dừng
        để render loop không hoạt động tích cực trong lúc dialog mở — phòng vệ
        bổ sung cho việc ép dialog Qt thuần.
        """
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._video_widget.is_playing:
                self._video_widget.pause()

    def _on_open_video_clicked(self) -> None:
        if not self._confirm_discard_unsaved():
            return
        self._prepare_for_file_dialog()
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            self._translator.translate("editor.cap_open_video"),
            "",
            self._translator.translate("extract.dialog_filter_video"),
            options=self._FILE_DIALOG_OPTIONS,
        )
        if path_str: self.load_video(Path(path_str))

    def load_events(self, events: list[SubtitleEvent], *, confirm: bool = True) -> bool:
        """Nạp danh sách phụ đề vào trình chỉnh sửa.

        [v3.23.112] Bảo vệ chống mất dữ liệu: khi nạp từ luồng khác (trích xuất, dịch, mở
        dự án, gỡ lỗi) mà trình chỉnh sửa đang có thay đổi CHƯA LƯU, sẽ hỏi xác nhận trước
        khi ghi đè. Trả về ``True`` nếu đã nạp, ``False`` nếu người dùng huỷ để giữ bản đang sửa.

        Args:
            events: Danh sách câu phụ đề cần nạp.
            confirm: Hỏi xác nhận nếu đang có thay đổi chưa lưu (mặc định bật). Đặt ``False``
                khi nơi gọi đã tự hỏi xác nhận để tránh hỏi hai lần.
        """
        if confirm and not self._confirm_discard_unsaved():
            return False
        self._view_model.load_from_events(events)
        return True

    def get_current_events(self) -> list[SubtitleEvent]:
        """Trả về bản sao danh sách phụ đề hiện tại (cho trang Dịch lấy nguồn)."""
        return list(self._table_model.events)

    def get_current_video_path(self) -> str | None:
        """[#11] Trả về đường dẫn video đang mở (cho Auto-Attach của trang Dịch)."""
        path = getattr(self, "_current_video_path", None)
        return str(path) if path else None

    def load_video(self, path: Path | str) -> None:
        """Nạp video vào trình phát và sóng âm.

        Args:
            path: Đường dẫn video; nhận cả chuỗi (dự án lưu trong CSDL là chuỗi).
        """
        path = Path(path)
        try:
            # [Ghost OCR Boxes] Xoá overlay OCR của video cũ TRƯỚC khi nạp video mới,
            # tránh các hộp vàng của phim trước còn lơ lửng trên khung hình mới.
            with contextlib.suppress(AttributeError, RuntimeError):
                self._video_widget.clear_ocr_overlay()
            self._video_widget.load(path)
            self._current_video_path = path
            self._view_model.set_current_video(path)

            saved_state = self._container.video_state_repository.get(str(path.resolve()))
            if saved_state and saved_state.roi:
                self._current_roi = saved_state.roi
                with contextlib.suppress(AttributeError, RuntimeError): self._video_widget.set_committed_roi(saved_state.roi)

            self._autosave_timer.start()
            self._video_stack.setCurrentIndex(1)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Không nạp được video vào editor: {}.", exc)
            InfoBar.error(title=self._translator.translate("editor.ib_load_err_t"), content=self._translator.translate("editor.ib_load_err_c").replace("{name}", path.name).replace("{exc}", str(exc)), parent=self, position=InfoBarPosition.TOP, duration=5000)
            return

        self._waveform_widget.load_video(path)

    def apply_ui_settings(self, font_size: int, show_waveform: bool, show_ocr_overlay: bool) -> None:
        self._waveform_widget.setVisible(show_waveform)

    def _on_waveform_seek(self, timestamp_sec: float) -> None:
        self._break_loop_if_outside(timestamp_sec)
        
        now = time.monotonic()
        if now - getattr(self, '_last_waveform_seek_time', 0.0) < 0.05:
            return
        self._last_waveform_seek_time = now

        # [v3.23.650] try/finally: ngoại lệ giữa chừng (seek, cuộn bảng…) từng để cờ kẹt True + selectionChanged
        # mất kết nối vĩnh viễn → khung sửa không còn theo dòng chọn. Và _on_selection_changed phải CHẠY (bỏ qua
        # chỉ phần tua video) — trước đây nó thoát sớm khiến khung sửa giữ chữ dòng cũ rồi focus-out ghi đè
        # chữ đó lên dòng mới (mất dữ liệu).
        self._is_syncing_from_waveform = True
        selection_model = self._table.selectionModel()
        disconnected = False
        try:
            self._video_widget.seek(timestamp_sec)

            row_idx = self._view_model._service.find_event_index_at_time(timestamp_sec)
            if row_idx >= 0 and row_idx != self._current_row():
                source_index = self._table_model.index(row_idx, 0)
                proxy_index = self._proxy_model.mapFromSource(source_index)
                if proxy_index.isValid():
                    with contextlib.suppress(RuntimeError, TypeError):
                        selection_model.selectionChanged.disconnect(self._on_selection_changed)
                        disconnected = True
                    self._table.selectRow(proxy_index.row())

                    if self._btn_auto_scroll.isChecked():
                        self._table.scrollTo(proxy_index, QAbstractItemView.ScrollHint.PositionAtCenter)
                    self._on_selection_changed()
        finally:
            if disconnected:
                with contextlib.suppress(RuntimeError, TypeError):
                    selection_model.selectionChanged.connect(self._on_selection_changed)
            self._is_syncing_from_waveform = False

    def _on_open_clicked(self) -> None:
        if not self._confirm_discard_unsaved():
            return
        self._prepare_for_file_dialog()
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            self._translator.translate("editor.dialog_open"),
            "",
            self._translator.translate("editor.dialog_filter_open"),
            options=self._FILE_DIALOG_OPTIONS,
        )
        if path_str: self._view_model.load_from_file(Path(path_str))

    def _on_insert_clicked(self) -> None:
        idx = self._current_row()
        self._insert_event_at(idx if idx >= 0 else -1)

    def _on_delete_clicked(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes: return
        rows = sorted({self._proxy_model.mapToSource(item).row() for item in indexes}, reverse=True)
        self._view_model.batch_delete_events(rows)

    def _on_split_clicked(self) -> None:
        idx = self._current_row()
        if idx < 0 or idx >= len(self._table_model.events):
            return
        # [v3.6 bugfix SC-2]: Dùng _table_model.events thay vì current_events.
        event = self._table_model.events[idx]
        start = event.start_sec
        end = event.end_sec
        midpoint = (start + end) / 2.0

        # [v3.23.613] Truyền THEO VỊ TRÍ: tên tham số khác nhau giữa các binding (PySide6
        # "minValue"/"maxValue", PyQt6 "min"/"max"). Bản cũ dùng "min"/"max" nên với
        # PySide6 getDouble ném TypeError ngay → nút Tách dòng chết im lặng.
        value, ok = QInputDialog.getDouble(
            self,
            self._translator.translate("editor.dialog_split_title"),
            self._translator.translate("editor.dialog_split_prompt"),
            midpoint,
            start + 0.001,
            end - 0.001,
            3,
        )
        if not ok:
            return
        # [v3.23.650] Danh sách có thể đã đổi trong lúc hộp thoại mở (Re-OCR xong): chỉ tách nếu vẫn đúng câu đó.
        current = self._table_model.events
        if idx >= len(current) or current[idx].uid != event.uid:
            InfoBar.warning(
                self._translator.translate("editor.ib_scan_t"),
                self._translator.translate("editor.ib_list_changed"), parent=self, duration=4000)
            return
        self._view_model.split_event(idx, value)

    def _on_merge_clicked(self) -> None:
        idx = self._current_row()
        if idx >= 0: self._view_model.merge_with_next(idx)

    def _on_shift_clicked(self) -> None:
        offset = self._shift_spin.value()
        if offset != 0.0: self._view_model.shift_all(offset)

    def _on_selection_changed(self) -> None:
        # Đang đồng bộ từ waveform: video đã được tua đúng vị trí, chỉ cập nhật khung sửa (không tua lại về đầu dòng).
        syncing = getattr(self, '_is_syncing_from_waveform', False)

        idx = self._current_row()
        has_row = idx >= 0

        self._delete_button.setEnabled(has_row)
        self._split_button.setEnabled(has_row)
        self._merge_button.setEnabled(has_row and idx < self._table_model.rowCount() - 1)
        self._edit_apply_timing.setEnabled(has_row)

        self._btn_bold.setEnabled(has_row)
        self._btn_italic.setEnabled(has_row)
        self._btn_strip_tags.setEnabled(has_row)

        if not has_row: return

        try:
            start_sec = self._table_model.events[idx].start_sec
            end_sec = self._table_model.events[idx].end_sec
        except (AttributeError, ValueError, IndexError): return

        pending_uid = getattr(self, "_pending_edit_uid", None)
        keep_typing = pending_uid is not None and pending_uid == self._table_model.events[idx].uid
        if pending_uid is not None and not keep_typing:
            # [v3.23.659] Lựa chọn chuyển sang dòng khác khi chữ chưa ghi → ghi vào dòng cũ trước khi nạp dòng mới.
            self._commit_pending_text()
            idx = self._current_row()
            if not (0 <= idx < len(self._table_model.events)):
                return
            start_sec = self._table_model.events[idx].start_sec
            end_sec = self._table_model.events[idx].end_sec

        self._edit_start.blockSignals(True)
        self._edit_end.blockSignals(True)
        self._edit_text.blockSignals(True)
        try:
            self._edit_start.setValue(int(start_sec * 1000))
            self._edit_end.setValue(int(end_sec * 1000))
            if not keep_typing:
                # Cùng dòng đang gõ dở (bảng vừa làm mới do Re-OCR / lưu tự động…) → GIỮ chữ đang gõ, không đè.
                self._edit_text.setPlainText(self._table_model.events[idx].text)
        finally:
            self._edit_start.blockSignals(False)
            self._edit_end.blockSignals(False)
            self._edit_text.blockSignals(False)

        duration = end_sec - start_sec
        cps = len(self._table_model.events[idx].text.replace("\n", "").replace(" ", "")) / duration if duration > 0 else 0.0
        self._cps_gauge.set_cps(cps)
        self._cps_label.setText(f"{cps:.1f}")

        if hasattr(self, '_waveform_widget'):
            self._waveform_widget.set_active_row(idx)

        if not syncing and not self._video_widget.is_playing:
            self._pending_seek_row = idx
            self._debounced_seek_timer.start()

        if self._loop_active:
            self._loop_start_time = start_sec
            self._loop_end_time = end_sec
            self._waveform_widget.set_loop_region(start_sec, end_sec)

    @staticmethod
    def _merged_selected_row_ranges(selection_model) -> list[tuple[int, int]]:
        """Các khoảng hàng (top, bottom) đang chọn, đã gộp liền kề, tính trực tiếp từ range (không duyệt ô)."""
        spans = sorted((r.top(), r.bottom()) for r in selection_model.selection())
        merged: list[tuple[int, int]] = []
        for top, bottom in spans:
            if merged and top <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], bottom))
            else:
                merged.append((top, bottom))
        return merged

    def _refresh_table(self, state: EditorState) -> None:
        selection_model = self._table.selectionModel()
        v_scroll = self._table.verticalScrollBar().value()

        # [v3.23.374] BẢO VỆ khôi phục lựa chọn: nếu selection model đã bị tái tạo
        # (disconnect ném RuntimeError) thì KHÔNG được để cả hàm chết — vì phần cuối
        # còn phải BẬT LẠI nút Xuất. Trước đây một lỗi ở đây làm nút Xuất kẹt disable
        # vĩnh viễn → người dùng bấm "Lưu tệp" không thấy gì (đúng lỗi đang gặp).
        try:
            # [v3.23.650] Gom thành các khoảng hàng liên tiếp: Ctrl+A rồi sửa một dòng từng tạo 3649 range riêng,
            # khiến mọi thao tác sau đó chậm 10–30 lần (UI đơ) — nay luôn là vài range.
            selected_ranges = self._merged_selected_row_ranges(selection_model)

            with contextlib.suppress(RuntimeError, TypeError):
                selection_model.selectionChanged.disconnect(self._on_selection_changed)

            self._table_model.set_events(state.events)

            if selected_ranges:
                selection_model.clearSelection()
                batch_selection = QItemSelection()
                cols = self._proxy_model.columnCount() - 1
                max_row = self._proxy_model.rowCount() - 1
                last_valid_proxy_index = None
                for top, bottom in selected_ranges:
                    top, bottom = min(top, max_row), min(bottom, max_row)
                    if top < 0:
                        continue
                    batch_selection.select(self._proxy_model.index(top, 0), self._proxy_model.index(bottom, cols))
                    last_valid_proxy_index = self._proxy_model.index(bottom, 0)

                if last_valid_proxy_index is not None and last_valid_proxy_index.isValid():
                    selection_model.select(batch_selection, QItemSelectionModel.SelectionFlag.Select)
                    selection_model.setCurrentIndex(last_valid_proxy_index, QItemSelectionModel.SelectionFlag.NoUpdate)

            self._table.verticalScrollBar().setValue(v_scroll)
        except (RuntimeError, TypeError):
            # Khôi phục lựa chọn thất bại KHÔNG được chặn việc bật lại nút bên dưới.
            logger.exception("Khôi phục lựa chọn bảng thất bại — vẫn tiếp tục cập nhật nút.")
            with contextlib.suppress(RuntimeError, TypeError):
                self._table_model.set_events(state.events)
        finally:
            with contextlib.suppress(RuntimeError, TypeError):
                selection_model.selectionChanged.connect(self._on_selection_changed)

        self._on_selection_changed()

        self._undo_button.setEnabled(state.can_undo)
        self._redo_button.setEnabled(state.can_redo)
        self._shift_button.setEnabled(bool(state.events))
        self._replace_all_button.setEnabled(bool(state.events))
        self._insert_button.setEnabled(True)
        self._btn_autofix.setEnabled(bool(state.events))
        self._btn_merge_similar.setEnabled(bool(state.events))

        # [V3.47 PERF FIX] Trả lại trạng thái cho Nút Xuất File
        self._export_button.setEnabled(bool(state.events) and not self._view_model._is_busy)
        self._export_button.setText("💾 " + self._translator.translate("editor.btn_export"))

        
        service = self._view_model._service
        if state.can_undo and service._undo_stack:
            last_action = service._undo_stack[-1].description
            self._undo_button.setToolTip(self._translator.translate("editor.tt_undo_action").replace("{action}", str(last_action)))
        else:
            self._undo_button.setToolTip(self._translator.translate("editor.ed_tt_undo"))

        if state.can_redo and service._redo_stack:
            next_action = service._redo_stack[-1].description
            self._redo_button.setToolTip(self._translator.translate("editor.tt_redo_action").replace("{action}", str(next_action)))
        else:
            self._redo_button.setToolTip(self._translator.translate("editor.ed_tt_redo"))

        if state.is_dirty:
            self._status_label.setStyleSheet(f"color: {_c.warning()};")
            # [v3.23.379] Guard bất biến ngôn ngữ: KHÔNG ghi đè status đang hiển thị
            # tiến trình Re-OCR (chứa "Re-OCR") hoặc xác nhận đã lưu/xuất (tiền tố "✓").
            # Trước đây so khớp chuỗi "Lưu" tiếng Việt → vỡ khi chạy tiếng Anh.
            _status_text = self._status_label.text()
            if "Re-OCR" not in _status_text and not _status_text.startswith("✓"):
                self._status_label.setText(self._translator.translate("editor.status_dirty", count=len(state.events)))
        else:
            self._status_label.setStyleSheet(f"color: {_c.on_surface_muted()};")
            self._status_label.setText(self._translator.translate("editor.status_clean", count=len(state.events)))

        self._schedule_preview()
        
        current_timing_hash = hash(tuple((e.start_sec, e.end_sec, e.text) for e in state.events))
        if getattr(self, '_last_timing_hash', None) != current_timing_hash:
            if hasattr(self, '_waveform_widget'):
                self._waveform_widget.set_events(state.events)
                self._waveform_widget.clear_reocr_region()
            self._last_timing_hash = current_timing_hash

        active_focus = QApplication.focusWidget()
        if active_focus not in (self._edit_text, self._edit_start, self._edit_end, self._find_edit, self._replace_edit):
            self._table.setFocus()

    def _show_error(self, message: str) -> None:
        # Phục hồi nút Export về trạng thái hoạt động (phòng khi lỗi từ export flow).
        # [v3.6 bugfix]: setText() trước đây không kèm setEnabled(True) →
        # nút vẫn bị disable sau lỗi nếu busy_changed signal không đến kịp.
        self._export_button.setText("💾 " + self._translator.translate("editor.btn_export"))
        has_events = bool(self._table_model.events)
        self._export_button.setEnabled(has_events and not self._view_model._is_busy)

        InfoBar.error(
            title=self._translator.translate("editor.error_title"),
            content=message,
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=6000,
        )

    def _on_strip_tags_clicked(self) -> None:
        idx = self._current_row()
        if idx < 0:
            return
        text = self._edit_text.toPlainText()
        stripped_count = sum(len(m.group()) for m in self._ass_override_regex.finditer(text))
        self._view_model.strip_tags(idx, text)
        msg = self._translator.translate("editor.strip_done").replace("{n}", str(stripped_count)) if stripped_count else self._translator.translate("editor.strip_none")
        InfoBar.info(self._translator.translate("editor.ib_stripfmt_t"), msg, parent=self, duration=1500)

__all__ = ["EditorPage"]
