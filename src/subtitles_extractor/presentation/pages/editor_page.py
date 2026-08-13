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
from typing import Any

from loguru import logger
from subtitles_extractor.presentation.qt_compat import is_valid
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, QItemSelection, QItemSelectionModel
from PySide6.QtGui import (
    QCloseEvent, QColor, QDragEnterEvent, QDropEvent, QImage, QKeyEvent, QShowEvent, QHideEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QColorDialog, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QListWidget, QMenu, QMessageBox, QStackedLayout, QTableView, QVBoxLayout, QWidget,
)
from subtitles_extractor.presentation.fluent_compat import (
    themeColor,
    CaptionLabel, CheckBox, ComboBox, DoubleSpinBox, FluentIcon, InfoBar,
    InfoBarPosition, LineEdit, PrimaryPushButton, ProgressBar, PushButton, ScrollArea,
    Slider, SpinBox, StrongBodyLabel, TextEdit, ToolButton,
)

from subtitles_extractor.application.services.subtitle_editor_service import EditorState
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.presentation.utils.time_format import seconds_to_display
from subtitles_extractor.presentation.utils.ass_preview_builder import (
    AssPreviewStyle,
    build_ass_header,
    render_dialogue_line,
)
from subtitles_extractor.presentation.utils.text_replace import replace_in_text_safe
from subtitles_extractor.presentation.view_models.editor_page_view_model import EditorPageViewModel
from subtitles_extractor.presentation.widgets import create_video_widget
from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme.styles import caption_style, mono_label_style
from subtitles_extractor.presentation.widgets.section_card import SectionCard
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name
from subtitles_extractor.presentation.widgets.audio_waveform_widget import AudioWaveformWidget
from subtitles_extractor.presentation.widgets.editor_widgets import CpsGauge, SubtitleTagHighlighter, TimeSpinBox
from subtitles_extractor.presentation.widgets.mpv_video_widget import MpvVideoWidget as RoiMpvVideoWidget  # noqa: F401
from subtitles_extractor.presentation.widgets.subtitle_table_model import (
    COL_DUR, COL_END, COL_GAP, COL_INDEX, COL_SCORE, COL_START, COL_TEXT, COL_WARN,
    SubtitleFilterProxyModel, SubtitleTableModel,
)
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets

_MAX_LINE_CHARS: int = 40
_DEBOUNCED_SEEK_MS: int = 250
_PREVIEW_DEBOUNCE_MS: int = 250
_FILTER_DEBOUNCE_MS: int = 150
_SYNC_HIGHLIGHT_MS: int = 100
_AUTOSAVE_INTERVAL_MS: int = 3 * 60 * 1000
_AUTO_SCROLL_PAUSE_AFTER_USER_SEC: float = 2.0
_MIN_PLAYBACK_SPEED: float = 0.25
_MAX_PLAYBACK_SPEED: float = 4.0
_SLIDER_RESOLUTION: int = 1000
_LOW_CONFIDENCE_THRESHOLD: float = 0.6
_TOO_FAST_CPS_THRESHOLD: float = 20.0
_TOO_SHORT_DURATION_SEC: float = 0.5

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
    ],
    "sc_grp_timing": [
        ("Alt+← / →", "sc_shift_start"),
        ("Alt+Shift+← / →", "sc_shift_end"),
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
    ],
}


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


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

class AdvancedReOcrDialog(QDialog):
    _MIN_RANGE_SEC: float = 0.5

    def __init__(
        self, video_path: Path, start_sec: float, end_sec: float,
        current_roi: Roi | None, container: ApplicationContainer,
        parent: QWidget | None = None, remembered_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._translator = container.translator
        self.setWindowTitle(self._translator.translate("editor.ed_win_reocr"))

        screen_rect = QApplication.primaryScreen().availableGeometry()
        dialog_width = min(1100, int(screen_rect.width() * 0.8))
        dialog_height = min(720, int(screen_rect.height() * 0.85))
        self.resize(dialog_width, dialog_height)

        self.video_path = video_path
        self.start_sec = start_sec
        self.end_sec = max(end_sec, start_sec + self._MIN_RANGE_SEC)
        self.current_roi = current_roi
        self._container = container

        self._current_position_sec: float = self.start_sec
        self._was_playing_before_seek: bool = False
        self._updating_slider_from_playback: bool = False
        # [v3.6 bugfix DLG-1+]: Cờ báo dialog đang đóng → các queued signal
        # handler bỏ qua, tránh truy cập video_canvas đã release.
        self._is_closing: bool = False

        self._build_ui()
        protect_scroll_widgets(self)

        cfg = self._container.settings_service.current
        self.spin_limit_side.setValue(cfg.ocr.limit_side_len)
        self.spin_det_thresh.setValue(cfg.ocr.det_thresh)
        self.spin_box_thresh.setValue(cfg.ocr.det_box_thresh)
        self.spin_unclip_ratio.setValue(cfg.ocr.det_unclip_ratio)
        self.chk_upscale.setChecked(cfg.preprocess.upscale_small_text)
        self.spin_upscale_h.setValue(cfg.preprocess.upscale_target_height_px)
        self.chk_contrast.setChecked(cfg.preprocess.apply_contrast_boost)
        self.spin_contrast_val.setValue(cfg.preprocess.contrast_factor)
        self.chk_sharpen.setChecked(cfg.preprocess.apply_sharpen)
        self.chk_clahe.setChecked(cfg.preprocess.apply_clahe)
        self.spin_clahe_clip.setValue(cfg.preprocess.clahe_clip_limit)
        self.spin_clahe_tile.setValue(cfg.preprocess.clahe_tile_size)
        self.spin_border.setValue(cfg.preprocess.border_thickness_px)
        self.chk_median_blend.setChecked(cfg.preprocess.apply_median_blend)
        self.spin_conf.setValue(cfg.threshold.ocr_min_confidence)

        # [v3.23.105] Combo model phản ánh ĐÚNG version OCR đang cấu hình (trước ép cứng
        # index 0). Nếu có cấu hình đã nhớ từ lần Re-OCR trước thì ưu tiên hiển thị lại.
        remembered = remembered_config or {}
        self._select_model_version(remembered.get("model") or cfg.ocr.version)
        self._apply_remembered_tweaks(remembered.get("tweaks") or {})

        self._seek_service = self._create_seek_service()
        QTimer.singleShot(100, self._init_player)

    def _select_model_version(self, version: str | None) -> None:
        """Chọn mục combo model theo chuỗi version; không khớp -> giữ index 0."""
        if not version:
            return
        for i in range(self.cb_model.count()):
            if self.cb_model.itemData(i) == version:
                self.cb_model.setCurrentIndex(i)
                return

    def _apply_remembered_tweaks(self, tweaks: dict[str, Any]) -> None:
        """Áp lại tham số tiền xử lý đã nhớ từ lần Re-OCR trước (nếu có)."""
        if not tweaks:
            return
        setters = {
            "limit_side_len": self.spin_limit_side.setValue,
            "det_thresh": self.spin_det_thresh.setValue,
            "box_thresh": self.spin_box_thresh.setValue,
            "det_unclip_ratio": self.spin_unclip_ratio.setValue,
            "upscale": self.chk_upscale.setChecked,
            "upscale_h": self.spin_upscale_h.setValue,
            "sharpen": self.chk_sharpen.setChecked,
            "contrast": self.chk_contrast.setChecked,
            "contrast_factor": self.spin_contrast_val.setValue,
            "border": self.spin_border.setValue,
            "min_conf": self.spin_conf.setValue,
            "clahe": self.chk_clahe.setChecked,
            "clahe_clip": self.spin_clahe_clip.setValue,
            "clahe_tile": self.spin_clahe_tile.setValue,
            "median_blend": self.chk_median_blend.setChecked,
        }
        for key, setter in setters.items():
            if key in tweaks and tweaks[key] is not None:
                setter(tweaks[key])

    def _create_seek_service(self) -> Any:
        from subtitles_extractor.presentation.widgets.preview_seek_service import PreviewSeekService
        service = PreviewSeekService(self.video_path, parent=self)
        service.frame_ready.connect(self._on_seek_frame_ready)
        service.start()
        return service

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)

        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.video_canvas = create_video_widget(mpv_options=self._container.build_mpv_player_kwargs(), parent=self, translator=self._translator)
        self.video_canvas.enable_roi_drawing(True)
        self.video_canvas.roi_changed.connect(self._on_roi_drawn)
        self.video_canvas.position_changed.connect(self._on_canvas_position_changed)
        self.video_canvas.state_changed.connect(self._on_canvas_state_changed)
        self.video_canvas.seek_fallback_requested.connect(self._on_seek_fallback_requested)

        left_layout.addWidget(self.video_canvas, stretch=1)

        player_toolbar = QHBoxLayout()
        self.btn_play = ToolButton(FluentIcon.PLAY)
        set_accessible_name(self.btn_play, self._translator.translate("editor.ed_acc_playpause"))
        self.btn_play.clicked.connect(self.video_canvas.toggle_play_pause)

        self.timeline_slider = Slider(Qt.Orientation.Horizontal)
        set_accessible_name(self.timeline_slider, self._translator.translate("editor.ed_acc_timeline"), set_tooltip=False)
        self.timeline_slider.setRange(0, _SLIDER_RESOLUTION)
        self.timeline_slider.valueChanged.connect(self._on_slider_value_changed)
        self.timeline_slider.sliderPressed.connect(self._on_slider_pressed)
        self.timeline_slider.sliderReleased.connect(self._on_slider_released)

        self.lbl_time = CaptionLabel("00:00 / 00:00")

        player_toolbar.addWidget(self.btn_play)
        player_toolbar.addWidget(self.timeline_slider, stretch=1)
        player_toolbar.addWidget(self.lbl_time)
        left_layout.addLayout(player_toolbar)

        info_draw = QLabel(self._translator.translate("editor.ed_lbl_guide"))
        info_draw.setStyleSheet(f"color: {_c.success()};")
        left_layout.addWidget(info_draw)

        self.btn_clear_roi = PushButton(self._translator.translate("editor.ed_btn_clear_roi"))
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        left_layout.addWidget(self.btn_clear_roi)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)

        grp_time = SectionCard(self._translator.translate("editor.ed_sec_timerange"))
        form_time = QFormLayout()
        grp_time.add_layout(form_time)

        self.start_spin = TimeSpinBox()
        self.start_spin.setValue(int(self.start_sec * 1000))
        self.start_spin.valueChanged.connect(self._on_time_spin_changed)

        self.end_spin = TimeSpinBox()
        self.end_spin.setValue(int(self.end_sec * 1000))
        self.end_spin.valueChanged.connect(self._on_time_spin_changed)

        form_time.addRow(self._translator.translate("editor.ed_start"), self.start_spin)
        form_time.addRow(self._translator.translate("editor.ed_end"), self.end_spin)
        lbl_time_warning = QLabel(self._translator.translate("editor.ed_lbl_overwrite_warn"))
        lbl_time_warning.setWordWrap(True)
        lbl_time_warning.setStyleSheet(f"{caption_style(_c.danger())} margin-top: 2px;")
        form_time.addRow("", lbl_time_warning)
        right_layout.addWidget(grp_time)

        scroll_area = ScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 8, 0)

        grp_model = SectionCard(self._translator.translate("editor.ed_sec_ai_core"))
        form_model = QFormLayout()
        grp_model.add_layout(form_model)
        self.cb_model = ComboBox()
        self.cb_model.addItem(self._translator.translate("editor.ed_m_medium"), userData="PP-OCRv6_medium")
        self.cb_model.addItem(self._translator.translate("editor.ed_m_small"), userData="PP-OCRv6_small")
        self.cb_model.addItem(self._translator.translate("editor.ed_m_tiny"), userData="PP-OCRv6_tiny")
        self.cb_model.addItem("PP-OCRv5 (Mobile) - Nhanh", userData="PP-OCRv5_mobile")
        self.cb_model.addItem(self._translator.translate("editor.ed_m_server"), userData="PP-OCRv5_server")
        self.cb_model.addItem(self._translator.translate("editor.ed_m_v4"), userData="PP-OCRv4")
        self.cb_model.setCurrentIndex(0)  # PP-OCRv6 medium mặc định
        form_model.addRow(self._translator.translate("editor.ed_model"), self.cb_model)
        scroll_layout.addWidget(grp_model)

        grp_adv_ai = SectionCard(self._translator.translate("editor.ed_sec_detect_cfg"))
        form_adv_ai = QFormLayout()
        grp_adv_ai.add_layout(form_adv_ai)
        self.spin_limit_side = SpinBox()
        self.spin_limit_side.setRange(0, 4096); self.spin_limit_side.setSingleStep(32)
        self.spin_det_thresh = DoubleSpinBox()
        self.spin_det_thresh.setRange(0.01, 1.0); self.spin_det_thresh.setSingleStep(0.05)
        self.spin_box_thresh = DoubleSpinBox()
        self.spin_box_thresh.setRange(0.01, 1.0); self.spin_box_thresh.setSingleStep(0.05)
        self.spin_unclip_ratio = DoubleSpinBox()
        self.spin_unclip_ratio.setRange(0.5, 3.0); self.spin_unclip_ratio.setSingleStep(0.1)
        self.spin_conf = DoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0); self.spin_conf.setSingleStep(0.05)

        form_adv_ai.addRow(self._translator.translate("editor.ed_limit_side"), self.spin_limit_side)
        form_adv_ai.addRow(self._translator.translate("editor.ed_det_thresh"), self.spin_det_thresh)
        form_adv_ai.addRow(self._translator.translate("editor.ed_box_thresh"), self.spin_box_thresh)
        form_adv_ai.addRow(self._translator.translate("editor.ed_unclip"), self.spin_unclip_ratio)
        form_adv_ai.addRow(self._translator.translate("editor.ed_text_thresh"), self.spin_conf)
        scroll_layout.addWidget(grp_adv_ai)

        grp_basic = SectionCard(self._translator.translate("editor.ed_sec_preprocess"))
        form_basic = QFormLayout()
        grp_basic.add_layout(form_basic)

        self.chk_upscale = CheckBox(self._translator.translate("editor.ed_chk_upscale"))
        self.chk_upscale.setChecked(False)
        self.spin_upscale_h = SpinBox()
        self.spin_upscale_h.setRange(32, 512)
        self.spin_upscale_h.setSuffix(" px")
        self.chk_upscale.toggled.connect(self.spin_upscale_h.setEnabled)

        self.chk_contrast = CheckBox(self._translator.translate("editor.ed_chk_contrast"))
        self.chk_contrast.setChecked(True)
        self.spin_contrast_val = DoubleSpinBox()
        self.spin_contrast_val.setRange(0.5, 3.0)
        self.spin_contrast_val.setSingleStep(0.1)
        self.chk_contrast.toggled.connect(self.spin_contrast_val.setEnabled)

        self.chk_sharpen = CheckBox(self._translator.translate("editor.ed_chk_sharpen"))
        self.chk_sharpen.setChecked(True)

        self.chk_clahe = CheckBox(self._translator.translate("editor.ed_chk_clahe"))
        self.chk_clahe.setChecked(False)
        self.spin_clahe_clip = DoubleSpinBox()
        self.spin_clahe_clip.setRange(1.0, 10.0)
        self.spin_clahe_clip.setSingleStep(0.5)
        self.spin_clahe_tile = SpinBox()
        self.spin_clahe_tile.setRange(2, 32)
        self.chk_clahe.toggled.connect(self.spin_clahe_clip.setEnabled)
        self.chk_clahe.toggled.connect(self.spin_clahe_tile.setEnabled)

        self.spin_border = SpinBox()
        self.spin_border.setRange(0, 100)
        self.spin_border.setSuffix(" px")
        self.chk_median_blend = CheckBox(self._translator.translate("editor.ed_chk_median"))

        box_prep_up = QHBoxLayout(); box_prep_up.setContentsMargins(0,0,0,0)
        box_prep_up.addWidget(self.chk_upscale); box_prep_up.addWidget(self.spin_upscale_h)

        box_prep_con = QHBoxLayout(); box_prep_con.setContentsMargins(0,0,0,0)
        box_prep_con.addWidget(self.chk_contrast); box_prep_con.addWidget(self.spin_contrast_val)

        box_prep_clahe = QHBoxLayout(); box_prep_clahe.setContentsMargins(0,0,0,0)
        box_prep_clahe.addWidget(self.chk_clahe); box_prep_clahe.addWidget(self.spin_clahe_clip); box_prep_clahe.addWidget(self.spin_clahe_tile)

        form_basic.addRow(self._translator.translate("editor.ed_row_upscale"), box_prep_up)
        form_basic.addRow(self._translator.translate("editor.ed_row_contrast"), box_prep_con)
        form_basic.addRow(self._translator.translate("editor.ed_row_clahe"), box_prep_clahe)
        form_basic.addRow(self._translator.translate("editor.ed_row_sharpen"), self.chk_sharpen)
        form_basic.addRow(self._translator.translate("editor.ed_row_border"), self.spin_border)
        form_basic.addRow(self._translator.translate("editor.ed_row_denoise"), self.chk_median_blend)

        scroll_layout.addWidget(grp_basic)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        right_layout.addWidget(scroll_area, stretch=1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        btn_ok = btns.button(QDialogButtonBox.StandardButton.Ok)
        if btn_ok:
            btn_ok.setText(self._translator.translate("editor.ed_btn_start_rescan"))
            btn_ok.setStyleSheet(
                f"background-color: {themeColor().name()}; color: white; "
                f"font-weight: bold; padding: 6px 12px;"
            )

        right_layout.addWidget(btns)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 3)
        root_layout.addWidget(splitter)

    def _init_player(self) -> None:
        self.video_canvas.load(self.video_path)
        self.video_canvas.seek(self.start_sec)
        if self.current_roi:
            self.video_canvas.set_committed_roi(self.current_roi)

    def _clear_roi(self) -> None:
        self.current_roi = None
        self.video_canvas.set_committed_roi(None)

    def _on_canvas_state_changed(self, is_playing: bool) -> None:
        if hasattr(self, 'btn_play'):
            self.btn_play.setIcon(
                (FluentIcon.PAUSE if is_playing else FluentIcon.PLAY).icon()
            )

    def _on_canvas_position_changed(self, pos: float) -> None:
        # [v3.6 bugfix DLG-1+]: Bỏ qua nếu dialog đang đóng (queued signal).
        if getattr(self, "_is_closing", False) or getattr(self, "video_canvas", None) is None:
            return
        if pos < self.start_sec or pos >= self.end_sec:
            self.video_canvas.seek(self.start_sec)
            pos = self.start_sec

        dur = self.end_sec - self.start_sec
        if dur > 0 and hasattr(self, 'timeline_slider') and hasattr(self, 'lbl_time'):
            val = int(((pos - self.start_sec) / dur) * _SLIDER_RESOLUTION)
            self._updating_slider_from_playback = True
            self.timeline_slider.setValue(val)
            self._updating_slider_from_playback = False
            self.lbl_time.setText(f"{_fmt_mmss(pos)} / {_fmt_mmss(self.end_sec)}")

        self._current_position_sec = pos

    def _on_slider_pressed(self) -> None:
        if self.video_canvas.is_playing:
            self._was_playing_before_seek = True
            self.video_canvas.pause()
        else:
            self._was_playing_before_seek = False

    def _on_slider_released(self) -> None:
        if self._was_playing_before_seek:
            self.video_canvas.play()

    def _on_slider_value_changed(self, value: int) -> None:
        if self._updating_slider_from_playback: return
        dur = self.end_sec - self.start_sec
        if dur > 0:
            ts = self.start_sec + (value / float(_SLIDER_RESOLUTION)) * dur
            self.video_canvas.seek(ts)

    def _on_time_spin_changed(self) -> None:
        self.start_sec = self.start_spin.value() / 1000.0
        
        new_end_sec = max(self.end_spin.value() / 1000.0, self.start_sec + 0.1)
        if new_end_sec != self.end_sec:
            self.end_sec = new_end_sec
            self.end_spin.blockSignals(True)
            self.end_spin.setValue(int(self.end_sec * 1000))
            self.end_spin.blockSignals(False)
        else:
            self.end_sec = self.end_spin.value() / 1000.0

        if self._current_position_sec < self.start_sec or self._current_position_sec > self.end_sec:
            self.video_canvas.seek(self.start_sec)

    def _on_seek_fallback_requested(self, ts: float) -> None:
        if getattr(self, "_seek_service", None) is not None:
            self._seek_service.request_seek(ts)

    def _on_seek_frame_ready(self, qimage: QImage, width: int, height: int, timestamp_sec: float) -> None:
        # [v3.6 bugfix DLG-1+]: Seek service async có thể emit frame_ready đã
        # queued ngay trước khi stop() → handler fire sau cleanup. Guard để
        # tránh truy cập video_canvas đã release/đang hủy.
        if getattr(self, "_is_closing", False) or getattr(self, "video_canvas", None) is None:
            return
        self.video_canvas.set_video_size(width, height)
        if self.video_canvas.player() is None:
            try: self.video_canvas.set_frame(qimage, width, height)  # type: ignore
            except AttributeError: pass
        if self.current_roi: self.video_canvas.set_committed_roi(self.current_roi)

    def _on_roi_drawn(self, new_roi: Roi | None) -> None:
        self.current_roi = new_roi

    def get_values(self) -> tuple[float, float, dict[str, Any], Roi | None, str]:
        s = self.start_spin.value() / 1000.0
        e = self.end_spin.value() / 1000.0
        if s >= e: e = s + 0.1
        model_version = self.cb_model.currentData()
        return (
            s, e,
            {
                "limit_side_len": self.spin_limit_side.value(),
                "det_thresh": self.spin_det_thresh.value(),
                "box_thresh": self.spin_box_thresh.value(),
                "det_unclip_ratio": self.spin_unclip_ratio.value(),
                "upscale": self.chk_upscale.isChecked(),
                "upscale_h": self.spin_upscale_h.value(),
                "sharpen": self.chk_sharpen.isChecked(),
                "contrast": self.chk_contrast.isChecked(),
                "contrast_factor": self.spin_contrast_val.value(),
                "border": self.spin_border.value(),
                "min_conf": self.spin_conf.value(),
                "clahe": self.chk_clahe.isChecked(),
                "clahe_clip": self.spin_clahe_clip.value(),
                "clahe_tile": self.spin_clahe_tile.value(),
                "median_blend": self.chk_median_blend.isChecked(),
            },
            self.current_roi,
            model_version
        )

    def _cleanup_services(self) -> None:
        # [v3.6 bugfix DLG-1+]: Đặt cờ _is_closing để các queued signal handler
        # (position_changed, frame_ready...) biết dialog đang đóng và bỏ qua.
        self._is_closing = True

        # Tách biệt 2 bước cleanup để một bước fail KHÔNG ngăn bước kia chạy.
        # Trước đây _seek_service.stop() không bọc suppress → nếu raise thì
        # release_player() không bao giờ chạy → MPV leak.
        if getattr(self, "_seek_service", None) is not None:
            with contextlib.suppress(AttributeError, RuntimeError):
                self._seek_service.stop()
            self._seek_service = None

        # KHÔNG set self.video_canvas = None ở đây — các queued signal handler
        # vẫn có thể tham chiếu nó. release_player() là idempotent và an toàn.
        # video_canvas QWidget sẽ được hủy khi dialog.deleteLater() chạy (cascade
        # tới mọi widget con), giải phóng native window handle + GPU surface.
        if getattr(self, "video_canvas", None) is not None:
            with contextlib.suppress(AttributeError, RuntimeError):
                self.video_canvas.release_player()

    def accept(self) -> None:
        self._cleanup_services()
        super().accept()

    def reject(self) -> None:
        self._cleanup_services()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._cleanup_services()
        super().closeEvent(event)


class EditorSettingsDialog(QDialog):
    def __init__(self, settings_repo: Any, translator: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._translator = translator
        self.setWindowTitle(self._translator.translate("editor.ed_win_video_settings"))
        self.resize(450, 400)
        self._repo = settings_repo

        main_lay = QVBoxLayout(self)

        grp_sub = SectionCard(self._translator.translate("editor.ed_sec_textfmt"))
        form_sub = QFormLayout()
        grp_sub.add_layout(form_sub)

        self.cb_font = ComboBox()
        from PySide6.QtGui import QFontDatabase
        fams = QFontDatabase.families()
        if not fams: fams = ["Arial", "Consolas", "Times New Roman"]
        self.cb_font.addItems(fams)
        self.cb_font.setCurrentText(str(self._repo.load("editor_preview_font", "Arial")))
        form_sub.addRow(self._translator.translate("editor.ed_row_font"), self.cb_font)

        self.spin_size = SpinBox()
        self.spin_size.setRange(8, 120)
        self.spin_size.setValue(int(self._repo.load("editor_preview_size", 48)))
        form_sub.addRow(self._translator.translate("editor.ed_row_size"), self.spin_size)

        self.btn_color_pri = PushButton(self._translator.translate("editor.ed_btn_pick_color"))
        self._pri_color = str(self._repo.load("editor_preview_color_primary", "&H00FFFFFF"))
        self._update_color_btn(self.btn_color_pri, self._pri_color)
        self.btn_color_pri.clicked.connect(lambda: self._pick_color('primary', self.btn_color_pri, use_alpha=False))
        form_sub.addRow(self._translator.translate("editor.ed_row_color"), self.btn_color_pri)

        self.btn_color_out = PushButton(self._translator.translate("editor.ed_btn_pick_color"))
        self._out_color = str(self._repo.load("editor_preview_color_outline", "&H00000000"))
        self._update_color_btn(self.btn_color_out, self._out_color)
        self.btn_color_out.clicked.connect(lambda: self._pick_color('outline', self.btn_color_out, use_alpha=False))
        form_sub.addRow(self._translator.translate("editor.ed_row_outline"), self.btn_color_out)

        self.btn_color_bg = PushButton(self._translator.translate("editor.ed_btn_pick_color"))
        self._bg_color = str(self._repo.load("editor_preview_color_bg", "&H99000000"))
        self._update_color_btn(self.btn_color_bg, self._bg_color)
        self.btn_color_bg.clicked.connect(lambda: self._pick_color('bg', self.btn_color_bg, use_alpha=True))
        form_sub.addRow(self._translator.translate("editor.ed_row_opacity"), self.btn_color_bg)

        self.spin_margin = SpinBox()
        self.spin_margin.setRange(0, 500)
        self.spin_margin.setValue(int(self._repo.load("editor_preview_margin_v", 25)))
        form_sub.addRow(self._translator.translate("editor.ed_row_marginv"), self.spin_margin)

        main_lay.addWidget(grp_sub)

        grp_wf = SectionCard(self._translator.translate("editor.ed_sec_waveform"))
        form_wf = QFormLayout()
        grp_wf.add_layout(form_wf)
        self.spin_wf_edge = SpinBox()
        self.spin_wf_edge.setRange(5, 50); self.spin_wf_edge.setSuffix(" px")
        self.spin_wf_edge.setValue(int(self._repo.load("editor_waveform_edge_px", 10)))
        form_wf.addRow(self._translator.translate("editor.ed_row_hitbox"), self.spin_wf_edge)
        main_lay.addWidget(grp_wf)

        main_lay.addStretch()

        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)

        btn_lay = QHBoxLayout()
        btn_lay.addStretch()
        btn_lay.addWidget(bbox)
        main_lay.addLayout(btn_lay)

    def _ass_to_qcolor(self, ass_str: str) -> QColor:
        try:
            if len(ass_str) >= 10 and ass_str.startswith("&H"):
                a_ass = int(ass_str[2:4], 16); b = int(ass_str[4:6], 16)
                g = int(ass_str[6:8], 16); r = int(ass_str[8:10], 16)
                return QColor(r, g, b, 255 - a_ass)
        except (ValueError, IndexError): pass
        return QColor(255, 255, 255, 255)

    def _update_color_btn(self, btn: PushButton, ass_str: str) -> None:
        c = self._ass_to_qcolor(ass_str)
        text_col = "black" if (c.red() * 0.299 + c.green() * 0.587 + c.blue() * 0.114) > 128 else "white"
        bg_style = f"rgba({c.red()},{c.green()},{c.blue()},{c.alpha() / 255.0})"
        btn.setStyleSheet(f"QPushButton {{ background-color: {bg_style}; color: {text_col}; font-weight: bold; border: 1px solid {_c.border()}; }}")

    def _pick_color(self, key_type: str, btn: PushButton, use_alpha: bool = False) -> None:
        initial_color = self._ass_to_qcolor(self._pri_color if key_type == 'primary' else self._out_color if key_type == 'outline' else self._bg_color)
        options = QColorDialog.ColorDialogOption.ShowAlphaChannel if use_alpha else QColorDialog.ColorDialogOption(0)
        color = QColorDialog.getColor(initial_color, self, "Chọn màu", options)

        if color.isValid():
            a_ass = 255 - color.alpha() if use_alpha else 0
            ass_val = f"&H{a_ass:02X}{color.blue():02X}{color.green():02X}{color.red():02X}"

            if key_type == 'primary': self._pri_color = ass_val
            elif key_type == 'outline': self._out_color = ass_val
            else: self._bg_color = ass_val
            self._update_color_btn(btn, ass_val)

    def save_to_repo(self) -> None:
        self._repo.save("editor_preview_font", self.cb_font.currentText())
        self._repo.save("editor_preview_size", self.spin_size.value())
        self._repo.save("editor_preview_color_primary", self._pri_color)
        self._repo.save("editor_preview_color_outline", self._out_color)
        self._repo.save("editor_preview_color_bg", self._bg_color)
        self._repo.save("editor_preview_margin_v", self.spin_margin.value())
        self._repo.save("editor_waveform_edge_px", self.spin_wf_edge.value())


class _DetachedVideoWindow(QWidget):
    def __init__(self, parent_page: EditorPage) -> None:
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)
        self._translator = parent_page._translator
        self.setWindowTitle(self._translator.translate("editor.ed_win_video_preview"))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(900, 540)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)
        self._parent_page = parent_page

    def attach_widget(self, video_widget: QWidget) -> None:
        self.lay.addWidget(video_widget, stretch=1)
        video_widget.show()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        k = event.key()
        if k == Qt.Key.Key_Escape: self._parent_page._on_detach_reattach()
        elif k == Qt.Key.Key_F11: self.showNormal() if self.isFullScreen() else self.showFullScreen()
        else: super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._parent_page._on_detach_reattach()
        event.ignore()


class EditorPage(QWidget):

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

        root.addLayout(self._build_file_toolbar())
        root.addLayout(self._build_edit_toolbar())
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

    def _build_file_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
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
        layout.addStretch(1)
        layout.addWidget(self._undo_button)
        layout.addWidget(self._redo_button)
        layout.addWidget(self._history_button)
        layout.addWidget(self._btn_shortcuts)
        return layout

    def _build_edit_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
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
        layout.addSpacing(15)
        layout.addWidget(self._btn_autofix); layout.addWidget(self._btn_merge_similar)
        layout.addStretch(1)

        layout.addWidget(StrongBodyLabel(self._translator.translate("editor.shift_label")))
        self._shift_spin = DoubleSpinBox()
        self._shift_spin.setRange(-3600.0, 3600.0); self._shift_spin.setSingleStep(0.1)
        self._shift_spin.setDecimals(3); self._shift_spin.setSuffix(" s")
        layout.addWidget(self._shift_spin)
        self._shift_button = PushButton(self._translator.translate("editor.btn_shift_apply"))
        self._shift_button.setEnabled(False)
        layout.addWidget(self._shift_button)
        return layout

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
        self._filter_combo.addItems([self._translator.translate("editor.ed_flt_all"), self._translator.translate("editor.ed_flt_ocr_err"), self._translator.translate("editor.ed_flt_warn")])
        self._filter_combo.setFixedWidth(120)
        layout.addWidget(self._filter_combo); layout.addWidget(self._lbl_find_count)
        layout.addSpacing(10); layout.addWidget(StrongBodyLabel(self._translator.translate("editor.ed_lbl_replace")))

        self._replace_edit = LineEdit()
        layout.addWidget(self._replace_edit, stretch=1)
        self._replace_one_button = PushButton("Thay 1"); self._replace_all_button = PushButton(self._translator.translate("editor.ed_btn_replace_all"))
        self._replace_one_button.setEnabled(False); self._replace_all_button.setEnabled(False)
        layout.addWidget(self._replace_one_button); layout.addWidget(self._replace_all_button)

        layout.addSpacing(20)
        self._btn_fast_reocr = PushButton(self._translator.translate("editor.ed_btn_reocr_fast"))
        self._btn_fast_reocr.setStyleSheet(f"color: {_c.success()}; font-weight: bold;")
        self._btn_fast_reocr.setToolTip(
            self._translator.translate("editor.ed_tt_reocr_fast")
        )
        layout.addWidget(self._btn_fast_reocr)

        self._btn_reocr = PrimaryPushButton(self._translator.translate("editor.ed_btn_reocr_adv"))
        self._btn_reocr.setToolTip(self._translator.translate("editor.ed_tt_reocr_adv"))
        layout.addWidget(self._btn_reocr)
        return layout

    def _build_top_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._table = QTableView(parent=self)
        self._table_model = SubtitleTableModel(self)
        self._proxy_model = SubtitleFilterProxyModel(self)
        self._proxy_model.setSourceModel(self._table_model)
        self._table.setModel(self._proxy_model)

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
        row_s = QHBoxLayout(); row_s.addWidget(StrongBodyLabel("Show:")); self._edit_start = TimeSpinBox(); row_s.addWidget(self._edit_start)
        row_e = QHBoxLayout(); row_e.addWidget(StrongBodyLabel("Hide:  ")); self._edit_end = TimeSpinBox(); row_e.addWidget(self._edit_end)
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

        self._table_model.text_edit_requested.connect(self._view_model.update_text)
        self._table_model.time_edit_requested.connect(self._view_model.update_timing)
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
        if hasattr(self, 'btn_play'):
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
        indexes = self._table.selectionModel().selectedRows()
        if not indexes: return -1
        return self._proxy_model.mapToSource(indexes[0]).row()

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

        mpv_player = self._video_widget.player()
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

    def _on_text_editing(self) -> None:
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
        if k == Qt.Key.Key_BracketLeft: self._change_speed(-0.25); return True
        if k == Qt.Key.Key_BracketRight: self._change_speed(0.25); return True
        if k == Qt.Key.Key_Left and alt and not shift: self._nudge_time(is_start=True, delta=-0.1); return True
        if k == Qt.Key.Key_Right and alt and not shift: self._nudge_time(is_start=True, delta=0.1); return True
        if k == Qt.Key.Key_Left and alt and shift: self._nudge_time(is_start=False, delta=-0.1); return True
        if k == Qt.Key.Key_Right and alt and shift: self._nudge_time(is_start=False, delta=0.1); return True
        
        if k == Qt.Key.Key_C and not (ctrl or alt or shift): 
            self._btn_center_wf.setChecked(not self._btn_center_wf.isChecked())
            return True
            
        if k == Qt.Key.Key_M and not (ctrl or alt or shift):
            player = self._video_widget.player()
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
            indexes = self._table.selectionModel().selectedRows()
            if indexes:
                pr = max(0, indexes[0].row() - 1)
                self._table.selectRow(pr)
                if not self._video_widget.is_playing:
                    self._jump_to_event_at(pr)
            return True
        if k == Qt.Key.Key_Down and ctrl:
            indexes = self._table.selectionModel().selectedRows()
            if indexes:
                pr = min(self._proxy_model.rowCount() - 1, indexes[0].row() + 1)
                self._table.selectRow(pr)
                if not self._video_widget.is_playing:
                    self._jump_to_event_at(pr)
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
        player = self._video_widget.player()
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

    def _apply_filters(self) -> None:
        query = self._find_edit.text().strip().lower()
        filter_mode = self._filter_combo.currentIndex()

        self._proxy_model.set_filter_params(query, filter_mode)

        match_cnt = self._proxy_model.rowCount()
        if query or filter_mode > 0: self._lbl_find_count.setText(self._translator.translate("editor.ed_find_count").replace("{n}", str(match_cnt)))
        else: self._lbl_find_count.setText("")

        has_query = bool(query)
        self._replace_one_button.setEnabled(has_query)
        self._replace_all_button.setEnabled(has_query)

    def _find_next(self) -> None: self._find(forward=True)
    def _find_prev(self) -> None: self._find(forward=False)

    def _find(self, forward: bool = True) -> None:
        term = self._find_edit.text().strip().lower()
        if not term: return

        matches = [i for i, text in enumerate(self._table_model._lower_text_cache) if term in text]
        if not matches: return

        cur = self._current_row()
        if forward:
            nxt = next((m for m in matches if m > cur), matches[0])
        else:
            nxt = next((m for m in reversed(matches) if m < cur), matches[-1])

        source_index = self._table_model.index(nxt, 0)
        proxy_index = self._proxy_model.mapFromSource(source_index)

        if proxy_index.isValid():
            self._table.selectRow(proxy_index.row())
            self._table.scrollTo(proxy_index, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _replace_one(self) -> None:
        term = self._find_edit.text().strip()
        repl = self._replace_edit.text()
        if not term:
            return
        row_index = self._current_row()
        if row_index < 0:
            return

        # [v3.6 bugfix RO-1]: Dùng _table_model nhất quán thay vì current_events.
        # Trước đây: cache từ _table_model + event.text từ current_events → hai nguồn
        # khác nhau có thể bất đồng bộ. Sau fix mutation bug, cache luôn sync với
        # _table_model.events, nên dùng cả hai từ cùng một nguồn.
        model_events = self._table_model.events
        cache = self._table_model._lower_text_cache
        if row_index >= len(model_events) or row_index >= len(cache):
            return

        if term.lower() in cache[row_index]:
            new_text = self._replace_in_text_safe(term, repl, model_events[row_index].text, count=1)
            self._view_model.update_text(row_index, new_text)
            self._find_next()

    def _on_apply_timing_clicked(self) -> None:
        idx = self._current_row()
        if idx >= 0: 
            self._view_model.update_timing(idx, self._edit_start.value() / 1000.0, self._edit_end.value() / 1000.0)
            self._jump_to_event_at(idx)

    @staticmethod
    def _replace_in_text_safe(term: str, replacement: str, source: str, count: int) -> str:
        # [v3.20] Logic đã tách sang module thuần ``utils.text_replace`` (testable).
        return replace_in_text_safe(term, replacement, source, count)

    def _replace_all(self) -> None:
        term = self._find_edit.text().strip()
        repl = self._replace_edit.text()
        if not term:
            return

        # [v3.6 bugfix]: Dùng nhất quán từ _table_model thay vì kết hợp
        # ViewModel + table cache.  Trước đây:
        #   events = self._view_model.current_events   ← từ ViewModel
        #   cache  = self._table_model._lower_text_cache ← từ table model
        # Nếu cache stale (vd do bug mutation cũ), tìm kiếm sẽ miss/hit sai.
        # Sau khi fix mutation bug, cache luôn được rebuild qua set_events().
        # Dùng _table_model.events và cache cùng nguồn → luôn nhất quán.
        model_events = self._table_model.events
        cache = self._table_model._lower_text_cache

        # Safety guard: cache và events phải cùng độ dài (đảm bảo không IndexError).
        if len(cache) != len(model_events):
            # Rebuild cache nếu mất sync (edge case phòng thủ).
            cache = [e.text.lower() for e in model_events]

        term_lower = term.lower()
        replacements: dict[int, str] = {}

        for index, event in enumerate(model_events):
            if term_lower in cache[index]:
                new_text = self._replace_in_text_safe(term, repl, event.text, count=0)
                replacements[index] = new_text

        if replacements:
            self._view_model.batch_replace_text(replacements)
            InfoBar.success(
                self._translator.translate("editor.ib_replace_t"), self._translator.translate("editor.ib_replace_c").replace("{n}", str(len(replacements))),
                parent=self, duration=2000,
            )
        else:
            # [v3.23.115] Trước đây im lặng -> người dùng tưởng nút hỏng. Báo rõ không khớp.
            InfoBar.info(
                self._translator.translate("editor.ib_noreplace_t"), self._translator.translate("editor.ib_noreplace_c").replace("{term}", term),
                parent=self, duration=2500,
            )

    def _on_auto_fix_timeline(self) -> None:
        events = self._view_model.current_events
        if not events:
            return

        # [v3.6 bugfix AF-1]: Đếm đúng điều kiện của auto_fix_timeline:
        # bao gồm cả overlap (gap < 0) VÀ small positive gap (0 < gap < 0.150s).
        # Trước đây chỉ đếm overlap → thông báo số lỗi thấp hơn thực tế.
        error_count = sum(
            1
            for i in range(len(events) - 1)
            if (events[i + 1].start_sec - events[i].end_sec) < 0.150
            and events[i + 1].start_sec >= events[i].start_sec  # bỏ qua fully-inverted
        )
        if error_count == 0:
            InfoBar.info(self._translator.translate("editor.ib_perfect_t"), self._translator.translate("editor.ib_perfect_c"), parent=self, duration=2000)
            return

        reply = QMessageBox.question(
            self, self._translator.translate("editor.dlg_autofix_t"),
            self._translator.translate("editor.dlg_autofix_b").replace("{n}", str(error_count)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes: return

        fixes = self._view_model.auto_fix_timeline()
        if fixes > 0:
            self._video_widget._schedule_geometry_update()
            InfoBar.success(self._translator.translate("editor.ib_autofix_t"), self._translator.translate("editor.ib_autofix_c").replace("{n}", str(fixes)), parent=self, duration=3000)

    def _on_merge_similar_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._translator.translate("editor.ed_win_merge_dup"))
        dlg.resize(400, 200)
        form = QFormLayout(dlg)

        gap_sb = DoubleSpinBox(dlg)
        gap_sb.setRange(0.0, 10.0); gap_sb.setSingleStep(0.1); gap_sb.setValue(0.5); gap_sb.setSuffix(" s")
        sim_sb = DoubleSpinBox(dlg)
        sim_sb.setRange(0.5, 1.0); sim_sb.setSingleStep(0.05); sim_sb.setValue(0.9); sim_sb.setDecimals(2)

        form.addRow(self._translator.translate("editor.ed_row_maxgap"), gap_sb)
        form.addRow(self._translator.translate("editor.ed_row_minsim"), sim_sb)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        # [v3.6 bugfix DLG-4]: deleteLater cho dialog cấu hình. Đọc giá trị
        # trong try, hủy trong finally.
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            gap_value = gap_sb.value()
            sim_value = sim_sb.value()
        finally:
            dlg.deleteLater()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try: groups = self._view_model.find_similar_groups(gap_value, sim_value)
        finally: QApplication.restoreOverrideCursor()

        if not groups:
            InfoBar.info(self._translator.translate("editor.ib_scan_t"), self._translator.translate("editor.ib_scan_nodup"), parent=self)
            return

        preview_dlg = QDialog(self)
        preview_dlg.setWindowTitle(self._translator.translate("editor.dlg_dup_title").replace("{n}", str(len(groups))))
        preview_dlg.resize(600, 400)
        p_lay = QVBoxLayout(preview_dlg)
        p_lay.addWidget(QLabel(self._translator.translate("editor.ed_lbl_merge_prompt"), preview_dlg))

        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
        table = QTableWidget(preview_dlg)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels([self._translator.translate("editor.th_merge"), self._translator.translate("editor.th_time"), self._translator.translate("editor.th_content")])
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.setRowCount(len(groups))

        checkboxes = []
        events = self._view_model.current_events

        for row_idx, grp in enumerate(groups):
            chk = CheckBox(preview_dlg)
            chk.setChecked(True)
            checkboxes.append(chk)
            chk_widget = QWidget()
            layout = QHBoxLayout(chk_widget); layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(chk)
            table.setCellWidget(row_idx, 0, chk_widget)
            first_ev = events[grp[0]]
            table.setItem(row_idx, 1, QTableWidgetItem(self._translator.translate("editor.dup_row").replace("{time}", seconds_to_display(first_ev.start_sec)).replace("{n}", str(len(grp)))))
            best_text = max((events[idx].text for idx in grp), key=len).replace('\n', ' ')
            table.setItem(row_idx, 2, QTableWidgetItem(best_text))

        p_lay.addWidget(table)
        p_bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel, preview_dlg)
        btn_all = PushButton(self._translator.translate("editor.ed_btn_select_all"), preview_dlg)
        btn_all.clicked.connect(lambda: [c.setChecked(True) for c in checkboxes])
        btn_none = PushButton(self._translator.translate("editor.ed_btn_deselect_all"), preview_dlg)
        btn_none.clicked.connect(lambda: [c.setChecked(False) for c in checkboxes])

        p_bb.addButton(btn_all, QDialogButtonBox.ButtonRole.ActionRole)
        p_bb.addButton(btn_none, QDialogButtonBox.ButtonRole.ActionRole)
        p_bb.accepted.connect(preview_dlg.accept)
        p_bb.rejected.connect(preview_dlg.reject)
        p_lay.addWidget(p_bb)

        # [v3.6 bugfix DLG-4]: deleteLater cho preview dialog. Đọc trạng thái
        # checkbox TRONG try (trước khi hủy), apply merge ngoài try.
        try:
            if preview_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            groups_to_merge = [grp for i, grp in enumerate(groups) if checkboxes[i].isChecked()]
        finally:
            preview_dlg.deleteLater()

        if groups_to_merge:
            applied = self._view_model.apply_merge_groups(groups_to_merge)
            InfoBar.success(self._translator.translate("editor.ib_done_t"), self._translator.translate("editor.ib_merged_c").replace("{n}", str(applied)), parent=self, duration=3000)

    def _toggle_detach(self) -> None:
        if self._detached_window is None:
            self._detached_window = _DetachedVideoWindow(self)

        if self._detached_window.isVisible():
            self._on_detach_reattach()
        else:
            self._player_panel.layout().removeWidget(self._video_widget)
            self._detached_window.attach_widget(self._video_widget)
            self._detached_window.show()

    def _on_detach_reattach(self) -> None:
        if self._detached_window is not None and is_valid(self._detached_window):
            self._detached_window.lay.removeWidget(self._video_widget)
            self._video_stack.insertWidget(1, self._video_widget)
            self._video_stack.setCurrentIndex(1)
            self._video_widget.show()
            self._detached_window.hide()
            QTimer.singleShot(50, self._video_widget._schedule_geometry_update)

    def _on_video_roi_changed(self, roi: Roi) -> None: pass

    def _on_reocr_progress(self, current: int, total: int, msg: str) -> None:
        if total > 0:
            self._progress_bar.setRange(0, total); self._progress_bar.setValue(current)
            self._progress_bar.setVisible(True)
        else:
            # [v3.6 bugfix R1]: Chỉ ẩn thanh progress, KHÔNG xóa status_label.
            # Trước đây setVisible(False) + setText("") cùng một lúc → thông báo
            # "Re-OCR hoàn tất trong X.Xs" bị xóa ngay sau khi hiện ra, user
            # không kịp đọc. Giờ giữ lại text nếu msg là chuỗi rỗng.
            self._progress_bar.setVisible(False)
        if msg:  # Chỉ cập nhật status label khi có nội dung thực sự
            self._status_label.setText(msg)

    def _on_reocr_busy(self, is_busy: bool) -> None:
        self._btn_reocr.setEnabled(not is_busy)
        self._btn_fast_reocr.setEnabled(not is_busy)
        
        # [V3.47 PERF FIX] Vô hiệu hóa Nút Xuất khi hệ thống bận I/O để tránh Spam
        self._export_button.setEnabled(not is_busy and bool(self._table_model.events))
        
        self._status_label.setStyleSheet(f"color: {_c.warning()}; font-weight: bold;" if is_busy else f"color: {_c.on_surface_muted()};")

    def _on_reocr_clicked(self) -> None:
        if not self._current_video_path:
            InfoBar.warning(self._translator.translate("editor.ib_no_video_t"), self._translator.translate("editor.ib_no_video_open_c"), parent=self, duration=3000)
            return

        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            InfoBar.warning(self._translator.translate("editor.ib_no_sel_t"), self._translator.translate("editor.ib_no_sel_c"), parent=self, duration=3000)
            return

        rows = sorted({self._proxy_model.mapToSource(item).row() for item in indexes})
        events = self._view_model.current_events
        preview_start = min(events[r].start_sec for r in rows)
        preview_end = max(events[r].end_sec for r in rows)

        if self._video_widget.is_playing:
            self._video_widget.toggle_play_pause()

        dlg = AdvancedReOcrDialog(
            video_path=self._current_video_path, start_sec=preview_start, end_sec=preview_end,
            current_roi=getattr(self, "_current_roi", None), container=self._container, parent=self,
            remembered_config=getattr(self, "_reocr_remembered", None),
        )

        # [v3.6 bugfix DLG-1]: PHẢI hủy dialog sau khi dùng. Dialog tạo với
        # parent=self → Qt giữ sống đến khi parent bị hủy. Mỗi lần Re-OCR tạo
        # một AdvancedReOcrDialog mới (kèm MPV player + native window handle).
        # Không hủy → tích lũy 8+ dialog ẩn → khi QFileDialog mở (cùng parent)
        # tương tác với các MPV widget/GPU context tích lũy → TREO ứng dụng.
        # try/finally đảm bảo deleteLater() luôn chạy kể cả khi early-return.
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            new_start, new_end, tweaks, new_roi, model_override = dlg.get_values()
            # [v3.23.105] Ghi nhớ cấu hình vừa dùng để "Re-OCR Nhanh" lặp lại + Studio mở lại
            # hiển thị đúng lần trước (đáp ứng kỳ vọng "cài đặt đã nhớ trong Re-OCR Studio").
            self._reocr_remembered = {"model": model_override, "tweaks": dict(tweaks)}
        finally:
            dlg.deleteLater()

        overlapping_rows = set(rows)
        for i, ev in enumerate(events):
            if ev.start_sec < new_end and ev.end_sec > new_start:
                overlapping_rows.add(i)

        final_rows = sorted(overlapping_rows)

        # [v3.6 bugfix R2/R4]: Không gọi update_timing() trước Re-OCR nữa.
        # Trước đây expand timing của first/last event để OCR quét đúng vùng:
        #   - Bug R2: khi first_row==last_row, lần 2 update_timing dùng stale
        #     events ref → revert lần 1 → start expansion bị mất.
        #   - Bug R4: nếu Re-OCR không tìm thấy gì, timing đã bị thay đổi vẫn
        #     giữ nguyên trong undo stack → user phải undo thủ công.
        # Thay bằng explicit_time_ranges: truyền vùng quét trực tiếp cho
        # start_reocr(), không cần sửa timing trước.
        from subtitles_extractor.application.dtos.reocr_dto import TimeRange
        import contextlib
        explicit_ranges = []
        with contextlib.suppress(ValueError):
            explicit_ranges.append(TimeRange(start_sec=new_start, end_sec=new_end))

        if not explicit_ranges:
            self.error_occurred.emit(self._translator.translate("editor.err_bad_range"))
            return

        self._waveform_widget.set_reocr_region(new_start, new_end)
        self._view_model.start_reocr(
            video_path=self._current_video_path, rows_to_replace=final_rows, roi=new_roi,
            tweaks=tweaks, model_override=model_override,
            explicit_time_ranges=explicit_ranges,
        )

    def _on_fast_reocr_clicked(self) -> None:
        if not self._current_video_path:
            InfoBar.warning(self._translator.translate("editor.ib_no_video_t"), self._translator.translate("editor.ib_no_video_c"), parent=self, duration=3000)
            return

        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            InfoBar.warning(self._translator.translate("editor.ib_no_sel_t"), self._translator.translate("editor.ib_no_sel_c"), parent=self, duration=3000)
            return

        rows_to_process = sorted({self._proxy_model.mapToSource(item).row() for item in indexes})
        current_events_list = self._view_model.current_events

        preview_start_sec = min(current_events_list[row_idx].start_sec for row_idx in rows_to_process)
        preview_end_sec = max(current_events_list[row_idx].end_sec for row_idx in rows_to_process)
        if preview_end_sec - preview_start_sec < 0.5: preview_end_sec = preview_start_sec + 0.5

        if self._video_widget.is_playing:
            self._video_widget.toggle_play_pause()

        self._waveform_widget.set_reocr_region(preview_start_sec, preview_end_sec)

        # [v3.23.105] "Re-OCR Nhanh" = lặp lại cấu hình Re-OCR đã nhớ gần nhất (model +
        # tham số). Nếu chưa từng mở Studio trong phiên này -> dùng cài đặt OCR mặc định của
        # ứng dụng (tweaks rỗng -> _build_reocr_request lấy theo settings; model None -> v6).
        remembered = getattr(self, "_reocr_remembered", None)
        if remembered:
            advanced_default_tweaks = dict(remembered.get("tweaks") or {})
            advanced_model_override = remembered.get("model")
        else:
            advanced_default_tweaks = {}
            advanced_model_override = None

        current_active_roi = None
        saved_video_state = self._container.video_state_repository.get(str(self._current_video_path.resolve()))
        if saved_video_state and saved_video_state.roi:
            current_active_roi = saved_video_state.roi

        overlapping_row_indices = set(rows_to_process)
        for index, event_obj in enumerate(current_events_list):
            if event_obj.start_sec < preview_end_sec and event_obj.end_sec > preview_start_sec:
                overlapping_row_indices.add(index)

        final_rows_to_replace = sorted(overlapping_row_indices)

        # [v3.6 bugfix R2/R4]: Xây explicit_time_ranges từ preview range thay vì
        # gọi update_timing() trước. Loại bỏ hoàn toàn:
        #   - Bug R2: stale events ref khi single-row có cả start lẫn end expand
        #   - Bug R4: orphaned timing entries khi Re-OCR không tìm thấy kết quả
        from subtitles_extractor.application.dtos.reocr_dto import TimeRange
        import contextlib as _ctx
        explicit_ranges = []
        with _ctx.suppress(ValueError):
            explicit_ranges.append(
                TimeRange(start_sec=preview_start_sec, end_sec=preview_end_sec)
            )

        if not explicit_ranges:
            self._waveform_widget.clear_reocr_region()
            InfoBar.warning(self._translator.translate("editor.ib_bad_range_t"), self._translator.translate("editor.ib_range_short_c"), parent=self, duration=3000)
            return

        self._view_model.start_reocr(
            video_path=self._current_video_path, rows_to_replace=final_rows_to_replace,
            roi=current_active_roi, tweaks=advanced_default_tweaks,
            model_override=advanced_model_override,
            explicit_time_ranges=explicit_ranges,
        )

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

    def load_video(self, path: Path) -> None:
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

        self._is_syncing_from_waveform = True 
        self._video_widget.seek(timestamp_sec)
        
        row_idx = self._view_model._service.find_event_index_at_time(timestamp_sec)
        if row_idx >= 0 and row_idx != self._current_row():
            source_index = self._table_model.index(row_idx, 0)
            proxy_index = self._proxy_model.mapFromSource(source_index)
            if proxy_index.isValid():
                self._table.selectionModel().selectionChanged.disconnect(self._on_selection_changed)
                self._table.selectRow(proxy_index.row())
                
                if self._btn_auto_scroll.isChecked():
                    self._table.scrollTo(proxy_index, QAbstractItemView.ScrollHint.PositionAtCenter)
                    
                self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)
                self._on_selection_changed()
        
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

    def _on_export_clicked(self) -> None:
        # [v3.23.374] Log ở đầu để phân biệt "nút bị disable" (không có log) với "handler
        # chạy nhưng dialog không mở" (có log). Bọc try/except vì PySide6 NUỐT exception
        # trong slot → trước đây lỗi ở đây làm nút "không thấy chạy gì" mà không báo.
        logger.debug(
            "Editor: bấm Xuất tệp (events=%d, is_busy=%s).",
            len(self._table_model.events),
            self._view_model._is_busy,
        )
        try:
            self._do_export_clicked()
        except Exception as exc:  # noqa: BLE001 — chặn Qt nuốt lỗi im lặng; báo cho người dùng
            logger.exception("Lỗi không mong đợi khi mở hộp thoại xuất tệp.")
            self._export_button.setText("💾 " + self._translator.translate("editor.btn_export"))
            self._export_button.setEnabled(bool(self._table_model.events))
            InfoBar.error(
                title=self._translator.translate("editor.error_title"),
                content=(
                    self._translator.translate("editor.export_unexpected_error")
                    + f"\n[{type(exc).__name__}] {exc}"
                ),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=8000,
            )

    def _do_export_clicked(self) -> None:
        default_dir = ""
        if self._current_video_path:
            # Quy ước an toàn: phụ đề gốc từ Biên tập đặt là <tên>.original.srt
            # để không ghi đè bản dịch (.translate.) hay bản TTS (.tts.).
            from subtitles_extractor.domain.value_objects.output_naming import (
                extracted_subtitle_path,
            )

            default_dir = str(extracted_subtitle_path(self._current_video_path))
        self._prepare_for_file_dialog()
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            self._translator.translate("editor.dialog_export"),
            default_dir,
            self._translator.translate("editor.dialog_filter_export"),
            options=self._FILE_DIALOG_OPTIONS,
        )
        if not path_str:
            return

        # [v3.6 bugfix EXPORT-FIX-3]: Kiểm tra _is_busy SAU KHI dialog đóng.
        # QFileDialog.getSaveFileName() là blocking và chạy event loop bên trong.
        # Trong khoảng thời gian dialog mở (~vài giây), các sự kiện khác có thể:
        #   • Re-OCR hoàn tất → set _is_busy=False, sau đó bắt đầu lại → True
        #   • Watchdog hết giờ → force-reset nhưng lại có task khác bắt đầu
        # Nếu _is_busy=True khi export_to_file() được gọi, nó sẽ trả về False
        # NHƯNG nút đã bị disable + text "⏳" → UI kẹt. Guard này ngăn điều đó.
        if self._view_model._is_busy:
            InfoBar.warning(
                title=self._translator.translate("editor.dlg_busy_t"),
                content=self._translator.translate("editor.dlg_busy_b"),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
            )
            return

        output_path = Path(path_str)
        output_format = (
            SubtitleFormat.ASS
            if output_path.suffix.lower() == ".ass"
            else SubtitleFormat.SRT
        )

        # Cập nhật nút trước khi gọi export
        self._export_button.setText(self._translator.translate("editor.ed_btn_saving"))
        self._export_button.setEnabled(False)

        started = self._view_model.export_to_file(output_path, output_format)
        if not started:
            # export_to_file bị từ chối (race condition rất hiếm sau guard ở trên)
            # → phục hồi nút để user thử lại được.
            self._export_button.setText("💾 " + self._translator.translate("editor.btn_export"))
            self._export_button.setEnabled(bool(self._table_model.events))
            InfoBar.warning(
                title=self._translator.translate("editor.dlg_cantexport_t"),
                content=self._translator.translate("editor.dlg_cantexport_b"),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
            )

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

        value, ok = QInputDialog.getDouble(
            self,
            self._translator.translate("editor.dialog_split_title"),
            self._translator.translate("editor.dialog_split_prompt"),
            value=midpoint,
            min=start + 0.001,
            max=end - 0.001,
            decimals=3,
        )
        if ok: self._view_model.split_event(idx, value)

    def _on_merge_clicked(self) -> None:
        idx = self._current_row()
        if idx >= 0: self._view_model.merge_with_next(idx)

    def _on_shift_clicked(self) -> None:
        offset = self._shift_spin.value()
        if offset != 0.0: self._view_model.shift_all(offset)

    def _on_selection_changed(self) -> None:
        if getattr(self, '_is_syncing_from_waveform', False):
            return

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

        self._edit_start.blockSignals(True)
        self._edit_end.blockSignals(True)
        self._edit_text.blockSignals(True)
        try:
            self._edit_start.setValue(int(start_sec * 1000))
            self._edit_end.setValue(int(end_sec * 1000))
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

        if not self._video_widget.is_playing:
            self._pending_seek_row = idx
            self._debounced_seek_timer.start()

        if self._loop_active:
            self._loop_start_time = start_sec
            self._loop_end_time = end_sec
            self._waveform_widget.set_loop_region(start_sec, end_sec)

    def _refresh_table(self, state: EditorState) -> None:
        selection_model = self._table.selectionModel()
        v_scroll = self._table.verticalScrollBar().value()

        # [v3.23.374] BẢO VỆ khôi phục lựa chọn: nếu selection model đã bị tái tạo
        # (disconnect ném RuntimeError) thì KHÔNG được để cả hàm chết — vì phần cuối
        # còn phải BẬT LẠI nút Xuất. Trước đây một lỗi ở đây làm nút Xuất kẹt disable
        # vĩnh viễn → người dùng bấm "Lưu tệp" không thấy gì (đúng lỗi đang gặp).
        try:
            selected_rows = [idx.row() for idx in selection_model.selectedRows()]

            with contextlib.suppress(RuntimeError, TypeError):
                selection_model.selectionChanged.disconnect(self._on_selection_changed)

            self._table_model.set_events(state.events)

            if selected_rows:
                selection_model.clearSelection()
                batch_selection = QItemSelection()
                cols = self._proxy_model.columnCount() - 1

                last_valid_proxy_index = None
                max_row = self._proxy_model.rowCount() - 1

                for r in selected_rows:
                    safe_r = min(r, max_row)
                    if safe_r >= 0:
                        idx_start = self._proxy_model.index(safe_r, 0)
                        idx_end = self._proxy_model.index(safe_r, cols)
                        batch_selection.select(idx_start, idx_end)
                        last_valid_proxy_index = idx_start

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

    def _on_export_finished(self, output_path: Path) -> None:
        self._status_label.setStyleSheet(f"color: {_c.success()};")
        self._status_label.setText(self._translator.translate("editor.status_exported", path=str(output_path)))
        # [V3.47 PERF FIX] Khôi phục Nút
        self._export_button.setText("💾 " + self._translator.translate("editor.btn_export"))
        self._export_button.setEnabled(True)

    def _schedule_preview(self) -> None: self._preview_timer.start()

    def _generate_ass_preview(self) -> None:
        player = self._video_widget.player()
        if player is None or not player.is_loaded: return
        events = self._view_model.current_events
        if not events: return

        play_res_x, play_res_y = 1920, 1080

        try:
            ass_content = self._build_ass_content(events, play_res_x, play_res_y)

            self._preview_idx ^= 1
            path = self._preview_files[self._preview_idx]

            with open(path, "w", encoding="utf-8") as fp:
                fp.write(ass_content)
                fp.flush()

            resolved_path = str(Path(path).resolve()).replace("\\", "/")

            player.send_command("sub-add", resolved_path, "select")
            QTimer.singleShot(100, lambda: self._cleanup_old_preview_tracks(resolved_path))

        except (OSError, AttributeError) as exc:
            logger.debug("generate_ass_preview thất bại: {}.", exc)

    def _cleanup_old_preview_tracks(self, current_path: str) -> None:
        player = self._video_widget.player()
        if player is None:
            return
        raw_mpv = getattr(player, "_mpv", None)
        if raw_mpv is None:
            return
        try:
            tracks = raw_mpv.track_list or []
        except (AttributeError, RuntimeError):
            return
        for track in tracks:
            if track.get("type") != "sub":
                continue
            ext_path = str(track.get("external-filename", "") or "")
            if (
                "se_preview_" not in ext_path
                or ext_path.replace("\\", "/") == current_path
            ):
                continue
            try:
                player.send_command("sub-remove", str(track["id"]))
            except (RuntimeError, KeyError) as exc:
                logger.debug("Không gỡ được sub track cũ: {}.", exc)

    def _build_ass_content(self, events: list[SubtitleEvent], play_res_x: int = 1920, play_res_y: int = 1080) -> str:
        settings_repo = self._container.settings_service._repo
        style = AssPreviewStyle(
            font_name=str(settings_repo.load("editor_preview_font", "Arial")),
            font_size=int(settings_repo.load("editor_preview_size", 48)),
            color_primary=str(settings_repo.load("editor_preview_color_primary", "&H00FFFFFF")),
            color_outline=str(settings_repo.load("editor_preview_color_outline", "&H00000000")),
            color_background=str(settings_repo.load("editor_preview_color_bg", "&H99000000")),
            margin_vertical=int(settings_repo.load("editor_preview_margin_v", 25)),
            play_res_x=play_res_x,
            play_res_y=play_res_y,
        )
        header = build_ass_header(style)

        active_idx = self._current_row()
        is_editing = self._edit_text.hasFocus()
        active_text = self._edit_text.toPlainText()

        lines = [
            render_dialogue_line(
                active_text if (is_editing and i == active_idx) else ev.text,
                ev.start_sec, ev.end_sec,
            )
            for i, ev in enumerate(events)
        ]
        return header + "\n".join(lines) + "\n"

    def _on_strip_tags_clicked(self) -> None:
        idx = self._current_row()
        if idx < 0:
            return
        text = self._edit_text.toPlainText()
        stripped_count = sum(len(m.group()) for m in self._ass_override_regex.finditer(text))
        self._view_model.strip_tags(idx, text)
        msg = self._translator.translate("editor.strip_done").replace("{n}", str(stripped_count)) if stripped_count else self._translator.translate("editor.strip_none")
        InfoBar.info(self._translator.translate("editor.ib_stripfmt_t"), msg, parent=self, duration=1500)

    def cancel_reocr(self) -> None:
        if hasattr(self, "_view_model"): self._view_model.cancel_reocr()

__all__ = ["EditorPage"]
