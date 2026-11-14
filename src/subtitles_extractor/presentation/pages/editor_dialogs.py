"""[v3.23.693] Hộp thoại và cửa sổ phụ của trang Chỉnh sửa: Re-OCR nâng cao, cài đặt trình chỉnh sửa, cửa sổ video tách rời.

Tách nguyên văn khỏi ``editor_page.py`` (U6) — hành vi không đổi.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QColor, QImage, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.presentation.fluent_compat import (
    CaptionLabel,
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    FluentIcon,
    PushButton,
    ScrollArea,
    Slider,
    SpinBox,
    ToolButton,
    themeColor,
)
from subtitles_extractor.presentation.pages.editor_constants import _SLIDER_RESOLUTION
from subtitles_extractor.presentation.qt_compat import is_valid
from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.theme.styles import caption_style
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets
from subtitles_extractor.presentation.widgets import create_video_widget
from subtitles_extractor.presentation.widgets.editor_widgets import TimeSpinBox
from subtitles_extractor.presentation.widgets.section_card import SectionCard

if TYPE_CHECKING:
    from subtitles_extractor.presentation.pages.editor_page import EditorPage


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
        from subtitles_extractor.presentation.widgets.preview_seek_service import (
            PreviewSeekService,
        )
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

    #: [v3.23.619] Đặt True khi ứng dụng đóng: cửa sổ được đóng hẳn thay vì gắn video lại.
    closing_for_good: bool = False

    def keyPressEvent(self, event: QKeyEvent) -> None:
        k = event.key()
        if k == Qt.Key.Key_Escape and is_valid(self._parent_page): self._parent_page._on_detach_reattach()
        elif k == Qt.Key.Key_F11: self.showNormal() if self.isFullScreen() else self.showFullScreen()
        else: super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        # [v3.23.619] Ứng dụng đang đóng, hoặc trang Biên tập đã bị huỷ → đóng hẳn. Trước
        # đây luôn "gắn video lại + ẩn" và bỏ qua lệnh đóng: đóng cửa sổ chính khi video
        # đang tách thì ứng dụng KHÔNG thoát (còn cửa sổ video), bấm X tiếp thì không còn
        # cửa sổ nào nhưng tiến trình vẫn chạy ngầm; gắn vào trang đã huỷ thì RuntimeError.
        if self.closing_for_good or not is_valid(self._parent_page):
            event.accept()
            return
        self._parent_page._on_detach_reattach()
        event.ignore()


__all__ = ['_fmt_mmss', 'AdvancedReOcrDialog', 'EditorSettingsDialog', '_DetachedVideoWindow']
