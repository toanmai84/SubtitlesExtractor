"""Trang "Trích xuất phụ đề" — MpvVideoWidget + SeekWorker + Điều phối UI.

CẢI TIẾN ĐỘT PHÁ (V3.13.1 - ROI Rendering Fix):
    * [CRITICAL FIX] Sửa lỗi Mất ROI: Ép VideoCanvas nhận kích thước chuẩn tuyệt đối
      từ VideoMetadata ngay khi Load, vượt qua hiện tượng trễ Header (0x0) của tiến trình MPV.
    * [CRITICAL FIX] Sửa lỗi Desync ROI: Cho phép truyền `None` vào ViewModel để
      đồng bộ hóa thao tác Hủy/Xóa vùng chọn của người dùng trên Canvas.
"""

from __future__ import annotations

import contextlib
import time
import logging
from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QObject, QEvent
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QImage, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from subtitles_extractor.presentation.fluent_compat import (
    CheckBox,
    ComboBox,
    LineEdit,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    Slider,
    ToolButton,
    themeColor,
)

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesResponse,
)
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.ports.hardsub_detector_port import HardsubDetectionResult
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment
from subtitles_extractor.presentation.view_models.extract_page_view_model import (
    ExtractPageViewModel,
)
from subtitles_extractor.presentation.widgets import create_video_widget
from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.theme.styles import caption_style, mono_label_style
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.widgets.section_card import SectionCard
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name
# Import kiểu tường minh để type checker biết canvas có thể là MpvVideoWidget (v2.30 migration)
from subtitles_extractor.presentation.widgets.mpv_video_widget import MpvVideoWidget  # noqa: F401
from subtitles_extractor.presentation.workers.seek_worker import (
    PersistentVideoReader,
    SeekWorker,
)
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets

_SEEK_DEBOUNCE_MS: int = 80
_SLIDER_RESOLUTION: int = 10000


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _preset_roi(preset_key: str, video_w: int, video_h: int) -> Roi | None:
    # "auto_subtitle": ROI được phát hiện động tại runtime (không trả cố định ở đây).
    if preset_key in ("full", "custom", "auto_subtitle"): return None
    if preset_key == "bottom_third":
        h = max(1, video_h // 3)
        return Roi(x=0, y=video_h - h, width=video_w, height=h, alignment=TextAlignment.CENTER)
    if preset_key == "bottom_quarter":
        h = max(1, video_h // 4)
        return Roi(x=0, y=video_h - h, width=video_w, height=h, alignment=TextAlignment.CENTER)
    if preset_key == "bottom_half":
        h = max(1, video_h // 2)
        return Roi(x=0, y=video_h - h, width=video_w, height=h, alignment=TextAlignment.CENTER)
    return None


logger = logging.getLogger(__name__)


class ExtractPage(QWidget):
    extraction_completed = Signal(object)

    def __init__(self, container: ApplicationContainer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("extractPage")
        self.setAcceptDrops(True)

        self._container = container
        self._translator = container.translator
        self._view_model = ExtractPageViewModel(container, parent=self)

        self._video_metadata: VideoMetadata | None = None
        self._pending_video_path: Path | None = None

        self._seek_thread: QThread | None = None
        self._seek_worker: SeekWorker | None = None
        self._video_reader: PersistentVideoReader | None = None

        self._current_position_sec: float = 0.0
        self._pending_seek_sec: float | None = None
        self._was_playing_before_seek: bool = False
        self._updating_slider_from_playback: bool = False

        self._extract_start_time: float = 0.0
        self._is_cancelled_by_user: bool = False  # [#7] cờ huỷ thủ công

        self._seek_debounce_timer = QTimer(self)
        self._seek_debounce_timer.setSingleShot(True)
        self._seek_debounce_timer.setInterval(_SEEK_DEBOUNCE_MS)
        self._seek_debounce_timer.timeout.connect(self._do_seek)

        self._build_ui()
        protect_scroll_widgets(self)
        self._connect_signals()

        self.installEventFilter(self)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel(self._translator.translate("extract.title"))
        title.setStyleSheet(f"font-size: {_m.FONT_SIZE_HEADING}px; font-weight: 600;")
        root.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        left_panel = self._build_canvas_panel()
        left_panel.setMinimumWidth(360)
        splitter.addWidget(left_panel)

        control_panel = self._build_control_panel()
        scroll = QScrollArea(self)
        scroll.setWidget(control_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(340)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([800, 480])
        root.addWidget(splitter, stretch=1)

        self._progress_bar = ProgressBar(self)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        self._status_label = QLabel(self._translator.translate("extract.status_idle"))
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

    def _build_canvas_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._canvas = create_video_widget(
            mpv_options=self._container.build_mpv_player_kwargs(),
            parent=self
        )
        self._canvas.roi_changed.connect(self._on_roi_drawn)
        self._canvas.roi_preview.connect(self._on_roi_preview)

        self._canvas.position_changed.connect(self._on_canvas_position_changed)
        self._canvas.state_changed.connect(self._on_canvas_state_changed)
        self._canvas.seek_fallback_requested.connect(self._on_seek_fallback_requested)

        layout.addWidget(self._canvas, stretch=1)
        layout.addLayout(self._build_player_controls())
        return panel

    def _build_player_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._btn_step_back = ToolButton(FluentIcon.CARE_LEFT_SOLID)
        self._btn_step_back.setToolTip(self._translator.translate("extract.ex_step_back"))
        self._btn_step_back.clicked.connect(lambda: self._seek_relative(-5.0))

        self._btn_play_pause = ToolButton(FluentIcon.PLAY)
        set_accessible_name(self._btn_play_pause, self._translator.translate("extract.ex_acc_playpause"))
        self._btn_play_pause.clicked.connect(self._canvas.toggle_play_pause)

        self._btn_step_forward = ToolButton(FluentIcon.CARE_RIGHT_SOLID)
        self._btn_step_forward.setToolTip(self._translator.translate("extract.ex_step_fwd"))
        self._btn_step_forward.clicked.connect(lambda: self._seek_relative(5.0))

        self._position_slider = Slider(Qt.Orientation.Horizontal)
        set_accessible_name(self._position_slider, self._translator.translate("extract.ex_acc_timeline"), set_tooltip=False)
        self._position_slider.setRange(0, _SLIDER_RESOLUTION)
        self._position_slider.valueChanged.connect(self._on_slider_value_changed)
        self._position_slider.sliderPressed.connect(self._on_slider_pressed)
        self._position_slider.sliderReleased.connect(self._on_slider_released)

        self._position_label = QLabel("00:00 / 00:00")
        self._position_label.setStyleSheet(mono_label_style())
        self._position_label.setMinimumWidth(110)

        row.addWidget(self._btn_step_back)
        row.addWidget(self._btn_play_pause)
        row.addWidget(self._btn_step_forward)
        row.addWidget(self._position_slider, stretch=1)
        row.addWidget(self._position_label)

        row.addWidget(QLabel(self._translator.translate("extract.ex_speed")))
        self._speed_combo = ComboBox()
        for label, value in [("0.25×", 0.25), ("0.5×", 0.5), ("1.0×", 1.0), ("2.0×", 2.0)]:
            self._speed_combo.addItem(label, userData=value)
        self._speed_combo.setCurrentIndex(2)
        row.addWidget(self._speed_combo)
        return row

    def _build_control_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(12)

        # [v3.23.109] Video là bước CHUNG cho mọi loại trích xuất -> luôn ở trên cùng.
        layout.addWidget(self._build_video_group())

        # [v3.23.109] Gom theo LOẠI trích xuất bằng tab: mỗi tab chỉ hiện control + nút
        # của loại đó -> người dùng không phải cuộn qua các loại không dùng, bố cục gọn rõ.
        self._method_tabs = QTabWidget()
        self._method_tabs.setObjectName("ExtractMethodTabs")
        self._method_tabs.addTab(self._build_hardsub_tab(), self._translator.translate("extract.ex_tab_hardsub"))
        self._method_tabs.addTab(self._build_embedded_group(), self._translator.translate("extract.ex_tab_embedded"))
        self._method_tabs.addTab(self._build_stt_group(), self._translator.translate("extract.ex_tab_stt"))
        self._method_tabs.setTabToolTip(
            0, self._translator.translate("extract.ex_tt_hardsub")
        )
        self._method_tabs.setTabToolTip(
            1, self._translator.translate("extract.ex_tt_embedded")
        )
        self._method_tabs.setTabToolTip(
            2, self._translator.translate("extract.ex_tt_stt")
        )
        layout.addWidget(self._method_tabs)

        # [v3.23.109] Khu vực CHUNG dưới cùng: huỷ tiến trình + thao tác với kết quả OCR.
        self._cancel_button = PushButton(self._translator.translate("extract.btn_cancel"))
        self._cancel_button.setEnabled(False)
        self._cancel_button.setToolTip(self._translator.translate("extract.ex_cancel_tip"))
        layout.addWidget(self._cancel_button)

        self._btn_load_cache = PushButton(self._translator.translate("extract.ex_btn_load_cache"))
        self._btn_load_cache.setStyleSheet(f"color: {_c.success()}; font-weight: bold;")
        self._btn_load_cache.setVisible(False)
        self._btn_load_cache.clicked.connect(self._view_model.load_cached_subtitles)
        layout.addWidget(self._btn_load_cache)

        self._btn_export_raw = PushButton(self._translator.translate("extract.ex_btn_export_raw"))
        self._btn_export_raw.setStyleSheet(f"color: {_c.warning()}; font-weight: bold;")
        self._btn_export_raw.setToolTip(self._translator.translate("extract.ex_export_raw_tip"))
        self._btn_export_raw.setVisible(False)
        self._btn_export_raw.clicked.connect(self._on_export_raw_clicked)
        layout.addWidget(self._btn_export_raw)

        layout.addStretch(1)
        return panel

    def _build_hardsub_tab(self) -> QWidget:
        """[v3.23.109] Tab 'Chữ cháy (OCR)': ROI + tự dò + nút Bắt đầu của loại này."""
        tab = QWidget()
        inner = QVBoxLayout(tab)
        inner.setContentsMargins(0, 8, 0, 0)
        inner.setSpacing(12)
        inner.addWidget(self._build_roi_group())
        inner.addWidget(self._build_detect_group())

        # [v3.23.321] Chọn ngôn ngữ OCR ngay tại đây. Phim bộ CJK phải đổi Trung/Nhật/
        # Hàn theo từng bộ; trước đây chỉ đổi được ở trang Cài đặt nên mỗi lần phải
        # rời trang này. self._translator.translate("extract.ex_by_settings") = không ghi đè.
        from subtitles_extractor.application.services.embedded_ocr_language import (
            UI_LANGUAGE_CHOICES,
        )

        self._main_lang_label = QLabel(self._translator.translate("extract.ex_sub_lang"))
        self._main_lang_label.setStyleSheet(caption_style())
        self._cb_main_ocr_language = ComboBox()
        self._cb_main_ocr_language.addItem(self._translator.translate("extract.ex_by_settings"), "")
        for _label, _code in UI_LANGUAGE_CHOICES:
            if _code:
                self._cb_main_ocr_language.addItem(_label, _code)
        self._cb_main_ocr_language.setToolTip(
            self._translator.translate("extract.ex_tt_mainlang")
        )
        self._cb_main_ocr_language.currentIndexChanged.connect(
            self._on_main_ocr_language_changed
        )

        # [v3.23.320] Thử nhanh 60 giây trước khi chạy cả tập — ROI/ngôn ngữ sai thì
        # biết ngay sau ~10 giây thay vì mất hàng chục phút chạy hết rồi mới phát hiện.
        # [v3.23.322] Dựng lại phụ đề từ cache OCR — đổi tham số dựng câu rồi thấy kết
        # quả sau vài giây, thay vì chạy lại OCR cả tập (hàng chục phút).
        self._rebuild_button = PushButton(self._translator.translate("extract.ex_btn_rebuild"))
        self._rebuild_button.setToolTip(
            self._translator.translate("extract.ex_tt_rebuild")
        )
        self._rebuild_button.clicked.connect(self._on_rebuild_clicked)

        # [v3.23.321] Cho chọn thử ở đoạn nào. Mặc định GIỮA phim vì đầu là intro/logo
        # và cuối là credits — thử ở đó dễ ra rỗng khiến tưởng OCR hỏng.
        self._cb_probe_position = ComboBox()
        for _label, _ratio in (
            (self._translator.translate("extract.ex_pos_mid"), 0.5), (self._translator.translate("extract.ex_pos_start"), 0.05), (self._translator.translate("extract.ex_pos_q1"), 0.25),
            (self._translator.translate("extract.ex_pos_q3"), 0.75), (self._translator.translate("extract.ex_pos_end"), 0.95),
        ):
            self._cb_probe_position.addItem(_label, _ratio)
        self._cb_probe_position.setToolTip(
            self._translator.translate("extract.ex_tt_probepos")
        )

        self._probe_button = PushButton(self._translator.translate("extract.ex_btn_probe"))
        self._probe_button.setToolTip(
            self._translator.translate("extract.ex_tt_probe")
        )
        self._probe_button.clicked.connect(self._on_probe_clicked)
        self._start_button = PrimaryPushButton(self._translator.translate("extract.btn_start"))
        self._start_button.setEnabled(False)
        self._start_button.setToolTip(
            self._translator.translate("extract.ex_tt_extract")
        )
        inner.addWidget(self._main_lang_label)
        inner.addWidget(self._cb_main_ocr_language)
        inner.addWidget(self._rebuild_button)
        inner.addWidget(self._cb_probe_position)
        inner.addWidget(self._probe_button)
        inner.addWidget(self._start_button)
        inner.addStretch(1)
        return tab

    def _build_video_group(self) -> QWidget:
        card = SectionCard(self._translator.translate("extract.group_video"))
        inner = QVBoxLayout()
        inner.setSpacing(_m.SPACING_XS)
        select_row = QHBoxLayout()
        # [v3.23.319] Trích xuất hàng loạt nhiều tập — phim bộ CJK thường vài chục tập
        # cùng vị trí phụ đề, trước đây phải lặp lại toàn bộ thao tác cho từng tập.
        self._batch_button = PushButton(self._translator.translate("extract.ex_btn_batch"))
        self._batch_button.setToolTip(
            self._translator.translate("extract.ex_tt_batch")
        )
        self._batch_button.clicked.connect(self._on_batch_clicked)
        # [v3.23.367] Nối các tập GỐC thành 1 video TRƯỚC khi trích xuất — hỗ trợ quy trình
        # "nối trọn bộ → trích xuất → dịch → … → xuất bản" trên một tệp duy nhất.
        self._concat_button = PushButton(self._translator.translate("extract.ex_btn_concat"))
        self._concat_button.setToolTip(
            self._translator.translate("extract.ex_tt_concat")
        )
        self._concat_button.clicked.connect(self._on_concat_series_clicked)
        self._select_button = PushButton(self._translator.translate("extract.btn_select"))
        select_row.addWidget(self._select_button)
        select_row.addWidget(self._batch_button)
        select_row.addWidget(self._concat_button)
        select_row.addStretch(1)
        inner.addLayout(select_row)

        self._video_label = QLabel(self._translator.translate("extract.no_video"))
        self._video_label.setStyleSheet(f"color: {_c.on_surface_muted()};")
        self._video_label.setWordWrap(True)
        inner.addWidget(self._video_label)

        hint = QLabel(self._translator.translate("extract.ex_dnd_hint"))
        hint.setStyleSheet(caption_style())
        inner.addWidget(hint)
        card.add_layout(inner)
        return card

    def _build_roi_group(self) -> QWidget:
        card = SectionCard(self._translator.translate("extract.group_roi"))
        inner = QVBoxLayout()
        inner.setSpacing(_m.SPACING_XS)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = ComboBox()
        for label, key in [
            (self._translator.translate("extract.ex_roi_auto"), "auto_subtitle"),
            (self._translator.translate("extract.ex_roi_custom"), "custom"),
            (self._translator.translate("extract.ex_roi_full"), "full"),
            (self._translator.translate("extract.ex_roi_q"), "bottom_quarter"),
            (self._translator.translate("extract.ex_roi_t"), "bottom_third"),
            (self._translator.translate("extract.ex_roi_h"), "bottom_half"),
        ]:
            self._preset_combo.addItem(label, userData=key)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo, stretch=1)
        inner.addLayout(preset_row)

        self._draw_toggle = PushButton(self._translator.translate("extract.btn_draw_roi"))
        self._draw_toggle.setCheckable(True)
        # [#6 v3.18] Nút vẽ LUÔN mở khoá: bấm là vẽ ngay, không cần chọn "Tuỳ chỉnh"
        # trước. Handler sẽ tự gạt ComboBox sang "Tuỳ chỉnh" khi bắt đầu vẽ.
        self._draw_toggle.toggled.connect(self._on_draw_toggle_toggled)
        self._draw_toggle.setEnabled(True)
        inner.addWidget(self._draw_toggle)

        info_row = QHBoxLayout()
        self._roi_label = QLabel(self._translator.translate("extract.no_roi"))
        self._roi_label.setStyleSheet(f"color: {_c.muted_italic()}; font-style: italic;")
        self._roi_label.setWordWrap(True)
        info_row.addWidget(self._roi_label, stretch=1)

        self._clear_roi_button = PushButton(self._translator.translate("extract.btn_clear_roi"))
        self._clear_roi_button.setEnabled(False)
        info_row.addWidget(self._clear_roi_button)
        inner.addLayout(info_row)

        self._detected_rois_group = SectionCard(self._translator.translate("extract.ex_roi_group"))
        self._detected_rois_group.setVisible(False)

        hint = QLabel(
            self._translator.translate("extract.ex_tt_roihint")
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(caption_style())
        self._detected_rois_group.add_widget(hint)

        self._roi_buttons_container = QWidget()
        self._roi_buttons_layout = QVBoxLayout(self._roi_buttons_container)
        self._roi_buttons_layout.setSpacing(_m.SPACING_XS)
        self._roi_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._detected_rois_group.add_widget(self._roi_buttons_container)
        inner.addWidget(self._detected_rois_group)

        card.add_layout(inner)
        return card

    def _build_detect_group(self) -> QWidget:
        card = SectionCard(self._translator.translate("extract.group_detection"))
        row = QHBoxLayout()
        row.setSpacing(_m.SPACING_XS)

        self._detect_hardsub_button = PushButton(self._translator.translate("extract.btn_detect_hardsub"))
        self._detect_hardsub_button.setToolTip(
            self._translator.translate("extract.ex_tt_detect_hardsub")
        )
        self._detect_roi_button = PushButton(self._translator.translate("extract.btn_detect_roi"))
        self._detect_roi_button.setToolTip(
            self._translator.translate("extract.ex_tt_detect_roi")
        )

        self._review_again_button = PushButton(self._translator.translate("extract.ex_btn_review_roi"))
        self._review_again_button.setVisible(False)
        self._review_again_button.setStyleSheet(f"color: {_c.warning()}; font-weight: bold;")

        for btn in (self._detect_hardsub_button, self._detect_roi_button):
            btn.setEnabled(False)

        row.addWidget(self._detect_hardsub_button)
        row.addWidget(self._detect_roi_button)
        row.addWidget(self._review_again_button)
        card.add_layout(row)
        return card

    def _build_embedded_group(self) -> QWidget:
        """[v3.21] Nhóm trích phụ đề NHÚNG sẵn trong video (text + bitmap OCR)."""
        card = SectionCard(self._translator.translate("extract.ex_embedded_group"))
        outer = QVBoxLayout()
        outer.setSpacing(_m.SPACING_SM)

        self._btn_scan_embedded = PushButton(self._translator.translate("extract.ex_btn_scan_embedded"))
        self._btn_scan_embedded.setEnabled(False)
        self._btn_scan_embedded.setToolTip(
            self._translator.translate("extract.ex_tt_scan_embedded")
        )
        outer.addWidget(self._btn_scan_embedded)

        # [v3.23.168] Phụ đề RỜI cùng tên cạnh video (Movie.srt/Movie.vi.srt/Movie.ass):
        # nạp trực tiếp, nhanh hơn OCR và không cần track nhúng. Tự dò khi mở video.
        sidecar_row = QHBoxLayout()
        self._cb_sidecar_subtitle = ComboBox()
        self._cb_sidecar_subtitle.setEnabled(False)
        self._cb_sidecar_subtitle.setToolTip(
            self._translator.translate("extract.ex_tt_sidecar")
        )
        self._btn_load_sidecar = PushButton(self._translator.translate("extract.ex_btn_load_sidecar"))
        self._btn_load_sidecar.setEnabled(False)
        self._btn_load_sidecar.setStyleSheet(
            f"color: {_c.success()}; font-weight: bold;"
        )
        self._btn_load_sidecar.setToolTip(
            self._translator.translate("extract.ex_tt_load_sidecar")
        )
        sidecar_row.addWidget(self._cb_sidecar_subtitle, stretch=1)
        sidecar_row.addWidget(self._btn_load_sidecar)
        outer.addLayout(sidecar_row)

        track_row = QHBoxLayout()
        self._cb_embedded_track = ComboBox()
        self._cb_embedded_track.setEnabled(False)
        self._cb_embedded_track.setToolTip(self._translator.translate("extract.ex_embedded_track_tip"))
        self._btn_extract_embedded = PushButton(self._translator.translate("extract.ex_btn_extract_embedded"))
        self._btn_extract_embedded.setEnabled(False)
        self._btn_extract_embedded.setStyleSheet(f"color: {_c.success()}; font-weight: bold;")
        track_row.addWidget(self._cb_embedded_track, stretch=1)
        track_row.addWidget(self._btn_extract_embedded)
        outer.addLayout(track_row)

        # [v3.23.102] Ngôn ngữ OCR cho track ảnh (VobSub/PGS). self._translator.translate("extract.ex_dev_auto") suy từ ngôn ngữ
        # track; cho phép chọn tay khi metadata thiếu/sai để tránh nhiễu □ do sai model.
        from subtitles_extractor.application.services.embedded_ocr_language import (
            UI_LANGUAGE_CHOICES,
        )

        lang_row = QHBoxLayout()
        lang_label = QLabel(self._translator.translate("extract.ex_img_ocr_lang"))
        lang_label.setStyleSheet(caption_style())
        self._cb_ocr_language = ComboBox()
        for label, _code in UI_LANGUAGE_CHOICES:
            self._cb_ocr_language.addItem(label)
        self._ocr_language_codes = [code for _label, code in UI_LANGUAGE_CHOICES]
        self._cb_ocr_language.setToolTip(
            self._translator.translate("extract.ex_tt_imglang")
        )
        lang_row.addWidget(lang_label)
        lang_row.addWidget(self._cb_ocr_language, stretch=1)
        outer.addLayout(lang_row)

        self._lbl_embedded_hint = QLabel("")
        self._lbl_embedded_hint.setWordWrap(True)
        self._lbl_embedded_hint.setStyleSheet(caption_style())
        outer.addWidget(self._lbl_embedded_hint)
        card.add_layout(outer)
        return card

    def _build_stt_group(self) -> QWidget:
        """[v3.21] Nhóm trích phụ đề từ TIẾNG NÓI (Speech-to-Text — WhisperX)."""
        card = SectionCard(self._translator.translate("extract.ex_stt_group"))
        outer = QVBoxLayout()
        outer.setSpacing(_m.SPACING_SM)

        config_row = QHBoxLayout()
        self._cb_stt_language = ComboBox()
        self._cb_stt_language.addItems([
            self._translator.translate("extract.ex_stt_auto"), self._translator.translate("extract.ex_stt_vi"), self._translator.translate("extract.ex_stt_en"),
            self._translator.translate("extract.ex_stt_zh"), self._translator.translate("extract.ex_stt_ja"), self._translator.translate("extract.ex_stt_ko"),
        ])
        self._cb_stt_language.setToolTip(self._translator.translate("extract.ex_stt_lang_tip"))
        self._cb_stt_model = ComboBox()
        # [v3.23.363] large-v3-turbo: nhanh gấp ~6 lần large-v3, mất mát độ chính xác tối
        # thiểu, ~6GB VRAM (hợp GPU phổ thông) và VẪN đa ngôn ngữ (giữ CJK/tiếng Việt) —
        # lựa chọn cân bằng tốt nhất hiện nay cho trích phụ đề. large-v3 cho chất lượng tối
        # đa (nặng ~10GB VRAM, chậm hơn).
        self._cb_stt_model.addItems(
            ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
        )
        self._cb_stt_model.setCurrentText("large-v3-turbo")
        self._cb_stt_model.setToolTip(
            self._translator.translate("extract.ex_tt_model")
        )
        self._cb_stt_device = ComboBox()
        self._cb_stt_device.addItems([self._translator.translate("extract.ex_dev_auto"), "GPU (CUDA)", "CPU"])
        self._cb_stt_device.setToolTip(
            self._translator.translate("extract.ex_tt_stt_device")
        )
        config_row.addWidget(self._cb_stt_language, stretch=1)
        config_row.addWidget(self._cb_stt_model)
        config_row.addWidget(self._cb_stt_device)
        outer.addLayout(config_row)

        self._btn_transcribe = PushButton(self._translator.translate("extract.ex_btn_transcribe"))
        self._btn_transcribe.setEnabled(False)
        self._btn_transcribe.setStyleSheet(f"color: {_c.secondary()}; font-weight: bold;")
        outer.addWidget(self._btn_transcribe)

        self._chk_stt_align = CheckBox(self._translator.translate("extract.ex_chk_align"))
        self._chk_stt_align.setChecked(True)
        self._chk_stt_align.setToolTip(
            self._translator.translate("extract.ex_tt_align")
        )
        outer.addWidget(self._chk_stt_align)

        align_dev_row = QHBoxLayout()
        align_dev_row.addWidget(QLabel(self._translator.translate("extract.ex_align_device")))
        self._cb_stt_align_device = ComboBox()
        self._cb_stt_align_device.addItems([self._translator.translate("extract.ex_aligndev_cpu"), self._translator.translate("extract.ex_aligndev_gpu")])
        self._cb_stt_align_device.setToolTip(
            self._translator.translate("extract.ex_tt_aligndev")
        )
        align_dev_row.addWidget(self._cb_stt_align_device, stretch=1)
        outer.addLayout(align_dev_row)

        self._chk_stt_diarize = CheckBox(self._translator.translate("extract.ex_chk_diarize"))
        self._chk_stt_diarize.setChecked(False)
        self._chk_stt_diarize.setToolTip(
            self._translator.translate("extract.ex_tt_diarize")
        )
        outer.addWidget(self._chk_stt_diarize)

        token_row = QHBoxLayout()
        token_row.addWidget(QLabel("HuggingFace token:"))
        self._edit_hf_token = LineEdit()
        self._edit_hf_token.setPlaceholderText(self._translator.translate("extract.ex_hf_token_ph"))
        self._edit_hf_token.setEchoMode(LineEdit.EchoMode.Password)
        token_row.addWidget(self._edit_hf_token, stretch=1)
        outer.addLayout(token_row)

        self._chk_stt_split = CheckBox(self._translator.translate("extract.ex_chk_split"))
        self._chk_stt_split.setChecked(True)
        self._chk_stt_split.setToolTip(
            self._translator.translate("extract.ex_tt_split")
        )
        outer.addWidget(self._chk_stt_split)

        self._chk_stt_jieba = CheckBox(self._translator.translate("extract.ex_chk_jieba"))
        self._chk_stt_jieba.setChecked(True)
        self._chk_stt_jieba.setToolTip(
            self._translator.translate("extract.ex_tt_jieba")
        )
        outer.addWidget(self._chk_stt_jieba)

        len_row = QHBoxLayout()
        len_row.addWidget(QLabel(self._translator.translate("extract.ex_max_line_len")))
        self._cb_stt_max_chars = ComboBox()
        self._cb_stt_max_chars.addItems([self._translator.translate("extract.ex_mc_12"), self._translator.translate("extract.ex_mc_16"), self._translator.translate("extract.ex_mc_20"), self._translator.translate("extract.ex_mc_30")])
        self._cb_stt_max_chars.setCurrentIndex(1)
        self._cb_stt_max_chars.setToolTip(self._translator.translate("extract.ex_max_chars_tip"))
        len_row.addWidget(self._cb_stt_max_chars, stretch=1)
        outer.addLayout(len_row)

        self._lbl_stt_hint = QLabel("")
        self._lbl_stt_hint.setWordWrap(True)
        self._lbl_stt_hint.setStyleSheet(caption_style())
        outer.addWidget(self._lbl_stt_hint)

        # [v3.23.336] Nút cài WhisperX ngay trong ứng dụng. AN TOÀN vì cài vào môi
        # trường RIÊNG `whisperx_env` — khác hẳn nút cài cũ (đã gỡ ở v3.23.335) vốn
        # cài vào môi trường chính và làm hạ cấp huggingface-hub.
        self._btn_install_stt = PushButton(self._translator.translate("extract.ex_btn_install_stt"))
        self._btn_install_stt.setToolTip(
            self._translator.translate("extract.ex_tt_install")
        )
        self._btn_install_stt.clicked.connect(self._on_install_whisperx_clicked)
        outer.addWidget(self._btn_install_stt)

        self._stt_install_progress = QProgressBar()
        self._stt_install_progress.setRange(0, 100)
        self._stt_install_progress.setVisible(False)
        outer.addWidget(self._stt_install_progress)
        card.add_layout(outer)
        return card

    def _connect_signals(self) -> None:
        self._select_button.clicked.connect(self._on_select_clicked)
        self._start_button.clicked.connect(self._on_start_clicked)
        # [#7] Bấm Hủy đi qua handler để đặt cờ _is_cancelled_by_user.
        self._cancel_button.clicked.connect(self._on_cancel_clicked)

        # [#9] Tốc độ phát + click thẳng vào video để Phát/Dừng.
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self._canvas.video_clicked.connect(self._on_video_clicked)

        self._detect_hardsub_button.clicked.connect(self._on_detect_hardsub_clicked)
        self._detect_roi_button.clicked.connect(self._on_detect_roi_clicked)
        # [#5] Xoá ROI: đồng bộ ComboBox + tắt đèn nút ROI trong danh sách.
        self._clear_roi_button.clicked.connect(self._on_clear_roi_clicked)
        self._review_again_button.clicked.connect(self._view_model.review_auto_roi_again)
        self._view_model.has_last_analysis.connect(self._review_again_button.setVisible)

        # [v3.21] Phụ đề nhúng.
        self._btn_scan_embedded.clicked.connect(self._on_scan_embedded_clicked)
        self._btn_extract_embedded.clicked.connect(self._on_extract_embedded_clicked)
        self._view_model.embedded_tracks_listed.connect(self._on_embedded_tracks_listed)
        self._view_model.embedded_extract_finished.connect(self._on_embedded_extract_finished)
        self._view_model.embedded_failed.connect(self._on_embedded_failed)

        # [v3.23.168] Phụ đề rời cùng tên.
        self._btn_load_sidecar.clicked.connect(self._on_load_sidecar_clicked)
        self._view_model.sidecar_load_finished.connect(self._on_sidecar_load_finished)
        self._view_model.sidecar_load_failed.connect(self._on_embedded_failed)

        # [v3.21] STT WhisperX.
        self._btn_transcribe.clicked.connect(self._on_transcribe_clicked)
        self._view_model.transcribe_finished.connect(self._on_transcribe_finished)
        self._view_model.transcribe_failed.connect(self._on_transcribe_failed)
        self._view_model.transcribe_raw_ready.connect(self._on_transcribe_raw_ready)

        self._view_model.video_loaded.connect(self._on_video_loaded)
        self._view_model.cached_subtitles_found.connect(self._btn_load_cache.setVisible)
        self._view_model.progress_changed.connect(self._on_progress_changed)
        self._view_model.busy_changed.connect(self._on_busy_changed)
        self._view_model.extraction_finished.connect(self._on_extraction_finished)
        self._view_model.extraction_failed.connect(self._on_extraction_failed)

        self._view_model.raw_export_finished.connect(self._on_raw_export_finished)

        self._view_model.hardsub_detected.connect(self._on_hardsub_detected)
        self._view_model.auto_roi_detected.connect(self._on_auto_roi_detected)
        self._view_model.detection_failed.connect(self._on_detection_failed)
        self._view_model.roi_changed.connect(self._on_roi_changed)
        self._view_model.detected_rois_changed.connect(self._on_detected_rois_changed)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            k = event.key()
            mods = event.modifiers()

            focus_widget = QApplication.focusWidget()
            if focus_widget:
                cls_name = focus_widget.metaObject().className()
                input_classes = ["LineEdit", "TextEdit", "QTextEdit", "QLineEdit", "SpinBox", "TimeSpinBox", "QSpinBox", "QDoubleSpinBox", "ComboBox"]
                if any(name in cls_name for name in input_classes):
                    return super().eventFilter(obj, event)

            if k in (Qt.Key.Key_Space, Qt.Key.Key_K):
                self._canvas.toggle_play_pause()
                return True
            if k == Qt.Key.Key_Left:
                delta = -10.0 if (mods & Qt.KeyboardModifier.ShiftModifier) else -5.0
                self._seek_relative(delta)
                return True
            if k == Qt.Key.Key_Right:
                delta = 10.0 if (mods & Qt.KeyboardModifier.ShiftModifier) else 5.0
                self._seek_relative(delta)
                return True
            if k == Qt.Key.Key_J:
                self._seek_relative(-5.0)
                return True
            if k == Qt.Key.Key_L:
                self._seek_relative(5.0)
                return True

        return super().eventFilter(obj, event)

    @property
    def current_video(self) -> VideoMetadata | None:
        return self._view_model.video

    def _on_rebuild_clicked(self) -> None:
        """Dựng lại phụ đề từ cache OCR đã lưu (không chạy lại OCR)."""
        from PySide6.QtWidgets import QMessageBox

        if self._view_model.video is None:
            return
        if not self._view_model.has_cached_ocr():
            QMessageBox.information(
                self, self._translator.translate("extract.dlg_nocache_title"),
                self._translator.translate("extract.dlg_nocache_body"),
            )
            return
        self._extract_start_time = time.time()
        self._view_model.rebuild_from_cached_ocr()

    def _refresh_rebuild_button(self) -> None:
        """Chỉ bật nút dựng lại khi video hiện tại THỰC SỰ có cache OCR."""
        button = getattr(self, "_rebuild_button", None)
        if button is None:
            return
        try:
            button.setEnabled(self._view_model.has_cached_ocr())
        except Exception as exc:  # noqa: BLE001 — không được phá luồng nạp video
            logger.debug("Không kiểm được cache OCR: %s", exc)
            button.setEnabled(False)

    def _on_main_ocr_language_changed(self) -> None:
        """Áp ngôn ngữ OCR vừa chọn cho phiên làm việc hiện tại."""
        code = self._cb_main_ocr_language.currentData() or ""
        self._view_model.set_ocr_language_override(str(code))

    def _report_empty_extraction(self, response: ExtractSubtitlesResponse) -> None:
        """Báo rõ vì sao không ra câu nào, thay vì báo thành công nhầm."""
        from PySide6.QtWidgets import QMessageBox

        from subtitles_extractor.application.services.extraction_preflight import (
            diagnose_empty_result,
        )

        roi = self._view_model.roi
        message = diagnose_empty_result(
            frames_processed=getattr(response, "frames_processed", 0) or 0,
            roi=(roi.x, roi.y, roi.width, roi.height) if roi is not None else None,
            ocr_language=self._view_model.ocr_language_override
            or self._container.settings_service.current.ocr.language,
        )
        logger.warning("Trích xuất không ra câu nào: %s", message.replace("\n", " | "))

        if hasattr(self, "_status_label"):
            self._status_label.setStyleSheet(
                f"color: {_c.warning()}; font-weight: bold;"
            )
            self._status_label.setText(self._translator.translate("extract.st_no_lines"))
        if hasattr(self, "_btn_export_raw"):
            self._btn_export_raw.setVisible(True)

        box = QMessageBox(self)
        box.setWindowTitle(self._translator.translate("extract.dlg_nolines_title"))
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(message)
        box.exec()

    def _run_preflight(self) -> bool:
        """Kiểm điều kiện trước khi chạy. Trả ``True`` nếu được phép tiếp tục.

        [v3.23.325] Trước đây KHÔNG kiểm gì — ROI vẽ lệch ra ngoài khung hay tệp đã bị
        di chuyển đều chỉ lộ ra sau khi đã chờ xong. Nay chặn sớm.
        """
        from PySide6.QtWidgets import QMessageBox

        from subtitles_extractor.application.services.extraction_preflight import (
            check_before_extraction,
            has_blocker,
            summarise_issues,
        )

        video = self._view_model.video
        roi = self._view_model.roi
        issues = check_before_extraction(
            video_path=getattr(video, "path", None),
            video_size=(video.width, video.height) if video is not None else None,
            roi=(roi.x, roi.y, roi.width, roi.height) if roi is not None else None,
            duration_sec=getattr(video, "duration_sec", 0.0) or 0.0,
        )
        if not issues:
            return True

        summary = summarise_issues(issues)
        if has_blocker(issues):
            QMessageBox.warning(self, self._translator.translate("extract.dlg_cannot_title"), summary)
            return False

        answer = QMessageBox.question(
            self, self._translator.translate("extract.dlg_preflight_title"),
            self._translator.translate("extract.dlg_preflight_body").replace("{summary}", summary),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # ── Cài WhisperX ─────────────────────────────────────────────────────────
    def _on_install_whisperx_clicked(self) -> None:
        """Cài WhisperX vào môi trường riêng, chạy nền có báo tiến độ."""
        from PySide6.QtWidgets import QMessageBox

        from subtitles_extractor.infrastructure.stt.whisperx_installer import (
            ESTIMATED_DOWNLOAD_GB,
            find_system_python,
        )

        if getattr(self, "_stt_install_thread", None) is not None:
            return

        if find_system_python() is None:
            QMessageBox.warning(
                self, self._translator.translate("extract.dlg_nopython_title"),
                self._translator.translate("extract.dlg_nopython_body"),
            )
            return

        answer = QMessageBox.question(
            self, self._translator.translate("extract.dlg_installwx_title"),
            self._translator.translate("extract.dlg_installwx_body").replace("{gb}", f"{ESTIMATED_DOWNLOAD_GB:.0f}"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        from PySide6.QtCore import QThread

        from subtitles_extractor.presentation.workers.install_whisperx_worker import (
            InstallWhisperXWorker,
        )

        project_root = Path(__file__).resolve().parents[4]
        self._btn_install_stt.setEnabled(False)
        self._stt_install_progress.setValue(0)
        self._stt_install_progress.setVisible(True)

        self._stt_install_thread = QThread(self)
        self._stt_install_worker = InstallWhisperXWorker(project_root)
        self._stt_install_worker.moveToThread(self._stt_install_thread)

        # QueuedConnection: worker ở luồng khác, tín hiệu phải xếp hàng về luồng UI.
        self._stt_install_thread.started.connect(self._stt_install_worker.run)
        self._stt_install_worker.progress.connect(
            self._on_install_progress, Qt.ConnectionType.QueuedConnection
        )
        self._stt_install_worker.finished.connect(
            self._on_install_finished, Qt.ConnectionType.QueuedConnection
        )
        self._stt_install_worker.failed.connect(
            self._on_install_failed, Qt.ConnectionType.QueuedConnection
        )
        self._stt_install_worker.done.connect(
            self._cleanup_install_thread, Qt.ConnectionType.QueuedConnection
        )
        self._stt_install_thread.start()

    def _on_install_progress(self, percent: int, label: str) -> None:
        self._stt_install_progress.setValue(percent)
        self._lbl_stt_hint.setText(self._translator.translate("extract.st_installing").replace("{{label}}", str(label)))

    def _on_install_finished(self, env_python: str) -> None:
        """Cài xong: làm mới trạng thái để dùng được NGAY, không cần mở lại app."""
        from PySide6.QtWidgets import QMessageBox

        # Xoá cache khả dụng để lần kiểm sau nhận ra môi trường mới.
        if hasattr(self._view_model, "_stt_available_cache"):
            self._view_model._stt_available_cache = None  # noqa: SLF001
        self._refresh_stt_state()
        QMessageBox.information(
            self, self._translator.translate("extract.dlg_wxdone_title"),
            self._translator.translate("extract.dlg_wxdone_body").replace("{env}", str(env_python)),
        )

    def _on_install_failed(self, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        self._lbl_stt_hint.setText(self._translator.translate("extract.st_install_failed").replace("{{msg}}", str(message.splitlines()[0])))
        QMessageBox.warning(self, self._translator.translate("extract.dlg_wxfailed_title"), message)

    def _cleanup_install_thread(self) -> None:
        """Dừng và xoá luồng cài (kể cả khi lỗi)."""
        thread = getattr(self, "_stt_install_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
        self._stt_install_thread = None
        self._stt_install_worker = None
        self._btn_install_stt.setEnabled(True)
        self._stt_install_progress.setVisible(False)

    def _refresh_stt_state(self) -> None:
        """Cập nhật nhãn gợi ý + hiện/ẩn nút cài theo trạng thái WhisperX."""
        try:
            available = self._view_model.is_stt_available()
        except Exception as exc:  # noqa: BLE001 — không được phá trang
            logger.debug("Không kiểm được WhisperX: %s", exc)
            return
        if available:
            self._lbl_stt_hint.setText(
                "Phiên âm toàn bộ giọng nói trong video thành phụ đề."
            )
        else:
            self._lbl_stt_hint.setText(self._view_model.get_stt_diagnosis_hint())
        self._btn_install_stt.setVisible(not available)
        self._update_action_states()

    def _on_probe_clicked(self) -> None:
        """Chạy thử nhanh: bó OCR vào một đoạn ngắn rồi hiện kết quả để đối chiếu."""
        if self._view_model.video is None or not self._run_preflight():
            return
        self._canvas.pause()
        self._is_probing = True
        self._extract_start_time = time.time()
        ratio = self._cb_probe_position.currentData()
        self._view_model.start_probe_extraction(
            center_ratio=float(ratio) if ratio is not None else 0.5
        )

    def _show_probe_result(self, response: ExtractSubtitlesResponse) -> None:
        """Hiện tóm tắt kết quả thử nhanh (không chuyển tab, không lưu tệp)."""
        from PySide6.QtWidgets import QMessageBox

        from subtitles_extractor.application.services.ocr_probe_window import (
            compute_probe_window,
            summarise_probe_result,
        )

        window = getattr(self._view_model, "last_probe_window", None)
        if window is None:
            video = self._view_model.video
            window = compute_probe_window(video.duration_sec if video else 0.0)

        events = list(response.events or [])
        texts = [event.text for event in events if getattr(event, "text", "")]
        message = summarise_probe_result(len(events), window, texts)

        box = QMessageBox(self)
        box.setWindowTitle(self._translator.translate("extract.dlg_probe_title"))
        box.setIcon(
            QMessageBox.Icon.Information if events else QMessageBox.Icon.Warning
        )
        box.setText(message)
        box.exec()

    # ── Trích xuất hàng loạt ─────────────────────────────────────────────────
    def _on_concat_series_clicked(self) -> None:
        """[v3.23.367] Nối các video GỐC thành 1 tệp trọn bộ (để trích xuất sau)."""
        from subtitles_extractor.application.services.concat_plan import (
            default_concat_output,
            find_concat_videos,
        )

        start_dir = ""
        if self._pending_video_path:
            start_dir = str(self._pending_video_path.parent)
        folder = QFileDialog.getExistingDirectory(
            self, self._translator.translate("extract.dlg_choose_concat_dir"), start_dir
        )
        if not folder:
            return
        folder_path = Path(folder)

        # Nối các video GỐC — loại trừ tệp trọn bộ đầu ra nếu đã tồn tại từ lần trước.
        output_path = default_concat_output(folder_path, [])
        videos = find_concat_videos(folder_path, exclude_names={output_path.name})
        if len(videos) < 2:
            QMessageBox.information(
                self, "Không đủ tập để nối",
                "Cần ít nhất 2 tệp video trong thư mục để nối thành phim trọn bộ.",
            )
            return
        output_path = default_concat_output(folder_path, videos)

        preview = "\n".join(f"  {i + 1}. {v.name}" for i, v in enumerate(videos[:8]))
        if len(videos) > 8:
            preview += f"\n  … và {len(videos) - 8} tệp nữa"
        answer = QMessageBox.question(
            self, self._translator.translate("extract.dlg_concat_title"),
            self._translator.translate("extract.dlg_concat_body").replace("{n}", str(len(videos))).replace("{preview}", preview).replace("{output}", output_path.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start_concat(videos, output_path)

    def _start_concat(
        self, videos: list[Path], output_path: Path, *, reencode: bool = False
    ) -> None:
        """Khởi chạy worker nối video trên QThread riêng."""
        if getattr(self, "_concat_thread", None) is not None:
            return
        from subtitles_extractor.presentation.workers.concat_video_worker import (
            ConcatVideoWorker,
        )

        self._concat_videos = videos
        self._concat_output = output_path
        self._concat_button.setEnabled(False)
        self._batch_button.setEnabled(False)
        self._set_status(self._translator.translate("extract.ss_concat_start").replace("{n}", str(len(videos))))

        thread = QThread(self)
        worker = ConcatVideoWorker(videos, output_path, 0.0, reencode=reencode)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda p: self._set_status(self._translator.translate("extract.ss_concat_prog").replace("{p}", str(p)))
        )
        worker.finished.connect(self._on_concat_finished)
        worker.failed.connect(self._on_concat_failed)
        worker.done.connect(
            self._cleanup_concat_thread, Qt.ConnectionType.QueuedConnection
        )
        self._concat_thread = thread
        self._concat_worker = worker
        thread.start()

    def _on_concat_finished(self, output: object) -> None:
        out_path = Path(str(output))
        self._set_status(self._translator.translate("extract.ss_concat_done").replace("{name}", out_path.name))
        answer = QMessageBox.question(
            self, "Nối cả bộ thành công",
            f"Đã tạo phim trọn bộ:\n{out_path}\n\n"
            "Nạp video này để bắt đầu TRÍCH XUẤT ngay?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._load_video(out_path)

    def _on_concat_failed(self, message: str) -> None:
        # Sao chép luồng lỗi do lệch thông số → mời nén lại (chắc chắn hơn).
        if not getattr(self, "_concat_reencode_offered", False) and (
            "copy" in message.lower() or "codec" in message.lower()
            or "Invalid" in message or "not match" in message.lower()
        ):
            self._concat_reencode_offered = True
            answer = QMessageBox.question(
                self, self._translator.translate("extract.dlg_streamcopy_title"),
                self._translator.translate("extract.dlg_streamcopy_body"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                videos = getattr(self, "_concat_videos", [])
                output = getattr(self, "_concat_output", None)
                if videos and output is not None:
                    self._start_concat(videos, output, reencode=True)
                    return
        self._set_status(self._translator.translate("extract.ss_concat_fail"))
        QMessageBox.warning(self, self._translator.translate("extract.dlg_concatfail_title"), message)

    def _cleanup_concat_thread(self) -> None:
        """Dọn luồng nối video sau khi xong/lỗi."""
        thread = getattr(self, "_concat_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait(3000)
        self._concat_thread = None
        self._concat_worker = None
        self._concat_button.setEnabled(True)
        self._batch_button.setEnabled(True)

    def _on_batch_clicked(self) -> None:
        """Chọn nhiều tập rồi chạy tuần tự với cùng ROI/tham số."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from subtitles_extractor.application.services.batch_extraction_plan import (
            build_batch_plan,
            summarise_plan,
        )

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._translator.translate("extract.dialog_pick_episodes"),
            "",
            self._translator.translate("extract.dialog_filter_video"),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not paths:
            return

        video = self._view_model.video
        reference_roi = self._view_model.roi
        reference_size = (video.width, video.height) if video is not None else None
        if reference_roi is not None and reference_size is None:
            reference_roi = None  # có ROI mà không biết cỡ gốc -> không dám dùng lại

        plan = build_batch_plan(
            [Path(p) for p in paths],
            reference_roi=reference_roi,
            reference_size=reference_size,
            skip_existing=True,
        )
        runnable = [item for item in plan if item.will_run]
        if not runnable:
            QMessageBox.information(
                self, "Không có gì để chạy",
                f"{summarise_plan(plan)}\n\nTất cả tập đã có phụ đề hoặc không hợp lệ.",
            )
            return

        roi_note = (
            "Dùng ROI đang đặt cho mọi tập."
            if reference_roi is not None
            else "KHÔNG có ROI mẫu — hệ thống sẽ tự dò ROI cho từng tập."
        )
        confirm = QMessageBox.question(
            self, "Xác nhận trích xuất hàng loạt",
            f"{summarise_plan(plan)}\n\n{roi_note}\n\n"
            "Quá trình chạy tuần tự, có thể mất nhiều giờ. Tiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._batch_queue = runnable
        self._batch_index = 0
        self._batch_failures = []
        self._run_next_batch_item()

    def _run_next_batch_item(self) -> None:
        """Nạp và chạy tập kế tiếp trong hàng đợi; báo tổng kết khi hết."""
        if not getattr(self, "_batch_queue", None):
            return
        if self._batch_index >= len(self._batch_queue):
            self._finish_batch()
            return

        item = self._batch_queue[self._batch_index]
        total = len(self._batch_queue)
        logger.info(
            "Hàng loạt: tập %d/%d — %s", self._batch_index + 1, total, item.video_path.name
        )
        self._set_status(self._translator.translate("extract.ss_batch").replace("{i}", str(self._batch_index + 1)).replace("{total}", str(total)).replace("{name}", item.video_path.name))

        # Nạp video rồi áp ROI của kế hoạch (đã co giãn nếu cần) trước khi chạy.
        self._pending_batch_item = item
        self._view_model.load_video(item.video_path)

    def _start_batch_item_extraction(self) -> None:
        """Chạy trích xuất cho tập vừa nạp xong (gọi từ ``_on_video_loaded``)."""
        item = getattr(self, "_pending_batch_item", None)
        if item is None:
            return
        self._pending_batch_item = None
        if item.roi is not None:
            self._view_model.set_roi(item.roi)
        self._extract_start_time = time.time()
        self._view_model.start_extraction()

    def _advance_batch(self, *, failed_name: str | None = None) -> bool:
        """Chuyển sang tập kế tiếp. Trả ``True`` nếu đang chạy hàng loạt."""
        if not getattr(self, "_batch_queue", None):
            return False
        if failed_name:
            self._batch_failures.append(failed_name)
        # [v3.23.328] Người dùng đã bấm Huỷ -> dừng hẳn, KHÔNG chạy tập kế.
        if getattr(self, "_batch_cancelled", False):
            self._finish_batch()
            return True
        self._batch_index += 1
        # Dùng timer 0ms để trả quyền về vòng lặp sự kiện trước khi nạp tập kế tiếp,
        # tránh đệ quy sâu và giúp giao diện kịp vẽ lại.
        QTimer.singleShot(0, self._run_next_batch_item)
        return True

    def _finish_batch(self) -> None:
        """Báo tổng kết và dọn trạng thái hàng đợi."""
        from PySide6.QtWidgets import QMessageBox

        total = len(self._batch_queue)
        failures = list(self._batch_failures)
        cancelled = getattr(self, "_batch_cancelled", False)
        processed = self._batch_index + (0 if cancelled else 0)
        skipped = max(0, total - processed - (0 if cancelled else 0)) if cancelled else 0
        self._batch_queue = []
        self._batch_index = 0
        self._batch_failures = []
        self._batch_cancelled = False
        self._is_cancelled_by_user = False

        done = max(0, processed - len(failures))
        if cancelled:
            message = (
                f"Đã huỷ hàng loạt. Hoàn tất {done}/{total} tập, "
                f"bỏ {skipped} tập còn lại."
            )
        else:
            message = f"Đã trích xuất xong {done}/{total} tập."
        if failures:
            message += "\n\nThất bại:\n• " + "\n• ".join(failures[:10])
            if len(failures) > 10:
                message += f"\n… và {len(failures) - 10} tập nữa."
        QMessageBox.information(self, self._translator.translate("extract.dlg_batchdone_title"), message)
        self._set_status(message.splitlines()[0])

    def _set_status(self, text: str) -> None:
        """Cập nhật dòng trạng thái nếu trang có nhãn trạng thái."""
        label = getattr(self, "_status_label", None)
        if label is not None:
            label.setText(text)

    def _on_select_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            self._translator.translate("extract.dialog_select_video"),
            "",
            self._translator.translate("extract.dialog_filter_video"),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path_str:
            self._load_video(Path(path_str))

    def _load_video(self, path: Path) -> None:
        self._pending_video_path = path
        self._view_model.load_video(path)

    def _on_start_clicked(self) -> None:
        if self._view_model.video is None or not self._run_preflight():
            return

        if self._view_model.has_cached_subtitles:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(self._translator.translate("extract.dlg_found_title"))
            msg_box.setText(self._translator.translate("extract.dlg_found_text"))
            msg_box.setInformativeText(self._translator.translate("extract.dlg_found_info"))
            msg_box.setIcon(QMessageBox.Icon.Question)

            btn_open = msg_box.addButton(self._translator.translate("extract.dlg_found_open"), QMessageBox.ButtonRole.AcceptRole)
            msg_box.addButton(self._translator.translate("extract.dlg_found_redo"), QMessageBox.ButtonRole.DestructiveRole)
            btn_cancel = msg_box.addButton(self._translator.translate("extract.dlg_found_cancel"), QMessageBox.ButtonRole.RejectRole)

            msg_box.exec()

            if msg_box.clickedButton() == btn_open:
                self._view_model.load_cached_subtitles()
                return
            if msg_box.clickedButton() == btn_cancel:
                return

        self._btn_load_cache.setVisible(False)
        self._is_cancelled_by_user = False
        # [v3.23.319] Hàng đợi trích xuất hàng loạt.
        self._is_probing: bool = False
        self._stt_install_thread = None
        self._stt_install_worker = None
        self._batch_queue: list = []
        self._batch_cancelled: bool = False
        self._batch_index: int = 0
        self._batch_failures: list[str] = []
        self._pending_batch_item = None  # [#7] reset cờ cho lần chạy mới
        # [v3.20.3 #4] Reset bộ làm mượt ETA cho phiên trích xuất mới.
        self._eta_ema_fps = None
        self._eta_last_time = None
        self._eta_last_frame = None
        self._eta_display_sec = None
        self._canvas.pause()                # [#8] tắt tiếng video khi gọi AI
        self._extract_start_time = time.time()
        self._view_model.start_extraction()

    def _on_cancel_clicked(self) -> None:
        """[#7] Người dùng ép dừng: đặt cờ rồi yêu cầu huỷ. Cờ này khiến
        :meth:`_on_extraction_finished` hiển thị cảnh báo vàng thay vì báo hoàn tất.

        [v3.23.328] SỬA LỖI: khi đang chạy HÀNG LOẠT, bấm Huỷ trước đây chỉ dừng tập
        hiện tại rồi hàng đợi TỰ CHẠY TIẾP tập sau — người dùng không thoát ra được,
        phải tắt cả ứng dụng. Nay đánh dấu huỷ cả hàng đợi.
        """
        self._is_cancelled_by_user = True
        if getattr(self, "_batch_queue", None):
            self._batch_cancelled = True
            remaining = len(self._batch_queue) - self._batch_index - 1
            logger.info("Huỷ hàng loạt — bỏ %d tập còn lại.", max(0, remaining))
            self._set_status(self._translator.translate("extract.ss_batch_cancel").replace("{n}", str(max(0, remaining))))
        self._view_model.cancel_extraction()

    def _on_detect_hardsub_clicked(self) -> None:
        self._canvas.pause()  # [#8]
        self._view_model.detect_hardsub()

    def _on_detect_roi_clicked(self) -> None:
        self._canvas.pause()  # [#8]
        self._view_model.detect_auto_roi()

    def _on_clear_roi_clicked(self) -> None:
        """[#5] Xoá ROI: dọn ROI thực, đồng bộ ComboBox về 'Toàn khung' và tắt đèn
        mọi nút ROI trong danh sách phát hiện (để UI khớp trạng thái thực)."""
        self._view_model.set_roi(None)
        self._sync_preset_combo_to("full")
        if hasattr(self, "_draw_toggle"):
            self._draw_toggle.blockSignals(True)
            self._draw_toggle.setChecked(False)
            self._draw_toggle.blockSignals(False)
        self._canvas.enable_roi_drawing(False)
        # Tắt đèn các nút ROI đã phát hiện.
        self._on_detected_rois_changed(self._view_model.detected_rois)

    def _on_preset_changed(self) -> None:
        if self._video_metadata is None:
            return
        preset_key = self._preset_combo.currentData()
        if preset_key == "custom":
            self._draw_toggle.setEnabled(True)
            return

        # [#6] Không còn ép tắt nút vẽ ở preset khác — nút vẽ luôn dùng được.
        self._draw_toggle.setChecked(False)

        if preset_key == "auto_subtitle":
            # [Phần 6] Tự nhận diện vùng phụ đề chính: chạy phân tích rồi tự chọn
            # cụm đậm đặc nhất làm ROI duy nhất (không mở dialog kiểm duyệt).
            self._canvas.pause()  # [#8] tắt tiếng video khi gọi AI
            self._view_model.detect_primary_subtitle_roi()
            return

        roi = _preset_roi(preset_key, self._video_metadata.width, self._video_metadata.height)
        self._view_model.set_roi(roi)

    def _sync_preset_combo_to(self, preset_key: str) -> None:
        """[#5] Ép ComboBox preset về đúng trạng thái thực tế MÀ KHÔNG kích hoạt
        lại detection (chặn tín hiệu). Dùng khi vẽ tay / xoá ROI / AI thất bại."""
        if not hasattr(self, "_preset_combo"):
            return
        index = self._preset_combo.findData(preset_key)
        if index < 0:
            return
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(index)
        self._preset_combo.blockSignals(False)

    def _on_draw_toggle_toggled(self, checked: bool) -> None:
        """[#6] Bật/tắt chế độ vẽ ROI. Khi bắt đầu vẽ, tự gạt ComboBox sang
        'Tuỳ chỉnh' để trạng thái UI khớp thao tác (không kích hoạt detection)."""
        if checked:
            self._sync_preset_combo_to("custom")
            self._draw_toggle.setEnabled(True)
        self._canvas.enable_roi_drawing(checked)

    def _on_speed_changed(self, _index: int) -> None:
        """[#9] Áp tốc độ phát đã chọn vào canvas/player."""
        speed = self._speed_combo.currentData()
        if speed is not None:
            self._canvas.set_playback_speed(float(speed))

    def _on_video_clicked(self, _button: object) -> None:
        """[#9] Click thẳng vào video để Phát/Dừng (như YouTube). Bỏ qua khi đang
        ở chế độ vẽ ROI để không xung đột với thao tác kéo vẽ."""
        if getattr(self._draw_toggle, "isChecked", lambda: False)():
            return
        self._canvas.toggle_play_pause()

    def _on_roi_drawn(self, roi: Roi | None) -> None:
        # [V3.13.1 CRITICAL FIX] Chấp nhận cả giá trị `None` để xóa đồng bộ ROI
        self._view_model.set_roi(roi)
        # [#5] Vẽ tay tạo ROI → ComboBox phải khớp trạng thái "Tuỳ chỉnh".
        if roi is not None:
            self._sync_preset_combo_to("custom")

    def _on_roi_preview(self, roi: Roi | None) -> None:
        pass

    def _on_canvas_state_changed(self, is_playing: bool) -> None:
        if hasattr(self, '_btn_play_pause'):
            self._btn_play_pause.setIcon(
                (FluentIcon.PAUSE if is_playing else FluentIcon.PLAY).icon()
            )

    def _on_canvas_position_changed(self, pos: float) -> None:
        if self._video_metadata is None:
            return
        dur = self._video_metadata.duration_sec
        if dur > 0:
            val = int((pos / dur) * _SLIDER_RESOLUTION)
            if hasattr(self, '_position_slider'):
                self._updating_slider_from_playback = True
                self._position_slider.setValue(val)
                self._updating_slider_from_playback = False
            if hasattr(self, '_position_label'):
                self._position_label.setText(f"{_fmt_mmss(pos)} / {_fmt_mmss(dur)}")
        self._current_position_sec = pos

    def _on_slider_pressed(self) -> None:
        if self._canvas.is_playing:
            self._was_playing_before_seek = True
            self._canvas.pause()
        else:
            self._was_playing_before_seek = False

    def _on_slider_released(self) -> None:
        if self._was_playing_before_seek:
            self._canvas.play()

    def _on_slider_value_changed(self, value: int) -> None:
        if self._updating_slider_from_playback:
            return
        if self._video_metadata is None:
            return
        dur = self._video_metadata.duration_sec
        if dur > 0:
            ts = (value / float(_SLIDER_RESOLUTION)) * dur
            self._canvas.seek(ts)

    def _seek_relative(self, offset_sec: float) -> None:
        if self._video_metadata is None: return
        new_ts = max(0.0, min(self._current_position_sec + offset_sec, self._video_metadata.duration_sec))
        self._canvas.seek(new_ts)

    def _on_seek_fallback_requested(self, ts: float) -> None:
        self._pending_seek_sec = ts
        self._seek_debounce_timer.start()

    def _do_seek(self) -> None:
        """Seek đến vị trí đang chờ.

        Nếu MPV player đang hoạt động, dùng ``mpv_player.seek()`` trực tiếp
        để tránh overhead decode frame (fast-path). Nếu không có MPV, dùng
        SeekWorker để decode frame bằng CPU và hiển thị qua ``set_frame``.
        """
        if self._pending_seek_sec is None:
            return
        if self._seek_thread is not None and self._seek_thread.isRunning():
            # [Stuck Preview] Luồng đọc ảnh đang bận → KHÔNG bỏ lệnh, mà đặt lịch
            # chạy lại sau 50ms. Đảm bảo khung hình cuối cùng (khi buông chuột) luôn
            # được load, không kẹt ở mốc cũ khi kéo thanh trượt quá nhanh.
            QTimer.singleShot(50, self._do_seek)
            return
        timestamp_sec = self._pending_seek_sec
        self._pending_seek_sec = None

        # Fast-path: MPV đang hoạt động → seek trực tiếp, không cần decode frame
        mpv_player = self._canvas.player()
        if mpv_player is not None:
            mpv_player.seek(timestamp_sec)
            return

        if self._pending_video_path is None:
            return
        thread = QThread(self)
        worker = SeekWorker(self._pending_video_path, timestamp_sec, reader=self._video_reader, sequential=False)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.frame_ready.connect(self._on_seek_frame_ready)
        worker.failed.connect(lambda msg: logger.debug("Seek thất bại: {}.", msg))
        worker.frame_ready.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_seek_thread_finished)

        self._seek_thread = thread
        self._seek_worker = worker
        thread.start()

    def _on_seek_thread_finished(self) -> None:
        self._seek_thread = None
        self._seek_worker = None
        if self._pending_seek_sec is not None: self._do_seek()

    def _on_seek_frame_ready(self, img: QImage, w: int, h: int) -> None:
        """Hiển thị frame sau khi SeekWorker decode xong.

        Khi MPV player không khả dụng (player() is None — fallback VideoCanvas),
        gọi set_frame trực tiếp để vẽ lên canvas. Khi MPV đang chạy, set_frame
        là no-op và MPV tự render qua native window.
        """
        self._canvas.set_video_size(w, h)
        if self._canvas.player() is None:
            self._canvas.set_frame(img, w, h)
        if self.current_roi:
            self._canvas.set_committed_roi(self.current_roi)

    def _on_video_loaded(self, metadata: VideoMetadata) -> None:
        # [v3.23.319] Đang chạy hàng loạt: nạp xong là chạy ngay tập này.
        if getattr(self, "_pending_batch_item", None) is not None:
            QTimer.singleShot(0, self._start_batch_item_extraction)
        # [v3.23.322] Video mới -> cập nhật trạng thái nút "Dựng lại từ OCR đã lưu".
        QTimer.singleShot(0, self._refresh_rebuild_button)
        self._video_metadata = metadata
        if hasattr(self, '_video_label'):
            self._video_label.setText(
                self._translator.translate("extract.video_info",
                    filename=metadata.filename, width=metadata.width,
                    height=metadata.height, fps=f"{metadata.fps:.2f}",
                    duration=metadata.duration_str)
            )
            self._video_label.setStyleSheet(f"color: {_c.on_surface()};")

        if hasattr(self, '_start_button'): self._start_button.setEnabled(True)
        if hasattr(self, '_detect_hardsub_button'): self._detect_hardsub_button.setEnabled(True)
        if hasattr(self, '_detect_roi_button'): self._detect_roi_button.setEnabled(True)
        if hasattr(self, '_btn_export_raw'): self._btn_export_raw.setVisible(True)
        # [v3.21] Cho phép quét phụ đề nhúng; reset combo của video cũ.
        if hasattr(self, '_btn_scan_embedded'):
            self._btn_scan_embedded.setEnabled(True)
            self._cb_embedded_track.clear()
            self._cb_embedded_track.setEnabled(False)
            self._btn_extract_embedded.setEnabled(False)
            self._embedded_tracks = []
            self._lbl_embedded_hint.setText("")
        # [v3.23.168] Tự dò phụ đề rời cùng tên cạnh video, đổ vào combo.
        if hasattr(self, "_cb_sidecar_subtitle"):
            self._populate_sidecar_combo()
        # [v3.21] Bật STT nếu WhisperX khả dụng; nếu không thì chẩn đoán chi tiết.
        if hasattr(self, '_btn_transcribe'):
            stt_ok = self._view_model.is_stt_available()
            self._btn_transcribe.setEnabled(stt_ok)
            if stt_ok:
                self._lbl_stt_hint.setText(self._translator.translate("extract.st_transcribe_hint"))
            else:
                self._lbl_stt_hint.setText(self._view_model.get_stt_diagnosis_hint())

        path = self._pending_video_path or metadata.path
        # [v3.23.366] Nếu reader ĐÃ mở đúng file này thì GIỮ NGUYÊN — tránh mở lại PyAV
        # thừa (quan sát log batch: cùng 1 video bị mở lại tới 56 lần, phí I/O).
        if (
            self._video_reader is not None
            and getattr(self._video_reader, "_video_path", None) == Path(path)
        ):
            self._current_position_sec = 0.0
            self._canvas.load(path)
        else:
            if self._video_reader is not None:
                self._video_reader.close()
            self._video_reader = PersistentVideoReader(Path(path))
            self._current_position_sec = 0.0
            self._canvas.load(path)

        # [V3.13.1 CRITICAL FIX]: Ép Video Canvas ghi nhận ngay Kích thước Video
        # Tránh việc Widget tự Hủy vẽ ROI do thông số Video C-Core load chậm trả về 0
        self._canvas.set_video_size(metadata.width, metadata.height)

        self._canvas.seek(0.0)

        snapshot = self._container.settings_service.current

        # [#3 v3.18] CHỐNG RACE CONDITION cháy VRAM: tại một thời điểm chỉ cho phép
        # DUY NHẤT một luồng AI khởi chạy. Dùng if/elif loại trừ lẫn nhau giữa:
        #   (a) ROI tuỳ chỉnh có sẵn  → không chạy AI;
        #   (b) preset mặc định = self._translator.translate("extract.ex_stt_auto") → 1 luồng detect_primary_subtitle_roi;
        #   (c) bật "tự phát hiện khi mở video" → 1 luồng detect_auto_roi;
        #   (d) còn lại → preset tĩnh, không AI.
        if hasattr(self, '_preset_combo'):
            default_preset = snapshot.roi.default_preset

            if self._view_model.roi is not None:
                self._sync_preset_combo_to("custom")
                if hasattr(self, '_draw_toggle'):
                    self._draw_toggle.setEnabled(True)

            elif default_preset == "auto_subtitle":
                self._sync_preset_combo_to("auto_subtitle")
                self._canvas.pause()  # [#8]
                self._view_model.detect_primary_subtitle_roi()  # AI (1 luồng)

            elif snapshot.roi.auto_detect_on_load:
                idx = self._preset_combo.findData(default_preset)
                if idx >= 0:
                    self._sync_preset_combo_to(default_preset)
                self._canvas.pause()  # [#8]
                QTimer.singleShot(300, self._view_model.detect_auto_roi)  # AI (1 luồng)

            else:
                self._sync_preset_combo_to(default_preset)
                roi = _preset_roi(default_preset, metadata.width, metadata.height)
                self._view_model.set_roi(roi)

    def _on_progress_changed(self, current: int, total: int, message: str) -> None:
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setVisible(True)
            if total > 0:
                self._progress_bar.setRange(0, total)
                self._progress_bar.setValue(current)
            else:
                self._progress_bar.setRange(0, 0)

        if hasattr(self, '_status_label') and message:
            if "Đã quét OCR" in message and total > 0 and current > 10:
                # [v3.20.3 #4] ETA mượt bằng EMA (Exponential Moving Average) trên
                # TỐC ĐỘ TỨC THỜI thay vì tốc độ trung bình toàn cục. Tốc độ trung
                # bình bị CPU lag kéo lệch → ETA nhảy loạn (2 phút → 5 tiếng). EMA
                # làm trơn dao động; ETA hiển thị trôi lùi đều đặn.
                now = time.time()
                last_t = getattr(self, "_eta_last_time", None)
                last_frame = getattr(self, "_eta_last_frame", None)
                if last_t is not None and last_frame is not None and current > last_frame:
                    dt = max(1e-3, now - last_t)
                    instant_fps = (current - last_frame) / dt
                    prev_ema = getattr(self, "_eta_ema_fps", instant_fps)
                    # alpha thấp → mượt hơn (ít nhạy với gai lag).
                    ema_fps = 0.2 * instant_fps + 0.8 * prev_ema
                else:
                    elapsed_time = now - getattr(self, "_extract_start_time", now)
                    ema_fps = max(0.001, current / max(1.0, elapsed_time))
                self._eta_ema_fps = max(0.001, ema_fps)
                self._eta_last_time = now
                self._eta_last_frame = current

                estimated_seconds_left = (total - current) / self._eta_ema_fps
                # Monotonic: ETA chỉ giảm (hoặc giữ), không nhảy vọt lên — trừ khi
                # lệch quá lớn (>20%) thì mới cho cập nhật tăng.
                prev_eta = getattr(self, "_eta_display_sec", None)
                if prev_eta is not None and estimated_seconds_left > prev_eta * 1.2:
                    estimated_seconds_left = prev_eta
                self._eta_display_sec = estimated_seconds_left

                message += f" (Còn khoảng ~{_fmt_mmss(estimated_seconds_left)})"

            self._status_label.setText(message)

    def _on_busy_changed(self, busy: bool) -> None:
        has_video = self._view_model.video is not None
        if hasattr(self, '_select_button'): self._select_button.setEnabled(not busy)
        if hasattr(self, '_batch_button'): self._batch_button.setEnabled(not busy)
        if hasattr(self, '_probe_button'): self._probe_button.setEnabled(not busy)
        if hasattr(self, '_cb_main_ocr_language'): self._cb_main_ocr_language.setEnabled(not busy)
        if hasattr(self, '_cb_probe_position'): self._cb_probe_position.setEnabled(not busy)
        if hasattr(self, '_rebuild_button'): self._rebuild_button.setEnabled((not busy) and self._view_model.has_cached_ocr())
        if hasattr(self, '_start_button'): self._start_button.setEnabled(not busy and has_video)
        if hasattr(self, '_cancel_button'): self._cancel_button.setEnabled(busy)
        if hasattr(self, '_detect_hardsub_button'): self._detect_hardsub_button.setEnabled(not busy and has_video)
        if hasattr(self, '_detect_roi_button'): self._detect_roi_button.setEnabled(not busy and has_video)
        # [v3.22] Khoá các nút nguồn MỚI khi đang chạy tác vụ nền (chống chạy chồng
        # nhiều thread cùng lúc gây tranh chấp GPU/IO).
        if hasattr(self, '_btn_scan_embedded'):
            self._btn_scan_embedded.setEnabled(not busy and has_video)
        if hasattr(self, '_btn_extract_embedded'):
            # Chỉ mở lại nếu đã có track được quét.
            has_tracks = bool(getattr(self, '_embedded_tracks', []))
            self._btn_extract_embedded.setEnabled(not busy and has_tracks)
        if hasattr(self, '_btn_transcribe'):
            stt_ok = not busy and has_video and self._view_model.is_stt_available()
            self._btn_transcribe.setEnabled(stt_ok)

        if not busy and hasattr(self, '_progress_bar'):
            self._progress_bar.setVisible(False)
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(0)

    def _on_extraction_finished(self, response: ExtractSubtitlesResponse) -> None:
        # [v3.23.320] Thử nhanh: chỉ báo kết quả, KHÔNG nạp sang Biên tập/lưu dự án.
        if getattr(self, "_is_probing", False):
            self._is_probing = False
            self._show_probe_result(response)
            return
        # [v3.23.319] Đang chạy hàng loạt -> sang tập kế, KHÔNG chuyển tab/hiện hộp thoại.
        if getattr(self, "_batch_queue", None):
            # [v3.23.364] SỬA BUG NGHIÊM TRỌNG: trước đây batch nhảy tập kế mà KHÔNG lưu
            # gì → trích 84 tập xong nhưng trang Dịch/TTS quét đĩa thấy "chưa trích xuất".
            # Nay lưu phụ đề gốc từng tập (ghi .original.srt + lưu CSDL) trước khi tiếp.
            self._persist_batch_extraction(response)
            self._advance_batch()
            return
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)

        # [#7] Nếu người dùng đã ép dừng giữa chừng: cảnh báo vàng, KHÔNG báo hoàn
        # tất, KHÔNG tự chuyển sang tab Editor.
        if self._is_cancelled_by_user:
            self._is_cancelled_by_user = False
            if hasattr(self, '_status_label'):
                self._status_label.setStyleSheet(f"color: {_c.warning()}; font-weight: bold;")
                self._status_label.setText(self._translator.translate("extract.st_cancelled"))
            if hasattr(self, '_btn_export_raw'):
                self._btn_export_raw.setVisible(True)
            return

        # [v3.23.325] Ra 0 câu KHÔNG phải thành công. Trước đây vẫn báo xanh
        # "✓ Hoàn tất! 0 câu → Đã lưu vào Database" — vừa sai (không lưu gì) vừa khiến
        # người dùng tưởng phim không có phụ đề, trong khi thường là do ROI/ngôn ngữ.
        if not response.events:
            self._report_empty_extraction(response)
            return

        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.success()};")
            self._status_label.setText(
                f"✓ Hoàn tất! {len(response.events)} câu trong {response.elapsed_seconds:.1f}s → Đã lưu vào Database"
            )

        if hasattr(self, '_btn_export_raw'): self._btn_export_raw.setVisible(True)

        snapshot = self._container.settings_service.current
        if snapshot.ui.show_ocr_overlay and response.events:
            boxes: list[tuple[Any, Any, Any, Any]] = []
            for event in response.events:
                if hasattr(event, "bounding_box") and event.bounding_box is not None:
                    x_min, y_min, x_max, y_max = event.bounding_box
                    boxes.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))
            if boxes:
                self._canvas.set_ocr_overlay(boxes, visible=True)
        else:
            self._canvas.clear_ocr_overlay()

        if self.extraction_completed is not None:
            self.extraction_completed.emit(response.events)

    def _persist_batch_extraction(self, response: ExtractSubtitlesResponse) -> None:
        """[v3.23.364] Lưu phụ đề gốc của MỘT tập vừa trích trong chế độ hàng loạt.

        Ghi tệp ``<tên>.original.srt`` cạnh video (khâu Dịch/TTS/Xuất bản hàng loạt quét
        ĐĨA tìm tệp này) và lưu vào CSDL (để hiện trong Thư viện ở khâu "Đã trích xuất").
        Không ném lỗi ra ngoài: một tập ghi hỏng chỉ log cảnh báo, không chặn cả hàng đợi.
        """
        if not response.events:
            return
        queue = getattr(self, "_batch_queue", None)
        if not queue:
            return
        index = min(self._batch_index, len(queue) - 1)
        video_path = queue[index].video_path

        from subtitles_extractor.domain.value_objects.output_naming import (
            extracted_subtitle_path,
        )
        from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import (
            SrtExporter,
        )

        # 1) Ghi tệp .original.srt cạnh video — điều kiện để các khâu hàng loạt sau nhận ra.
        srt_text = SrtExporter._build_content(response.events)
        try:
            SrtExporter().export(response.events, extracted_subtitle_path(video_path))
        except (OSError, ValueError) as exc:
            logger.warning("Hàng loạt: không ghi được .original.srt cho %s: %s",
                           video_path.name, exc)

        # 2) Lưu CSDL (thư viện) theo hash video — để mở lại và hiện đúng khâu.
        try:
            from subtitles_extractor.domain.entities.project_record import (
                ProjectRecord, WorkflowStage,
            )
            from subtitles_extractor.infrastructure.video.video_hasher import (
                compute_video_hash,
            )

            video_hash = compute_video_hash(str(video_path))
            repo = self._container.project_repository
            record = repo.get(video_hash) or ProjectRecord(video_hash=video_hash)
            record.video_path = str(video_path)
            record.video_name = video_path.name
            record.original_subtitle = srt_text
            record.subtitle_format = "srt"
            if record.stage < WorkflowStage.EXTRACTED:
                record.stage = WorkflowStage.EXTRACTED
            repo.save(record)
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("Hàng loạt: không lưu CSDL cho %s: %s", video_path.name, exc)

    def _on_extraction_failed(self, message: str) -> None:
        if getattr(self, "_is_probing", False):
            self._is_probing = False
        # [v3.23.319] Hàng loạt: ghi nhận tập lỗi rồi chạy tiếp, không dừng cả hàng đợi.
        if getattr(self, "_batch_queue", None):
            index = min(self._batch_index, len(self._batch_queue) - 1)
            failed = self._batch_queue[index].video_path.name
            logger.warning("Hàng loạt: tập %s lỗi — %s", failed, message)
            self._advance_batch(failed_name=failed)
            return
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(0)
        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.danger()};")
            self._status_label.setText(self._translator.translate("extract.status_error", message=message))

        InfoBar.error(
            title="Lỗi Trích xuất",
            content=message,
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=6000
        )

    def _on_scan_embedded_clicked(self) -> None:
        self._btn_scan_embedded.setEnabled(False)
        self._lbl_embedded_hint.setText(self._translator.translate("extract.st_scanning"))
        self._view_model.list_embedded_tracks()

    def _on_embedded_tracks_listed(self, tracks: list) -> None:
        self._btn_scan_embedded.setEnabled(True)
        self._embedded_tracks = tracks
        self._cb_embedded_track.clear()
        if not tracks:
            self._cb_embedded_track.setEnabled(False)
            self._btn_extract_embedded.setEnabled(False)
            self._lbl_embedded_hint.setText(self._translator.translate("extract.st_no_embedded"))
            InfoBar.info(
                "Không có phụ đề nhúng",
                "File video không chứa track phụ đề tích hợp.",
                parent=self, position=InfoBarPosition.TOP_RIGHT, duration=4000,
            )
            return
        for track in tracks:
            self._cb_embedded_track.addItem(track.display_label)
        self._cb_embedded_track.setEnabled(True)
        self._btn_extract_embedded.setEnabled(True)
        has_bitmap = any(getattr(t, "is_bitmap", False) for t in tracks)
        hint = f"Tìm thấy {len(tracks)} track phụ đề nhúng."
        if has_bitmap:
            hint += " Track dạng ảnh sẽ được OCR bằng PaddleOCR (lâu hơn)."
        self._lbl_embedded_hint.setText(hint)

    def _on_extract_embedded_clicked(self) -> None:
        index = self._cb_embedded_track.currentIndex()
        tracks = getattr(self, "_embedded_tracks", [])
        if not (0 <= index < len(tracks)):
            return
        track = tracks[index]
        self._is_cancelled_by_user = False
        self._eta_ema_fps = None
        self._eta_last_time = None
        self._eta_last_frame = None
        self._eta_display_sec = None
        self._btn_extract_embedded.setEnabled(False)
        self._btn_scan_embedded.setEnabled(False)
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 0)  # busy indefinite cho tới khi có progress
        kind = "OCR ảnh phụ đề" if getattr(track, "is_bitmap", False) else "trích phụ đề văn bản"
        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.info()};")
            self._status_label.setText(self._translator.translate("extract.st_extracting_embedded").replace("{{kind}}", str(kind)))
        ocr_language = ""
        if hasattr(self, "_ocr_language_codes"):
            lang_index = self._cb_ocr_language.currentIndex()
            if 0 <= lang_index < len(self._ocr_language_codes):
                ocr_language = self._ocr_language_codes[lang_index]
        self._view_model.start_embedded_extraction(track, ocr_language)

    def _on_embedded_extract_finished(self, events: list) -> None:
        self._btn_extract_embedded.setEnabled(True)
        self._btn_scan_embedded.setEnabled(True)
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)
        if not events:
            if hasattr(self, '_status_label'):
                self._status_label.setStyleSheet(f"color: {_c.warning()}; font-weight: bold;")
                self._status_label.setText(self._translator.translate("extract.st_embedded_empty"))
            return
        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.success()};")
            self._status_label.setText(self._translator.translate("extract.st_embedded_done").replace("{{n}}", str(len(events))))
        InfoBar.success(
            "Trích phụ đề nhúng xong",
            f"Đã lấy {len(events)} câu. Đang chuyển sang trang Chỉnh sửa…",
            parent=self, position=InfoBarPosition.TOP_RIGHT, duration=3500,
        )
        # Đổ sang editor đúng cơ chế chung của trang.
        if self.extraction_completed is not None:
            self.extraction_completed.emit(events)

    def _on_embedded_failed(self, message: str) -> None:
        self._btn_extract_embedded.setEnabled(True)
        self._btn_scan_embedded.setEnabled(True)
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(0)
        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.danger()};")
            self._status_label.setText(f"✗ {message}")
        InfoBar.error(
            "Lỗi phụ đề nhúng", message,
            parent=self, position=InfoBarPosition.TOP_RIGHT, duration=6000,
        )

    # ── [v3.23.168] Phụ đề rời cùng tên ──────────────────────────────────────
    def _populate_sidecar_combo(self) -> None:
        """Dò phụ đề rời cạnh video, đổ vào combo và bật/tắt nút nạp tương ứng."""
        self._sidecar_candidates = self._view_model.find_sidecar_subtitles()
        self._cb_sidecar_subtitle.clear()
        has_any = bool(self._sidecar_candidates)
        if has_any:
            for item in self._sidecar_candidates:
                label = item.path.name
                if item.language_tag:
                    label = f"{item.path.name}  ·  [{item.language_tag}]"
                self._cb_sidecar_subtitle.addItem(label)
        else:
            self._cb_sidecar_subtitle.addItem(self._translator.translate("extract.ex_no_sidecar"))
        self._cb_sidecar_subtitle.setEnabled(has_any)
        self._btn_load_sidecar.setEnabled(has_any)

    def _on_load_sidecar_clicked(self) -> None:
        candidates = getattr(self, "_sidecar_candidates", [])
        index = self._cb_sidecar_subtitle.currentIndex()
        if not candidates or not (0 <= index < len(candidates)):
            return
        self._btn_load_sidecar.setEnabled(False)
        self._view_model.load_sidecar_subtitle(candidates[index].path)

    def _on_sidecar_load_finished(self, events: list) -> None:
        self._btn_load_sidecar.setEnabled(True)
        if not events:
            InfoBar.warning(
                "Phụ đề rời rỗng", "Tệp không có câu phụ đề nào đọc được.",
                parent=self, position=InfoBarPosition.TOP_RIGHT, duration=4000,
            )
            return
        InfoBar.success(
            "Nạp phụ đề rời xong",
            f"Đã lấy {len(events)} câu. Đang chuyển sang trang Chỉnh sửa…",
            parent=self, position=InfoBarPosition.TOP_RIGHT, duration=3500,
        )
        if self.extraction_completed is not None:
            self.extraction_completed.emit(events)

    def _on_transcribe_raw_ready(self, raw_segments: list, language: str, model_size: str) -> None:
        # [v3.22.7] Tự lưu dữ liệu STT thô (.sestt.json) cạnh video để hiệu chuẩn
        # thuật toán tách câu offline.
        try:
            from subtitles_extractor import __version__
            from subtitles_extractor.infrastructure.serializers.raw_stt_serializer import (
                save_raw_stt,
            )

            video = self._view_model.video
            if video is None:
                return
            from pathlib import Path

            video_path = Path(video.path)
            output_path = video_path.with_suffix(".sestt.json")
            save_raw_stt(
                output_path, raw_segments, str(video_path), language, model_size, __version__
            )
            self._last_stt_raw_path = output_path
            InfoBar.success(
                "Đã xuất dữ liệu thô STT",
                f"Lưu {output_path.name} cạnh video (để hiệu chuẩn thuật toán tách câu).",
                parent=self, position=InfoBarPosition.TOP_RIGHT, duration=5000,
            )
        except (OSError, ImportError, ValueError) as exc:
            InfoBar.warning(
                "Không xuất được dữ liệu thô STT", str(exc),
                parent=self, position=InfoBarPosition.TOP_RIGHT, duration=4000,
            )

    def _on_transcribe_clicked(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        lang_map = {0: "", 1: "vi", 2: "en", 3: "zh", 4: "ja", 5: "ko"}
        language = lang_map.get(self._cb_stt_language.currentIndex(), "")
        device_map = {0: "cuda", 1: "cuda", 2: "cpu"}  # self._translator.translate("extract.ex_dev_auto") cũng thử cuda; adapter tự fallback
        device = device_map.get(self._cb_stt_device.currentIndex(), "cuda")
        max_chars_map = {0: 12, 1: 16, 2: 20, 3: 30}
        max_chars = max_chars_map.get(self._cb_stt_max_chars.currentIndex(), 16)
        align_device = "cpu" if self._cb_stt_align_device.currentIndex() == 0 else "cuda"
        config = TranscriptionConfig(
            language=language,
            model_size=self._cb_stt_model.currentText(),
            device=device,
            enable_align=self._chk_stt_align.isChecked(),
            align_device=align_device,
            enable_diarize=self._chk_stt_diarize.isChecked(),
            hf_token=self._edit_hf_token.text().strip(),
            enable_sentence_split=self._chk_stt_split.isChecked(),
            max_chars_per_cue=max_chars,
            use_word_segmentation=self._chk_stt_jieba.isChecked(),
        )
        self._is_cancelled_by_user = False
        self._btn_transcribe.setEnabled(False)
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.secondary()};")
            self._status_label.setText(self._translator.translate("extract.st_transcribing"))
        self._view_model.start_transcription(config)

    def _on_transcribe_finished(self, events: list) -> None:
        self._btn_transcribe.setEnabled(True)
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)
        if not events:
            if hasattr(self, '_status_label'):
                self._status_label.setStyleSheet(f"color: {_c.warning()}; font-weight: bold;")
                self._status_label.setText(self._translator.translate("extract.st_no_speech"))
            return
        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.success()};")
            self._status_label.setText(self._translator.translate("extract.st_speech_done").replace("{{n}}", str(len(events))))
        InfoBar.success(
            "Phiên âm xong",
            f"Đã nhận dạng {len(events)} câu. Đang chuyển sang trang Chỉnh sửa…",
            parent=self, position=InfoBarPosition.TOP_RIGHT, duration=3500,
        )
        if self.extraction_completed is not None:
            self.extraction_completed.emit(events)

    def _on_transcribe_failed(self, message: str) -> None:
        # [v3.23.340] Ghi log ở tầng giao diện nữa: có đường lỗi không đi qua worker.
        logger.error("Phiên âm thất bại: %s", message.replace("\n", " | "))
        self._btn_transcribe.setEnabled(True)
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(0)
        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.danger()};")
            self._status_label.setText(f"✗ {message}")
        InfoBar.error(
            "Lỗi phiên âm", message,
            parent=self, position=InfoBarPosition.TOP_RIGHT, duration=6000,
        )

    def _on_export_raw_clicked(self) -> None:
        if not self._view_model.video_path:
            InfoBar.warning("Thiếu Video", "Vui lòng trích xuất phụ đề trước khi xuất dữ liệu thô.", parent=self, duration=4000)
            return

        reply = QMessageBox.question(
            self, "Lưu kèm hình ảnh OCR?",
            "Bạn có muốn xuất kèm hình ảnh (Input và Output) không?\n⚠️ Lưu ý: Tốn khá nhiều dung lượng ổ đĩa.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        export_images = (reply == QMessageBox.StandardButton.Yes)

        video_name = Path(self._view_model.video_path).stem
        default_name = f"{video_name}.seraw.json"
        default_dir = Path(self._view_model.video_path).parent

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu dữ liệu OCR thô", str(default_dir / default_name), "SubtitleExtractor Raw OCR (*.seraw.json *.seraw.json.gz);;Tất cả (*.*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not save_path: return

        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {_c.info()};")
            self._status_label.setText(self._translator.translate("extract.st_raw_running"))
        if hasattr(self, '_btn_export_raw'): self._btn_export_raw.setEnabled(False)

        self._extract_start_time = time.time()
        self._view_model.start_extraction_with_raw_export(raw_ocr_output_path=Path(save_path), export_images=export_images)

    def _on_raw_export_finished(self, raw_path: Path | None, response: ExtractSubtitlesResponse) -> None:
        if hasattr(self, '_btn_export_raw'): self._btn_export_raw.setEnabled(True)

        if hasattr(self, '_progress_bar'):
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)

        if raw_path is None or not raw_path.exists():
            InfoBar.error("Xuất thất bại", "Không lưu được file dữ liệu thô.", parent=self, duration=5000)
            if hasattr(self, '_status_label'):
                self._status_label.setStyleSheet(f"color: {_c.danger()};")
                self._status_label.setText(self._translator.translate("extract.st_raw_failed"))
            return

        size_mb = raw_path.stat().st_size / (1024 * 1024)
        msg = f"Đã lưu dữ liệu OCR thô:\n\n  {raw_path.name}  ({size_mb:.2f} MB)\n\nNhấn OK để nạp dữ liệu và chuyển sang trang Chỉnh sửa."

        QMessageBox.information(self, self._translator.translate("extract.dlg_rawok_title"), msg)
        self._on_extraction_finished(response)

    def _on_hardsub_detected(self, result: HardsubDetectionResult) -> None:
        if result.has_hardsub:
            QMessageBox.information(
                self, self._translator.translate("extract.hardsub_dialog_title"),
                self._translator.translate("extract.hardsub_yes", confidence=f"{result.confidence:.0%}", reason=result.reason)
            )
        else:
            QMessageBox.warning(
                self, self._translator.translate("extract.hardsub_dialog_title"),
                self._translator.translate("extract.hardsub_no", confidence=f"{result.confidence:.0%}", reason=result.reason)
            )

    def _on_auto_roi_detected(self, roi: Roi | list[Roi] | str | None) -> None:
        if roi == "CANCELED": return
        if roi is None:
            # [#5] AI không tìm thấy vùng chữ → đồng bộ ComboBox về self._translator.translate("extract.ex_roi_full")
            # thay vì kẹt ở self._translator.translate("extract.ex_stt_auto") (trạng thái ảo).
            self._sync_preset_combo_to("full")
            QMessageBox.warning(self, self._translator.translate("extract.roi_dialog_title"), self._translator.translate("extract.roi_not_found"))

    def _on_detected_rois_changed(self, rois: list[Roi]) -> None:
        if hasattr(self, '_roi_buttons_layout'):
            while self._roi_buttons_layout.count():
                item = self._roi_buttons_layout.takeAt(0)
                if item and item.widget(): item.widget().deleteLater()

        if not rois or not isinstance(rois, list):
            if hasattr(self, '_detected_rois_group'): self._detected_rois_group.setVisible(False)
            self._canvas.set_secondary_rois([])
            return

        active_roi = self._view_model.roi
        secondary = [r for r in rois if r != active_roi]
        self._canvas.set_secondary_rois(secondary)

        for idx, roi in enumerate(rois):
            is_active = (roi is active_roi) or (
                active_roi is not None and
                roi.x == active_roi.x and roi.y == active_roi.y and
                roi.width == active_roi.width and roi.height == active_roi.height
            )
            label = f"{'▶ ' if is_active else ''}ROI #{idx + 1}: ({roi.x},{roi.y}) {roi.width}×{roi.height}"
            btn = PushButton(label)
            btn.setCheckable(True)
            btn.setChecked(is_active)
            if is_active:
                # Dùng màu nhấn của theme (tự thích ứng dark/light) cho ROI đang chọn
                accent = themeColor().name()
                btn.setStyleSheet(
                    f"text-align: left; padding: 4px 8px; "
                    f"background-color: {accent}; color: white; font-weight: 600;"
                )
            else:
                # Inactive: chỉ căn lề + padding, để qfluentwidgets tự tô theo theme
                btn.setStyleSheet("text-align: left; padding: 4px 8px;")
            btn.clicked.connect(lambda checked, i=idx: self._on_roi_button_clicked(i))
            if hasattr(self, '_roi_buttons_layout'): self._roi_buttons_layout.addWidget(btn)

        if hasattr(self, '_detected_rois_group'): self._detected_rois_group.setVisible(True)

    def _on_roi_button_clicked(self, roi_index: int) -> None:
        self._view_model.select_detected_roi(roi_index)
        self._on_detected_rois_changed(self._view_model.detected_rois)

    def _on_detection_failed(self, message: str) -> None:
        # [#5] AI báo lỗi → đồng bộ ComboBox về self._translator.translate("extract.ex_roi_full") (tránh trạng thái ảo).
        self._sync_preset_combo_to("full")
        InfoBar.error(
            title=self._translator.translate("extract.detection_error_title"),
            content=message,
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=5000
        )

    def _on_roi_changed(self, roi: Roi | None) -> None:
        self._canvas.set_committed_roi(roi)
        detected = self._view_model.detected_rois
        if detected:
            secondary = [r for r in detected if r != roi]
            self._canvas.set_secondary_rois(secondary)

        if roi is None:
            if hasattr(self, '_roi_label'):
                self._roi_label.setText(self._translator.translate("extract.no_roi"))
                self._roi_label.setStyleSheet(f"color: {_c.muted_italic()}; font-style: italic;")
            if hasattr(self, '_clear_roi_button'): self._clear_roi_button.setEnabled(False)
        else:
            if hasattr(self, '_roi_label'):
                self._roi_label.setText(self._translator.translate("extract.roi_info", x=roi.x, y=roi.y, w=roi.width, h=roi.height))
                self._roi_label.setStyleSheet(f"color: {_c.success()};")
            if hasattr(self, '_clear_roi_button'): self._clear_roi_button.setEnabled(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_file(): self._load_video(path)

    def cancel_extraction(self) -> None:
        if hasattr(self, "_view_model"): self._view_model.cancel_extraction()

    def closeEvent(self, event: QCloseEvent) -> None:
        with contextlib.suppress(RuntimeError): self._seek_debounce_timer.stop()

        if self._seek_thread is not None:
            try:
                if self._seek_worker is not None:
                    with contextlib.suppress(RuntimeError):
                        self._seek_worker.frame_ready.disconnect()
                        self._seek_worker.failed.disconnect()
                if self._seek_thread.isRunning():
                    self._seek_thread.requestInterruption()
                    self._seek_thread.quit()
                    self._seek_thread.wait(2000)
            except RuntimeError: pass
            finally:
                self._seek_thread = None
                self._seek_worker = None

        if self._video_reader is not None:
            self._video_reader.close()
            self._video_reader = None

        self.cancel_extraction()
        if hasattr(self, '_view_model'):
            # [BUG FIX v2.9+]: Tăng timeout lên 8s và thêm terminate fallback.
            # TRƯỚC: wait(2000) — GPU batch dễ vượt 2s → app exit với GPU đang chạy.
            # SAU: wait(8000) + terminate() nếu vẫn còn chạy để đảm bảo cleanup sạch.
            _WAIT_MS = 8_000

            extract_thread = getattr(self._view_model, 'active_extract_thread', None)
            if extract_thread and extract_thread.isRunning():
                logger.info("Chờ luồng Extract kết thúc an toàn (tối đa %ds)...", _WAIT_MS // 1000)
                if not extract_thread.wait(_WAIT_MS):
                    logger.warning("Extract thread timeout — buộc terminate để tránh crash.")
                    extract_thread.terminate()
                    extract_thread.wait(2000)

            detect_thread = getattr(self._view_model, 'active_detect_thread', None)
            if detect_thread and detect_thread.isRunning():
                logger.info("Chờ luồng Detect kết thúc an toàn...")
                if not detect_thread.wait(4000):
                    detect_thread.terminate()
                    detect_thread.wait(1000)

        with contextlib.suppress(AttributeError, RuntimeError): self._canvas.release_player()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._container.preload_ocr_engine_async()

__all__ = ["ExtractPage"]
