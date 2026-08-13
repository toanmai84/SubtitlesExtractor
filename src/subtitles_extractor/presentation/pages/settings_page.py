"""Trang "Cài đặt" — 9 tabs đầy đủ.

CẢI TIẾN ĐỘT PHÁ (V3.15 - The Control Center Polish):
    * [UX POLISH] Smart Dirty State: Các nút Lưu/Hủy tự động vô hiệu hóa nếu
      chưa có thay đổi nào. Giao diện báo cáo ngay khi người dùng chỉnh sửa.
    * [SAFETY] Factory Reset Guard: Hộp thoại xác nhận chống ấn nhầm khôi phục mặc định.
    * [FEATURE] Thêm Nút Dọn dẹp Bộ nhớ đệm (Cache Cleaner) vào tab Nâng cao.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)
from subtitles_extractor.presentation.fluent_compat import InfoBar, InfoBarPosition, themeColor
from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme.styles import caption_style
from subtitles_extractor.presentation.widgets.section_card import SectionCard

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.value_objects.device_kind import (
    DeviceKind,
    PrecisionMode,
    SubtitleFormat,
)
from subtitles_extractor.infrastructure.settings.application_settings import (
    ApplicationSettings,
)
from subtitles_extractor.presentation.view_models.settings_page_view_model import (
    SettingsPageViewModel,
)
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets

logger = logging.getLogger(__name__)

class SettingsPage(QWidget):
    def __init__(
        self, container: ApplicationContainer, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsPage")
        self._translator = container.translator
        self._view_model = SettingsPageViewModel(container, parent=self)

        # [V3.15 UX] Cờ khóa vòng lặp Event khi đang đổ dữ liệu vào Form
        self._is_populating = False

        self._build_ui()
        protect_scroll_widgets(self)

        self._connect_signals()

        # [V3.15 UX] Theo dõi tương tác của toàn bộ Form
        self._monitor_form_changes()

        self._view_model.reload()

    @property
    def view_model(self) -> SettingsPageViewModel:
        return self._view_model

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        title = QLabel(self._translator.translate("settings.title"))
        title.setStyleSheet(f"font-size: {_m.FONT_SIZE_HEADING}px; font-weight: 600;")
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab_ocr_hardware(), self._translator.translate("settings.tab_ocr_hw"))
        self._tabs.addTab(self._build_tab_nlp(), self._translator.translate("settings.tab_nlp"))
        self._tabs.addTab(self._build_tab_mpv(), self._translator.translate("settings.tab_mpv"))
        self._tabs.addTab(self._build_tab_roi(), self._translator.translate("settings.tab_roi"))
        self._tabs.addTab(self._build_tab_threshold(), self._translator.translate("settings.tab_threshold"))
        self._tabs.addTab(self._build_tab_frame(), self._translator.translate("settings.tab_frame"))
        self._tabs.addTab(self._build_tab_post(), self._translator.translate("settings.tab_post"))
        self._tabs.addTab(self._build_tab_preprocess(), self._translator.translate("settings.tab_preprocess"))
        self._tabs.addTab(self._build_tab_video_translation(), self._translator.translate("settings.tab_video_translation"))
        self._tabs.addTab(self._build_tab_ui(), self._translator.translate("settings.tab_ui"))
        self._tabs.addTab(self._build_tab_advanced(), self._translator.translate("settings.tab_advanced"))
        root.addWidget(self._tabs, stretch=1)

        actions = QHBoxLayout()
        self._save_button = QPushButton(self._translator.translate("settings.btn_save"))
        self._save_button.setStyleSheet(
            f"QPushButton {{ background-color: {themeColor().name()}; color: white; "
            f"padding: 6px 14px; font-weight: 600; border-radius: 4px; }}"
            "QPushButton:disabled { background-color: #6b7280; color: #d1d5db; }"
        )
        self._cancel_button = QPushButton(self._translator.translate("settings.btn_cancel_changes"))
        self._reset_button = QPushButton(self._translator.translate("settings.btn_reset"))
        self._reset_button.setStyleSheet(f"color: {_c.danger()}; font-weight: bold;")

        actions.addWidget(self._save_button)
        actions.addWidget(self._cancel_button)
        actions.addStretch(1)
        actions.addWidget(self._reset_button)
        root.addLayout(actions)

        self._status_label = QLabel("")
        root.addWidget(self._status_label)

    def _create_scroll_container(self, parent: QWidget | None = None) -> tuple[QWidget, QVBoxLayout, QScrollArea]:
        scroll_area = QScrollArea(parent)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 8, 0)
        scroll_layout.setSpacing(12)

        return scroll_widget, scroll_layout, scroll_area

    def _build_tab_ocr_hardware(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        ocr_group = SectionCard(self._translator.translate("settings.sec_engine"))
        ocr_form = QFormLayout()
        ocr_group.add_layout(ocr_form)

        self._cb_ocr_version = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_ocrv6_medium"), "PP-OCRv6_medium"),
            (self._translator.translate("settings.co_ocrv6_small"), "PP-OCRv6_small"),
            (self._translator.translate("settings.co_ocrv6_tiny"), "PP-OCRv6_tiny"),
            (self._translator.translate("settings.co_ocrv5_server"), "PP-OCRv5_server"),
            (self._translator.translate("settings.co_ocrv5_mobile"), "PP-OCRv5_mobile"),
            ("PP-OCRv4", "PP-OCRv4"),
        ]:
            self._cb_ocr_version.addItem(label, userData=key)
        self._cb_ocr_version.setToolTip(
            self._translator.translate("settings.tip_ocr_model")
        )
        ocr_form.addRow(self._translator.translate("settings.rl_ocr_version"), self._cb_ocr_version)

        self._cb_ocr_language = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_lang_vi"), "vi"), (self._translator.translate("settings.co_lang_en"), "en"),
            (self._translator.translate("settings.co_lang_ch"), "ch"), (self._translator.translate("settings.co_lang_ja"), "japan"),
            (self._translator.translate("settings.co_lang_ko"), "korean"), (self._translator.translate("settings.co_lang_latin"), "latin"),
        ]:
            self._cb_ocr_language.addItem(label, userData=key)
        ocr_form.addRow(self._translator.translate("settings.rl_language"), self._cb_ocr_language)

        self._sp_score_threshold = _make_double_spin(0.0, 1.0, 0.05, 2)
        ocr_form.addRow(self._translator.translate("settings.rl_ocr_score"), self._sp_score_threshold)
        scroll_layout.addWidget(ocr_group)

        adv_ocr_group = SectionCard(self._translator.translate("settings.sec_detection"))
        adv_ocr_form = QFormLayout()
        adv_ocr_group.add_layout(adv_ocr_form)

        self._cb_limit_type = QComboBox()
        self._cb_limit_type.addItems(["max", "min"])
        adv_ocr_form.addRow(self._translator.translate("settings.rl_limit_type"), self._cb_limit_type)

        self._sp_limit_side = QSpinBox()
        self._sp_limit_side.setRange(0, 4096)
        self._sp_limit_side.setSingleStep(32)
        self._sp_limit_side.setToolTip(self._translator.translate("settings.tip_limit_side"))
        adv_ocr_form.addRow(self._translator.translate("settings.rl_limit_side"), self._sp_limit_side)

        self._sp_det_thresh = _make_double_spin(0.1, 1.0, 0.05, 2)
        adv_ocr_form.addRow(self._translator.translate("settings.rl_det_thresh"), self._sp_det_thresh)

        self._sp_det_box_thresh = _make_double_spin(0.1, 1.0, 0.05, 2)
        adv_ocr_form.addRow(self._translator.translate("settings.rl_det_box_thresh"), self._sp_det_box_thresh)

        self._sp_det_unclip_ratio = _make_double_spin(0.5, 3.0, 0.1, 2)
        adv_ocr_form.addRow(self._translator.translate("settings.rl_unclip"), self._sp_det_unclip_ratio)

        self._chk_textline_orient = QCheckBox(self._translator.translate("settings.cb_rotate_text"))
        adv_ocr_form.addRow(self._translator.translate("settings.rl_line_dir"), self._chk_textline_orient)

        self._chk_doc_orient = QCheckBox(self._translator.translate("settings.cb_doc_orient"))
        adv_ocr_form.addRow(self._translator.translate("settings.rl_doc_reverse"), self._chk_doc_orient)

        self._chk_doc_unwarp = QCheckBox(self._translator.translate("settings.cb_doc_unwarp"))
        adv_ocr_form.addRow(self._translator.translate("settings.rl_dewarp"), self._chk_doc_unwarp)

        info_limit = QLabel(
            self._translator.translate("settings.tip_limit_hint")
        )
        info_limit.setWordWrap(True)
        info_limit.setStyleSheet(caption_style(_c.warning()))
        adv_ocr_form.addRow("", info_limit)

        scroll_layout.addWidget(adv_ocr_group)

        hw_group = SectionCard(self._translator.translate("settings.sec_hardware"))
        hw_form = QFormLayout()
        hw_group.add_layout(hw_form)

        self._cb_device = QComboBox()
        for d in DeviceKind:
            self._cb_device.addItem(d.value.upper(), userData=d.value)
        hw_form.addRow(self._translator.translate("settings.rl_ocr_device"), self._cb_device)

        self._cb_precision = QComboBox()
        for p in PrecisionMode:
            self._cb_precision.addItem(p.value.upper(), userData=p.value)
        hw_form.addRow(self._translator.translate("settings.rl_precision"), self._cb_precision)

        self._sp_batch_ocr = QSpinBox(); self._sp_batch_ocr.setRange(1, 512)
        hw_form.addRow(self._translator.translate("settings.rl_batch_ocr"), self._sp_batch_ocr)

        self._sp_batch_roi = QSpinBox(); self._sp_batch_roi.setRange(1, 128)
        hw_form.addRow(self._translator.translate("settings.rl_batch_det"), self._sp_batch_roi)

        self._sp_workers = QSpinBox(); self._sp_workers.setRange(1, 16)
        hw_form.addRow(self._translator.translate("settings.rl_frame_workers"), self._sp_workers)

        self._chk_auto_tune = QCheckBox(self._translator.translate("settings.cb_autotune_batch"))
        hw_form.addRow(self._translator.translate("settings.rl_autotune"), self._chk_auto_tune)

        self._chk_mkldnn = QCheckBox(self._translator.translate("settings.cb_mkldnn"))
        hw_form.addRow(self._translator.translate("settings.rl_mkldnn"), self._chk_mkldnn)

        self._chk_tensorrt = QCheckBox(self._translator.translate("settings.cb_tensorrt"))
        hw_form.addRow(self._translator.translate("settings.rl_tensorrt"), self._chk_tensorrt)

        # [v3.23.375] Bản build nhỏ KHÔNG nhúng CUDA (~2.3GB) — tải lúc chạy để bật GPU OCR.
        from PySide6.QtWidgets import QHBoxLayout

        gpu_row = QHBoxLayout()
        self._btn_enable_gpu_ocr = QPushButton(self._translator.translate("settings.gpu_ocr_btn"))
        self._btn_enable_gpu_ocr.setToolTip(self._translator.translate("settings.gpu_ocr_tip"))
        self._btn_enable_gpu_ocr.clicked.connect(self._on_enable_gpu_ocr_clicked)
        self._lbl_gpu_ocr_status = QLabel("")
        self._lbl_gpu_ocr_status.setWordWrap(True)
        gpu_row.addWidget(self._btn_enable_gpu_ocr)
        gpu_row.addWidget(self._lbl_gpu_ocr_status, 1)
        hw_form.addRow(self._translator.translate("settings.gpu_ocr_row"), gpu_row)
        self._refresh_gpu_ocr_status()

        # [v3.23.386] Bản build nhỏ (one-file) KHÔNG nhúng lõi paddlepaddle-gpu (~810MB) —
        # tải lúc chạy để chạy được OCR. Nút riêng, cùng mẫu với nút CUDA ở trên.
        paddle_row = QHBoxLayout()
        self._btn_download_paddle = QPushButton(
            self._translator.translate("settings.paddle_btn")
        )
        self._btn_download_paddle.setToolTip(self._translator.translate("settings.paddle_tip"))
        self._btn_download_paddle.clicked.connect(self._on_download_paddle_clicked)
        self._lbl_paddle_status = QLabel("")
        self._lbl_paddle_status.setWordWrap(True)
        paddle_row.addWidget(self._btn_download_paddle)
        paddle_row.addWidget(self._lbl_paddle_status, 1)
        hw_form.addRow(self._translator.translate("settings.paddle_row"), paddle_row)
        self._refresh_paddle_status()

        scroll_layout.addWidget(hw_group)

        decoder_group = SectionCard(self._translator.translate("settings.sec_decode_backend"))
        decoder_form = QFormLayout()
        decoder_group.add_layout(decoder_form)

        self._cb_frame_backend = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_fb_opencv"), "opencv"),
            ("Mpv (HW cross-platform: D3D11VA/Vulkan/NVDEC/VAAPI/VT)", "mpv"),
            (self._translator.translate("settings.co_fb_pynv"), "pynvvideocodec"),
        ]:
            self._cb_frame_backend.addItem(label, userData=key)
        decoder_form.addRow(self._translator.translate("settings.rl_frame_sampler"), self._cb_frame_backend)

        self._cb_metadata_backend = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_mb_opencv"), "opencv"),
            (self._translator.translate("settings.co_mb_pyav"), "pyav"),
            (self._translator.translate("settings.co_mb_mpv"), "mpv"),
        ]:
            self._cb_metadata_backend.addItem(label, userData=key)
        decoder_form.addRow(self._translator.translate("settings.rl_read_meta"), self._cb_metadata_backend)

        scroll_layout.addWidget(decoder_group)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_nlp(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        group = SectionCard(self._translator.translate("settings.sec_embeddings"))
        form = QFormLayout()
        group.add_layout(form)

        self._chk_enable_nlp = QCheckBox(self._translator.translate("settings.cb_semantic"))
        form.addRow(self._translator.translate("settings.rl_feature"), self._chk_enable_nlp)

        self._cb_nlp_model = QComboBox()
        self._cb_nlp_model.setEditable(True)

        from subtitles_extractor.infrastructure.nlp.fastembed_adapter import (
            FastEmbedAdapter,
        )
        safe_models = FastEmbedAdapter.get_supported_models()

        for m in safe_models:
            self._cb_nlp_model.addItem(m, userData=m)

        form.addRow(self._translator.translate("settings.rl_embed_model"), self._cb_nlp_model)

        self._cb_nlp_mode = QComboBox()
        self._cb_nlp_mode.addItem(self._translator.translate("settings.nlp_hybrid"), userData="hybrid")
        self._cb_nlp_mode.addItem(self._translator.translate("settings.nlp_semantic"), userData="semantic")
        form.addRow(self._translator.translate("settings.rl_match_mode"), self._cb_nlp_mode)

        self._chk_enable_nlp.toggled.connect(self._cb_nlp_model.setEnabled)
        self._chk_enable_nlp.toggled.connect(self._cb_nlp_mode.setEnabled)

        scroll_layout.addWidget(group)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_mpv(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        dll_group = SectionCard(self._translator.translate("settings.sec_libmpv"))
        dll_layout = QVBoxLayout()
        dll_group.add_layout(dll_layout)

        self._lbl_mpv_status = QLabel(self._translator.translate("settings.lbl_checking"))
        self._lbl_mpv_status.setWordWrap(True)
        dll_layout.addWidget(self._lbl_mpv_status)

        dll_btn_row = QHBoxLayout()
        self._btn_install_mpv = QPushButton(self._translator.translate("settings.btn_install_mpv"))
        self._btn_install_mpv.clicked.connect(self._on_install_mpv_clicked)
        dll_btn_row.addWidget(self._btn_install_mpv)

        self._btn_remove_mpv = QPushButton(self._translator.translate("settings.btn_remove_mpv"))
        self._btn_remove_mpv.clicked.connect(self._on_remove_mpv_clicked)
        dll_btn_row.addWidget(self._btn_remove_mpv)
        dll_btn_row.addStretch(1)
        dll_layout.addLayout(dll_btn_row)

        from PySide6.QtWidgets import QProgressBar
        self._mpv_progress = QProgressBar()
        self._mpv_progress.setVisible(False)
        dll_layout.addWidget(self._mpv_progress)
        scroll_layout.addWidget(dll_group)
        self._refresh_mpv_dll_status()

        hwdec_group = SectionCard(self._translator.translate("settings.sec_hw_decode"))
        hwdec_form = QFormLayout()
        hwdec_group.add_layout(hwdec_form)

        self._cb_hwdec_mode = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_hw_autosafe"), "auto-safe"),
            (self._translator.translate("settings.co_hw_auto"), "auto"),
            (self._translator.translate("settings.co_hw_autocopy"), "auto-copy"),
            (self._translator.translate("settings.co_hw_no"), "no"),
            ("d3d11va — Windows DirectX 11", "d3d11va"),
            (self._translator.translate("settings.co_hw_d3d11vacopy"), "d3d11va-copy"),
            ("dxva2 — Windows DirectX 9", "dxva2"),
            ("dxva2-copy", "dxva2-copy"),
            ("nvdec — NVIDIA NVDEC", "nvdec"),
            (self._translator.translate("settings.co_hw_nvdeccopy"), "nvdec-copy"),
            ("cuda — NVIDIA CUDA", "cuda"),
            ("cuda-copy", "cuda-copy"),
            ("vaapi — Linux Intel/AMD", "vaapi"),
            ("vaapi-copy", "vaapi-copy"),
            ("vulkan — Cross-platform Vulkan", "vulkan"),
            ("videotoolbox — macOS", "videotoolbox"),
            ("videotoolbox-copy", "videotoolbox-copy"),
        ]:
            self._cb_hwdec_mode.addItem(label, userData=key)
        hwdec_form.addRow(self._translator.translate("settings.rl_hw_decode_mode"), self._cb_hwdec_mode)

        self._le_hwdec_codecs = QLineEdit()
        self._le_hwdec_codecs.setPlaceholderText("h264,vc1,hevc,vp8,vp9,av1,mpeg2video,mpeg4")
        hwdec_form.addRow(self._translator.translate("settings.rl_hw_codecs"), self._le_hwdec_codecs)
        scroll_layout.addWidget(hwdec_group)

        vo_group = SectionCard(self._translator.translate("settings.sec_video_output"))
        vo_form = QFormLayout()
        vo_group.add_layout(vo_form)

        self._cb_video_output = QComboBox()
        for label, key in[
            ("auto", "auto"),
            (self._translator.translate("settings.co_vo_gpu"), "gpu"),
            ("gpu-next — Renderer next-gen (libplacebo)", "gpu-next"),
            ("libmpv — Embed-friendly", "libmpv"),
            ("opengl — Legacy OpenGL", "opengl"),
            ("d3d11 — Windows native", "d3d11"),
            ("vulkan — Vulkan native", "vulkan"),
            ("direct3d — DirectX 9", "direct3d"),
            (self._translator.translate("settings.co_vo_null"), "null"),
            ("drm — Linux DRM", "drm"),
        ]:
            self._cb_video_output.addItem(label, userData=key)
        vo_form.addRow(self._translator.translate("settings.rl_vo"), self._cb_video_output)

        self._cb_gpu_api = QComboBox()
        for label, key in[
            ("auto", "auto"),
            ("opengl — OpenGL (cross-platform)", "opengl"),
            ("vulkan — Vulkan (low overhead)", "vulkan"),
            ("d3d11 — Direct3D 11 (Windows native)", "d3d11"),
        ]:
            self._cb_gpu_api.addItem(label, userData=key)
        vo_form.addRow(self._translator.translate("settings.rl_gpu_api"), self._cb_gpu_api)

        self._le_gpu_context = QLineEdit()
        self._le_gpu_context.setPlaceholderText("auto / win / angle / x11 / wayland / macos")
        vo_form.addRow(self._translator.translate("settings.rl_gpu_context"), self._le_gpu_context)
        scroll_layout.addWidget(vo_group)

        perf_group = SectionCard(self._translator.translate("settings.sec_cache_perf"))
        perf_form = QFormLayout()
        perf_group.add_layout(perf_form)

        self._cb_cache = QComboBox()
        for key in["yes", "no", "auto"]:
            self._cb_cache.addItem(key, userData=key)
        perf_form.addRow(self._translator.translate("settings.rl_cache"), self._cb_cache)

        self._sp_cache_secs = QSpinBox()
        self._sp_cache_secs.setRange(0, 600)
        self._sp_cache_secs.setSuffix(" s")
        perf_form.addRow(self._translator.translate("settings.rl_cache_ahead"), self._sp_cache_secs)

        self._sp_demuxer_max = QSpinBox()
        self._sp_demuxer_max.setRange(10, 2048)
        self._sp_demuxer_max.setSuffix(" MB")
        perf_form.addRow(self._translator.translate("settings.rl_demuxer_max"), self._sp_demuxer_max)

        self._cb_video_sync = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_vs_audio"), "audio"),
            (self._translator.translate("settings.co_vs_resample"), "display-resample"),
            (self._translator.translate("settings.co_vs_vdrop"), "display-vdrop"),
            ("display-adrop — Drop audio", "display-adrop"),
            (self._translator.translate("settings.co_vs_desync"), "desync"),
        ]:
            self._cb_video_sync.addItem(label, userData=key)
        perf_form.addRow(self._translator.translate("settings.rl_video_sync"), self._cb_video_sync)
        scroll_layout.addWidget(perf_group)

        quality_group = SectionCard(self._translator.translate("settings.sec_quality"))
        quality_form = QFormLayout()
        quality_group.add_layout(quality_form)

        self._cb_profile = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_pf_default"), "default"),
            (self._translator.translate("settings.co_pf_fast"), "fast"),
            (self._translator.translate("settings.co_pf_gpuhq"), "gpu-hq"),
            (self._translator.translate("settings.co_pf_hq"), "high-quality"),
            ("low-latency — Streaming low-latency", "low-latency"),
        ]:
            self._cb_profile.addItem(label, userData=key)
        quality_form.addRow(self._translator.translate("settings.rl_profile"), self._cb_profile)

        self._chk_deinterlace = QCheckBox(self._translator.translate("settings.cb_deinterlace"))
        quality_form.addRow(self._translator.translate("settings.rl_deinterlace"), self._chk_deinterlace)
        scroll_layout.addWidget(quality_group)

        log_group = SectionCard(self._translator.translate("settings.sec_mpv_log"))
        log_form = QFormLayout()
        log_group.add_layout(log_form)

        self._cb_mpv_log = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_log_no"), "no"), ("fatal", "fatal"), ("error", "error"),
            ("warn", "warn"), ("info", "info"), ("status", "status"),
            ("v", "v"), ("debug", "debug"), ("trace", "trace"),
        ]:
            self._cb_mpv_log.addItem(label, userData=key)
        log_form.addRow(self._translator.translate("settings.rl_mpv_loglevel"), self._cb_mpv_log)
        scroll_layout.addWidget(log_group)

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_roi(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        group = SectionCard(self._translator.translate("settings.sec_roi"))
        form = QFormLayout()
        group.add_layout(form)

        self._cb_roi_preset = QComboBox()
        for label, key in[
            (self._translator.translate("settings.co_roi_auto"), "auto_subtitle"),
            (self._translator.translate("settings.co_roi_custom"), "custom"), (self._translator.translate("settings.co_roi_full"), "full"),
            (self._translator.translate("settings.co_roi_q"), "bottom_quarter"), (self._translator.translate("settings.co_roi_t"), "bottom_third"),
            (self._translator.translate("settings.co_roi_h"), "bottom_half"),
        ]:
            self._cb_roi_preset.addItem(label, userData=key)
        form.addRow(self._translator.translate("settings.rl_default_preset"), self._cb_roi_preset)

        # [v3.19] Tinh chỉnh lõi AI dò ROI (BBoxAnalyzer) — cho người dùng tuỳ biến.
        self._chk_auto_band_refine = QCheckBox(self._translator.translate("settings.cb_band_tune"))
        form.addRow("", self._chk_auto_band_refine)

        self._sp_auto_keep = QDoubleSpinBox(); self._sp_auto_keep.setRange(0.05, 0.95); self._sp_auto_keep.setSingleStep(0.05); self._sp_auto_keep.setDecimals(2)
        self._sp_auto_keep.setToolTip(self._translator.translate("settings.tip_auto_keep"))
        form.addRow(self._translator.translate("settings.rl_core_band"), self._sp_auto_keep)

        self._sp_auto_extend = QDoubleSpinBox(); self._sp_auto_extend.setRange(0.02, 0.95); self._sp_auto_extend.setSingleStep(0.05); self._sp_auto_extend.setDecimals(2)
        self._sp_auto_extend.setToolTip(self._translator.translate("settings.tip_auto_extend"))
        form.addRow(self._translator.translate("settings.rl_expand_thresh"), self._sp_auto_extend)

        self._sp_auto_botpad = QDoubleSpinBox(); self._sp_auto_botpad.setRange(1.0, 3.0); self._sp_auto_botpad.setSingleStep(0.1); self._sp_auto_botpad.setDecimals(1)
        self._sp_auto_botpad.setToolTip(self._translator.translate("settings.tip_auto_botpad"))
        form.addRow(self._translator.translate("settings.rl_lower_margin"), self._sp_auto_botpad)

        self._sp_auto_sensitivity = QDoubleSpinBox(); self._sp_auto_sensitivity.setRange(0.3, 2.0); self._sp_auto_sensitivity.setSingleStep(0.1); self._sp_auto_sensitivity.setDecimals(1)
        self._sp_auto_sensitivity.setToolTip(self._translator.translate("settings.tip_auto_sensitivity"))
        form.addRow(self._translator.translate("settings.rl_text_detect"), self._sp_auto_sensitivity)

        self._chk_remember_roi = QCheckBox(self._translator.translate("settings.cb_remember_roi"))
        form.addRow(self._translator.translate("settings.rl_save_roi"), self._chk_remember_roi)

        self._chk_auto_detect_load = QCheckBox(self._translator.translate("settings.cb_autodetect_roi"))
        form.addRow(self._translator.translate("settings.rl_auto"), self._chk_auto_detect_load)

        self._sp_roi_step_ms = QSpinBox()
        self._sp_roi_step_ms.setRange(100, 60000)
        self._sp_roi_step_ms.setSingleStep(100)
        self._sp_roi_step_ms.setSuffix(" ms")
        form.addRow(self._translator.translate("settings.rl_roi_step"), self._sp_roi_step_ms)

        scroll_layout.addWidget(group)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_threshold(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        group = SectionCard(self._translator.translate("settings.sec_thresholds"))
        form = QFormLayout()
        group.add_layout(form)

        self._sp_min_confidence = _make_double_spin(0.0, 1.0, 0.05, 2)
        form.addRow(self._translator.translate("settings.rl_min_conf"), self._sp_min_confidence)

        self._sp_text_similarity = _make_double_spin(0.0, 1.0, 0.05, 2)
        form.addRow(self._translator.translate("settings.rl_text_sim"), self._sp_text_similarity)

        self._sp_line_similarity = _make_double_spin(0.0, 1.0, 0.05, 2)
        form.addRow(self._translator.translate("settings.rl_line_sim"), self._sp_line_similarity)

        self._sp_drop_short = QSpinBox(); self._sp_drop_short.setRange(0, 20)
        form.addRow(self._translator.translate("settings.rl_drop_short"), self._sp_drop_short)

        scroll_layout.addWidget(group)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_frame(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        group = SectionCard(self._translator.translate("settings.sec_frame_sampling"))
        form = QFormLayout()
        group.add_layout(form)

        self._sp_sample_step = _make_double_spin(0.01, 5.0, 0.05, 2, suffix=" s")
        form.addRow(self._translator.translate("settings.rl_sample_step"), self._sp_sample_step)

        self._sp_phash = QSpinBox(); self._sp_phash.setRange(0, 64)
        form.addRow(self._translator.translate("settings.rl_phash"), self._sp_phash)

        self._sp_pixel_diff = _make_double_spin(0.0, 1.0, 0.005, 3)
        form.addRow(self._translator.translate("settings.rl_pixel_diff"), self._sp_pixel_diff)

        self._sp_skip_intro = _make_double_spin(0.0, 600.0, 1.0, 1, suffix=" s")
        form.addRow(self._translator.translate("settings.rl_skip_start"), self._sp_skip_intro)

        self._sp_skip_outro = _make_double_spin(0.0, 600.0, 1.0, 1, suffix=" s")
        form.addRow(self._translator.translate("settings.rl_skip_end"), self._sp_skip_outro)

        scroll_layout.addWidget(group)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_post(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        group_algo = SectionCard(self._translator.translate("settings.sec_merge_time"))
        form_algo = QFormLayout()
        group_algo.add_layout(form_algo)

        self._sp_similarity = _make_double_spin(0.0, 1.0, 0.05, 2)
        form_algo.addRow(self._translator.translate("settings.rl_merge_sim"), self._sp_similarity)

        self._sp_min_dur = _make_double_spin(0.0, 60.0, 0.05, 2, suffix=" s")
        form_algo.addRow(self._translator.translate("settings.rl_min_dur"), self._sp_min_dur)

        self._sp_max_dur = _make_double_spin(0.1, 600.0, 1.0, 2, suffix=" s")
        form_algo.addRow(self._translator.translate("settings.rl_max_dur"), self._sp_max_dur)

        self._sp_merge_gap = _make_double_spin(0.0, 10.0, 0.05, 2, suffix=" s")
        form_algo.addRow(self._translator.translate("settings.rl_max_gap"), self._sp_merge_gap)

        self._cb_output_format = QComboBox()
        for f in SubtitleFormat:
            self._cb_output_format.addItem(f.value.upper(), userData=f.value)
        form_algo.addRow(self._translator.translate("settings.rl_default_format"), self._cb_output_format)

        self._chk_use_viterbi = QCheckBox(self._translator.translate("settings.cb_viterbi"))
        form_algo.addRow(self._translator.translate("settings.rl_merge_algo"), self._chk_use_viterbi)

        self._sp_viterbi_penalty = _make_double_spin(0.0, 2.0, 0.05, 2)
        form_algo.addRow(self._translator.translate("settings.rl_viterbi_penalty"), self._sp_viterbi_penalty)
        self._chk_use_viterbi.toggled.connect(self._sp_viterbi_penalty.setEnabled)

        scroll_layout.addWidget(group_algo)

        adv_group = SectionCard(self._translator.translate("settings.sec_spatial_filter"))
        adv_form = QFormLayout()
        adv_group.add_layout(adv_form)

        self._sp_align_center = _make_double_spin(0.0, 0.5, 0.05, 2)
        adv_form.addRow(self._translator.translate("settings.rl_tol_center"), self._sp_align_center)

        self._sp_align_margin = _make_double_spin(0.0, 0.5, 0.05, 2)
        adv_form.addRow(self._translator.translate("settings.rl_tol_margin"), self._sp_align_margin)

        self._sp_align_min = _make_double_spin(0.0, 200.0, 5.0, 1, suffix=" px")
        adv_form.addRow(self._translator.translate("settings.rl_tol_min"), self._sp_align_min)
        scroll_layout.addWidget(adv_group)

        group_y = SectionCard(self._translator.translate("settings.sec_merge_line"))
        form_y = QFormLayout()
        group_y.add_layout(form_y)

        self._sp_y_ratio = _make_double_spin(0.0, 1.0, 0.05, 2)
        form_y.addRow(self._translator.translate("settings.rl_tol_line_merge"), self._sp_y_ratio)

        self._sp_y_min = _make_double_spin(0.0, 50.0, 1.0, 1, suffix=" px")
        form_y.addRow(self._translator.translate("settings.rl_tol_y_min"), self._sp_y_min)

        scroll_layout.addWidget(group_y)

        group_other = SectionCard(self._translator.translate("settings.sec_other"))
        form_other = QFormLayout()
        group_other.add_layout(form_other)
        self._sp_temporal_padding = _make_double_spin(0.0, 1.0, 0.01, 2, suffix=" s")
        form_other.addRow(self._translator.translate("settings.rl_tail_padding"), self._sp_temporal_padding)
        scroll_layout.addWidget(group_other)

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_preprocess(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        group_basic = SectionCard(self._translator.translate("settings.sec_preprocess_basic"))
        form_basic = QFormLayout()
        group_basic.add_layout(form_basic)

        self._chk_upscale = QCheckBox(self._translator.translate("settings.cb_upscale"))
        form_basic.addRow(self._translator.translate("settings.rl_upscale"), self._chk_upscale)

        self._sp_upscale_target = QSpinBox()
        self._sp_upscale_target.setRange(32, 512); self._sp_upscale_target.setSuffix(" px")
        form_basic.addRow(self._translator.translate("settings.rl_target_height"), self._sp_upscale_target)
        self._chk_upscale.toggled.connect(self._sp_upscale_target.setEnabled)

        self._chk_border = QCheckBox(self._translator.translate("settings.cb_black_border"))
        form_basic.addRow(self._translator.translate("settings.rl_black_border"), self._chk_border)

        self._sp_border = QSpinBox()
        self._sp_border.setRange(0, 64); self._sp_border.setSuffix(" px")
        form_basic.addRow(self._translator.translate("settings.rl_border_thickness"), self._sp_border)
        self._chk_border.toggled.connect(self._sp_border.setEnabled)

        self._chk_sharpen = QCheckBox(self._translator.translate("settings.cb_sharpen"))
        form_basic.addRow(self._translator.translate("settings.rl_sharpen"), self._chk_sharpen)

        self._chk_contrast = QCheckBox(self._translator.translate("settings.cb_contrast"))
        form_basic.addRow(self._translator.translate("settings.rl_contrast"), self._chk_contrast)

        self._sp_contrast_factor = _make_double_spin(0.5, 3.0, 0.05, 2)
        form_basic.addRow(self._translator.translate("settings.rl_contrast_factor"), self._sp_contrast_factor)
        self._chk_contrast.toggled.connect(self._sp_contrast_factor.setEnabled)

        scroll_layout.addWidget(group_basic)

        group_adv = SectionCard(self._translator.translate("settings.sec_preprocess_adv"))
        form_adv = QFormLayout()
        group_adv.add_layout(form_adv)

        self._chk_clahe = QCheckBox(self._translator.translate("settings.cb_clahe"))
        form_adv.addRow(self._translator.translate("settings.rl_clahe"), self._chk_clahe)

        self._sp_clahe_clip = _make_double_spin(1.0, 10.0, 0.5, 1)
        form_adv.addRow(self._translator.translate("settings.rl_clahe_clip"), self._sp_clahe_clip)

        self._sp_clahe_tile = QSpinBox()
        self._sp_clahe_tile.setRange(2, 32)
        form_adv.addRow(self._translator.translate("settings.rl_clahe_tile"), self._sp_clahe_tile)

        self._chk_clahe.toggled.connect(self._sp_clahe_clip.setEnabled)
        self._chk_clahe.toggled.connect(self._sp_clahe_tile.setEnabled)

        scroll_layout.addWidget(group_adv)

        group_gpu = SectionCard(self._translator.translate("settings.sec_preprocess_gpu"))
        form_gpu = QFormLayout()
        group_gpu.add_layout(form_gpu)

        self._chk_median_blend = QCheckBox(self._translator.translate("settings.cb_median_blend"))
        self._chk_median_blend.setToolTip(self._translator.translate("settings.tip_median_blend"))
        form_gpu.addRow(self._translator.translate("settings.rl_denoise_bg"), self._chk_median_blend)

        self._cb_median_frames = QComboBox()
        for label, val in[(self._translator.translate("settings.co_blend_3"), 3), (self._translator.translate("settings.co_blend_5"), 5), (self._translator.translate("settings.co_blend_7"), 7)]:
            self._cb_median_frames.addItem(label, userData=val)
        form_gpu.addRow(self._translator.translate("settings.rl_blend_frames"), self._cb_median_frames)
        self._chk_median_blend.toggled.connect(self._cb_median_frames.setEnabled)

        scroll_layout.addWidget(group_gpu)

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_video_translation(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        # ── Nhóm 1: Nén video ngữ cảnh ───────────────────────────────────
        enc_group = SectionCard(self._translator.translate("settings.sec_context_compress"))
        enc_form = QFormLayout()
        enc_group.add_layout(enc_form)

        self._sp_vidctx_height = QSpinBox()
        self._sp_vidctx_height.setRange(144, 1080)
        self._sp_vidctx_height.setSingleStep(36)
        self._sp_vidctx_height.setSuffix(" px")
        self._sp_vidctx_height.setToolTip(
            self._translator.translate("settings.tip_vidctx_height")
        )
        enc_form.addRow(self._translator.translate("settings.rl_resolution_height"), self._sp_vidctx_height)

        self._sp_vidctx_fps = _make_double_spin(0.2, 5.0, 0.1, 1, suffix=" fps")
        self._sp_vidctx_fps.setToolTip(
            self._translator.translate("settings.tip_vidctx_fps")
        )
        enc_form.addRow(self._translator.translate("settings.rl_framerate"), self._sp_vidctx_fps)

        self._sp_vidctx_cq = QSpinBox()
        self._sp_vidctx_cq.setRange(18, 45)
        self._sp_vidctx_cq.setToolTip(
            self._translator.translate("settings.tip_vidctx_cq")
        )
        enc_form.addRow(self._translator.translate("settings.rl_gpu_cq"), self._sp_vidctx_cq)

        self._sp_vidctx_crf = QSpinBox()
        self._sp_vidctx_crf.setRange(18, 45)
        self._sp_vidctx_crf.setToolTip(
            self._translator.translate("settings.tip_vidctx_crf")
        )
        enc_form.addRow(self._translator.translate("settings.rl_cpu_crf"), self._sp_vidctx_crf)
        scroll_layout.addWidget(enc_group)

        # ── Nhóm 2: Cắt đoạn theo token ──────────────────────────────────
        chunk_group = SectionCard(self._translator.translate("settings.sec_video_segment"))
        chunk_form = QFormLayout()
        chunk_group.add_layout(chunk_form)

        self._sp_vidctx_tokens_chunk = QSpinBox()
        self._sp_vidctx_tokens_chunk.setRange(50_000, 900_000)
        self._sp_vidctx_tokens_chunk.setSingleStep(50_000)
        self._sp_vidctx_tokens_chunk.setGroupSeparatorShown(True)
        self._sp_vidctx_tokens_chunk.setToolTip(
            self._translator.translate("settings.tip_tokens_chunk")
        )
        chunk_form.addRow(self._translator.translate("settings.rl_max_token_seg"), self._sp_vidctx_tokens_chunk)

        self._sp_vidctx_chunk_min = _make_double_spin(1.0, 60.0, 1.0, 1, suffix=" " + self._translator.translate("settings.unit_min"))
        self._sp_vidctx_chunk_min.setToolTip(
            self._translator.translate("settings.tip_chunk_min")
        )
        chunk_form.addRow(self._translator.translate("settings.rl_max_dur_seg"), self._sp_vidctx_chunk_min)

        self._sp_vidctx_tps = QSpinBox()
        self._sp_vidctx_tps.setRange(50, 400)
        self._sp_vidctx_tps.setToolTip(
            self._translator.translate("settings.tip_tps")
        )
        chunk_form.addRow(self._translator.translate("settings.rl_token_per_sec"), self._sp_vidctx_tps)
        scroll_layout.addWidget(chunk_group)

        # ── Nhóm 3: Điều tiết dịch ───────────────────────────────────────
        trans_group = SectionCard(self._translator.translate("settings.sec_translate_process"))
        trans_form = QFormLayout()
        trans_group.add_layout(trans_form)

        self._sp_trans_batch = QSpinBox()
        self._sp_trans_batch.setRange(10, 200)
        self._sp_trans_batch.setToolTip(
            self._translator.translate("settings.tip_trans_batch")
        )
        trans_form.addRow(self._translator.translate("settings.rl_default_batch"), self._sp_trans_batch)

        self._sp_trans_ctx = QSpinBox()
        self._sp_trans_ctx.setRange(0, 40)
        self._sp_trans_ctx.setToolTip(
            self._translator.translate("settings.tip_trans_ctx")
        )
        trans_form.addRow(self._translator.translate("settings.rl_ctx_per_batch"), self._sp_trans_ctx)

        self._sp_trans_retry = QSpinBox()
        self._sp_trans_retry.setRange(1, 10)
        self._sp_trans_retry.setToolTip(
            self._translator.translate("settings.tip_trans_retry")
        )
        trans_form.addRow(self._translator.translate("settings.rl_retries"), self._sp_trans_retry)

        self._sp_trans_timeout = QSpinBox()
        self._sp_trans_timeout.setRange(30, 600)
        self._sp_trans_timeout.setSuffix(" s")
        self._sp_trans_timeout.setToolTip(
            self._translator.translate("settings.tip_trans_timeout")
        )
        trans_form.addRow(self._translator.translate("settings.rl_req_timeout"), self._sp_trans_timeout)
        scroll_layout.addWidget(trans_group)

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_ui(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        group = SectionCard(self._translator.translate("settings.sec_ui"))
        form = QFormLayout()
        group.add_layout(form)

        self._cb_theme = QComboBox()
        for label, key in[(self._translator.translate("settings.co_theme_auto"), "auto"), (self._translator.translate("settings.co_theme_light"), "light"), (self._translator.translate("settings.co_theme_dark"), "dark")]:
            self._cb_theme.addItem(label, userData=key)
        form.addRow(self._translator.translate("settings.rl_theme"), self._cb_theme)

        self._cb_locale = QComboBox()
        for label, key in[(self._translator.translate("settings.co_locale_vi"), "vi"), ("English", "en")]:
            self._cb_locale.addItem(label, userData=key)
        form.addRow(self._translator.translate("settings.rl_language"), self._cb_locale)

        self._sp_font_size = QSpinBox()
        self._sp_font_size.setRange(8, 18); self._sp_font_size.setSuffix(" pt")
        form.addRow(self._translator.translate("settings.rl_font_size"), self._sp_font_size)

        self._chk_show_ocr_overlay = QCheckBox(self._translator.translate("settings.cb_show_ocr_overlay"))
        form.addRow("Overlay OCR:", self._chk_show_ocr_overlay)

        self._chk_show_waveform = QCheckBox(self._translator.translate("settings.cb_show_waveform"))
        form.addRow("Waveform:", self._chk_show_waveform)

        scroll_layout.addWidget(group)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _build_tab_advanced(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget, scroll_layout, scroll_area = self._create_scroll_container(self)

        group = SectionCard(self._translator.translate("settings.sec_advanced_opts"))
        form = QFormLayout()
        group.add_layout(form)

        self._cb_log_level = QComboBox()
        for level in["DEBUG", "INFO", "WARNING", "ERROR"]:
            self._cb_log_level.addItem(level, userData=level)
        form.addRow(self._translator.translate("settings.rl_log_level"), self._cb_log_level)

        self._chk_save_debug = QCheckBox(self._translator.translate("settings.cb_save_debug_frame"))
        form.addRow("Debug frames:", self._chk_save_debug)

        debug_dir_row = QHBoxLayout()
        self._le_debug_frames_dir = QLineEdit()
        self._le_debug_frames_dir.setPlaceholderText(
            self._translator.translate("settings.debug_dir_placeholder")
        )
        self._btn_browse_debug = QPushButton(self._translator.translate("settings.btn_browse"))
        self._btn_browse_debug.clicked.connect(self._on_browse_debug_dir)
        debug_dir_row.addWidget(self._le_debug_frames_dir, stretch=1)
        debug_dir_row.addWidget(self._btn_browse_debug)
        form.addRow(self._translator.translate("settings.rl_debug_dir"), debug_dir_row)

        self._chk_save_debug.toggled.connect(self._le_debug_frames_dir.setEnabled)
        self._chk_save_debug.toggled.connect(self._btn_browse_debug.setEnabled)

        self._chk_keep_temp = QCheckBox(self._translator.translate("settings.cb_keep_temp"))
        form.addRow(self._translator.translate("settings.rl_temp_file"), self._chk_keep_temp)

        self._chk_disable_pdx_check = QCheckBox(self._translator.translate("settings.cb_skip_pdx"))
        form.addRow(self._translator.translate("settings.rl_paddle_net"), self._chk_disable_pdx_check)

        # [V3.15 FEATURE] Cache Cleaner Button
        self._btn_clean_cache = QPushButton(self._translator.translate("settings.btn_clean_cache"))
        self._btn_clean_cache.setStyleSheet(f"color: {_c.warning()}; font-weight: bold;")
        self._btn_clean_cache.clicked.connect(self._view_model.clear_temp_cache)
        form.addRow(self._translator.translate("settings.rl_maintenance"), self._btn_clean_cache)

        # [v3.23.88] Nút dọn DATABASE: xoá sạch dữ liệu mọi bảng -> như mới tạo.
        self._btn_clean_db = QPushButton(self._translator.translate("settings.btn_clean_db"))
        self._btn_clean_db.setStyleSheet(f"color: {_c.danger()}; font-weight: bold;")
        self._btn_clean_db.clicked.connect(self._on_clean_database_clicked)
        form.addRow("Database:", self._btn_clean_db)

        scroll_layout.addWidget(group)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return tab

    def _on_browse_debug_dir(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        current = self._le_debug_frames_dir.text().strip()
        chosen = QFileDialog.getExistingDirectory(
            self, self._translator.translate("settings.dlg_choose_debug_dir"), current
        )
        if chosen:
            self._le_debug_frames_dir.setText(chosen)

    def _connect_signals(self) -> None:
        self._save_button.clicked.connect(self._on_save_clicked)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        self._reset_button.clicked.connect(self._on_reset_clicked)

        self._view_model.settings_loaded.connect(self._populate_form)
        self._view_model.settings_saved.connect(self._on_saved)
        self._view_model.validation_failed.connect(self._on_validation_failed)
        self._view_model.cache_cleaned.connect(self._on_cache_cleaned)
        self._view_model.database_reset.connect(self._on_database_reset)

    def _monitor_form_changes(self) -> None:
        """[V3.15 UX] Bắt sự kiện người dùng thay đổi giá trị ở bất cứ ô nào."""
        for widget in self.findChildren(QSpinBox):
            widget.valueChanged.connect(self._on_any_field_changed)
        for widget in self.findChildren(QDoubleSpinBox):
            widget.valueChanged.connect(self._on_any_field_changed)
        for widget in self.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self._on_any_field_changed)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self._on_any_field_changed)
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self._on_any_field_changed)

    def _on_any_field_changed(self) -> None:
        if self._is_populating:
            return
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        """[v3.20.3 #4] Có thay đổi chưa lưu → mở khoá Save/Cancel."""
        self._save_button.setEnabled(True)
        self._cancel_button.setEnabled(True)

    def _mark_clean(self) -> None:
        """[v3.20.3 #4] Không còn thay đổi nào → khoá Save/Cancel (Smart Dirty)."""
        self._save_button.setEnabled(False)
        self._cancel_button.setEnabled(False)

    def _populate_form(self, snapshot: ApplicationSettings) -> None:
        # [v3.6 bugfix POP-1]: try/finally đảm bảo _is_populating luôn được reset.
        # Trước đây không có guard → nếu bất kỳ setValue/setChecked nào raise
        # exception, _is_populating kẹt True mãi mãi → _on_any_field_changed()
        # trả về sớm → Save button không bao giờ được bật → settings không lưu được.
        self._is_populating = True
        try:
            _set_combo(self._cb_ocr_version, snapshot.ocr.version)
            _set_combo(self._cb_ocr_language, snapshot.ocr.language)
            self._sp_score_threshold.setValue(snapshot.ocr.score_threshold)

            self._sp_limit_side.setValue(snapshot.ocr.limit_side_len)
            _set_combo(self._cb_limit_type, snapshot.ocr.limit_type)
            self._sp_det_thresh.setValue(snapshot.ocr.det_thresh)
            self._sp_det_box_thresh.setValue(snapshot.ocr.det_box_thresh)
            self._sp_det_unclip_ratio.setValue(snapshot.ocr.det_unclip_ratio)
            self._chk_textline_orient.setChecked(snapshot.ocr.use_textline_orientation)
            self._chk_doc_orient.setChecked(snapshot.ocr.use_doc_orientation_classify)
            self._chk_doc_unwarp.setChecked(snapshot.ocr.use_doc_unwarping)

            _set_combo(self._cb_device, snapshot.hardware.device.value)
            _set_combo(self._cb_precision, snapshot.hardware.precision.value)
            self._sp_batch_ocr.setValue(snapshot.hardware.batch_size_ocr)
            self._sp_batch_roi.setValue(snapshot.hardware.batch_size_roi)
            self._sp_workers.setValue(snapshot.hardware.workers)
            self._chk_auto_tune.setChecked(snapshot.hardware.auto_tune_batch)
            self._chk_mkldnn.setChecked(snapshot.hardware.enable_mkldnn)
            self._chk_tensorrt.setChecked(snapshot.hardware.use_tensorrt)
            _set_combo(self._cb_frame_backend, snapshot.hardware.frame_decoder_backend)
            _set_combo(self._cb_metadata_backend, snapshot.hardware.metadata_reader_backend)

            self._chk_enable_nlp.setChecked(snapshot.nlp.enable_vector_embeddings)
            _set_combo(self._cb_nlp_mode, snapshot.nlp.similarity_mode)
            idx = self._cb_nlp_model.findData(snapshot.nlp.model_name)
            if idx >= 0:
                self._cb_nlp_model.setCurrentIndex(idx)
            else:
                self._cb_nlp_model.setCurrentText(snapshot.nlp.model_name)

            _set_combo(self._cb_hwdec_mode, snapshot.mpv.hwdec_mode)
            self._le_hwdec_codecs.setText(snapshot.mpv.hwdec_codecs)
            _set_combo(self._cb_video_output, snapshot.mpv.video_output)
            _set_combo(self._cb_gpu_api, snapshot.mpv.gpu_api)
            self._le_gpu_context.setText(snapshot.mpv.gpu_context)
            _set_combo(self._cb_cache, snapshot.mpv.cache)
            self._sp_cache_secs.setValue(snapshot.mpv.cache_secs)
            self._sp_demuxer_max.setValue(snapshot.mpv.demuxer_max_bytes)
            _set_combo(self._cb_video_sync, snapshot.mpv.video_sync_mode)
            _set_combo(self._cb_profile, snapshot.mpv.profile)
            self._chk_deinterlace.setChecked(snapshot.mpv.deinterlace)
            _set_combo(self._cb_mpv_log, snapshot.mpv.log_level)

            _set_combo(self._cb_roi_preset, snapshot.roi.default_preset)
            self._chk_auto_band_refine.setChecked(snapshot.roi.auto_enable_band_refinement)
            self._sp_auto_keep.setValue(snapshot.roi.auto_band_keep_ratio)
            self._sp_auto_extend.setValue(snapshot.roi.auto_band_extend_ratio)
            self._sp_auto_botpad.setValue(snapshot.roi.auto_bottom_padding_factor)
            self._sp_auto_sensitivity.setValue(snapshot.roi.auto_sensitivity_multiplier)
            self._chk_remember_roi.setChecked(snapshot.roi.remember_last_roi)
            self._chk_auto_detect_load.setChecked(snapshot.roi.auto_detect_on_load)
            step_val = getattr(snapshot.roi, "auto_detect_step_ms", 1000)
            self._sp_roi_step_ms.setValue(step_val)

            self._sp_min_confidence.setValue(snapshot.threshold.ocr_min_confidence)
            self._sp_text_similarity.setValue(snapshot.threshold.text_similarity)
            self._sp_line_similarity.setValue(snapshot.threshold.line_similarity)
            self._sp_drop_short.setValue(snapshot.threshold.drop_short_text_chars)

            self._sp_sample_step.setValue(snapshot.frame.sample_step_sec)
            self._sp_phash.setValue(snapshot.frame.phash_distance)
            self._sp_pixel_diff.setValue(snapshot.frame.pixel_diff_ratio)
            self._sp_skip_intro.setValue(snapshot.frame.skip_intro_sec)
            self._sp_skip_outro.setValue(snapshot.frame.skip_outro_sec)

            self._sp_similarity.setValue(snapshot.post_process.similarity_threshold)
            self._sp_min_dur.setValue(snapshot.post_process.min_duration_sec)
            self._sp_max_dur.setValue(snapshot.post_process.max_duration_sec)
            self._sp_merge_gap.setValue(snapshot.post_process.merge_gap_sec)
            _set_combo(self._cb_output_format, snapshot.post_process.output_format.value)
            self._chk_use_viterbi.setChecked(snapshot.post_process.use_viterbi)
            self._sp_viterbi_penalty.setValue(snapshot.post_process.viterbi_open_penalty)

            self._sp_temporal_padding.setValue(snapshot.post_process.temporal_padding_sec)
            self._sp_y_ratio.setValue(snapshot.post_process.y_clustering_tolerance_ratio)
            self._sp_y_min.setValue(snapshot.post_process.y_clustering_tolerance_min_px)
            self._sp_align_center.setValue(snapshot.post_process.alignment_center_tolerance_ratio)
            self._sp_align_margin.setValue(snapshot.post_process.alignment_margin_tolerance_ratio)
            self._sp_align_min.setValue(snapshot.post_process.alignment_tolerance_min_px)

            self._chk_upscale.setChecked(snapshot.preprocess.upscale_small_text)
            self._sp_upscale_target.setValue(snapshot.preprocess.upscale_target_height_px)
            self._chk_border.setChecked(snapshot.preprocess.add_white_border)
            self._sp_border.setValue(snapshot.preprocess.border_thickness_px)
            self._chk_sharpen.setChecked(snapshot.preprocess.apply_sharpen)
            self._chk_contrast.setChecked(snapshot.preprocess.apply_contrast_boost)
            self._sp_contrast_factor.setValue(snapshot.preprocess.contrast_factor)
            self._sp_contrast_factor.setEnabled(snapshot.preprocess.apply_contrast_boost)

            self._chk_clahe.setChecked(snapshot.preprocess.apply_clahe)
            self._sp_clahe_clip.setValue(snapshot.preprocess.clahe_clip_limit)
            self._sp_clahe_tile.setValue(snapshot.preprocess.clahe_tile_size)
            self._sp_clahe_clip.setEnabled(snapshot.preprocess.apply_clahe)
            self._sp_clahe_tile.setEnabled(snapshot.preprocess.apply_clahe)

            self._chk_median_blend.setChecked(snapshot.preprocess.apply_median_blend)
            _set_combo(self._cb_median_frames, snapshot.preprocess.median_blend_frames)
            self._cb_median_frames.setEnabled(snapshot.preprocess.apply_median_blend)

            _set_combo(self._cb_theme, snapshot.ui.theme)
            _set_combo(self._cb_locale, snapshot.ui.locale)
            self._sp_font_size.setValue(snapshot.ui.safe_font_size)
            self._chk_show_ocr_overlay.setChecked(snapshot.ui.show_ocr_overlay)
            self._chk_show_waveform.setChecked(snapshot.ui.show_waveform)

            _set_combo(self._cb_log_level, snapshot.advanced.log_level)
            self._chk_save_debug.setChecked(snapshot.advanced.save_debug_frames)
            self._le_debug_frames_dir.setText(snapshot.advanced.debug_frames_dir or "")
            self._le_debug_frames_dir.setEnabled(snapshot.advanced.save_debug_frames)
            self._btn_browse_debug.setEnabled(snapshot.advanced.save_debug_frames)
            self._chk_keep_temp.setChecked(snapshot.advanced.keep_temp_files)
            self._chk_disable_pdx_check.setChecked(snapshot.advanced.disable_paddle_network_check)

            vc = getattr(snapshot, "video_context", None)
            if vc is not None:
                self._sp_vidctx_height.setValue(vc.resolution_height)
                self._sp_vidctx_fps.setValue(vc.fps)
                self._sp_vidctx_cq.setValue(vc.nvenc_cq)
                self._sp_vidctx_crf.setValue(vc.cpu_crf)
                self._sp_vidctx_tokens_chunk.setValue(vc.tokens_per_chunk)
                self._sp_vidctx_chunk_min.setValue(vc.max_chunk_minutes)
                self._sp_vidctx_tps.setValue(vc.tokens_per_second)
            tr = getattr(snapshot, "translation", None)
            if tr is not None:
                self._sp_trans_batch.setValue(tr.default_batch_size)
                self._sp_trans_ctx.setValue(tr.default_context_size)
                self._sp_trans_retry.setValue(tr.retry_count)
                self._sp_trans_timeout.setValue(tr.request_timeout_sec)

        finally:
            # [v3.6 bugfix POP-1]: Luôn reset cờ dù có exception.
            self._is_populating = False
            # [V3.15 UX] Trả lại trạng thái Khóa (Disabled) cho nút Lưu/Hủy khi đã Load xong
            self._save_button.setEnabled(False)
            self._cancel_button.setEnabled(False)
            self._status_label.setText("")

    def _build_patch(self) -> dict[str, Any]:
        model_name = self._cb_nlp_model.currentData()
        if not model_name:
            model_name = self._cb_nlp_model.currentText()
            if " (" in model_name:
                model_name = model_name.split(" (")[0].strip()

        return {
            "ocr": {
                "version": self._cb_ocr_version.currentData(),
                "language": self._cb_ocr_language.currentData(),
                "score_threshold": self._sp_score_threshold.value(),
                "limit_side_len": self._sp_limit_side.value(),
                "limit_type": self._cb_limit_type.currentText(),
                "det_thresh": self._sp_det_thresh.value(),
                "det_box_thresh": self._sp_det_box_thresh.value(),
                "det_unclip_ratio": self._sp_det_unclip_ratio.value(),
                "use_textline_orientation": self._chk_textline_orient.isChecked(),
                "use_doc_orientation_classify": self._chk_doc_orient.isChecked(),
                "use_doc_unwarping": self._chk_doc_unwarp.isChecked(),
            },
            "hardware": {
                "device": self._cb_device.currentData(),
                "precision": self._cb_precision.currentData(),
                "batch_size_ocr": self._sp_batch_ocr.value(),
                "batch_size_roi": self._sp_batch_roi.value(),
                "workers": self._sp_workers.value(),
                "auto_tune_batch": self._chk_auto_tune.isChecked(),
                "enable_mkldnn": self._chk_mkldnn.isChecked(),
                "use_tensorrt": self._chk_tensorrt.isChecked(),
                "frame_decoder_backend": self._cb_frame_backend.currentData(),
                "metadata_reader_backend": self._cb_metadata_backend.currentData(),
            },
            "nlp": {
                "enable_vector_embeddings": self._chk_enable_nlp.isChecked(),
                "model_name": model_name,
                "similarity_mode": self._cb_nlp_mode.currentData(),
            },
            "mpv": {
                "hwdec_mode": self._cb_hwdec_mode.currentData(),
                "hwdec_codecs": self._le_hwdec_codecs.text().strip() or "h264,vc1,hevc,vp8,vp9,av1,mpeg2video,mpeg4",
                "video_output": self._cb_video_output.currentData(),
                "gpu_api": self._cb_gpu_api.currentData(),
                "gpu_context": self._le_gpu_context.text().strip() or "auto",
                "cache": self._cb_cache.currentData(),
                "cache_secs": self._sp_cache_secs.value(),
                "demuxer_max_bytes": self._sp_demuxer_max.value(),
                "video_sync_mode": self._cb_video_sync.currentData(),
                "profile": self._cb_profile.currentData(),
                "deinterlace": self._chk_deinterlace.isChecked(),
                "log_level": self._cb_mpv_log.currentData(),
            },
            "roi": {
                "default_preset": self._cb_roi_preset.currentData(),
                "remember_last_roi": self._chk_remember_roi.isChecked(),
                "auto_detect_on_load": self._chk_auto_detect_load.isChecked(),
                "auto_detect_step_ms": self._sp_roi_step_ms.value(),
                "auto_enable_band_refinement": self._chk_auto_band_refine.isChecked(),
                "auto_band_keep_ratio": self._sp_auto_keep.value(),
                "auto_band_extend_ratio": self._sp_auto_extend.value(),
                "auto_bottom_padding_factor": self._sp_auto_botpad.value(),
                "auto_sensitivity_multiplier": self._sp_auto_sensitivity.value(),
            },
            "threshold": {
                "ocr_min_confidence": self._sp_min_confidence.value(),
                "text_similarity": self._sp_text_similarity.value(),
                "line_similarity": self._sp_line_similarity.value(),
                "drop_short_text_chars": self._sp_drop_short.value(),
            },
            "frame": {
                "sample_step_sec": self._sp_sample_step.value(),
                "phash_distance": self._sp_phash.value(),
                "pixel_diff_ratio": self._sp_pixel_diff.value(),
                "skip_intro_sec": self._sp_skip_intro.value(),
                "skip_outro_sec": self._sp_skip_outro.value(),
            },
            "post_process": {
                "similarity_threshold": self._sp_similarity.value(),
                "min_duration_sec": self._sp_min_dur.value(),
                "max_duration_sec": self._sp_max_dur.value(),
                "merge_gap_sec": self._sp_merge_gap.value(),
                "output_format": self._cb_output_format.currentData(),
                "use_viterbi": self._chk_use_viterbi.isChecked(),
                "viterbi_open_penalty": self._sp_viterbi_penalty.value(),
                "temporal_padding_sec": self._sp_temporal_padding.value(),
                "y_clustering_tolerance_ratio": self._sp_y_ratio.value(),
                "y_clustering_tolerance_min_px": self._sp_y_min.value(),
                "alignment_center_tolerance_ratio": self._sp_align_center.value(),
                "alignment_margin_tolerance_ratio": self._sp_align_margin.value(),
                "alignment_tolerance_min_px": self._sp_align_min.value(),
            },
            "preprocess": {
                "upscale_small_text": self._chk_upscale.isChecked(),
                "upscale_target_height_px": self._sp_upscale_target.value(),
                "add_white_border": self._chk_border.isChecked(),
                "border_thickness_px": self._sp_border.value(),
                "apply_sharpen": self._chk_sharpen.isChecked(),
                "apply_contrast_boost": self._chk_contrast.isChecked(),
                "contrast_factor": self._sp_contrast_factor.value(),
                "apply_clahe": self._chk_clahe.isChecked(),
                "clahe_clip_limit": self._sp_clahe_clip.value(),
                "clahe_tile_size": self._sp_clahe_tile.value(),
                "apply_median_blend": self._chk_median_blend.isChecked(),
                "median_blend_frames": self._cb_median_frames.currentData(),
            },
            "ui": {
                "theme": self._cb_theme.currentData(),
                "locale": self._cb_locale.currentData(),
                "font_size": self._sp_font_size.value(),
                "show_ocr_overlay": self._chk_show_ocr_overlay.isChecked(),
                "show_waveform": self._chk_show_waveform.isChecked(),
            },
            "advanced": {
                "log_level": self._cb_log_level.currentData(),
                "save_debug_frames": self._chk_save_debug.isChecked(),
                "debug_frames_dir": self._le_debug_frames_dir.text().strip(),
                "keep_temp_files": self._chk_keep_temp.isChecked(),
                "disable_paddle_network_check": self._chk_disable_pdx_check.isChecked(),
            },
            "video_context": {
                "resolution_height": self._sp_vidctx_height.value(),
                "fps": self._sp_vidctx_fps.value(),
                "nvenc_cq": self._sp_vidctx_cq.value(),
                "cpu_crf": self._sp_vidctx_crf.value(),
                "tokens_per_chunk": self._sp_vidctx_tokens_chunk.value(),
                "max_chunk_minutes": self._sp_vidctx_chunk_min.value(),
                "tokens_per_second": self._sp_vidctx_tps.value(),
            },
            "translation": {
                "default_batch_size": self._sp_trans_batch.value(),
                "default_context_size": self._sp_trans_ctx.value(),
                "retry_count": self._sp_trans_retry.value(),
                "request_timeout_sec": self._sp_trans_timeout.value(),
            },
        }

    def _on_save_clicked(self) -> None:
        self._view_model.save(self._build_patch())

    def _on_cancel_clicked(self) -> None:
        self._view_model.reload()

    def _on_reset_clicked(self) -> None:
        """[V3.15 SAFETY] Hộp thoại Guard cảnh báo người dùng trước khi Reset."""
        reply = QMessageBox.question(
            self,
            self._translator.translate("settings.dlg_reset_confirm_title"),
            self._translator.translate("settings.dlg_reset_confirm_body"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._view_model.reset_to_defaults()

    def _on_saved(self, _snapshot: ApplicationSettings) -> None:
        # [v3.20.3 #4] Toast Fluent thay vì đổi chữ ở góc + khoá nút (Smart Dirty).
        self._mark_clean()
        InfoBar.success(
            self._translator.translate("settings.dlg_saved_title"),
            self._translator.translate("settings.dlg_saved_body"),
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3500,
        )

    def _on_validation_failed(self, message: str) -> None:
        InfoBar.error(
            self._translator.translate("settings.dlg_invalid_title"),
            message,
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=5000,
        )

    def _on_cache_cleaned(self, files_deleted: int, mb_freed: float) -> None:
        InfoBar.success(
            self._translator.translate("settings.dlg_clean_done_title"),
            self._translator.translate("settings.dlg_clean_done_body").replace("{files}", str(files_deleted)).replace("{mb}", f"{mb_freed:.2f}"),
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=4000
        )

    def _on_clean_database_clicked(self) -> None:
        """[v3.23.88] Hỏi xác nhận trước khi xoá sạch toàn bộ dữ liệu database."""
        reply = QMessageBox.warning(
            self,
            self._translator.translate("settings.dlg_cleandb_title"),
            self._translator.translate("settings.dlg_cleandb_body"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._view_model.reset_database()

    def _on_database_reset(self, total_tables: int) -> None:
        if total_tables > 0:
            InfoBar.success(
                self._translator.translate("settings.dlg_cleandb_done_title"),
                self._translator.translate("settings.dlg_cleandb_done_body").replace("{tables}", str(total_tables)),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=6000,
            )
        else:
            InfoBar.warning(
                self._translator.translate("settings.dlg_nothing_title"),
                self._translator.translate("settings.dlg_nothing_body"),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=4000,
            )

    def _refresh_mpv_dll_status(self) -> None:
        from subtitles_extractor.infrastructure.video.mpv_dll_manager import MpvDllManager
        container_dir = self._view_model.user_data_dir
        manager = MpvDllManager(app_data_dir=container_dir)
        status = manager.ensure_available()

        if not status.platform_supported:
            self._lbl_mpv_status.setStyleSheet(f"color: {_c.on_surface_muted()};")
            self._lbl_mpv_status.setText(self._translator.translate("settings.lbl_mpv_linux"))
            self._btn_install_mpv.setEnabled(False)
            self._btn_remove_mpv.setEnabled(False)
            return

        self._btn_install_mpv.setEnabled(True)
        if status.is_available and status.dll_path is not None:
            self._lbl_mpv_status.setStyleSheet(f"color: {_c.success()};")
            self._lbl_mpv_status.setText(self._translator.translate("settings.mpv_installed").replace("{path}", str(status.dll_path)))
            self._btn_remove_mpv.setEnabled(True)
        else:
            self._lbl_mpv_status.setStyleSheet(f"color: {_c.danger()};")
            self._lbl_mpv_status.setText(self._translator.translate("settings.mpv_missing").replace("{dir}", str(manager.dll_dir)))
            self._btn_remove_mpv.setEnabled(False)

    def _on_install_mpv_clicked(self) -> None:
        from PySide6.QtCore import QThread

        from subtitles_extractor.presentation.workers.mpv_dll_download_worker import (
            MpvDllDownloadWorker,
        )
        container_dir = self._view_model.user_data_dir
        self._mpv_progress.setVisible(True)
        self._mpv_progress.setRange(0, 0)
        self._btn_install_mpv.setEnabled(False)
        self._btn_remove_mpv.setEnabled(False)
        self._lbl_mpv_status.setStyleSheet(f"color: {_c.on_surface_muted()};")
        self._lbl_mpv_status.setText(self._translator.translate("settings.lbl_mpv_downloading"))

        thread = QThread(self)
        worker = MpvDllDownloadWorker(app_data_dir=container_dir)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_mpv_download_progress)
        worker.finished.connect(self._on_mpv_install_finished)
        worker.failed.connect(self._on_mpv_install_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._mpv_install_thread = thread
        self._mpv_install_worker = worker
        thread.start()

    def _on_mpv_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self._mpv_progress.setRange(0, total)
            self._mpv_progress.setValue(downloaded)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self._lbl_mpv_status.setText(self._translator.translate("settings.mpv_progress").replace("{done}", f"{mb_done:.1f}").replace("{total}", f"{mb_total:.1f}"))
        else:
            self._mpv_progress.setRange(0, 0)

    def _on_mpv_install_finished(self, _status) -> None:
        self._mpv_progress.setVisible(False)
        self._refresh_mpv_dll_status()
        QMessageBox.information(
            self, self._translator.translate("settings.mpv_done_title"), self._translator.translate("settings.mpv_done_body")
        )

    def _on_mpv_install_failed(self, message: str) -> None:
        self._mpv_progress.setVisible(False)
        self._refresh_mpv_dll_status()
        QMessageBox.critical(self, self._translator.translate("settings.mpv_error_title"), f"✗ {message}")

    def _on_remove_mpv_clicked(self) -> None:
        reply = QMessageBox.question(self, self._translator.translate("settings.mpv_remove_title"), self._translator.translate("settings.mpv_remove_body"))
        if reply != QMessageBox.StandardButton.Yes: return
        from subtitles_extractor.infrastructure.video.mpv_dll_manager import MpvDllManager
        container_dir = self._view_model.user_data_dir
        MpvDllManager(app_data_dir=container_dir).remove()
        self._refresh_mpv_dll_status()

    # ── [v3.23.375] Bật tăng tốc GPU (OCR): tải CUDA runtime lúc chạy ────────────
    def _cuda_dir(self) -> Path:
        """Thư mục sẽ tải CUDA runtime vào: ``<models>/cuda_runtime``."""
        from subtitles_extractor.infrastructure.model_store import (
            ensure_model_store_root,
        )
        from subtitles_extractor.infrastructure.ocr.cuda_runtime_plan import (
            CUDA_RUNTIME_DIRNAME,
        )

        return ensure_model_store_root() / CUDA_RUNTIME_DIRNAME

    def _refresh_gpu_ocr_status(self) -> None:
        """Cập nhật nhãn + trạng thái nút theo tình trạng CUDA runtime."""
        import sys

        from subtitles_extractor.infrastructure.ocr.cuda_runtime_plan import (
            CudaRuntimeStatus,
            evaluate_cuda_runtime,
        )

        meipass = getattr(sys, "_MEIPASS", None)
        bundled = bool(meipass) and (
            Path(meipass) / "nvidia" / "cudnn" / "bin"
        ).is_dir()
        try:
            plan = evaluate_cuda_runtime(self._cuda_dir(), bundled=bundled)
        except OSError:
            return
        if plan.status is CudaRuntimeStatus.BUNDLED:
            self._lbl_gpu_ocr_status.setText(self._translator.translate("settings.gpu_ocr_bundled"))
            self._btn_enable_gpu_ocr.setEnabled(False)
        elif plan.status is CudaRuntimeStatus.INSTALLED:
            self._lbl_gpu_ocr_status.setText(
                self._translator.translate("settings.gpu_ocr_installed")
            )
            self._btn_enable_gpu_ocr.setEnabled(False)
        else:
            self._lbl_gpu_ocr_status.setText(
                self._translator.translate("settings.gpu_ocr_needs")
            )
            self._btn_enable_gpu_ocr.setEnabled(True)

    def _on_enable_gpu_ocr_clicked(self) -> None:
        from subtitles_extractor.infrastructure.process.embedded_python import (
            resolve_installer_python,
        )
        from subtitles_extractor.infrastructure.stt.whisperx_installer import (
            find_system_python,
        )

        if getattr(self, "_cuda_thread", None) is not None:
            return
        # [v3.23.397] Ưu tiên Python embeddable NHÚNG (tự lập, không cần Python hệ thống).
        python_exe = resolve_installer_python(find_system_python())
        if not python_exe:
            QMessageBox.warning(
                self, self._translator.translate("settings.gpu_ocr_no_python_title"),
                self._translator.translate("settings.gpu_ocr_no_python_body"),
            )
            return
        answer = QMessageBox.question(
            self, self._translator.translate("settings.gpu_ocr_confirm_title"),
            self._translator.translate("settings.gpu_ocr_confirm_body"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        from PySide6.QtCore import QThread

        from subtitles_extractor.presentation.workers.install_cuda_runtime_worker import (
            InstallCudaRuntimeWorker,
        )

        self._btn_enable_gpu_ocr.setEnabled(False)
        self._lbl_gpu_ocr_status.setText(self._translator.translate("settings.gpu_ocr_downloading"))
        thread = QThread(self)
        worker = InstallCudaRuntimeWorker(python_exe, self._cuda_dir())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda pct, msg: self._lbl_gpu_ocr_status.setText(f"{msg} ({pct}%)")
        )
        worker.finished.connect(self._on_cuda_install_finished)
        worker.failed.connect(self._on_cuda_install_failed)
        worker.done.connect(self._cleanup_cuda_thread, Qt.ConnectionType.QueuedConnection)
        self._cuda_thread = thread
        self._cuda_worker = worker
        thread.start()

    def _on_cuda_install_finished(self, _cuda_dir: object) -> None:
        self._lbl_gpu_ocr_status.setText(self._translator.translate("settings.gpu_ocr_done_status"))
        QMessageBox.information(
            self, self._translator.translate("settings.gpu_ocr_done_title"),
            self._translator.translate("settings.gpu_ocr_done_body"),
        )

    def _on_cuda_install_failed(self, message: str) -> None:
        self._lbl_gpu_ocr_status.setText(self._translator.translate("settings.gpu_ocr_failed_status"))
        QMessageBox.warning(self, self._translator.translate("settings.gpu_ocr_failed_title"), message)

    def _cleanup_cuda_thread(self) -> None:
        thread = getattr(self, "_cuda_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait(3000)
        self._cuda_thread = None
        self._cuda_worker = None
        self._refresh_gpu_ocr_status()

    # ---- [v3.23.386] Tải lõi paddlepaddle-gpu lúc chạy (mirror mẫu CUDA) ----

    def _paddle_dir(self) -> Path:
        """Thư mục sẽ tải lõi paddle vào: ``<models>/paddle_runtime``."""
        from subtitles_extractor.infrastructure.model_store import (
            ensure_model_store_root,
        )
        from subtitles_extractor.infrastructure.ocr.paddle_runtime_plan import (
            PADDLE_RUNTIME_DIRNAME,
        )

        return ensure_model_store_root() / PADDLE_RUNTIME_DIRNAME

    def _refresh_paddle_status(self) -> None:
        """Cập nhật nhãn + trạng thái nút theo tình trạng lõi paddle."""
        import sys

        from subtitles_extractor.infrastructure.ocr.paddle_runtime_plan import (
            PaddleRuntimeStatus,
            evaluate_paddle_runtime,
        )

        # "Nhúng sẵn" = có gói paddle trong bản đóng gói (_MEIPASS/paddle) — bản đầy đủ.
        meipass = getattr(sys, "_MEIPASS", None)
        bundled = bool(meipass) and (Path(meipass) / "paddle").is_dir()
        try:
            plan = evaluate_paddle_runtime(self._paddle_dir(), bundled=bundled)
        except OSError:
            return
        if plan.status is PaddleRuntimeStatus.BUNDLED:
            self._lbl_paddle_status.setText(self._translator.translate("settings.paddle_bundled"))
            self._btn_download_paddle.setEnabled(False)
        elif plan.status is PaddleRuntimeStatus.INSTALLED:
            self._lbl_paddle_status.setText(self._translator.translate("settings.paddle_installed"))
            self._btn_download_paddle.setEnabled(False)
        else:
            self._lbl_paddle_status.setText(self._translator.translate("settings.paddle_needs"))
            self._btn_download_paddle.setEnabled(True)

    def _on_download_paddle_clicked(self) -> None:
        from subtitles_extractor.infrastructure.process.embedded_python import (
            resolve_installer_python,
        )
        from subtitles_extractor.infrastructure.stt.whisperx_installer import (
            find_system_python,
        )

        if getattr(self, "_paddle_thread", None) is not None:
            return
        # [v3.23.397] Ưu tiên Python embeddable NHÚNG (tự lập, không cần Python hệ thống).
        python_exe = resolve_installer_python(find_system_python())
        if not python_exe:
            QMessageBox.warning(
                self, self._translator.translate("settings.gpu_ocr_no_python_title"),
                self._translator.translate("settings.gpu_ocr_no_python_body"),
            )
            return
        answer = QMessageBox.question(
            self, self._translator.translate("settings.paddle_confirm_title"),
            self._translator.translate("settings.paddle_confirm_body"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        from PySide6.QtCore import QThread

        from subtitles_extractor.presentation.workers.install_paddle_runtime_worker import (
            InstallPaddleRuntimeWorker,
        )

        self._btn_download_paddle.setEnabled(False)
        self._lbl_paddle_status.setText(self._translator.translate("settings.paddle_downloading"))
        thread = QThread(self)
        worker = InstallPaddleRuntimeWorker(python_exe, self._paddle_dir())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda pct, msg: self._lbl_paddle_status.setText(f"{msg} ({pct}%)")
        )
        worker.finished.connect(self._on_paddle_install_finished)
        worker.failed.connect(self._on_paddle_install_failed)
        worker.done.connect(self._cleanup_paddle_thread, Qt.ConnectionType.QueuedConnection)
        self._paddle_thread = thread
        self._paddle_worker = worker
        thread.start()

    def _on_paddle_install_finished(self, _paddle_dir: object) -> None:
        self._lbl_paddle_status.setText(self._translator.translate("settings.paddle_done_status"))
        QMessageBox.information(
            self, self._translator.translate("settings.paddle_done_title"),
            self._translator.translate("settings.paddle_done_body"),
        )

    def _on_paddle_install_failed(self, message: str) -> None:
        self._lbl_paddle_status.setText(self._translator.translate("settings.paddle_failed_status"))
        QMessageBox.warning(self, self._translator.translate("settings.paddle_failed_title"), message)

    def _cleanup_paddle_thread(self) -> None:
        thread = getattr(self, "_paddle_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait(3000)
        self._paddle_thread = None
        self._paddle_worker = None
        self._refresh_paddle_status()

def _make_double_spin(minimum: float, maximum: float, step: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    if suffix: spin.setSuffix(suffix)
    return spin

def _set_combo(combo: QComboBox, data: Any) -> None:
    if data is None: return
    idx = combo.findData(data)
    if idx >= 0: combo.setCurrentIndex(idx)

__all__ = ["SettingsPage"]
