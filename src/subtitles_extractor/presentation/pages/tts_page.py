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

from PySide6.QtCore import Qt, QSettings, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QScrollArea, QSpinBox,
    QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)
from subtitles_extractor.presentation.fluent_compat import (
    PrimaryPushButton, PushButton, ToolButton,
)

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSSegmentResult
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import _EDGE_VOICE_MAP
from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import (
    GEMINI_TTS_VOICES, GEMINI_TTS_MODELS,
)
from subtitles_extractor.presentation.view_models.tts_page_view_model import TTSPageViewModel
from subtitles_extractor.presentation.utils import safe_dialogs as _safe_dialogs
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets
from subtitles_extractor.presentation.theme.styles import caption_style
from subtitles_extractor.presentation.theme import feedback as _feedback
from subtitles_extractor.presentation.utils.error_humanizer import (
    humanize_gemini_error as _humanize_gemini_error,
)
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.widgets.section_card import SectionCard
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name
from subtitles_extractor.infrastructure.process.hidden_process import (
    no_window_kwargs,
)

logger = logging.getLogger(__name__)


class _VieNeuVoiceLoader(QThread):
    """[v3.23.200] Nạp danh sách giọng VieNeu ở BACKGROUND thread (hết khựng UI ~9s).

    ``list_speakers`` phải nạp model VieNeu (llama+codec, ~9s lần đầu) — chạy trên UI
    thread làm ứng dụng đứng hình khi người dùng đổi engine/chế độ (log thực tế
    11:35:55 -> 11:36:03). Worker này nạp ở thread nền rồi phát tín hiệu về UI thread;
    kết quả kèm ``mode`` để UI bỏ qua kết quả STALE khi người dùng đã đổi chế độ giữa
    chừng. Model được cache thread-safe (``_ENGINE_LOCK``) nên Generate chạy song song
    không nạp trùng.
    """

    voices_loaded = Signal(str, list)  # (mode, danh sách tên giọng)
    load_failed = Signal(str)          # (mode)

    def __init__(self, container, mode: str, parent=None) -> None:
        super().__init__(parent)
        self._container = container
        self._mode = mode

    def run(self) -> None:  # noqa: D102 — QThread API
        try:
            adapter = self._container.make_vieneu_tts_adapter(mode=self._mode)
            if not adapter.is_available():
                self.load_failed.emit(self._mode)
                return
            voice_ids = adapter.list_speakers("vi-VN") or []
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("Nạp giọng VieNeu (nền) lỗi: %s", exc)
            self.load_failed.emit(self._mode)
            return
        self.voices_loaded.emit(self._mode, list(voice_ids))

# [v3.23.206] Ngưỡng CẢNH BÁO lấn (giây): câu chồng sang câu sau quá mức này được tô
# nhãn "Lấn" trong bảng kết quả + lọt bộ lọc "Có vấn đề" -> người dùng scan nhanh câu
# cần rút gọn bản dịch (ca thực tế #62: thoại ~4.9s trong khung 1.28s, kịch trần nén
# vẫn dư 1.26s). Lấn nhỏ hơn thường rơi vào khoảng lặng, vô hại.
_OVERLAP_WARN_S = 0.5

# [v3.23.214] Các tuỳ chọn CHỈ engine Edge đọc (VieNeu/Gemini bỏ qua hoàn toàn — đã
# đối chiếu mã nguồn 3 adapter). Trước đây UI hiện chúng cho MỌI engine -> người dùng
# chỉnh "Elastic timing", "Double Pass"… khi chạy VieNeu và tưởng có tác dụng, thực tế
# vô hiệu; debug config cũng in ra gây hiểu nhầm khi phân tích. Nay: tự vô hiệu hoá +
# ghi rõ "(chỉ Edge)" khi engine khác được chọn.
_EDGE_ONLY_HINT = " — chỉ áp dụng cho Edge TTS"

# Các trường TTSRequest CHỈ Edge đọc (đã quét mã 3 adapter). Khi engine khác chạy,
# debug config phải ĐÁNH DẤU rõ để không gây hiểu nhầm lúc phân tích kết quả.
_EDGE_ONLY_FIELDS = frozenset({
    "edge_concurrency", "timing_strategy", "elastic_timing", "double_pass",
    "high_quality", "max_drift_s", "last_line_max_extend_s",
    "min_stretch_ratio", "comfort_speed_ratio", "min_pause_ratio",
    "max_intra_gap_s", "anchor_gap_s", "max_segment_s",
})
# [v3.23.218] ``lead_in_s`` ("Ăn gian đầu") ĐÃ RA KHỎI danh sách trên: VieNeu/Gemini nay
# cũng dùng nó để đọc sớm vào khoảng lặng câu trước -> nén nhẹ hơn, giọng rõ hơn.


_ENGINE_EDGE   = "edge"
_ENGINE_GEMINI = "gemini"
_ENGINE_VIENEU = "vieneu"

_ENGINES = [
    (_ENGINE_EDGE,   "🌐 Edge TTS",        "Microsoft Edge Neural TTS (online, Tiếng Việt)"),
    (_ENGINE_GEMINI, "✨ Gemini TTS",       "Google Gemini Neural TTS (online, 30 voices)"),
    (_ENGINE_VIENEU, "🇻🇳 VieNeu-TTS",      "TTS tiếng Việt on-device, voice cloning (offline)"),
]

_EDGE_LANGUAGES = list(_EDGE_VOICE_MAP.keys())
_EDGE_LANGUAGE_DISPLAY = {
    "vi-VN": "🇻🇳 Tiếng Việt", "zh-CN": "🇨🇳 Trung (Giản thể)",
    "zh-TW": "🇹🇼 Trung (Phồn thể)", "en-US": "🇺🇸 English (US)",
    "en-GB": "🇬🇧 English (UK)", "ja-JP": "🇯🇵 日本語",
    "ko-KR": "🇰🇷 한국어", "fr-FR": "🇫🇷 Français",
    "es-ES": "🇪🇸 Español", "de-DE": "🇩🇪 Deutsch", "th-TH": "🇹🇭 Thai",
}


class TTSPage(QWidget):
    """Trang TTS phụ đề → WAV."""

    # Phát khi tạo TTS xong, mang đường dẫn file WAV — để liên thông lưu dự án.
    tts_completed = Signal(str)

    def __init__(self, container: ApplicationContainer, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ttsPage")
        self._container = container
        # [v3.23.372] Translator để externalize chuỗi UI (đa ngôn ngữ).
        self._translator = container.translator
        self._settings = QSettings("SubtitlesExtractorPaddleOCR", "TTSPage_v2")
        self._view_model = TTSPageViewModel(container)
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

    def _make_config_tab(self, *widgets: QWidget) -> QScrollArea:
        """Bọc các nhóm cấu hình vào tab cuộn dọc (mỗi tab ngắn, đỡ phải cuộn nhiều)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(4, 8, 8, 8)
        col.setSpacing(8)
        for w in widgets:
            col.addWidget(w)
        col.addStretch(1)
        scroll.setWidget(holder)
        return scroll

    def _build_ui(self) -> None:
        from PySide6.QtWidgets import QSplitter, QTabWidget

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # [v3.23.119] TRÁI = cấu hình gom theo TAB (mỗi tab ngắn) + nút chạy chung dưới;
        # PHẢI = kết quả (luôn thấy). Trước đây xếp dọc tất cả nên phải cuộn nhiều.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(0, 0, 8, 0)
        left_col.setSpacing(8)

        self._config_tabs = QTabWidget()
        self._config_tabs.addTab(
            self._make_config_tab(
                self._build_source_group(),
                self._build_engine_selector(),
                self._build_engine_stack(),
            ),
            "🎙 Engine & Nguồn",
        )
        self._config_tabs.addTab(
            self._make_config_tab(
                self._build_common_config(), self._build_output_group()
            ),
            "🎚 Âm thanh & Xuất",
        )
        left_col.addWidget(self._config_tabs, 1)
        left_col.addWidget(self._build_actions_group())
        left.setMinimumWidth(480)
        splitter.addWidget(left)

        result_panel = self._build_result_group()
        result_panel.setMinimumWidth(380)
        splitter.addWidget(result_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 700])
        root.addWidget(splitter, 1)

    # ── Source ────────────────────────────────────────────────────────────────

    def _build_source_group(self) -> QWidget:
        card = SectionCard(self._translator.translate("tts.tts_sec_source"))
        col = QVBoxLayout()
        col.setSpacing(_m.SPACING_SM)
        row = QHBoxLayout()
        row.setSpacing(_m.SPACING_SM)
        self._btn_load   = PushButton(self._translator.translate("tts.btn_load_srt"))
        self._btn_editor = PushButton(self._translator.translate("tts.btn_editor"))
        self._btn_trans  = PushButton(self._translator.translate("tts.btn_translation"))
        for btn in (self._btn_load, self._btn_editor, self._btn_trans):
            row.addWidget(btn, 1)
        col.addLayout(row)
        self._lbl_source = QLabel(self._translator.translate("tts.no_subtitle"))
        self._lbl_source.setStyleSheet("font-weight:bold;")
        self._lbl_source.setWordWrap(True)
        col.addWidget(self._lbl_source)
        self._btn_load.clicked.connect(self._on_load_clicked)
        self._btn_editor.clicked.connect(self._on_pull_editor)
        self._btn_trans.clicked.connect(self._on_pull_translate)
        card.add_layout(col)
        return card

    # ── Engine selector ───────────────────────────────────────────────────────

    def _build_engine_selector(self) -> QWidget:
        g = SectionCard("Engine TTS")
        lay = QVBoxLayout()
        lay.setSpacing(_m.SPACING_SM)
        # Combobox chọn engine
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Engine:"))
        self._engine_combo = QComboBox()
        for _id, name, tip in _ENGINES:
            self._engine_combo.addItem(name, userData=_id)
        self._engine_combo.setToolTip(self._translator.translate("tts.tip_engine"))
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        sel_row.addWidget(self._engine_combo, 1)
        lay.addLayout(sel_row)
        # Status
        status_row = QHBoxLayout()
        self._lbl_edge_st   = QLabel(self._translator.translate("tts.probe_edge"))
        self._lbl_gemini_st = QLabel(self._translator.translate("tts.probe_gemini"))
        self._lbl_vieneu_st = QLabel(self._translator.translate("tts.probe_vieneu"))
        for lbl in (self._lbl_edge_st, self._lbl_gemini_st, self._lbl_vieneu_st):
            lbl.setStyleSheet(caption_style())
            status_row.addWidget(lbl)
        status_row.addStretch()
        lay.addLayout(status_row)
        # Install hint
        hint = QLabel(
            "  💡 Edge TTS: <code>pip install edge-tts soundfile</code>"
            " &nbsp;|&nbsp; Gemini TTS: <code>pip install google-genai soundfile</code>"
            " &nbsp;|&nbsp; VieNeu-TTS: <code>pip install vieneu soundfile</code>"
        )
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setStyleSheet(caption_style())
        hint.setWordWrap(True)
        lay.addWidget(hint)
        g.add_layout(lay)
        return g

    # ── Engine-specific config panels ─────────────────────────────────────────

    def _build_engine_stack(self) -> QWidget:
        self._engine_stack = QStackedWidget()
        self._engine_stack.addWidget(self._build_edge_panel())    # idx 0
        self._engine_stack.addWidget(self._build_gemini_panel())  # idx 1
        self._engine_stack.addWidget(self._build_vieneu_panel())  # idx 2
        return self._engine_stack

    def _build_edge_panel(self) -> QWidget:
        w = SectionCard(self._translator.translate("tts.tts_sec_edge"))
        form = QFormLayout()
        form.setSpacing(_m.SPACING_SM)
        self._edge_lang = QComboBox()
        items = [(_EDGE_LANGUAGE_DISPLAY.get(k, k), k) for k in _EDGE_VOICE_MAP]
        for disp, val in items:
            self._edge_lang.addItem(disp, userData=val)
        self._edge_lang.currentIndexChanged.connect(self._update_edge_speakers)
        form.addRow(self._translator.translate("tts.tts_lang"), self._edge_lang)
        self._edge_voice = QComboBox()
        self._edge_voice.setEditable(True)
        form.addRow(self._translator.translate("tts.tts_voice"), self._edge_voice)

        self._edge_concurrency = QSpinBox()
        self._edge_concurrency.setRange(1, 64); self._edge_concurrency.setValue(16)
        self._edge_concurrency.setToolTip(
            "Số request Edge TTS chạy song song (1–64).\n"
            "Cao hơn = nhanh hơn, nhưng quá cao (>32) Microsoft có thể\n"
            "rate-limit (lỗi 429) hoặc rớt kết nối. Khuyến nghị 16–32.\n"
            "Nếu gặp nhiều lỗi/retry, giảm xuống."
        )
        form.addRow(self._translator.translate("tts.tts_concurrency"), self._edge_concurrency)

        info = QLabel(
            "ℹ️  Tốc độ đọc qua API (server-side) GIỮ PITCH + ngữ điệu tự nhiên,\n"
            "   chất lượng tốt hơn time-stretch → ưu tiên dùng tới 3.0×.\n"
            "   Elastic timing: dời câu trong dung sai thay vì nén; tự re-sync ở gap.\n"
            "   Chống cắt: audio tràn nhẹ thay vì cắt nội dung (bảo toàn 100%).\n"
            "   Chất lượng wave: RMS đồng đều + fade chống click + soft-limit.\n"
            "   Tốt nhất: <code>pip install pedalboard</code> (Rubber Band — giữ rõ giọng khi nén mạnh; lưu ý license GPL)."
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setWordWrap(True)
        info.setStyleSheet(caption_style())
        form.addRow(info)
        # [v3.23.67] Gắn form vào thẻ — nếu thiếu, SectionCard rỗng (chỉ còn tiêu đề).
        w.add_layout(form)
        return w

    def _build_gemini_panel(self) -> QWidget:
        w = SectionCard(self._translator.translate("tts.tts_sec_gemini"))
        form = QFormLayout()
        form.setSpacing(_m.SPACING_SM)
        # API Key
        key_row = QHBoxLayout()
        self._gemini_key = QLineEdit()
        self._gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._gemini_key.setPlaceholderText(self._translator.translate("tts.gemini_key_ph"))
        self._btn_eye = ToolButton()
        self._btn_eye.setIcon(__import__('subtitles_extractor.presentation.fluent_compat', fromlist=['FluentIcon']).FluentIcon.VIEW.icon())
        set_accessible_name(self._btn_eye, self._translator.translate("tts.tts_acc_eye"))
        self._btn_eye.setCheckable(True)
        self._btn_eye.toggled.connect(
            lambda on: self._gemini_key.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(self._gemini_key, 1)
        key_row.addWidget(self._btn_eye)
        form.addRow("API Key:", self._wrap(key_row))

        self._gemini_model = QComboBox()
        self._gemini_model.addItems(GEMINI_TTS_MODELS)
        self._gemini_model.setToolTip(
            "Standard TTS: nhanh, ổn định, 1 API call/dòng\n"
            "Native Audio Dialog: tự nhiên hơn, cảm xúc hơn, dùng Live API (WebSocket)"
        )
        self._gemini_model.setToolTip(
            "Standard TTS: nhanh, ổn định, 1 API call/dòng\n"
            "Native Audio Dialog (Live API): giọng tự nhiên, cảm xúc hơn\n"
            "  ★ gemini-2.5-flash-native-audio-latest — Recommended tracking alias\n"
            "  • gemini-2.5-flash-preview-native-audio-dialog — Affective Dialog (v1alpha)\n"
            "  • gemini-live-2.5-flash-native-audio — Stable pinned\n"
            "  • gemini-2.5-flash-native-audio-preview-12-2025 — Preview 12-2025"
        )
        self._gemini_model.currentIndexChanged.connect(self._on_gemini_model_changed)
        form.addRow("Model:", self._gemini_model)

        self._gemini_voice = QComboBox()
        self._gemini_voice.addItems(GEMINI_TTS_VOICES)
        self._gemini_voice.setToolTip(
            "Mô tả tính cách giọng (theo tài liệu Google):\n"
            "Zephyr — tươi sáng  |  Puck — sôi nổi (mặc định)\n"
            "Kore — chắc chắn, tự tin  |  Charon — rõ ràng, cung cấp thông tin\n"
            "Aoede — nhẹ nhàng, tự nhiên  |  Fenrir — dynamic, hào hứng\n"
            "Orus — dứt khoát  |  Leda — trẻ trung  |  Autonoe — lạc quan\n"
            "Callirrhoe — thoải mái  |  Achernar — mềm mại  |  Sulafat — ấm áp"
        )
        form.addRow(self._translator.translate("tts.tts_voice"), self._gemini_voice)

        # Native Audio: style prompt + affective dialog
        self._gemini_native_group = SectionCard(self._translator.translate("tts.tts_sec_native"))
        native_form = QFormLayout()
        native_form.setSpacing(_m.SPACING_SM)
        self._gemini_style = QTextEdit()
        self._gemini_style.setMaximumHeight(65)
        self._gemini_style.setPlaceholderText(
            "Phong cách đọc — để trống dùng mặc định:\n"
            "\"Đọc phụ đề phim tự nhiên, truyền cảm, phù hợp cảm xúc từng dòng.\""
        )
        native_form.addRow(self._translator.translate("tts.tts_gstyle"), self._gemini_style)
        self._gemini_affective = QCheckBox(self._translator.translate("tts.opt_affective"))
        self._gemini_affective.setChecked(True)
        self._gemini_affective.setToolTip(
            "Model hiểu cảm xúc trong văn bản và điều chỉnh giọng đọc tương ứng.\n"
            "VD: cảnh hành động → giọng căng thẳng, cảnh buồn → giọng nhẹ nhàng."
        )
        native_form.addRow(self._gemini_affective)
        # [v3.23.248] Nhiệt độ lấy mẫu — để "Tự động" (min) = dùng mặc định của model.
        # Hạ thấp làm giọng ỔN ĐỊNH hơn, giảm "ngân dài ngẫu nhiên" (hallucination); quá
        # thấp có thể bớt biểu cảm. specialValueText hiển thị "Tự động" khi ở giá trị min.
        self._gemini_temperature = QDoubleSpinBox()
        self._gemini_temperature.setRange(-0.1, 2.0)
        self._gemini_temperature.setSingleStep(0.1)
        self._gemini_temperature.setDecimals(1)
        self._gemini_temperature.setValue(-0.1)  # min -> hiển thị "Tự động"
        self._gemini_temperature.setSpecialValueText("Tự động (mặc định model)")
        self._gemini_temperature.setToolTip(
            "Nhiệt độ lấy mẫu (0.0-2.0). Để 'Tự động' dùng mặc định của model.\n"
            "Hạ thấp (vd 0.7) → giọng ổn định hơn, ít 'ngân dài ngẫu nhiên'.\n"
            "Quá thấp có thể làm giọng bớt biểu cảm."
        )
        native_form.addRow(self._translator.translate("tts.tts_gtemp"), self._gemini_temperature)
        info = QLabel(
            "ℹ️  Native Audio Dialog dùng Live API WebSocket — mỗi batch 40 dòng/session.\n"
            "   Affective Dialog chỉ hoạt động với model 'native-audio-dialog' (api_version=v1alpha).\n"
            "   Các model khác sẽ tự động bỏ qua cờ này."
        )
        info.setStyleSheet(caption_style())
        info.setWordWrap(True)
        native_form.addRow(info)
        # [v3.23.67] Gắn native_form vào thẻ Native Audio. THIẾU dòng này khiến thẻ rỗng VÀ
        # (do thẻ chưa có cha) lệnh setVisible(True) ở _on_gemini_model_changed bật nó thành
        # CỬA SỔ NỔI riêng lúc khởi động.
        self._gemini_native_group.add_layout(native_form)
        form.addRow(self._gemini_native_group)
        # [v3.23.67] Gắn form ngoài vào thẻ — đồng thời reparent _gemini_native_group vào w,
        # để setVisible chỉ ẩn/hiện trong thẻ thay vì tạo cửa sổ rời.
        w.add_layout(form)
        return w

    def _build_vieneu_panel(self) -> QWidget:
        """Panel cấu hình VieNeu-TTS: chế độ, sắc thái, và chọn giọng preset/cloning."""
        w = SectionCard(self._translator.translate("tts.tts_sec_vieneu"))
        form = QFormLayout()
        form.setSpacing(_m.SPACING_SM)

        # Chế độ engine (chất lượng vs tốc độ).
        self._vieneu_mode = QComboBox()
        self._vieneu_mode.addItem(self._translator.translate("tts.vn_mode_standard"), userData="standard")
        self._vieneu_mode.addItem(self._translator.translate("tts.vn_mode_turbo"), userData="turbo")
        self._vieneu_mode.addItem(self._translator.translate("tts.vn_mode_v3turbo"), userData="v3turbo")
        self._vieneu_mode.setToolTip(
            "Standard: chất lượng cao nhất (khuyến nghị cho phụ đề — nhiều câu ngắn).\n"
            "Turbo: nhanh hơn NHƯNG chất lượng thấp hơn và có thể lỗi/nhiễu với câu "
            "rất ngắn (< 5 từ) — theo cảnh báo của tác giả VieNeu.\n"
            "v3 Turbo: 48kHz, thử nghiệm."
        )
        self._vieneu_mode.currentIndexChanged.connect(self._on_vieneu_mode_changed)
        form.addRow(self._translator.translate("tts.tts_mode"), self._vieneu_mode)

        # Sắc thái giọng.
        self._vieneu_emotion = QComboBox()
        self._vieneu_emotion.addItem(self._translator.translate("tts.vn_emo_natural"), userData="natural")
        self._vieneu_emotion.addItem(self._translator.translate("tts.vn_emo_story"), userData="storytelling")
        self._vieneu_emotion.setToolTip(self._translator.translate("tts.tip_emotion"))
        form.addRow(self._translator.translate("tts.tts_emotion"), self._vieneu_emotion)

        # Ép chạy CPU/ONNX — né lỗi PyTorch/cuDNN (WinError 127) trên máy CUDA lệch bản.
        self._vieneu_force_cpu = QCheckBox(self._translator.translate("tts.opt_force_cpu"))
        self._vieneu_force_cpu.setChecked(True)
        self._vieneu_force_cpu.setToolTip(
            "Bật (khuyến nghị): chạy bằng ONNX Runtime trên CPU, né lỗi tải DLL "
            "PyTorch/cuDNN.\nTắt: thử dùng GPU — nhưng chỉ có tác dụng khi torch đã "
            "được cài (xem ghi chú bên dưới)."
        )
        form.addRow(self._translator.translate("tts.tts_device"), self._vieneu_force_cpu)

        # [v3.23.342] NÓI THẬT về khả năng dùng GPU. Trước đây bỏ tick ô trên KHÔNG có
        # tác dụng gì trong bản đóng gói: torch bị loại khỏi bundle, nên VieNeu gặp
        # ImportError rồi ÂM THẦM lùi về CPU — ô đánh dấu hứa điều nó không làm được.
        self._vieneu_gpu_note = QLabel("")
        self._vieneu_gpu_note.setWordWrap(True)
        self._vieneu_gpu_note.setStyleSheet(caption_style())
        form.addRow("", self._vieneu_gpu_note)

        # [v3.23.343] Nút TỰ CÀI thay vì bắt người dùng gõ lệnh tay. An toàn vì cài vào
        # môi trường RIÊNG `whisperx_env` — không đụng môi trường của ứng dụng.
        self._btn_install_gpu = PushButton(self._translator.translate("tts.btn_install_vieneu_gpu"))
        self._btn_install_gpu.setToolTip(
            "Thêm gói vieneu + sea-g2p vào môi trường riêng đã có torch.\n"
            "Nhẹ hơn nhiều so với cài WhisperX vì KHÔNG phải tải lại torch (~3GB)."
        )
        self._btn_install_gpu.clicked.connect(self._on_install_vieneu_gpu_clicked)
        form.addRow("", self._btn_install_gpu)

        self._gpu_install_progress = QProgressBar()
        self._gpu_install_progress.setRange(0, 100)
        self._gpu_install_progress.setVisible(False)
        form.addRow("", self._gpu_install_progress)

        self._refresh_vieneu_gpu_note()
        self._vieneu_force_cpu.toggled.connect(
            lambda _checked: self._refresh_vieneu_gpu_note()
        )

        # Giọng preset (điền động khi engine sẵn sàng).
        self._vieneu_voice = QComboBox()
        self._vieneu_voice.addItem(self._translator.translate("tts.voice_default"), userData="")
        self._vieneu_voice.setToolTip(self._translator.translate("tts.tip_voice"))
        form.addRow(self._translator.translate("tts.tts_preset"), self._vieneu_voice)

        # Voice cloning (tuỳ chọn — ưu tiên hơn preset nếu có).
        ref_row = QHBoxLayout()
        self._vieneu_ref_path = QLineEdit()
        self._vieneu_ref_path.setPlaceholderText(self._translator.translate("tts.clone_ph"))
        self._btn_vieneu_ref = PushButton(self._translator.translate("tts.btn_choose"))
        self._btn_vieneu_ref.clicked.connect(self._on_browse_vieneu_ref)
        ref_row.addWidget(self._vieneu_ref_path, 1)
        ref_row.addWidget(self._btn_vieneu_ref)
        form.addRow(self._translator.translate("tts.tts_clone"), self._wrap(ref_row))

        lbl = QLabel(
            "ℹ️  Chạy offline. Nếu chọn file nhân bản giọng, sẽ ưu tiên hơn giọng preset.\n"
            "✓  Model VieNeu dùng license Apache 2.0 — miễn phí, kể cả mục đích thương mại."
        )
        lbl.setStyleSheet(caption_style())
        lbl.setWordWrap(True)
        form.addRow(lbl)
        w.add_layout(form)
        return w

    def _build_common_config(self) -> QWidget:
        g = SectionCard(self._translator.translate("tts.tts_sec_common"))
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setSpacing(_m.SPACING_SM)
        # Speed
        spd_row = QHBoxLayout()
        self._speed = QDoubleSpinBox(); self._speed.setRange(0.5, 2.0)
        self._speed.setSingleStep(0.1); self._speed.setDecimals(1)
        # [v3.23.216] Mặc định 1.6 -> 1.1: từ v215 base_speed NÉN THẬT audio (trước
        # chỉ là nhãn ở VieNeu/Gemini) -> 1.6x làm mất ~8%% độ sắc phụ âm cho MỌI
        # câu (đo trên giọng thật; 1.1x chỉ mất ~5%%). Cũng khớp khuyến nghị
        # 1.0-1.2 ghi ngay trong tooltip bên dưới (trước đây MÂU THUẪN).
        self._speed.setValue(1.1)
        self._speed.setToolTip(
            "Tốc độ đọc CƠ BẢN — cũng là tốc độ TỐI THIỂU (sàn) của mọi câu.\n"
            "1.0 = bình thường, tự nhiên nhất.\n\n"
            "⚠️ QUAN TRỌNG: đặt CAO (vd 1.6) làm hẹp biên độ điều chỉnh (1.6→max),\n"
            "khiến câu dịch dài KHÔNG đủ chỗ nén → lệch đồng bộ (drift) TĂNG MẠNH.\n"
            "Khuyến nghị để 1.0–1.2 và để thuật toán tự nén khi cần — vừa tự nhiên,\n"
            "vừa giữ đồng bộ tốt nhất. Chỉ tăng nếu muốn cố tình đọc nhanh toàn bộ."
        )
        self._max_speed = QDoubleSpinBox(); self._max_speed.setRange(1.0, 5.0)
        # [v3.23.216] Nói rõ TRẦN CHẤT LƯỢNG: VieNeu/Gemini nén hậu kỳ nên bị chặn cứng
        # ở 2.0x (nén hơn làm tan formant — đo v201: mất 19%% độ sắc @2.0, 35%% @2.5)
        # -> đặt max > 2.0 KHÔNG có tác dụng với 2 engine này (Edge dùng được cao hơn
        # nhờ đọc nhanh phía máy chủ, không phải nén hậu kỳ).
        self._max_speed.setToolTip(
            "Trần tốc độ đọc TỔNG của một câu.\n\n"
            "• Edge TTS: dùng được tới 5.0× (đọc nhanh phía máy chủ, chất lượng tốt).\n"
            "• VieNeu / Gemini: nén hậu kỳ nên bị chặn ở 2.0× — đặt cao hơn KHÔNG có\n"
            "  tác dụng (nén quá 2× làm giọng mất rõ, thành 'tiếng gió').\n\n"
            "Câu quá dài không nén đủ sẽ TRÀN sang câu sau (nếu bật 'Cho phép chồng\n"
            "tiếng') — trọn nội dung, ưu tiên nghe rõ."
        )
        self._max_speed.setSingleStep(0.1); self._max_speed.setDecimals(1); self._max_speed.setValue(2.0)
        self._max_speed.setToolTip(
            "Tốc độ tối đa để khớp timing (post-processing time-stretch).\n"
            "Edge TTS: ưu tiên rate qua API tới 3.0× (giữ pitch tự nhiên).\n"
            "Gemini/VieNeu: time-stretch sau generate, 2-3× thường vẫn rõ ràng.\n"
            "Luôn ≥ tốc độ cơ bản."
        )
        # Ràng buộc UX: max_speed không bao giờ nhỏ hơn tốc độ cơ bản, tránh cấu
        # hình mâu thuẫn (base > max làm logic tốc độ vô nghĩa).
        self._speed.valueChanged.connect(self._on_base_speed_changed)
        self._max_speed.setMinimum(self._speed.value())
        spd_row.addWidget(self._speed); spd_row.addWidget(QLabel(self._translator.translate("tts.lbl_max"))); spd_row.addWidget(self._max_speed); spd_row.addWidget(QLabel("×")); spd_row.addStretch()
        form.addRow(self._translator.translate("tts.tts_speed_max"), self._wrap(spd_row))
        # Retry
        retry_row = QHBoxLayout()
        self._retry = QSpinBox(); self._retry.setRange(1, 10); self._retry.setValue(10)
        self._retry.setToolTip(self._translator.translate("tts.tip_retry"))
        self._retry_delay = QDoubleSpinBox(); self._retry_delay.setRange(0.5, 5.0)
        self._retry_delay.setSingleStep(0.5); self._retry_delay.setDecimals(1); self._retry_delay.setValue(2.0)
        self._retry_delay.setToolTip(self._translator.translate("tts.tip_delay"))
        retry_row.addWidget(self._retry); retry_row.addWidget(QLabel(self._translator.translate("tts.lbl_delay")))
        retry_row.addWidget(self._retry_delay); retry_row.addWidget(QLabel("s")); retry_row.addStretch()
        form.addRow(self._translator.translate("tts.tts_retry"), self._wrap(retry_row))
        # Normalize + loudness LUFS + device
        opt_row = QHBoxLayout()
        self._strategy = QComboBox()
        self._strategy.addItem(self._translator.translate("tts.strat_lipsync"), "lipsync")
        self._strategy.addItem(self._translator.translate("tts.strat_balanced"), "balanced")
        self._strategy.addItem(self._translator.translate("tts.strat_smooth"), "smooth")
        self._strategy.setToolTip(
            "Chiến lược căn thời gian thuyết minh:\n"
            "• Bám mốc (Lipsync chặt): mỗi câu khớp cả mốc đầu lẫn cuối phim, vượt\n"
            "  max mới mượn khoảng lặng. Đồng bộ hình–tiếng nhất, có thể chồng tiếng\n"
            "  và đọc nhanh khi thoại dày. Hợp phim cần khẩu hình chính xác.\n"
            "• Cân bằng: bám mốc đầu, mượn khoảng lặng để đọc thong thả hơn; mốc cuối\n"
            "  lỏng hơn. Phù hợp đa số phim.\n"
            "• Giọng mượt: KHÔNG chồng tiếng, dồn mốc + đọc chậm rõ từng câu; chấp\n"
            "  nhận lệch mốc nhiều hơn. Hợp phim ít thoại, ưu tiên nghe rõ."
        )
        opt_row.addWidget(QLabel(self._translator.translate("tts.lbl_strategy"))); opt_row.addWidget(self._strategy)
        self._normalize = QCheckBox(self._translator.translate("tts.opt_normalize"))
        self._normalize.setChecked(True)
        self._normalize.setToolTip(
            "Chuẩn hoá loudness: RMS đồng đều giữa câu + LUFS tổng thể (EBU R128)."
        )
        opt_row.addWidget(self._normalize)
        self._voice_clarity = QCheckBox(self._translator.translate("tts.opt_clarify"))
        self._voice_clarity.setChecked(True)
        self._voice_clarity.setToolTip(
            "Lọc bass rumble (<85Hz) — loại tiếng ù/ồn tần số thấp,\n"
            "giọng nói trong và rõ hơn. Tắt nếu muốn giữ nguyên dải trầm."
        )
        opt_row.addWidget(self._voice_clarity)
        self._high_quality = QCheckBox(self._translator.translate("tts.opt_hq"))
        self._high_quality.setChecked(True)
        self._high_quality.setToolTip(
            "Nén tốc độ CHỈ qua Edge TTS rate (server-side, giữ cao độ, chất lượng\n"
            "cao nhất), KHÔNG time-stretch trên máy (vốn gây rung/méo nhẹ).\n"
            "Đánh đổi: timing có thể lệch nhẹ ở câu cực tải (tự khớp lại ở khoảng lặng).\n"
            "Khuyến nghị BẬT để giọng tự nhiên nhất."
        )
        opt_row.addWidget(self._high_quality)
        self._allow_overlap = QCheckBox(self._translator.translate("tts.opt_overlap"))
        self._allow_overlap.setChecked(True)
        self._strategy.setCurrentIndex(0)
        self._allow_overlap.setToolTip(
            "Neo mỗi câu đúng mốc thời gian gốc để giữ đồng bộ hình–tiếng (lipsync).\n"
            "Khi câu trước chưa dứt mà câu sau tới mốc, cho phép chồng tiếng TTS.\n"
            "Phụ đề .srt xuất kèm LUÔN được cắt không chồng lấn. Tắt → câu sau lùi\n"
            "lại chờ câu trước xong (không chồng tiếng nhưng có thể lệch mốc)."
        )
        opt_row.addWidget(self._allow_overlap)
        self._lufs = QComboBox()
        # (nhãn hiển thị, giá trị LUFS) — sắp từ TO → nhỏ
        self._lufs.addItem(self._translator.translate("tts.lufs_12"), -12.0)
        self._lufs.addItem(self._translator.translate("tts.lufs_14"), -14.0)
        self._lufs.addItem(self._translator.translate("tts.lufs_16"), -16.0)
        self._lufs.addItem(self._translator.translate("tts.lufs_23"), -23.0)
        self._lufs.addItem(self._translator.translate("tts.lufs_off"), 0.0)
        self._lufs.setCurrentIndex(0)  # mặc định -12 LUFS: to/nổi rõ mà true-peak vẫn an toàn
        self._lufs.setToolTip(
            "Mức loudness mục tiêu cho toàn bộ track theo chuẩn EBU R128/ITU-R BS.1770.\n"
            "Thuyết minh lồng phim nên chọn -12 LUFS: nổi rõ trên tiếng gốc mà vẫn sạch,\n"
            "true-peak an toàn (≤ -0.5 dBTP) nên không méo khi encode lossy vào video.\n"
            "Đẩy to hơn (-10/-11) sẽ vượt 0 dBTP → méo khi nén, nên không khuyến nghị."
        )
        opt_row.addWidget(QLabel("  Loudness:")); opt_row.addWidget(self._lufs)
        self._device = QComboBox(); self._device.addItems(["auto", "cpu", "cuda", "mps"])
        self._device.setToolTip(self._translator.translate("tts.tip_device"))
        opt_row.addWidget(QLabel(self._translator.translate("tts.lbl_device"))); opt_row.addWidget(self._device); opt_row.addStretch()
        form.addRow(self._translator.translate("tts.tts_basic"), self._wrap(opt_row))

        # ── Định dạng & chất lượng file xuất ────────────────────────────────
        fmt_row = QHBoxLayout()
        self._fmt = QComboBox()
        # (nhãn, giá trị)
        self._fmt.addItem(self._translator.translate("tts.fmt_flac"), "flac")
        self._fmt.addItem(self._translator.translate("tts.fmt_wav"), "wav")
        self._fmt.addItem(self._translator.translate("tts.fmt_mp3"), "mp3")
        self._fmt.addItem(self._translator.translate("tts.fmt_opus"), "opus")
        self._fmt.addItem(self._translator.translate("tts.fmt_ogg"), "ogg")
        self._fmt.addItem(self._translator.translate("tts.fmt_m4a"), "m4a")
        self._fmt.setCurrentIndex(0)  # mặc định FLAC: giống WAV về chất lượng, nhẹ
        self._fmt.setToolTip(
            "Định dạng file âm thanh xuất ra, encode TRỰC TIẾP từ dữ liệu gốc (không\n"
            "qua chuyển đổi ngoài → không thêm nhiễu khi debug).\n"
            "• FLAC/WAV: lossless (giống hệt nhau về chất lượng); FLAC nhỏ hơn ~50%.\n"
            "• MP3/Opus/OGG/M4A: nén có mất, dung lượng nhỏ, dễ gửi đi debug.\n"
            "Lưu ý: ở 24kHz, MP3 tối đa ~160 kbps (giới hạn chuẩn MPEG-2) — vẫn rất tốt."
        )
        self._bitrate = QComboBox()
        for kb in (96, 128, 160, 192, 256, 320):
            self._bitrate.addItem(f"{kb} kbps", kb)
        self._bitrate.setCurrentIndex(5)  # 320 (sẽ tự kẹp theo định dạng/sr)
        self._bitrate.setToolTip(self._translator.translate("tts.tip_bitrate"))
        self._wav_subtype = QComboBox()
        self._wav_subtype.addItem("PCM 16-bit", "PCM_16")
        self._wav_subtype.addItem("PCM 24-bit", "PCM_24")
        self._wav_subtype.addItem("Float 32-bit", "FLOAT")
        self._wav_subtype.setToolTip(self._translator.translate("tts.tip_depth"))

        def _on_fmt_changed() -> None:
            f = self._fmt.currentData()
            self._bitrate.setVisible(f in ("mp3", "opus", "ogg", "m4a"))
            self._bitrate_lbl.setVisible(f in ("mp3", "opus", "ogg", "m4a"))
            self._wav_subtype.setVisible(f in ("wav", "flac"))
            self._wav_subtype_lbl.setVisible(f in ("wav", "flac"))
            # Cập nhật đuôi ô đường dẫn output cho khớp định dạng (nếu ô đã tạo).
            if hasattr(self, "_output"):
                txt = self._output.text().strip()
                if txt:
                    from pathlib import Path as _P
                    self._syncing_fmt = True
                    try:
                        self._output.setText(str(_P(txt).with_suffix(f".{f}")))
                    finally:
                        self._syncing_fmt = False

        self._fmt.currentIndexChanged.connect(lambda _i: _on_fmt_changed())
        fmt_row.addWidget(QLabel(self._translator.translate("tts.lbl_format"))); fmt_row.addWidget(self._fmt)
        self._bitrate_lbl = QLabel("  Bitrate:")
        fmt_row.addWidget(self._bitrate_lbl); fmt_row.addWidget(self._bitrate)
        self._wav_subtype_lbl = QLabel(self._translator.translate("tts.lbl_depth"))
        fmt_row.addWidget(self._wav_subtype_lbl); fmt_row.addWidget(self._wav_subtype)
        fmt_row.addStretch()
        form.addRow(self._translator.translate("tts.tts_export"), self._wrap(fmt_row))
        _on_fmt_changed()

        # ── Nhóm: Điều khiển Thời gian & Chồng lấp ──────────────────────────
        timing_group = SectionCard(
            self._translator.translate("tts.sec_timing"),
            collapsible=True, collapsed=True, translator=self._translator,
        )
        timing_form = QFormLayout()
        timing_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        timing_form.setSpacing(_m.SPACING_SM)

        t_opt_row = QHBoxLayout()
        self._clean_tags = QCheckBox(self._translator.translate("tts.opt_strip_tags"))
        self._clean_tags.setChecked(True)
        self._clean_tags.setToolTip(self._translator.translate("tts.tip_strip_tags"))
        self._double_pass = QCheckBox(self._translator.translate("tts.opt_double_pass"))
        self._double_pass.setChecked(True)
        self._double_pass.setToolTip(
            "Pass 1: generate song song ở tốc độ cơ bản, đo duration thực tế.\n"
            "Pass 2: CHỈ regenerate các dòng overflow với tốc độ tối thiểu vừa khít.\n"
            "  • Công thức: exact_speed = (speech_dur × base_speed) / target\n"
            "  • API tối đa 1.5× → phần còn lại dùng time-stretch (không gọi API thêm)\n"
            "Tắt để bỏ qua Pass 2 (nhanh hơn, nhiều cắt hơn)."
        )
        t_opt_row.addWidget(self._clean_tags); t_opt_row.addWidget(self._double_pass)
        timing_form.addRow(self._translator.translate("tts.tts_preprocess"), self._wrap(t_opt_row))

        self._dialog_pause = QSpinBox()
        self._dialog_pause.setRange(0, 1000); self._dialog_pause.setValue(100)
        self._dialog_pause.setSuffix(" ms")
        self._dialog_pause.setToolTip(self._translator.translate("tts.tip_dash_pause"))
        timing_form.addRow(self._translator.translate("tts.tts_dialog_pause"), self._dialog_pause)

        ov_row = QHBoxLayout()
        self._max_overlap = QSpinBox()
        self._max_overlap.setRange(0, 5000); self._max_overlap.setValue(300)
        self._max_overlap.setSuffix(" ms")
        self._max_overlap.setToolTip(self._translator.translate("tts.tip_bleed"))
        self._skip_overlap = QSpinBox()
        self._skip_overlap.setRange(0, 10000); self._skip_overlap.setValue(0)
        self._skip_overlap.setSuffix(" ms")
        self._skip_overlap.setToolTip(
            "Bỏ qua subtitle nếu audio lấn quá N ms.\n"
            "0 = TẮT (không bao giờ bỏ qua) — luôn truncate thay vì im lặng.\n"
            "Khuyến nghị: để 0 để đảm bảo đầy đủ nội dung."
        )
        ov_row.addWidget(QLabel(self._translator.translate("tts.lbl_cut_bleed"))); ov_row.addWidget(self._max_overlap)
        ov_row.addWidget(QLabel(self._translator.translate("tts.lbl_skip_bleed"))); ov_row.addWidget(self._skip_overlap)
        timing_form.addRow(self._translator.translate("tts.tts_overlap"), self._wrap(ov_row))

        # ── Elastic timing: dời câu thay vì nén tốc độ ──────────────────────
        self._elastic_timing = QCheckBox(self._translator.translate("tts.opt_elastic"))
        self._elastic_timing.setChecked(True)
        self._elastic_timing.setToolTip(
            "Tăng tốc độ đọc là giải pháp CUỐI CÙNG.\n"
            "Nếu câu quá dài nhưng sau đó có khoảng trống, DỜI các câu sau về sau\n"
            "(trong dung sai) để câu hiện tại đọc chậm, tự nhiên hơn.\n"
            "Tự lấy lại đồng bộ khi gặp khoảng lặng đủ lớn.\n"
            "→ Xuất kèm file .srt mới khớp lời thoại đã dời."
        )
        timing_form.addRow(self._translator.translate("tts.tts_elastic"), self._elastic_timing)

        # Dung sai cộng dồn (drift) — DÒNG RIÊNG để dễ thấy & điều chỉnh.
        drift_row = QHBoxLayout()
        self._max_drift = QSpinBox()
        self._max_drift.setRange(100, 5000); self._max_drift.setValue(300)
        self._max_drift.setSingleStep(100); self._max_drift.setSuffix(" ms")
        self._max_drift.setMinimumWidth(110)
        self._max_drift.setToolTip(
            "Dung sai dời mốc thời gian tích luỹ tối đa (mili giây).\n"
            "Chỉ dùng khi nén nhẹ (≤1.25×) không đủ. Ưu tiên giữ đồng bộ hình-tiếng.\n"
            "Khuyến nghị: 1000–1500ms cho phim có khẩu hình rõ; 2000–2500ms cho\n"
            "thuyết minh/voiceover (lệch sau dễ chấp nhận hơn lệch trước)."
        )
        drift_row.addWidget(self._max_drift)
        drift_hint = QLabel(self._translator.translate("tts.hint_max_shift"))
        drift_hint.setStyleSheet(caption_style())
        drift_row.addWidget(drift_hint); drift_row.addStretch()
        timing_form.addRow(self._translator.translate("tts.tts_drift"), self._wrap(drift_row))

        # "Ăn gian" đầu cụm — bắt đầu sớm vào khoảng lặng trước khi cụm quá tải.
        lead_row = QHBoxLayout()
        self._lead_in = QSpinBox()
        self._lead_in.setRange(0, 2000); self._lead_in.setValue(300)
        self._lead_in.setSingleStep(100); self._lead_in.setSuffix(" ms")
        self._lead_in.setMinimumWidth(110)
        self._lead_in.setToolTip(
            "Cụm quá tải được phép bắt đầu SỚM hơn mốc gốc tối đa bao nhiêu, lấn\n"
            "vào khoảng lặng TRƯỚC cụm (không đè câu trước) → thêm thời gian đọc,\n"
            "giảm nén/lệch. Nghe tiếng sớm vài trăm ms vào lúc im lặng gần như\n"
            "không nhận ra. 0 = tắt."
        )
        lead_row.addWidget(self._lead_in)
        lead_hint = QLabel(self._translator.translate("tts.hint_early_start"))
        lead_hint.setStyleSheet(caption_style())
        lead_row.addWidget(lead_hint); lead_row.addStretch()
        timing_form.addRow(self._translator.translate("tts.tts_leadin"), self._wrap(lead_row))

        # ── Khoảng lặng cuối phim ───────────────────────────────────────────
        self._last_extend = QSpinBox()
        self._last_extend.setRange(0, 10000); self._last_extend.setValue(300)
        self._last_extend.setSingleStep(100); self._last_extend.setSuffix(" ms")
        self._last_extend.setToolTip(
            "Câu CUỐI được kéo dài tối đa bao nhiêu vào khoảng trống cuối phim.\n"
            "0 = KHÔNG kéo dài (giữ đúng mốc kết thúc gốc) — tránh phát tiếng\n"
            "khi phim đã hết/màn hình đã đen."
        )
        timing_form.addRow(self._translator.translate("tts.tts_last_extend"), self._wrap(self._row_one(self._last_extend)))

        # ── Bỏ đọc mô tả/ký hiệu (TTS bỏ qua, SRT vẫn giữ) — tách riêng ──────
        skip_row = QHBoxLayout()
        self._skip_paren = QCheckBox("( )")
        self._skip_square = QCheckBox("[ ]")
        self._skip_curly = QCheckBox("{ }")
        self._skip_music_pair = QCheckBox("♪ ♪")
        self._skip_music_line = QCheckBox(self._translator.translate("tts.opt_music_line"))
        for cb in (self._skip_paren, self._skip_square, self._skip_curly,
                   self._skip_music_pair, self._skip_music_line):
            cb.setChecked(True)
            skip_row.addWidget(cb)
        self._skip_paren.setToolTip(self._translator.translate("tts.tip_skip_paren"))
        self._skip_square.setToolTip(self._translator.translate("tts.tip_skip_bracket"))
        self._skip_curly.setToolTip(self._translator.translate("tts.tip_skip_brace"))
        self._skip_music_pair.setToolTip(self._translator.translate("tts.tip_skip_music"))
        self._skip_music_line.setToolTip(
            "Bỏ đọc CẢ dòng bắt đầu bằng ♪ — dòng phụ đề nhạc.\n"
            "File .srt xuất kèm VẪN GIỮ nguyên để hiển thị đầy đủ."
        )
        skip_row.addStretch()
        timing_form.addRow(self._translator.translate("tts.tts_skip"), self._wrap(skip_row))

        el_hint = QLabel(
            "ℹ️  Bỏ đọc: TTS không phát các mô tả/ký hiệu trên, nhưng file <b>.srt</b> "
            "xuất kèm VẪN giữ nguyên. Khi Elastic timing dời dòng, .srt tự khớp lại."
        )
        el_hint.setTextFormat(Qt.TextFormat.RichText)
        el_hint.setStyleSheet(caption_style()); el_hint.setWordWrap(True)
        timing_form.addRow(el_hint)

        # ── Chiến lược phân cụm & co giãn (nâng cao — tầm nhìn vĩ mô) ────────
        strat_row = QHBoxLayout()
        self._anchor_gap = QDoubleSpinBox()
        self._anchor_gap.setRange(0.3, 3.0); self._anchor_gap.setSingleStep(0.1)
        self._anchor_gap.setDecimals(1); self._anchor_gap.setValue(0.7); self._anchor_gap.setSuffix(" s")
        self._anchor_gap.setToolTip(
            "Khoảng lặng gốc ≥ ngưỡng này = 'neo' đồng bộ cứng: thuyết minh cam kết\n"
            "khớp lại với hình tại đó. Nhỏ hơn → nhiều neo, đồng bộ chặt hơn nhưng\n"
            "ít dư địa co giãn. Lớn hơn → cụm dài hơn, co giãn linh hoạt hơn."
        )
        self._max_segment = QDoubleSpinBox()
        self._max_segment.setRange(4.0, 30.0); self._max_segment.setSingleStep(1.0)
        self._max_segment.setDecimals(0); self._max_segment.setValue(10); self._max_segment.setSuffix(" s")
        self._max_segment.setToolTip(
            "Cụm hội thoại dày dài hơn ngưỡng này được chia tại khoảng lặng lớn\n"
            "nhất bên trong → nén CỤC BỘ quanh chỗ quá tải, giới hạn lệch tích luỹ.\n"
            "Nhỏ hơn → chia nhiều, đồng bộ chặt; lớn hơn → co giãn mượt hơn."
        )
        self._comfort_ratio = QDoubleSpinBox()
        self._comfort_ratio.setRange(1.0, 1.6); self._comfort_ratio.setSingleStep(0.05)
        self._comfort_ratio.setDecimals(2); self._comfort_ratio.setValue(1.25); self._comfort_ratio.setPrefix("×")
        self._comfort_ratio.setToolTip(
            "Nén tới tốc độ cơ bản × hệ số này được coi là 'êm tai' → thuật toán\n"
            "ưu tiên nén nhẹ tới đây để GIỮ đồng bộ (drift 0) trước khi dời mốc.\n"
            "Cao hơn → chấp nhận đọc nhanh hơn để bám mốc; thấp hơn → ưu tiên dời mốc."
        )
        self._min_pause = QSpinBox()
        self._min_pause.setRange(10, 100); self._min_pause.setValue(35); self._min_pause.setSuffix(" %")
        self._min_pause.setToolTip(
            "Khoảng nghỉ giữa câu được rút tối đa còn bao nhiêu % khi cụm quá tải.\n"
            "Rút nghỉ KHÔNG méo giọng (khác nén) → ưu tiên dùng trước. Thấp hơn →\n"
            "rút nghỉ nhiều hơn (dồn thời gian cho lời), giữ tốc độ đọc tự nhiên hơn."
        )
        self._max_intra_gap = QDoubleSpinBox()
        self._max_intra_gap.setRange(0.05, 2.0); self._max_intra_gap.setSingleStep(0.05)
        self._max_intra_gap.setDecimals(2); self._max_intra_gap.setValue(0.5); self._max_intra_gap.setSuffix(" s")
        self._max_intra_gap.setToolTip(
            "Khoảng dừng TỐI ĐA giữa 2 câu liền nhau trong cùng cụm. Khi câu trước\n"
            "đọc xong sớm (bản dịch ngắn hơn thoại gốc) mà mốc gốc câu sau ở xa,\n"
            "kéo câu sau lên để không dừng quá lâu. Giảm → lời thoại liền mạch hơn;\n"
            "tăng → bám sát mốc thời gian gốc (giữ đồng bộ hình chặt hơn).\n"
            "(Chỉ áp dụng khi TẮT 'Cho phép chồng tiếng'.)"
        )
        self._min_stretch = QDoubleSpinBox()
        self._min_stretch.setRange(0.5, 1.0); self._min_stretch.setSingleStep(0.05)
        self._min_stretch.setDecimals(2); self._min_stretch.setValue(0.75); self._min_stretch.setPrefix("×")
        self._min_stretch.setToolTip(
            "Giãn câu tối đa: khi câu đọc xong sớm, giảm tốc độ đọc (chậm lại) để\n"
            "lấp khoảng trống tới câu sau mà vẫn giữ câu sau ĐÚNG mốc gốc (lipsync).\n"
            "Đây là tốc độ CHẬM NHẤT cho phép (× tốc độ cơ bản). 0.75 = chậm tối đa\n"
            "25%. Giảm → giãn nhiều hơn, lấp trống tốt hơn nhưng đọc chậm hơn."
        )
        for lbl, w in (("Neo:", self._anchor_gap), ("Cụm tối đa:", self._max_segment),
                       ("Nén êm:", self._comfort_ratio), ("Nghỉ tối thiểu:", self._min_pause),
                       ("Cách câu tối đa:", self._max_intra_gap), ("Giãn tối đa:", self._min_stretch)):
            strat_row.addWidget(QLabel(lbl)); strat_row.addWidget(w)
        strat_row.addStretch()
        timing_form.addRow(self._translator.translate("tts.lbl_strategy"), self._wrap(strat_row))
        timing_group.add_layout(timing_form)

        form.addRow(timing_group)
        g.add_layout(form)
        return g

    def _build_output_group(self) -> QWidget:
        g = SectionCard(self._translator.translate("tts.tts_sec_wav"))
        form = QFormLayout()
        form.setSpacing(_m.SPACING_SM)
        out_row = QHBoxLayout()
        self._output = QLineEdit(); self._output.setPlaceholderText(self._translator.translate("tts.audio_path_ph"))
        self._btn_out = PushButton(self._translator.translate("tts.btn_choose2")); self._btn_out.clicked.connect(self._on_browse_output)
        # [3.3] Đồng bộ hai chiều: gõ đuôi (.mp3/.flac…) vào ô đường dẫn → combo Định
        # dạng tự nhảy theo (cờ _syncing_fmt chống vòng lặp với _on_fmt_changed).
        self._output.textChanged.connect(self._on_output_text_changed)
        out_row.addWidget(self._output, 1); out_row.addWidget(self._btn_out)
        form.addRow(self._translator.translate("tts.tts_output"), self._wrap(out_row))
        hint = QLabel(self._translator.translate("tts.info_wav"))
        hint.setStyleSheet(caption_style()); hint.setWordWrap(True)
        form.addRow(hint)
        g.add_layout(form)
        return g

    def _build_actions_group(self) -> QWidget:
        g = SectionCard(self._translator.translate("tts.tts_sec_exec"))
        lay = QVBoxLayout()
        lay.setSpacing(_m.SPACING_SM)
        self._progress = QProgressBar(); self._progress.setRange(0, 100)
        self._stage_lbl = QLabel(self._translator.translate("tts.status_ready_short"))
        lay.addWidget(self._progress); lay.addWidget(self._stage_lbl)
        btn_row = QHBoxLayout()
        # [v3.23.331] Tổng hợp cả bộ — mô hình chỉ nạp MỘT LẦN cho toàn bộ hàng đợi,
        # tiết kiệm ~15 giây nạp lại cho mỗi tập.
        self._btn_batch = PushButton(self._translator.translate("tts.btn_batch"))
        self._btn_batch.setToolTip(
            "Quét thư mục phim bộ, tự tìm bản dịch của từng tập rồi tổng hợp lần lượt\n"
            "với cùng engine và thiết lập đang đặt."
        )
        self._btn_batch.clicked.connect(self._on_batch_tts_clicked)
        self._btn_gen = PrimaryPushButton(self._translator.translate("tts.btn_synthesize"))
        self._btn_can = PushButton(self._translator.translate("tts.btn_cancel")); self._btn_can.setEnabled(False)
        self._btn_reset = PushButton(self._translator.translate("tts.btn_reset"))
        self._btn_reset.setToolTip(self._translator.translate("tts.tip_reset"))
        self._btn_gen.clicked.connect(self._on_generate)
        self._btn_can.clicked.connect(self._on_cancel_clicked)
        self._btn_reset.clicked.connect(self._on_reset_defaults)
        btn_row.addWidget(self._btn_gen, 2); btn_row.addWidget(self._btn_batch, 1)
        btn_row.addWidget(self._btn_can, 1)
        btn_row.addWidget(self._btn_reset, 1)
        lay.addLayout(btn_row)
        g.add_layout(lay)
        return g

    def _build_result_group(self) -> QWidget:
        g = SectionCard(self._translator.translate("tts.tts_sec_result"))
        lay = QVBoxLayout()
        lay.setSpacing(_m.SPACING_SM)
        self._lbl_summary = QLabel(self._translator.translate("tts.status_no_result"))
        self._lbl_summary.setWordWrap(True)
        lay.addWidget(self._lbl_summary)

        # Bộ lọc nhanh: chỉ hiện các câu cần chú ý (bị bỏ/cắt/đọc nhanh) để khỏi
        # phải cuộn qua hàng nghìn dòng tìm câu lỗi.
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel(self._translator.translate("tts.lbl_filter")))
        self._row_filter = QComboBox()
        self._row_filter.addItem(self._translator.translate("tts.filt_all"), "all")
        self._row_filter.addItem(self._translator.translate("tts.filt_issues"), "issues")
        self._row_filter.addItem(self._translator.translate("tts.filt_skipped"), "skipped")
        self._row_filter.addItem(self._translator.translate("tts.filt_truncated"), "truncated")
        self._row_filter.addItem(self._translator.translate("tts.filt_fast"), "fast")
        self._row_filter.setToolTip(self._translator.translate("tts.tip_filter"))
        self._row_filter.currentIndexChanged.connect(lambda _i: self._apply_row_filter())
        filter_row.addWidget(self._row_filter)
        self._lbl_filter_count = QLabel("")
        filter_row.addWidget(self._lbl_filter_count)
        filter_row.addStretch()
        lay.addLayout(filter_row)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["#", "Thời gian", "Nội dung", "Thời lượng", "Tốc độ", "Loại", "Trạng thái"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setWordWrap(True); self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setDefaultSectionSize(36)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for c in (3, 4, 5, 6):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self._table, 1)
        btn_row = QHBoxLayout()
        self._btn_open = PushButton(self._translator.translate("tts.btn_open_folder"))
        self._btn_open.setEnabled(False)
        self._btn_open.clicked.connect(self._on_open_folder)
        self._btn_open_file = PushButton(self._translator.translate("tts.btn_open_file"))
        self._btn_open_file.setEnabled(False)
        self._btn_open_file.setToolTip(self._translator.translate("tts.tip_open_file"))
        self._btn_open_file.clicked.connect(self._on_open_file)
        self._btn_export = PushButton(self._translator.translate("tts.btn_export_debug"))
        self._btn_export.setEnabled(False)
        self._btn_export.setToolTip(
            "Xuất file CSV chi tiết từng dòng TTS: mốc gốc, mốc điều chỉnh, độ dời,\n"
            "tốc độ, thời lượng audio, trạng thái, lỗi — để điều tra/debug quá trình TTS."
        )
        self._btn_export.clicked.connect(self._on_export_details)
        btn_row.addWidget(self._btn_open)
        btn_row.addWidget(self._btn_open_file)
        btn_row.addWidget(self._btn_export)
        btn_row.addStretch()
        lay.addLayout(btn_row)
        g.add_layout(lay)
        return g

    # ── Signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        vm = self._view_model
        vm.source_changed.connect(self._on_source_changed)
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
        vm.tts_cancelled.connect(lambda: (
            self._progress.setValue(0),
            self._stage_lbl.setText(self._translator.translate("tts.status_cancelled")),
            self._show_info(self._translator.translate("tts.toast_cancelled_t"), self._translator.translate("tts.toast_cancelled_b"))
        ))
        vm.engine_available.connect(self._on_engine_available)

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _clear_results(self) -> None:
        """[1.5] Quét sạch bảng + tóm tắt kết quả TTS cũ khi đổi nguồn dữ liệu."""
        self._last_results = []
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

    def _on_export_details(self) -> None:
        """Xuất CSV chi tiết kết quả TTS + file cấu hình kèm theo để debug."""
        if not self._last_results:
            self._show_warning(self._translator.translate("tts.toast_nodata_t"), self._translator.translate("tts.toast_run_tts_first"))
            return
        import csv
        import statistics as _st

        path = _safe_dialogs.save_file(
            self, "Xuất chi tiết TTS", "tts_debug.csv", "CSV (*.csv)"
        )
        if not path:
            return

        def _b(v: bool) -> str:
            return "x" if v else ""

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "STT", "Văn bản", "Loại",
                    "Start gốc (s)", "End gốc (s)",
                    "Start điều chỉnh (s)", "End điều chỉnh (s)", "Dời (ms)",
                    "Khung gốc (s)", "Khung mở rộng (s)", "Khung an toàn (s)",
                    "Pass1 (s)", "Pass2 (s)", "Dùng Pass2",
                    "Tốc độ scheduler (×)", "Tốc độ Edge API (×)", "Tốc độ dùng (×)",
                    "Thời lượng audio (s)", "Lấn (s)",
                    "Nén (cách)", "Tỉ lệ nén (×)", "Cắt (ms)",
                    "Nghỉ (ms)", "Dìm câu trước (s)",
                    "Bị cắt", "Bị bỏ", "Lỗi",
                ])
                for i, r in enumerate(self._last_results, 1):
                    adj_start = r.adjusted_start_sec if r.adjusted_start_sec >= 0 else r.start_sec
                    adj_end = r.adjusted_end_sec if r.adjusted_end_sec >= 0 else r.end_sec
                    drift_ms = (adj_start - r.start_sec) * 1000.0
                    writer.writerow([
                        i, r.text, "Hội thoại" if r.is_dialog else "Thường",
                        f"{r.start_sec:.3f}", f"{r.end_sec:.3f}",
                        f"{adj_start:.3f}", f"{adj_end:.3f}", f"{drift_ms:.0f}",
                        f"{r.window_strict_s:.3f}", f"{r.window_ext_s:.3f}", f"{r.safe_window_s:.3f}",
                        f"{r.pass1_dur_s:.3f}", f"{r.pass2_dur_s:.3f}", _b(r.used_pass2),
                        f"{r.scheduled_speed:.3f}", f"{r.api_speed:.3f}", f"{r.speed_used:.3f}",
                        f"{r.audio_duration_s:.3f}", f"{r.overlap_s:.3f}",
                        r.stretch_method or "", f"{r.stretch_ratio:.3f}", f"{r.cut_amount_ms:.0f}",
                        f"{r.pause_ms:.0f}", f"{r.ducked_prev_s:.3f}",
                        _b(r.was_truncated), _b(r.was_skipped), r.error_msg or "",
                    ])

            self._write_debug_config(Path(path), _st)
            self._show_success(
                self._translator.translate("tts.toast_exported_t"),
                self._translator.translate("tts.toast_exported_b")
                .replace("{name}", Path(path).name)
                .replace("{config}", Path(path).with_suffix('.config.txt').name),
            )
        except OSError as exc:
            self._show_warning(self._translator.translate("tts.toast_write_err_t"), str(exc))

    def _write_debug_config(self, csv_path: Path, _st) -> None:
        """Ghi file .config.txt: toàn bộ cài đặt + tóm tắt thống kê để dễ debug."""
        from datetime import datetime

        rs = self._last_results
        req = getattr(self._view_model, "last_request", None)
        engine = getattr(self._view_model, "last_engine_name", "") or "?"
        cfg_path = csv_path.with_suffix(".config.txt")

        ok = [r for r in rs if not r.was_skipped]
        speeds = [r.speed_used for r in ok if r.speed_used > 0]
        drifts = [
            ((r.adjusted_start_sec if r.adjusted_start_sec >= 0 else r.start_sec) - r.start_sec) * 1000.0
            for r in ok
        ]
        overlaps = [r.overlap_s for r in ok if r.overlap_s > 0]
        stretched = [r for r in ok if r.stretch_method]
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("CẤU HÌNH & TÓM TẮT TTS (debug)")
        lines.append(f"Thời điểm xuất : {datetime.now():%Y-%m-%d %H:%M:%S}")
        lines.append(f"Engine         : {engine}")
        lines.append("=" * 60)
        lines.append("\n── TÓM TẮT KẾT QUẢ ──")
        lines.append(f"Tổng dòng              : {len(rs)}")
        lines.append(f"Đọc thành công         : {len(ok)}")
        lines.append(f"Bị bỏ (không đọc)      : {sum(1 for r in rs if r.was_skipped)}")
        lines.append(f"Bị cắt đuôi            : {sum(1 for r in rs if r.was_truncated)}")
        lines.append(f"Có lỗi                 : {sum(1 for r in rs if r.error_msg)}")
        lines.append(f"Cần nén thêm (stretch) : {len(stretched)}")
        lines.append(f"Có chồng tiếng (lấn)   : {len(overlaps)}")
        if speeds:
            lines.append(
                f"Tốc độ đọc (×)         : min {min(speeds):.2f} | "
                f"median {_st.median(speeds):.2f} | max {max(speeds):.2f}"
            )
        if drifts:
            lines.append(
                f"Dời mốc (ms)           : min {min(drifts):.0f} | "
                f"median {_st.median(drifts):.0f} | max {max(drifts):.0f}"
            )
        if overlaps:
            lines.append(
                f"Lấn (s)                : tổng {sum(overlaps):.2f} | max {max(overlaps):.2f}"
            )

        lines.append("\n── CÀI ĐẶT ĐÃ DÙNG ──")
        if req is not None:
            def _fmt(v: object) -> str:
                if isinstance(v, bool):
                    return "Bật" if v else "Tắt"
                if isinstance(v, float):
                    return f"{v:g}"
                return str(v)

            fields = [
                ("Ngôn ngữ", "language"), ("Giọng đọc", "speaker"),
                ("Thiết bị", "device"), ("Số luồng song song", "edge_concurrency"),
                ("Tốc độ cơ bản (×)", "base_speed"), ("Tốc độ tối đa (×)", "max_speed"),
                ("Chiến lược timing", "timing_strategy"),
                ("Elastic timing", "elastic_timing"), ("Double Pass", "double_pass"),
                (self._translator.translate("tts.opt_hq"), "high_quality"),
                ("Dung sai cộng dồn (s)", "max_drift_s"), ("Ăn gian đầu (s)", "lead_in_s"),
                ("Kéo dài câu cuối (s)", "last_line_max_extend_s"),
                ("Cắt lấn (ms)", "max_overlap_ms"), ("Bỏ qua lấn (ms)", "skip_overlap_ms"),
                (self._translator.translate("tts.opt_overlap"), "allow_audio_overlap"),
                ("Giãn tối đa (×base)", "min_stretch_ratio"),
                ("Nén êm (×base)", "comfort_speed_ratio"),
                ("Nghỉ tối thiểu (tỉ lệ)", "min_pause_ratio"),
                ("Cách câu tối đa (s)", "max_intra_gap_s"),
                ("Neo cụm (s)", "anchor_gap_s"), ("Cụm tối đa (s)", "max_segment_s"),
                ("Nghỉ hội thoại (ms)", "dialog_pause_ms"),
                ("Chuẩn hoá loudness", "normalize"), ("Loudness mục tiêu (LUFS)", "target_lufs"),
                ("Định dạng xuất", "output_format"), ("Bitrate (kbps)", "output_bitrate_kbps"),
                ("Độ sâu WAV/FLAC", "wav_subtype"),
                (self._translator.translate("tts.opt_clarify"), "voice_clarity"), (self._translator.translate("tts.opt_strip_tags"), "clean_tags"),
                ("Số lần thử lại", "retry_count"), ("Delay thử lại (s)", "retry_delay_s"),
                ("Bỏ đọc ( )", "skip_paren"), ("Bỏ đọc [ ]", "skip_square"),
                ("Bỏ đọc { }", "skip_curly"),
                ("Bỏ đọc ♪..♪", "skip_music_pair"), ("Bỏ đọc dòng ♪", "skip_music_line"),
            ]
            # [v3.23.214] Engine hiện tại có đọc trường này không? Trước đây in TẤT CẢ
            # -> người dùng (và cả việc phân tích debug) tưởng "Elastic timing: Bật" có
            # tác dụng khi chạy VieNeu, thực tế bị bỏ qua hoàn toàn.
            is_edge = (self._engine_combo.currentData() or _ENGINE_EDGE) == _ENGINE_EDGE
            for label, attr in fields:
                if hasattr(req, attr):
                    value = _fmt(getattr(req, attr))
                    if not is_edge and attr in _EDGE_ONLY_FIELDS:
                        value += "   (không áp dụng cho engine này)"
                    lines.append(f"{label:<24}: {value}")
        else:
            lines.append("(Không có snapshot cấu hình — chạy lại TTS để ghi nhận.)")

        cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _on_reset_defaults(self) -> None:
        """Đặt lại toàn bộ cấu hình TTS về mặc định khuyến nghị."""
        # [v3.23.112] Reset ghi đè mọi tinh chỉnh -> xác nhận tránh lỡ tay.
        confirm = QMessageBox.question(
            self, self._translator.translate("tts.dlg_reset_title"),
            self._translator.translate("tts.dlg_reset_body"),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._speed.setValue(1.1)
        self._max_speed.setValue(2.0)
        self._retry.setValue(10)
        self._retry_delay.setValue(2.0)
        self._normalize.setChecked(True)
        self._voice_clarity.setChecked(True)
        self._high_quality.setChecked(True)
        _li = self._lufs.findData(-23.0)
        if _li >= 0:
            self._lufs.setCurrentIndex(_li)
        self._device.setCurrentText("auto")
        self._clean_tags.setChecked(True)
        self._double_pass.setChecked(True)
        self._dialog_pause.setValue(100)
        self._max_overlap.setValue(300)
        self._skip_overlap.setValue(0)
        self._elastic_timing.setChecked(True)
        self._max_drift.setValue(300)
        self._lead_in.setValue(300)
        self._anchor_gap.setValue(0.7)
        self._max_segment.setValue(10)
        self._comfort_ratio.setValue(1.25)
        self._min_pause.setValue(35)
        self._max_intra_gap.setValue(0.5)
        self._allow_overlap.setChecked(True)
        self._min_stretch.setValue(0.75)
        self._last_extend.setValue(300)
        self._edge_concurrency.setValue(16)
        for cb in (self._skip_paren, self._skip_square, self._skip_curly,
                   self._skip_music_pair, self._skip_music_line):
            cb.setChecked(True)
        self._show_success(self._translator.translate("tts.toast_restored_t"), self._translator.translate("tts.toast_restored_b"))

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

    def _on_engine_changed(self, idx: int) -> None:
        engine_id = self._engine_combo.itemData(idx) or _ENGINE_EDGE
        stack_map = {_ENGINE_EDGE: 0, _ENGINE_GEMINI: 1, _ENGINE_VIENEU: 2}
        self._engine_stack.setCurrentIndex(stack_map.get(engine_id, 0))
        # max_speed dùng cho tất cả engines (time-stretch post-processing)
        self._max_speed.setVisible(True)
        self._sync_edge_only_controls(engine_id)
        # [v3.23.195] Nạp giọng VieNeu CHỈ khi trang đang hiển thị (người dùng tương tác
        # thật). Nạp lúc khởi động (restore engine đã lưu) sẽ tải model ~9s CHẶN app mở
        # (log thực tế: model nạp trước cả MainWindow). Trang chưa hiện -> hoãn tới
        # showEvent.
        if (
            engine_id == _ENGINE_VIENEU
            and not getattr(self, "_vieneu_voices_loaded", False)
            and self.isVisible()
        ):
            self._load_vieneu_voices()
        else:
            # [v3.23.252] Rời VieNeu (hoặc giọng đã nạp xong) -> cập nhật lại trạng thái
            # nút Tổng hợp. Nếu trước đó nút bị khoá vì VieNeu đang nạp mà nay chuyển sang
            # Edge/Gemini (không cần giọng VieNeu), nút phải bật lại.
            self._update_action_states()

    def _sync_edge_only_controls(self, engine_id: str) -> None:
        """[v3.23.214] Bật/tắt các tuỳ chọn CHỈ Edge đọc theo engine đang chọn.

        VieNeu/Gemini bỏ qua hoàn toàn nhóm này; để chúng bật gây hiểu nhầm là có tác
        dụng (người dùng chỉnh vô ích, debug config in ra sai lệch).

        Args:
            engine_id: Engine đang chọn (``_ENGINE_EDGE`` / ``_ENGINE_GEMINI`` /
                ``_ENGINE_VIENEU``).
        """
        is_edge = engine_id == _ENGINE_EDGE
        for widget in (
            self._strategy, self._high_quality, self._double_pass,
            self._elastic_timing, self._comfort_ratio, self._min_pause,
        ):
            widget.setEnabled(is_edge)
            edge_hint = self._translator.translate("tts.edge_only_hint")
            tip = widget.toolTip().split(edge_hint)[0]
            widget.setToolTip(tip if is_edge else tip + edge_hint)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Trang TTS hiện ra: nạp bù danh sách giọng VieNeu nếu đang chọn engine này."""
        super().showEvent(event)
        engine_id = self._engine_combo.currentData() or _ENGINE_EDGE
        if engine_id == _ENGINE_VIENEU and not getattr(
            self, "_vieneu_voices_loaded", False
        ):
            self._load_vieneu_voices()

    def _load_vieneu_voices(self) -> None:
        """Nạp danh sách giọng preset VieNeu (BACKGROUND thread) theo chế độ đang chọn.

        [v3.23.194] Mỗi chế độ (standard/v3turbo) có bộ giọng KHÁC NHAU (7 vs 10 giọng,
        tên khác) — nạp lại khi đổi chế độ, tránh "Voice not found".
        [v3.23.200] Chạy ở thread NỀN: nạp model ~9s trên UI thread làm app đứng hình.
        Trong lúc nạp combo hiển thị self._translator.translate("tts.voice_loading") và tạm khoá; không bao giờ có
        2 loader song song (đổi chế độ giữa chừng -> ghi nhận, nạp lại khi xong).
        """
        mode = self._vieneu_mode.currentData() or "standard"
        loader = getattr(self, "_vieneu_voice_loader", None)
        if loader is not None and loader.isRunning():
            # Đang nạp: handler khi xong sẽ tự so chế độ hiện tại và nạp lại nếu khác.
            return
        self._vieneu_voice.blockSignals(True)
        self._vieneu_voice.clear()
        self._vieneu_voice.addItem(self._translator.translate("tts.voice_loading"), userData="")
        self._vieneu_voice.setEnabled(False)
        self._vieneu_voice.blockSignals(False)
        # [v3.23.252] Khoá nút Tổng hợp trong lúc nạp giọng — tránh bấm chạy khi
        # giọng chưa sẵn sàng (sẽ lỗi). Chỉ khoá khi engine hiện tại là VieNeu.
        # Nút bật lại khi nạp xong/lỗi.
        if (self._engine_combo.currentData() or _ENGINE_EDGE) == _ENGINE_VIENEU:
            self._btn_gen.setEnabled(False)
        loader = _VieNeuVoiceLoader(self._view_model._container, mode, parent=self)
        loader.voices_loaded.connect(self._on_vieneu_voices_loaded)
        loader.load_failed.connect(self._on_vieneu_voices_failed)
        loader.finished.connect(loader.deleteLater)
        self._vieneu_voice_loader = loader
        loader.start()

    def _on_vieneu_voices_loaded(self, mode: str, voice_ids: list) -> None:
        """Nhận danh sách giọng từ thread nền; bỏ qua kết quả STALE nếu đã đổi chế độ."""
        self._vieneu_voice_loader = None
        current_mode = self._vieneu_mode.currentData() or "standard"
        if mode != current_mode:
            self._load_vieneu_voices()  # kết quả cũ — nạp lại theo chế độ hiện tại
            return
        self._vieneu_voice.setEnabled(True)
        if not voice_ids:
            self._reset_vieneu_voice_combo()
            return
        # Ưu tiên: giọng đã lưu QSettings (combo đang là placeholder) -> mặc định.
        current = getattr(self, "_vieneu_saved_voice", "")
        self._vieneu_voice.blockSignals(True)
        self._vieneu_voice.clear()
        self._vieneu_voice.addItem(self._translator.translate("tts.voice_default"), userData="")
        for voice_id in voice_ids:
            self._vieneu_voice.addItem(voice_id, userData=voice_id)
        restore = self._vieneu_voice.findData(current)
        if restore >= 0:
            self._vieneu_voice.setCurrentIndex(restore)
        else:
            self._vieneu_voice.setCurrentIndex(0)  # giọng cũ không có -> về mặc định
        self._vieneu_voice.blockSignals(False)
        self._vieneu_voices_loaded = True
        # [v3.23.252] Nạp xong -> khôi phục trạng thái nút Tổng hợp theo điều
        # kiện chuẩn (có phụ đề + không bận), bù cho việc khoá nút lúc bắt đầu.
        self._update_action_states()

    def _on_vieneu_voices_failed(self, mode: str) -> None:
        """Nạp giọng thất bại (engine chưa cài/lỗi): trả combo về trạng thái mặc định."""
        self._vieneu_voice_loader = None
        self._reset_vieneu_voice_combo()

    def _reset_vieneu_voice_combo(self) -> None:
        """Trả combo giọng về self._translator.translate("tts.voice_default") và mở khoá (dùng khi nạp lỗi/rỗng)."""
        self._vieneu_voice.blockSignals(True)
        self._vieneu_voice.clear()
        self._vieneu_voice.addItem(self._translator.translate("tts.voice_default"), userData="")
        self._vieneu_voice.setEnabled(True)
        self._vieneu_voice.blockSignals(False)
        # [v3.23.252] Nạp lỗi/rỗng cũng phải bật lại nút Tổng hợp (đã khoá lúc
        # bắt đầu nạp). Dùng giọng mặc định vẫn chạy được, không nên kẹt nút.
        self._update_action_states()

    def _on_vieneu_mode_changed(self, _idx: int = 0) -> None:
        """Đổi chế độ VieNeu -> nạp lại danh sách giọng của chế độ mới (thread nền)."""
        loader = getattr(self, "_vieneu_voice_loader", None)
        if getattr(self, "_vieneu_voices_loaded", False) or (
            loader is not None and loader.isRunning()
        ):
            self._vieneu_voices_loaded = False  # danh sách cũ không còn đúng chế độ
            self._load_vieneu_voices()

    def _on_gemini_model_changed(self, _idx: int = 0) -> None:
        """Ẩn/hiện Native Audio config dựa theo model đang chọn."""
        from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import _NATIVE_AUDIO_MODELS
        is_native = self._gemini_model.currentText() in _NATIVE_AUDIO_MODELS
        self._gemini_native_group.setVisible(is_native)

    def _update_edge_speakers(self) -> None:
        lang = self._edge_lang.currentData() or "vi-VN"
        speakers = _EDGE_VOICE_MAP.get(lang, [])
        curr = self._edge_voice.currentText()
        self._edge_voice.blockSignals(True)
        self._edge_voice.clear(); self._edge_voice.addItems(speakers)
        idx = self._edge_voice.findText(curr)
        if idx >= 0:
            self._edge_voice.setCurrentIndex(idx)
        self._edge_voice.blockSignals(False)

    def _on_browse_vieneu_ref(self) -> None:
        """Chọn file audio nhân bản giọng cho VieNeu-TTS (tuỳ chọn)."""
        path = _safe_dialogs.open_file(
            self, "Chọn audio nhân bản giọng (3–5 giây)", "",
            "Audio (*.wav *.mp3 *.flac *.m4a);;Tất cả (*)"
        )
        if path:
            self._vieneu_ref_path.setText(path)

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
            if not self._gemini_key.text().strip():
                self._show_warning(
                    self._translator.translate("tts.toast_noapikey_t"),
                    self._translator.translate("tts.toast_need_gemini_key"),
                )
                return
        elif engine_id == _ENGINE_VIENEU:
            # [v3.23.252] Đang nạp giọng VieNeu (thread nền) -> combo chưa sẵn
            # sàng, bấm chạy sẽ lỗi. Chặn sớm với thông báo rõ. Loader đang chạy
            # = giọng chưa nạp xong.
            _loader = getattr(self, "_vieneu_voice_loader", None)
            if _loader is not None and _loader.isRunning():
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
                          api_key=self._gemini_key.text().strip(),
                          style_prompt=self._gemini_style.toPlainText().strip(),
                          affective_dialog=self._gemini_affective.isChecked(),
                          gemini_temperature=gemini_temp)
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
                          vieneu_emotion=self._vieneu_emotion.currentData() or "natural",
                          vieneu_force_cpu=self._vieneu_force_cpu.isChecked())
        kwargs["media_duration_s"] = getattr(self, "_media_duration_s", None)
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

    def _on_install_vieneu_gpu_clicked(self) -> None:
        """Cài các gói còn thiếu để VieNeu chạy GPU, chạy nền có báo tiến độ."""
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QMessageBox

        from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
            build_gpu_tts_plan,
        )
        from subtitles_extractor.presentation.workers.install_vieneu_gpu_worker import (
            InstallVieneuGpuWorker,
        )

        if getattr(self, "_gpu_install_thread", None) is not None:
            return

        plan = build_gpu_tts_plan()
        if plan.is_ready:
            QMessageBox.information(
                self, self._translator.translate("tts.dlg_ready_title"),
                self._translator.translate("tts.dlg_ready_body"),
            )
            self._refresh_vieneu_gpu_note()
            return
        if plan.python_exe is None:
            QMessageBox.warning(
                self, self._translator.translate("tts.dlg_no_env_title"),
                self._translator.translate("tts.dlg_no_env_body"),
            )
            return

        answer = QMessageBox.question(
            self, self._translator.translate("tts.dlg_install_title"),
            self._translator.translate("tts.dlg_install_body")
            .replace("{packages}", ", ".join(plan.missing_packages))
            .replace("{mb}", f"{plan.download_estimate_gb * 1024:.0f}"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._btn_install_gpu.setEnabled(False)
        self._gpu_install_progress.setValue(0)
        self._gpu_install_progress.setVisible(True)

        self._gpu_install_thread = QThread(self)
        self._gpu_install_worker = InstallVieneuGpuWorker()
        self._gpu_install_worker.moveToThread(self._gpu_install_thread)

        # QueuedConnection: worker ở luồng khác, tín hiệu phải xếp hàng về luồng UI.
        self._gpu_install_thread.started.connect(self._gpu_install_worker.run)
        self._gpu_install_worker.progress.connect(
            self._on_gpu_install_progress, Qt.ConnectionType.QueuedConnection
        )
        self._gpu_install_worker.finished.connect(
            self._on_gpu_install_finished, Qt.ConnectionType.QueuedConnection
        )
        self._gpu_install_worker.failed.connect(
            self._on_gpu_install_failed, Qt.ConnectionType.QueuedConnection
        )
        self._gpu_install_worker.done.connect(
            self._cleanup_gpu_install_thread, Qt.ConnectionType.QueuedConnection
        )
        self._gpu_install_thread.start()

    def _on_gpu_install_progress(self, percent: int, label: str) -> None:
        self._gpu_install_progress.setValue(percent)
        self._vieneu_gpu_note.setText(label)

    def _on_gpu_install_finished(self, python_exe: str) -> None:
        """Cài xong: bỏ tick ép CPU giúp người dùng và làm mới ghi chú."""
        from PySide6.QtWidgets import QMessageBox

        self._vieneu_force_cpu.setChecked(False)   # kích hoạt đường GPU
        self._refresh_vieneu_gpu_note()
        QMessageBox.information(
            self, self._translator.translate("tts.dlg_installed_title"),
            self._translator.translate("tts.dlg_installed_body").replace("{env}", str(python_exe))
        )

    def _on_gpu_install_failed(self, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        logger.warning("Cài VieNeu GPU thất bại: %s", message.replace("\n", " | "))
        self._refresh_vieneu_gpu_note()
        QMessageBox.warning(self, self._translator.translate("tts.dlg_install_failed_title"), message)

    def _cleanup_gpu_install_thread(self) -> None:
        """Dừng và xoá luồng cài (kể cả khi lỗi)."""
        thread = getattr(self, "_gpu_install_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
        self._gpu_install_thread = None
        self._gpu_install_worker = None
        self._btn_install_gpu.setEnabled(True)
        self._gpu_install_progress.setVisible(False)

    def _refresh_vieneu_gpu_note(self) -> None:
        """Hiện tình trạng THẬT của khả năng chạy VieNeu trên GPU.

        VieNeu chỉ dùng GPU qua đường PyTorch — engine ONNX ghi cứng
        ``CPUExecutionProvider``. Mà torch bị loại khỏi bản đóng gói (nó xung đột CUDA
        với paddle), nên bỏ tick "Ép chạy CPU" một mình là KHÔNG đủ.
        """
        try:
            from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
                build_gpu_tts_plan,
            )

            plan = build_gpu_tts_plan()
        except Exception as exc:  # noqa: BLE001 — không được phá trang
            logger.debug("Không dò được khả năng GPU cho TTS: %s", exc)
            return

        if self._vieneu_force_cpu.isChecked():
            self._vieneu_gpu_note.setText(
                "Đang chạy ONNX/CPU — ổn định và đủ nhanh (khoảng 0,5 giây mỗi câu)."
            )
            self._btn_install_gpu.setVisible(False)
            return

        if plan.is_ready:
            self._vieneu_gpu_note.setText(
                "✓ Sẽ thử dùng GPU qua môi trường riêng. Đường GPU còn gộp lô nên "
                "nhanh hơn rõ rệt với tập nhiều câu."
            )
            self._btn_install_gpu.setVisible(False)
            return
        self._btn_install_gpu.setVisible(True)

        # Đây là điểm quan trọng: nói RÕ rằng bỏ tick không đủ.
        self._vieneu_gpu_note.setText(
            "⚠️ Bỏ tick MỘT MÌNH chưa bật được GPU: VieNeu chỉ dùng GPU qua PyTorch, "
            f"mà torch không nằm trong bản đóng gói. {plan.note} "
            "Trong lúc chờ, ONNX/CPU vẫn chạy bình thường."
        )

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
    def _on_batch_tts_clicked(self) -> None:
        """Quét thư mục phim bộ rồi tổng hợp giọng đọc lần lượt cho từng tập."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from subtitles_extractor.application.services.batch_tts_plan import (
            build_tts_plan,
            estimate_batch_minutes,
            find_episode_videos,
            summarise_tts_plan,
        )

        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa các tập", "")
        if not folder:
            return

        videos = find_episode_videos(Path(folder))
        if not videos:
            QMessageBox.information(
                self, self._translator.translate("tts.dlg_batch_none_title"),
                self._translator.translate("tts.dlg_batch_none_body"),
            )
            return

        plan = build_tts_plan(videos, skip_existing=True)
        runnable = [item for item in plan if item.will_run]
        summary = summarise_tts_plan(plan)
        if not runnable:
            QMessageBox.information(
                self, self._translator.translate("tts.dlg_batch_nothing_title"),
                self._translator.translate("tts.dlg_batch_nothing_body").replace("{summary}", summary),
            )
            return

        minutes = estimate_batch_minutes(runnable)
        answer = QMessageBox.question(
            self, self._translator.translate("tts.dlg_batch_confirm_title"),
            self._translator.translate("tts.dlg_batch_confirm_body")
            .replace("{n}", str(len(videos))).replace("{summary}", summary)
            .replace("{runnable}", str(len(runnable))).replace("{minutes}", f"{minutes:.0f}"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._batch_items = runnable
        self._batch_index = 0
        self._batch_failures = []
        self._batch_cancelled = False
        self._run_next_batch_tts()

    def _run_next_batch_tts(self) -> None:
        """Nạp phụ đề của tập kế tiếp; việc tổng hợp chạy khi nạp xong."""
        if not self._batch_items:
            return
        if self._batch_cancelled or self._batch_index >= len(self._batch_items):
            self._finish_batch_tts()
            return

        item = self._batch_items[self._batch_index]
        total = len(self._batch_items)
        self._stage_lbl.setText(
            f"Cả bộ {self._batch_index + 1}/{total}: {item.video_path.name}"
        )
        self._output.setText(str(item.output_path))
        # Nạp nguồn là thao tác bất đồng bộ -> đánh dấu để khi `source_changed` bắn ra
        # thì biết đây là tập của hàng đợi và tự bấm "Tổng hợp".
        self._batch_pending_generate = True
        self._clear_results()
        self._view_model.load_source_from_file(item.subtitle_path)

    def _on_source_changed_for_batch(self, count: int) -> None:
        """Nạp phụ đề xong trong lúc chạy hàng loạt -> tổng hợp ngay tập này."""
        if not self._batch_pending_generate:
            return
        self._batch_pending_generate = False
        if count <= 0:
            index = min(self._batch_index, len(self._batch_items) - 1)
            name = self._batch_items[index].video_path.name if self._batch_items else "?"
            logger.warning("Cả bộ: %s không nạp được phụ đề.", name)
            self._advance_batch_tts(failed=name)
            return
        QTimer.singleShot(0, self._on_generate)

    def _on_error_for_batch(self, message: str) -> None:
        """Nhận lỗi TTS: đang chạy hàng loạt thì ghi nhận rồi chạy tiếp."""
        if self._batch_items:
            index = min(self._batch_index, len(self._batch_items) - 1)
            name = self._batch_items[index].video_path.name
            logger.warning("Cả bộ: tập %s lỗi — %s", name, message)
            self._advance_batch_tts(failed=name)
            return
        self._show_warning(self._translator.translate("tts.toast_tts_err_t"), _humanize_gemini_error(message))
        self._stage_lbl.setText(self._translator.translate("tts.status_error"))

    def _advance_batch_tts(self, *, failed: str | None = None) -> bool:
        """Chuyển sang tập kế. Trả ``True`` nếu đang chạy hàng loạt."""
        if not self._batch_items:
            return False
        if failed:
            self._batch_failures.append(failed)
        if self._batch_cancelled:
            self._finish_batch_tts()
            return True
        self._batch_index += 1
        QTimer.singleShot(0, self._run_next_batch_tts)
        return True

    def _finish_batch_tts(self) -> None:
        """Báo tổng kết và dọn trạng thái hàng đợi."""
        from PySide6.QtWidgets import QMessageBox

        total = len(self._batch_items)
        failures = list(self._batch_failures)
        cancelled = self._batch_cancelled
        processed = self._batch_index
        done = max(0, processed - len(failures))

        self._batch_items = []
        self._batch_index = 0
        self._batch_failures = []
        self._batch_cancelled = False
        self._batch_pending_generate = False

        if cancelled:
            message = (
                f"Đã huỷ. Tổng hợp xong {done}/{total} tập, "
                f"bỏ {max(0, total - processed)} tập còn lại."
            )
        else:
            message = f"Đã tổng hợp xong {done}/{total} tập."
        if failures:
            message += "\n\nThất bại:\n• " + "\n• ".join(failures[:10])
            if len(failures) > 10:
                message += f"\n… và {len(failures) - 10} tập nữa."

        self._stage_lbl.setText(message.splitlines()[0])
        QMessageBox.information(self, self._translator.translate("tts.dlg_batch_done_title"), message)

    def _on_source_changed(self, n: int) -> None:
        # [v3.23.138] Nhất quán với trang Dịch: nạp nguồn mới -> xoá kết quả audio cũ
        # (bảng + tóm tắt + nút xuất). Trước đây chỉ đường "Nạp từ file" mới xoá; khi
        # phụ đề được ĐẨY sang (từ Editor) thì bảng vẫn giữ kết quả của nguồn CŨ.
        self._clear_results()
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
            _loader = getattr(self, "_vieneu_voice_loader", None)
            if _loader is not None and _loader.isRunning():
                self._btn_gen.setEnabled(False)
        # Disable nguồn + output khi đang bận
        for btn in (self._btn_load, self._btn_editor, self._btn_trans, self._btn_out):
            btn.setEnabled(not busy)
        if not busy:
            self._update_action_states()

    def _on_result_ready(self, results) -> None:
        rs: list[TTSSegmentResult] = list(results)
        logger.info("UI: bắt đầu hiển thị kết quả TTS (%d câu).", len(rs))
        self._last_results = rs  # lưu để xuất chi tiết debug
        self._populate_table(rs)
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
        )
        self._btn_open.setEnabled(bool(out and out.parent.exists()))
        self._btn_open_file.setEnabled(bool(out and out.exists()))
        self._btn_export.setEnabled(bool(rs))

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
            if out and Path(out).exists():
                self.tts_completed.emit(str(out))
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

    def _on_engine_available(self, name: str, available: bool) -> None:
        icon = "✅" if available else "❌"
        color = "#27ae60" if available else "#c0392b"
        note = "" if available else self._translator.translate("tts.res_not_installed")
        style = caption_style(color)
        n = name.lower()
        # [v3.23.115] Ghi nhớ tình trạng theo engine_id để kiểm tra trước khi chạy.
        if not hasattr(self, "_engine_avail"):
            self._engine_avail: dict[str, bool] = {}
        if "edge" in n:
            self._engine_avail[_ENGINE_EDGE] = available
            self._lbl_edge_st.setText(f"{icon} Edge TTS{note}")
            self._lbl_edge_st.setStyleSheet(style)
        elif "gemini" in n:
            self._engine_avail[_ENGINE_GEMINI] = available
            self._lbl_gemini_st.setText(f"{icon} Gemini TTS{note}")
            self._lbl_gemini_st.setStyleSheet(style)
        elif "vieneu" in n:
            self._engine_avail[_ENGINE_VIENEU] = available
            self._lbl_vieneu_st.setText(f"{icon} VieNeu-TTS{note}")
            self._lbl_vieneu_st.setStyleSheet(style)

        # [v3.23.291] Khi da kiem du 3 engine: neu engine dang chon CHUA san sang nhung co
        # engine khac san sang -> tu chuyen sang engine do (tranh nguoi dung bi ket o engine
        # chua cai nhu VieNeu/Edge, trong khi Gemini da san sang). Chi tu chuyen 1 lan.
        if len(self._engine_avail) >= 3 and not getattr(self, "_auto_switched_engine", False):
            current_id = self._engine_combo.currentData()
            if not self._engine_avail.get(current_id, False):
                # [v3.23.292] VieNeu bundle mac dinh (offline, khong can API key) -> uu tien
                # truoc; roi Gemini (can key); Edge cuoi (GPL, cai ngoai).
                for engine_id in (_ENGINE_VIENEU, _ENGINE_GEMINI, _ENGINE_EDGE):
                    if self._engine_avail.get(engine_id, False):
                        pos = self._engine_combo.findData(engine_id)
                        if pos >= 0:
                            self._auto_switched_engine = True
                            self._engine_combo.setCurrentIndex(pos)
                            break

    # ── Settings ──────────────────────────────────────────────────────────────

    def _save_settings(self) -> None:
        s = self._settings
        # [v3.23.195] Lưu theo engine_id (bền vững) thay vì index — index lệch khi danh
        # sách engine thay đổi giữa các phiên bản (chính là bug "engine một đàng, cấu
        # hình một nẻo" sau khi gỡ/thêm engine).
        s.setValue("engine_id", self._engine_combo.currentData() or _ENGINE_EDGE)
        s.setValue("edge_lang", self._edge_lang.currentData())
        s.setValue("edge_voice", self._edge_voice.currentText())
        s.setValue("gemini_key", self._gemini_key.text())
        s.setValue("gemini_model", self._gemini_model.currentText())
        s.setValue("gemini_voice", self._gemini_voice.currentText())
        s.setValue("gemini_style", self._gemini_style.toPlainText())
        s.setValue("gemini_affective", self._gemini_affective.isChecked())
        s.setValue("gemini_temperature", self._gemini_temperature.value())
        s.setValue("vieneu_mode", self._vieneu_mode.currentData() or "standard")
        s.setValue("vieneu_emotion", self._vieneu_emotion.currentData() or "natural")
        # Chỉ ghi đè giọng đã lưu khi danh sách giọng ĐÃ nạp (combo phản ánh lựa chọn
        # thật); chưa nạp -> giữ nguyên giá trị cũ, tránh mất giọng người dùng đã chọn.
        if getattr(self, "_vieneu_voices_loaded", False):
            s.setValue("vieneu_voice", self._vieneu_voice.currentData() or "")
        s.setValue("vieneu_ref_path", self._vieneu_ref_path.text())
        s.setValue("vieneu_force_cpu", self._vieneu_force_cpu.isChecked())
        s.setValue("speed", self._speed.value())
        s.setValue("max_speed", self._max_speed.value())
        s.setValue("retry", self._retry.value())
        s.setValue("retry_delay", self._retry_delay.value())
        s.setValue("normalize", self._normalize.isChecked())
        s.setValue("voice_clarity", self._voice_clarity.isChecked())
        s.setValue("high_quality", self._high_quality.isChecked())
        s.setValue("target_lufs", self._lufs.currentData())
        s.setValue("output_format", self._fmt.currentData())
        s.setValue("output_bitrate", self._bitrate.currentData())
        s.setValue("wav_subtype", self._wav_subtype.currentData())
        s.setValue("device", self._device.currentText())
        s.setValue("clean_tags", self._clean_tags.isChecked())
        s.setValue("double_pass", self._double_pass.isChecked())
        s.setValue("dialog_pause_ms", self._dialog_pause.value())
        s.setValue("max_overlap_ms", self._max_overlap.value())
        s.setValue("skip_overlap_ms", self._skip_overlap.value())
        s.setValue("elastic_timing", self._elastic_timing.isChecked())
        s.setValue("max_drift_ms", self._max_drift.value())
        s.setValue("lead_in_ms", self._lead_in.value())
        s.setValue("anchor_gap_s", self._anchor_gap.value())
        s.setValue("max_segment_s", self._max_segment.value())
        s.setValue("comfort_ratio", self._comfort_ratio.value())
        s.setValue("min_pause_pct", self._min_pause.value())
        s.setValue("max_intra_gap_s", self._max_intra_gap.value())
        s.setValue("allow_overlap", self._allow_overlap.isChecked())
        s.setValue("timing_strategy", self._strategy.currentData())
        s.setValue("min_stretch", self._min_stretch.value())
        s.setValue("last_extend_ms", self._last_extend.value())
        s.setValue("skip_paren", self._skip_paren.isChecked())
        s.setValue("skip_square", self._skip_square.isChecked())
        s.setValue("skip_curly", self._skip_curly.isChecked())
        s.setValue("skip_music_pair", self._skip_music_pair.isChecked())
        s.setValue("skip_music_line", self._skip_music_line.isChecked())
        s.setValue("edge_concurrency", self._edge_concurrency.value())
        s.sync()

    def _restore_settings(self) -> None:
        s = self._settings
        # [v3.23.195] Restore theo engine_id (bền vững). Tương thích ngược: bản cũ chỉ
        # có "engine_idx" (index) — index có thể lệch sau khi danh sách engine thay đổi
        # nên KHÔNG dùng lại; thiếu engine_id -> mặc định Edge (an toàn).
        saved_engine = s.value("engine_id", _ENGINE_EDGE, type=str)
        engine_pos = self._engine_combo.findData(saved_engine)
        self._engine_combo.setCurrentIndex(engine_pos if engine_pos >= 0 else 0)
        # Edge
        lang = s.value("edge_lang", "vi-VN", type=str)
        for i in range(self._edge_lang.count()):
            if self._edge_lang.itemData(i) == lang:
                self._edge_lang.setCurrentIndex(i); break
        self._update_edge_speakers()
        if v := s.value("edge_voice", type=str):
            self._edge_voice.setCurrentText(v)
        # Gemini
        if k := s.value("gemini_key", type=str):
            self._gemini_key.setText(k)
        if m := s.value("gemini_model", type=str):
            idx = self._gemini_model.findText(m)
            if idx >= 0: self._gemini_model.setCurrentIndex(idx)
        if v := s.value("gemini_voice", type=str):
            idx = self._gemini_voice.findText(v)
            if idx >= 0: self._gemini_voice.setCurrentIndex(idx)
        if st := s.value("gemini_style", type=str):
            self._gemini_style.setPlainText(st)
        if s.contains("gemini_affective"):
            self._gemini_affective.setChecked(s.value("gemini_affective", type=bool))
        if s.contains("gemini_temperature"):
            self._gemini_temperature.setValue(
                s.value("gemini_temperature", type=float)
            )
        self._on_gemini_model_changed()
        # VieNeu
        if m := s.value("vieneu_mode", type=str):
            mode_pos = self._vieneu_mode.findData(m)
            if mode_pos >= 0:
                self._vieneu_mode.setCurrentIndex(mode_pos)
        if e := s.value("vieneu_emotion", type=str):
            emo_pos = self._vieneu_emotion.findData(e)
            if emo_pos >= 0:
                self._vieneu_emotion.setCurrentIndex(emo_pos)
        if p := s.value("vieneu_ref_path", type=str):
            self._vieneu_ref_path.setText(p)
        if s.contains("vieneu_force_cpu"):
            self._vieneu_force_cpu.setChecked(s.value("vieneu_force_cpu", type=bool))
        # Giọng VieNeu: danh sách nạp lazy (khi mở panel) — lưu tạm để khôi phục sau nạp.
        self._vieneu_saved_voice = s.value("vieneu_voice", "", type=str)
        # Common
        for key, spin, default in [
            ("speed", self._speed, 1.0), ("max_speed", self._max_speed, 1.8),
            ("retry_delay", self._retry_delay, 1.0),
        ]:
            if s.contains(key): spin.setValue(s.value(key, type=float))
        if s.contains("retry"): self._retry.setValue(s.value("retry", type=int))
        if s.contains("normalize"): self._normalize.setChecked(s.value("normalize", type=bool))
        if s.contains("voice_clarity"): self._voice_clarity.setChecked(s.value("voice_clarity", type=bool))
        if s.contains("high_quality"): self._high_quality.setChecked(s.value("high_quality", type=bool))
        if s.contains("target_lufs"):
            _li = self._lufs.findData(s.value("target_lufs", type=float))
            if _li >= 0: self._lufs.setCurrentIndex(_li)
        if fv := s.value("output_format", type=str):
            _fi = self._fmt.findData(fv)
            if _fi >= 0: self._fmt.setCurrentIndex(_fi)
        if (bv := s.value("output_bitrate", type=int)):
            _bi = self._bitrate.findData(bv)
            if _bi >= 0: self._bitrate.setCurrentIndex(_bi)
        if sv := s.value("wav_subtype", type=str):
            _si = self._wav_subtype.findData(sv)
            if _si >= 0: self._wav_subtype.setCurrentIndex(_si)
        if d := s.value("device", type=str):
            idx = self._device.findText(d)
            if idx >= 0: self._device.setCurrentIndex(idx)
        if o := s.value("output", type=str):
            self._output.setText(o)
        for key, spin, _default in [
            ("dialog_pause_ms", self._dialog_pause, 300),
            ("max_overlap_ms", self._max_overlap, 500),
            ("skip_overlap_ms", self._skip_overlap, 0),
        ]:
            if s.contains(key):
                val = s.value(key, type=int)
                # Migration: skip_overlap_ms default cũ là 2000 → đổi sang 0 (disabled)
                if key == "skip_overlap_ms" and val == 2000:
                    val = 0
                spin.setValue(val)
        for key, chk, _default in [
            ("clean_tags", self._clean_tags, True),
            ("double_pass", self._double_pass, True),
            ("elastic_timing", self._elastic_timing, True),
        ]:
            if s.contains(key): chk.setChecked(s.value(key, type=bool))
        if s.contains("max_drift_ms"):
            self._max_drift.setValue(s.value("max_drift_ms", type=int))
        if s.contains("lead_in_ms"):
            self._lead_in.setValue(s.value("lead_in_ms", type=int))
        if s.contains("anchor_gap_s"): self._anchor_gap.setValue(s.value("anchor_gap_s", type=float))
        if s.contains("max_segment_s"): self._max_segment.setValue(s.value("max_segment_s", type=float))
        if s.contains("comfort_ratio"): self._comfort_ratio.setValue(s.value("comfort_ratio", type=float))
        if s.contains("min_pause_pct"): self._min_pause.setValue(s.value("min_pause_pct", type=int))
        if s.contains("max_intra_gap_s"): self._max_intra_gap.setValue(s.value("max_intra_gap_s", type=float))
        if s.contains("allow_overlap"): self._allow_overlap.setChecked(s.value("allow_overlap", type=bool))
        if s.contains("timing_strategy"):
            i = self._strategy.findData(s.value("timing_strategy", type=str))
            if i >= 0: self._strategy.setCurrentIndex(i)
        if s.contains("min_stretch"): self._min_stretch.setValue(s.value("min_stretch", type=float))
        if s.contains("last_extend_ms"):
            self._last_extend.setValue(s.value("last_extend_ms", type=int))
        for _key, _cb in (("skip_paren", self._skip_paren), ("skip_square", self._skip_square),
                          ("skip_curly", self._skip_curly), ("skip_music_pair", self._skip_music_pair),
                          ("skip_music_line", self._skip_music_line)):
            if s.contains(_key):
                _cb.setChecked(s.value(_key, type=bool))
        if s.contains("edge_concurrency"):
            self._edge_concurrency.setValue(s.value("edge_concurrency", type=int))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_action_states(self) -> None:
        has = self._view_model.has_source and not self._view_model.is_busy
        # [v3.23.252] Nếu ĐANG chọn VieNeu và giọng đang nạp, giữ nút khoá — giọng
        # chưa sẵn sàng thì chạy sẽ lỗi. Engine khác không phụ thuộc loader này.
        if (self._engine_combo.currentData() or _ENGINE_EDGE) == _ENGINE_VIENEU:
            _loader = getattr(self, "_vieneu_voice_loader", None)
            if _loader is not None and _loader.isRunning():
                has = False
        self._btn_gen.setEnabled(has)

    def _populate_table(self, results: list[TTSSegmentResult]) -> None:
        try:
            from subtitles_extractor.infrastructure.tts.edge_tts_adapter import _EDGE_API_SPEED_MAX
        except ImportError:
            _EDGE_API_SPEED_MAX = 1.5

        # [2.1] Khoá sắp xếp tự động + tín hiệu nội bộ + vẽ màn hình trong lúc nạp
        # hàng nghìn ô. Nếu để bật, mỗi setItem sẽ kích hoạt re-sort và phát tín hiệu
        # itemChanged → chậm theo cấp số nhân. Bật lại toàn bộ ở finally.
        self._table.setSortingEnabled(False)
        self._table.blockSignals(True)
        self._table.setUpdatesEnabled(False)
        try:
            # [SỬA HIỆU NĂNG QUAN TRỌNG] Xoá sạch nội dung bảng ở tầng C++ bằng một
            # lệnh duy nhất TRƯỚC khi đổ dữ liệu mới. Nếu không, việc gọi setItem đè
            # lên hàng nghìn ô cũ buộc Qt huỷ/tạo lại từng ô — gây treo UI khi chạy
            # TTS lần thứ hai trở đi (bảng đã đầy dữ liệu từ lần trước).
            self._table.clearContents()
            self._table.setRowCount(0)

            self._table.setRowCount(len(results))
            for row, r in enumerate(results):
                def _it(t, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter):
                    it = QTableWidgetItem(t); it.setTextAlignment(align); return it

                ts  = f"{int(r.start_sec//60):02d}:{r.start_sec%60:06.3f}"
                dur = f"{r.audio_duration_s:.2f}s" if not r.was_skipped else "—"
                spd = f"{r.speed_used:.2f}×" if not r.was_skipped else "—"

                # Cột "Loại": phân biệt cách tốc độ được tạo ra
                if r.was_skipped:
                    kind = "—"
                elif r.speed_used <= 1.02:
                    kind = self._translator.translate("tts.kind_base")           # tốc độ cơ bản, không tăng
                elif r.speed_used <= _EDGE_API_SPEED_MAX + 0.02:
                    kind = self._translator.translate("tts.kind_api")             # tăng qua Edge TTS rate param
                elif r.speed_used < 2.5:
                    kind = self._translator.translate("tts.kind_apits")          # API + time-stretch vừa
                else:
                    kind = self._translator.translate("tts.kind_tshigh")          # time-stretch nặng (≥2.5×, chất lượng giảm)

                if r.was_skipped:
                    st = self._translator.translate("tts.st_skip") + (f": {r.error_msg[:22]}" if r.error_msg else "")
                elif r.was_truncated:
                    st = self._translator.translate("tts.st_cut")
                elif r.overlap_s > _OVERLAP_WARN_S:
                    # [v3.23.206] Chồng đáng kể sang câu sau — gợi ý rút gọn bản dịch.
                    st = self._translator.translate("tts.st_overlap").replace("{s}", f"{r.overlap_s:.1f}")
                else:
                    st = self._translator.translate("tts.st_ok")

                self._table.setItem(row, 0, _it(str(row+1),
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter))
                self._table.setItem(row, 1, _it(ts))
                self._table.setItem(row, 2, _it(r.text))
                self._table.setItem(row, 3, _it(dur))
                self._table.setItem(row, 4, _it(spd))
                self._table.setItem(row, 5, _it(kind))
                self._table.setItem(row, 6, _it(st))
        finally:
            # Luôn bật lại updates dù có lỗi giữa chừng — tránh bảng "đóng băng".
            self._table.setUpdatesEnabled(True)
            self._table.blockSignals(False)
            # Giữ KHÔNG cho phép sort lại bằng click header để bảo toàn thứ tự dòng
            # phụ đề theo thời gian (sort theo cột sẽ phá vỡ trật tự timeline).
            self._table.setSortingEnabled(False)
        if len(results) <= 1500:
            self._table.resizeRowsToContents()
        self._apply_row_filter()

    def _apply_row_filter(self) -> None:
        """Ẩn/hiện hàng theo bộ lọc đang chọn (dùng setRowHidden — không tạo lại ô)."""
        mode = self._row_filter.currentData() if hasattr(self, "_row_filter") else "all"
        rs = self._last_results
        if not rs:
            if hasattr(self, "_lbl_filter_count"):
                self._lbl_filter_count.setText("")
            return
        shown = 0
        self._table.setUpdatesEnabled(False)
        try:
            for row, r in enumerate(rs):
                if mode == "skipped":
                    visible = r.was_skipped
                elif mode == "truncated":
                    visible = r.was_truncated
                elif mode == "fast":
                    visible = (not r.was_skipped) and r.speed_used >= 1.8
                elif mode == "issues":
                    visible = (
                        r.was_skipped or r.was_truncated
                        or ((not r.was_skipped) and r.speed_used >= 1.8)
                        or ((not r.was_skipped) and r.overlap_s > _OVERLAP_WARN_S)
                    )
                else:  # "all"
                    visible = True
                self._table.setRowHidden(row, not visible)
                if visible:
                    shown += 1
        finally:
            self._table.setUpdatesEnabled(True)
        self._lbl_filter_count.setText(
            "" if mode == "all" else self._translator.translate("tts.res_showing").replace("{shown}", str(shown)).replace("{total}", str(len(rs)))
        )

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget(); w.setLayout(layout); return w

    @staticmethod
    def _row_one(widget) -> QHBoxLayout:
        """Bọc 1 widget trong hàng có stretch (không chiếm hết chiều ngang)."""
        row = QHBoxLayout(); row.addWidget(widget); row.addStretch(); return row

    def _show_success(self, t, c): _feedback.show_success(self, t, c)
    def _show_info(self, t, c):    _feedback.show_info(self, t, c)
    def _show_warning(self, t, c): _feedback.show_warning(self, t, c)


__all__ = ["TTSPage"]
