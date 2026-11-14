"""[v3.23.695] Dựng giao diện trang Lồng tiếng: lồng tiếng theo vai, nguồn, chọn engine và bảng cấu hình từng engine, cấu hình chung, đầu ra, nút thao tác, bảng kết quả.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/tts_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TTSPage``.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import _EDGE_VOICE_MAP
from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import (
    GEMINI_TTS_MODELS,
    GEMINI_TTS_VOICES,
)
from subtitles_extractor.presentation.fluent_compat import PrimaryPushButton, PushButton
from subtitles_extractor.presentation.pages.tts_constants import (
    _DEFAULT_MAX_SPEED,
    _EDGE_LANGUAGE_DISPLAY,
    _ENGINES,
)
from subtitles_extractor.presentation.pages.tts_dubbing import TTSDubbingCoordinator
from subtitles_extractor.presentation.pages.tts_line_edits import (
    COLUMN_COUNT as _COLUMN_COUNT,
)
from subtitles_extractor.presentation.pages.tts_line_edits import TTSLineEditsController
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme.styles import caption_style
from subtitles_extractor.presentation.widgets.current_page_stack import CurrentPageStack
from subtitles_extractor.presentation.widgets.dubbing_cast_panel import DubbingCastPanel
from subtitles_extractor.presentation.widgets.flow_layout import flow_row
from subtitles_extractor.presentation.widgets.section_card import SectionCard

logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")



class TTSUiMixin:
    """Dựng giao diện trang Lồng tiếng: lồng tiếng theo vai, nguồn, chọn engine và bảng cấu hình từng engine, cấu hình chung, đầu ra, nút thao tác, bảng kết quả."""

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
            # [v3.23.674] Bám đỉnh ô: phần cao thừa (nếu có) nằm dưới cùng, không chen giữa các nhóm.
            col.addWidget(w, 0, Qt.AlignmentFlag.AlignTop)
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
            "🎙 Nguồn và Engine",
        )
        self._config_tabs.addTab(
            self._make_config_tab(
                self._build_common_config(), self._build_output_group()
            ),
            "🎚 Âm thanh và Xuất",
        )
        self._config_tabs.addTab(
            self._make_config_tab(self._build_dubbing_group()),
            self._translator.translate("tts.tab_dubbing"),
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

    def _build_dubbing_group(self) -> QWidget:
        """[v3.23.593] Thẻ lồng tiếng: mỗi nhân vật một giọng (điều phối: tts_dubbing)."""
        self._dubbing_panel = DubbingCastPanel(self._translator)
        self._dubbing = TTSDubbingCoordinator(
            self._dubbing_panel, self._view_model, self._settings,
            self._translator.translate, self._dubbing_catalog, parent=self,
            main_voice_provider=self._dubbing_main_voice,
        )
        return self._dubbing_panel

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
        # [v3.23.674] Gợi ý cài đặt CHỈ cho engine chưa cài (trước đây luôn hiện cả 3 lệnh pip dù đã cài đủ).
        self._install_hint = QLabel("")
        self._install_hint.setTextFormat(Qt.TextFormat.RichText)
        self._install_hint.setStyleSheet(caption_style())
        self._install_hint.setWordWrap(True)
        self._install_hint.setVisible(False)
        lay.addWidget(self._install_hint)
        g.add_layout(lay)
        return g

    def _build_engine_stack(self) -> QWidget:
        # [v3.23.674] Khung chỉ cao theo engine ĐANG CHỌN (xem CurrentPageStack).
        self._engine_stack = CurrentPageStack()
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

        # [v3.23.677] Chú thích bằng lời thường (bản cũ: "Elastic timing", "pip install pedalboard", "Rubber Band",
        # GPL… dồn thành một đoạn vì xuống dòng "\n" bị bỏ ở chế độ RichText). Chi tiết kỹ thuật chuyển vào tooltip.
        info = QLabel(self._translator.translate("tts.edge_info"))
        info.setToolTip(self._translator.translate("tts.edge_info_tip"))
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
        # [v3.23.677] Key Gemini dùng chung (Cài đặt → Khoá API và AI) thay ô "API Key" riêng của trang này.
        from subtitles_extractor.presentation.ai_access_store import (
            shared_ai_access_store,
        )
        from subtitles_extractor.presentation.widgets.ai_access_panel import (
            AiAccessSummary,
        )

        self._ai_store = shared_ai_access_store()
        self._gemini_key_summary = AiAccessSummary(self._translator, self._ai_store, gemini_only=True)
        form.addRow(self._translator.translate("ai.summary_label"), self._gemini_key_summary)

        self._gemini_model = QComboBox()
        self._gemini_model.addItems(GEMINI_TTS_MODELS)
        self._gemini_model.setToolTip(
            "★ gemini-3.8-live (Live API) — KHUYẾN NGHỊ với key miễn phí: không giới hạn lượt/ngày,\n"
            "  đọc đúng nguyên văn, ~3 phiên song song mỗi key (nhập nhiều key cách nhau dấu phẩy để nhanh hơn);\n"
            "  dòng có khung chật tự được dặn đọc nhanh hơn (ít phải nén âm thanh)\n"
            "  • gemini-3.8-live-extended-thinking — cùng họ, có suy nghĩ (mức thấp), giọng dài hơn một chút\n"
            "  • gemini-3.1-flash-live-preview — preview cũ, đọc nhanh nhưng sai chữ nhiều hơn\n"
            "Standard TTS: 1 API call/dòng — key miễn phí chỉ 10 lượt/ngày (không đủ một tập phim)\n"
            "  • gemini-3.8-flash-lite-tts — GA 9/2026, nhanh/rẻ\n"
            "  • gemini-3.8-flash-tts — GA 9/2026, giọng biểu cảm nhất\n"
            "  • gemini-3.1-flash-tts-preview — preview, được 3.8 Flash-Lite thay thế\n"
            "  • gemini-2.5-*-tts — đời cũ, Google giới hạn cho người đã dùng gần đây\n"
            "Native Audio đời 2.5 (Live API) — phát âm tiếng Việt sai nhiều (đo thật 24/40 câu sai), chỉ để tương thích\n"
            "  • gemini-2.5-flash-native-audio-latest — tracking alias\n"
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
        # [v3.23.688] Cảm xúc từng câu theo nhãn của phân tích ngữ cảnh (Gemini TTS 3.8, trường chỉ dẫn riêng).
        self._gemini_line_emotion = QCheckBox(self._translator.translate("tts.opt_line_emotion"))
        self._gemini_line_emotion.setChecked(False)
        self._gemini_line_emotion.setToolTip(self._translator.translate("tts.opt_line_emotion_tip"))
        form.addRow(self._gemini_line_emotion)

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
        # [v3.23.445] v3 Turbo (48kHz) nay là MẶC ĐỊNH của SDK VieNeu + chất lượng CAO NHẤT
        # (đã hết "thử nghiệm" từ bản 3.3.0) → đưa lên ĐẦU làm mặc định. Standard/Turbo (24kHz)
        # giữ lại cho máy yếu / tương thích ngược.
        self._vieneu_mode.addItem(self._translator.translate("tts.vn_mode_v3turbo"), userData="v3turbo")
        self._vieneu_mode.addItem(self._translator.translate("tts.vn_mode_standard"), userData="standard")
        self._vieneu_mode.addItem(self._translator.translate("tts.vn_mode_turbo"), userData="turbo")
        self._vieneu_mode.setToolTip(
            "v3 Turbo: 48kHz, chất lượng CAO NHẤT & là mặc định chính thức của VieNeu "
            "(kiến trúc mới, giọng tự nhiên nhất) — KHUYẾN NGHỊ.\n"
            "Standard: 24kHz, model cũ ổn định (dùng nếu máy yếu hoặc v3 Turbo lỗi).\n"
            "Turbo: 24kHz nhanh hơn NHƯNG chất lượng thấp & có thể nhiễu với câu rất "
            "ngắn (< 5 từ)."
        )
        self._vieneu_mode.currentIndexChanged.connect(self._on_vieneu_mode_changed)
        self._vieneu_mode.currentIndexChanged.connect(
            lambda _idx: self._sync_vieneu_mode_controls()
        )
        form.addRow(self._translator.translate("tts.tts_mode"), self._vieneu_mode)

        # Sắc thái giọng.
        self._vieneu_emotion = QComboBox()
        self._vieneu_emotion.addItem(self._translator.translate("tts.vn_emo_natural"), userData="natural")
        self._vieneu_emotion.addItem(self._translator.translate("tts.vn_emo_story"), userData="storytelling")
        self._vieneu_emotion.setToolTip(self._translator.translate("tts.tip_emotion"))
        form.addRow(self._translator.translate("tts.tts_emotion"), self._vieneu_emotion)

        # [v3.23.446] Tag cảm xúc VieNeu v3 Turbo (thử nghiệm): tự chuyển interjection cười/thở
        # dài trong bản dịch thành [cười]/[thở dài] để giọng đọc biểu cảm hơn. CHỈ tác động
        # audio (phụ đề giữ nguyên). Chỉ có tác dụng ở mode v3 Turbo.
        self._vieneu_emotion_cues = QCheckBox("Tag cảm xúc (v3 Turbo, thử nghiệm)")
        self._vieneu_emotion_cues.setChecked(False)
        self._vieneu_emotion_cues.setToolTip(
            "Thử nghiệm (chỉ v3 Turbo): tự chuyển tiếng cười/thở dài rõ ràng trong lời "
            "thoại (haha, khà khà, haizz…) thành tag [cười]/[thở dài] để giọng đọc phát "
            "biểu cảm THẬT thay vì đọc chữ. KHÔNG ảnh hưởng file phụ đề (chỉ audio)."
        )
        form.addRow("", self._vieneu_emotion_cues)

        # Ép chạy CPU/ONNX — né lỗi PyTorch/cuDNN (WinError 127) trên máy CUDA lệch bản.
        self._vieneu_force_cpu = QCheckBox(self._translator.translate("tts.opt_force_cpu"))
        self._vieneu_force_cpu.setChecked(True)
        self._vieneu_force_cpu.setToolTip(
            "Bật (khuyến nghị): chạy bằng ONNX Runtime trên CPU, né lỗi tải DLL "
            "PyTorch/cuDNN.\nTắt: thử dùng GPU — nhưng chỉ có tác dụng khi torch đã "
            "được cài (xem ghi chú bên dưới)."
        )
        form.addRow(self._translator.translate("tts.tts_device"), self._vieneu_force_cpu)

        # [v3.23.595] VieNeu 3.8: backbone ONNX/CPU của v3 Turbo có bản int8 — nhanh
        # ~3x mỗi frame theo chú thích SDK, nhưng cần CPU có VNNI để không méo tiếng.
        # Mặc định tắt.
        self._vieneu_int8 = QCheckBox(self._translator.translate("tts.opt_vieneu_int8"))
        self._vieneu_int8.setChecked(False)
        self._vieneu_int8.setToolTip(self._vieneu_int8_tooltip())
        form.addRow("", self._vieneu_int8)

        # [v3.23.416] Kích thước LÔ khi chạy GPU (tăng tốc phim dài). Tắt = từng câu (ổn định
        # nhất). Auto = tự chọn theo VRAM. Chỉ có tác dụng khi BỎ tick "Ép chạy CPU".
        self._vieneu_batch = QComboBox()
        self._vieneu_batch.addItem(
            self._translator.translate("tts.tts_batch_off"), userData=0
        )
        self._vieneu_batch.addItem(
            self._translator.translate("tts.tts_batch_auto"), userData=-1
        )
        self._vieneu_batch.addItem(
            self._translator.translate("tts.tts_batch_all"), userData=-2
        )
        _batch_n_tpl = self._translator.translate("tts.tts_batch_n")
        for _bs in (8, 16, 24, 32, 64, 128, 256, 512):
            label = (
                _batch_n_tpl.format(n=_bs)
                if "{n}" in _batch_n_tpl
                else f"{_bs} câu/lô"
            )
            self._vieneu_batch.addItem(label, userData=_bs)
        self._vieneu_batch.setCurrentIndex(0)  # mặc định TẮT (an toàn)
        self._vieneu_batch.setToolTip(
            "Gộp nhiều câu trong một lần suy luận GPU → nhanh hơn cho phim dài.\n"
            "• Tắt: tổng hợp từng câu (ổn định nhất, mặc định).\n"
            "• Một lô cho TẤT CẢ câu: gom hết vào một lần gọi (cần ~2GB RAM, không thấy\n"
            "  tiến độ tới khi xong; GPU vẫn xử lý chunk theo nhóm nên không nhanh hơn nhiều).\n"
            "• Tự động / số câu: số CÂU gom mỗi lần gọi GPU (giảm vòng lặp).\n"
            "• Số CHUNK/forward GPU được tự chọn theo VRAM (8GB→32, 16GB→64…) để tận dụng\n"
            "  VRAM dư — đây mới là yếu tố tăng tốc chính (mặc định vieneu chỉ 4 chunk).\n"
            "• Thiếu VRAM → tự lùi về từng câu cho lô đó (không lỗi).\n"
            "• Câu 'ngân dài' luôn thử lại từng câu (giữ chất lượng).\n"
            "Chỉ tác dụng khi BỎ tick 'Ép chạy CPU'."
        )
        form.addRow(
            self._translator.translate("tts.tts_batch"),
            self._vieneu_batch,
        )

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
        self._sync_vieneu_mode_controls()

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
            "Trần tốc độ đọc TỔNG của một câu. Ứng dụng LUÔN dùng ĐÚNG giá trị bạn đặt.\n\n"
            "• Edge TTS: tới 5.0× vẫn tốt (tăng tốc phía máy chủ, giữ chất lượng).\n"
            "• VieNeu / Gemini: nén hậu kỳ. KHUYẾN NGHỊ ≤ 2.0× — nén quá 2× dễ làm\n"
            "  giọng mất rõ, thành 'tiếng gió'. Đặt cao hơn VẪN được áp dụng (bạn tự\n"
            "  quyết đánh đổi tốc độ ↔ độ rõ).\n\n"
            "Câu quá dài không nén đủ trong trần này sẽ TRÀN sang câu sau (nếu bật\n"
            "'Cho phép chồng tiếng') — giữ trọn nội dung, ưu tiên nghe rõ."
        )
        self._max_speed.setSingleStep(0.1); self._max_speed.setDecimals(1); self._max_speed.setValue(_DEFAULT_MAX_SPEED)
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
            "→ Xuất kèm file .srt mới khớp lời thoại đã dời.\n"
            "VieNeu/Gemini: câu chật được nén tới mức 'Nén êm' trước, còn thiếu mới\n"
            "ngân trễ qua mốc câu sau tối đa min(Dung sai cộng dồn, 0,5 s) — câu sau\n"
            "đọc lùi tương ứng, không chồng tiếng."
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
        # [v3.23.630] Chỉ tổng hợp lại dòng đã đổi (dùng lại audio cũ) — log thật: 21 lượt cả
        # phim × 18 phút dù mỗi lần chỉ sửa vài chục dòng.
        self._incremental = QCheckBox(self._translator.translate("tts.opt_incremental"))
        self._incremental.setChecked(True)
        self._incremental.setToolTip(self._translator.translate("tts.tip_incremental"))
        lay.addWidget(self._incremental)
        # [v3.23.674] Hàng nút tự xuống dòng (nút chính bị cắt chữ "Tổng hợp tiếng nó…" ở màn hình 1366×768).
        btn_holder, btn_row = flow_row()
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
        btn_row.addWidget(self._btn_gen); btn_row.addWidget(self._btn_batch)
        btn_row.addWidget(self._btn_can)
        btn_row.addWidget(self._btn_reset)
        lay.addWidget(btn_holder)
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

        # [v3.23.598] Cột thứ 8 "Người nói · Giọng" do TTSLineEditsController dựng (hiện
        # cạnh cột Nội dung khi bật lồng tiếng).
        self._table = QTableWidget(0, _COLUMN_COUNT)
        self._table.setHorizontalHeaderLabels(
            ["#", "Thời gian", "Nội dung", "Thời lượng", "Tốc độ", "Loại", "Trạng thái"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # [v3.23.450] Cho SỬA nội dung bằng double-click (chỉ cột Nội dung được set editable
        # ở _populate_table). itemChanged → cập nhật kết quả + event nguồn để TTS lại đúng.
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self._table.itemChanged.connect(self._on_result_item_edited)
        self._table.itemSelectionChanged.connect(self._on_result_selection_changed)
        self._table.setWordWrap(True); self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setDefaultSectionSize(36)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for c in (3, 4, 5, 6):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self._table, 1)
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
        # [v3.23.450] Nghe lại + TTS lại các dòng ĐANG CHỌN (dùng audio segment per-line).
        self._btn_play_sel = PushButton("▶ Nghe dòng chọn")
        self._btn_play_sel.setEnabled(False)
        self._btn_play_sel.setToolTip("Phát audio của dòng đang chọn (đã tổng hợp).")
        self._btn_play_sel.clicked.connect(self._on_play_selected)
        self._btn_resynth = PushButton("🔄 TTS lại dòng chọn")
        self._btn_resynth.setEnabled(False)
        self._btn_resynth.setToolTip(
            "Tổng hợp lại các dòng đang chọn (kể cả nội dung vừa sửa) rồi ghép lại "
            "file audio — không phải chạy lại toàn bộ."
        )
        self._btn_resynth.clicked.connect(self._on_resynthesize_selected)
        # [v3.23.459] Chọn nhanh MỌI dòng "dài bất thường" để nghe/TTS lại (không phải nghe dò).
        self._btn_select_abnormal = PushButton("⚠ Chọn dòng bất thường")
        self._btn_select_abnormal.setEnabled(False)
        self._btn_select_abnormal.setToolTip(
            "Chọn tất cả dòng có audio dài bất thường (model ngân dài) để TTS lại hàng loạt."
        )
        self._btn_select_abnormal.clicked.connect(self._on_select_abnormal)
        # [v3.23.657] Dọn audio từng dòng THEO YÊU CẦU (không tự dọn sau xuất bản): giữ master + SRT.
        self._line_audio_bytes = 0
        self._btn_clean_line_audio = PushButton(self._translator.translate("tts.btn_clean_line_audio"))
        self._btn_clean_line_audio.setEnabled(False)
        self._btn_clean_line_audio.setToolTip(self._translator.translate("tts.tip_clean_line_audio"))
        self._btn_clean_line_audio.clicked.connect(self._on_clean_line_audio)
        # [v3.23.598] Sửa tay từng dòng: người nói (giọng) + lời thoại, trước/sau khi TTS.
        self._line_edits = TTSLineEditsController(
            self._table, self._view_model, self._dubbing, self._dubbing_panel,
            self._settings, self._translator.translate,
            commit_edit=self._commit_active_edit,
            show_status=self._stage_lbl.setText,
            context_buttons=(self._btn_play_sel, self._btn_resynth),
            rebuild_table=self._rebuild_after_structure,
            resynthesize=self._resynthesize_positions,
            shorten=self._start_shorten,
            parent=self,
            edits_store=self._make_edits_store(),
        )
        self._build_result_actions(lay)
        self._install_review_shortcuts()
        # [v3.23.682] Đếm việc cần duyệt sau mỗi thay đổi (gom 0,5 s; bảng vẽ lại người nói chạy trước đó).
        self._review_timer = QTimer(self)
        self._review_timer.setSingleShot(True)
        self._review_timer.setInterval(500)
        self._review_timer.timeout.connect(lambda: self.review_count_changed.emit(self.review_count()))
        self._view_model.result_ready.connect(self._schedule_review_count)
        self._view_model.line_edits_changed.connect(self._schedule_review_count)
        self._dubbing.line_speakers_changed.connect(self._schedule_review_count)
        # [v3.23.653] Người nói chọn tay → đồng bộ về Visual Cues của trang Dịch (gom 1,5 s sau lần sửa cuối).
        self._speaker_sync_timer = QTimer(self)
        self._speaker_sync_timer.setSingleShot(True)
        self._speaker_sync_timer.setInterval(1500)
        self._speaker_sync_timer.timeout.connect(self._emit_speaker_corrections)
        self._last_speaker_corrections: tuple[str, dict[int, str]] = ("", {})
        self._view_model.line_edits_changed.connect(self._speaker_sync_timer.start)
        self._dubbing.line_speakers_changed.connect(self._speaker_sync_timer.start)
        g.add_layout(lay)
        return g

    def _build_result_actions(self, lay: QVBoxLayout) -> None:
        """[v3.23.677] Dưới bảng dòng: 2 hàng nút hay dùng + 3 menu (trước: 22 nút xếp 6 hàng).

        Ảnh chụp 1366×768 bản 3.23.676: 22 nút xám khi chưa có dữ liệu chiếm nửa cột phải. Nút ít dùng (thêm/xoá/
        gộp/tách dòng, khôi phục, lịch sử, dọn audio, xuất debug…) nay nằm trong menu; mọi nút vẫn là đối tượng cũ.
        """
        from subtitles_extractor.presentation.widgets.menu_button import MenuButton

        tr = self._translator.translate
        le = self._line_edits.action_buttons()
        line_holder, line_row = flow_row()
        for button in (self._btn_play_sel, self._btn_resynth, le["speaker"], le["quick_review"],
                       le["shorten"], le["merge_fragments"], le["undo"]):
            line_row.addWidget(button)
        self._menu_edit_lines = MenuButton(tr("tts.menu_edit_lines"), [
            le["add_line"], le["delete_lines"], le["timing"], le["merge"], le["split"], le["redo"],
        ])
        # [v3.23.679] AI nghe lại từng câu, TTS lại câu đọc sai (tối đa 2 lượt), chọn sẵn câu còn sai.
        from subtitles_extractor.presentation.pages.tts_pronunciation import (
            PronunciationController,
        )

        self._pronunciation = PronunciationController(self, tr)
        self._btn_check_pron = PushButton(tr("tts.btn_check_pron"))
        self._btn_check_pron.setToolTip(tr("tts.btn_check_pron_tip"))
        self._btn_check_pron.setEnabled(False)
        self._btn_check_pron.clicked.connect(lambda _checked=False: self._pronunciation.start())
        self._view_model.result_ready.connect(self._pronunciation.on_result_ready)
        self._view_model.error_occurred.connect(self._pronunciation.on_run_failed)
        self._view_model.tts_cancelled.connect(self._pronunciation.on_run_failed)
        self._menu_select = MenuButton(tr("tts.menu_select"), [
            self._btn_check_pron, self._btn_select_abnormal, le["review"], le["select_edited"], le["revert"],
        ])
        line_row.addWidget(self._menu_edit_lines)
        line_row.addWidget(self._menu_select)
        lay.addWidget(line_holder)
        file_holder, file_row = flow_row()
        self._menu_more = MenuButton(tr("tts.menu_more"), [
            le["restore_deleted"], le["restore_original"], le["history"],
            self._btn_clean_line_audio, self._btn_export,
        ])
        for widget in (self._btn_open, self._btn_open_file, self._menu_more):
            file_row.addWidget(widget)
        lay.addWidget(file_holder)

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget(); w.setLayout(layout); return w

    @staticmethod
    def _row_one(widget) -> QHBoxLayout:
        """Bọc 1 widget trong hàng có stretch (không chiếm hết chiều ngang)."""
        row = QHBoxLayout(); row.addWidget(widget); row.addStretch(); return row


__all__ = ["TTSUiMixin"]
