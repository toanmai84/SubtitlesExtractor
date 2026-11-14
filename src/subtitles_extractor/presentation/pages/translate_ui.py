"""[v3.23.694] Dựng giao diện trang Dịch: nguồn, cấu hình AI, ngữ cảnh, video, các giai đoạn, nút thao tác, bảng kết quả.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/translate_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import contextlib
import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QScrollArea,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationStageKind
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.presentation.fluent_compat import PrimaryPushButton, PushButton
from subtitles_extractor.presentation.pages.translate_constants import (
    _DEFAULT_MODELS,
    _TARGET_LANGUAGES,
    DEFAULT_ANALYSIS_MODEL,
    DEFAULT_REFINE_MODEL,
)
from subtitles_extractor.presentation.pages.translate_widgets import _StagePanel
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.widgets.section_card import SectionCard

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")



class TranslateUiMixin:
    """Dựng giao diện trang Dịch: nguồn, cấu hình AI, ngữ cảnh, video, các giai đoạn, nút thao tác, bảng kết quả."""

    def _make_config_tab(self, *widgets: QWidget) -> QScrollArea:
        """Bọc các nhóm cấu hình vào tab cuộn dọc (mỗi tab ngắn, đỡ cuộn)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(_m.SPACING_XS, _m.SPACING_SM, _m.SPACING_SM, _m.SPACING_SM)
        lay.setSpacing(_m.SPACING_MD)
        for w in widgets:
            lay.addWidget(w)
        lay.addStretch(1)
        scroll.setWidget(holder)
        return scroll

    def _build_ui(self) -> None:
        from PySide6.QtWidgets import QSplitter, QTabWidget

        root = QVBoxLayout(self)
        root.setContentsMargins(_m.SPACING_MD, _m.SPACING_MD, _m.SPACING_MD, _m.SPACING_MD)
        root.setSpacing(_m.SPACING_SM)

        # [v3.23.119] TRÁI = cấu hình gom theo TAB (mỗi tab ngắn, gần như không cuộn) +
        # nút chạy chung dưới; PHẢI = kết quả. Trước đây xếp dọc tất cả -> cuộn.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(0, 0, _m.SPACING_SM, 0)
        left_col.setSpacing(_m.SPACING_SM)

        self._config_tabs = QTabWidget()
        self._config_tabs.addTab(
            self._make_config_tab(
                self._build_source_group(), self._build_ai_config_group()
            ),
            self._translator.translate("translate.tab_basic"),
        )
        self._config_tabs.addTab(
            self._make_config_tab(self._build_stages_group()), self._translator.translate("translate.tab_stages")
        )
        self._config_tabs.addTab(
            self._make_config_tab(
                self._build_context_group(), self._build_video_context_group()
            ),
            self._translator.translate("translate.tab_context"),
        )
        left_col.addWidget(self._config_tabs, 1)
        # Nút chạy + tiến trình luôn hiển thị dưới tab (không bị ẩn khi chuyển tab).
        left_col.addWidget(self._build_actions_group())
        left.setMinimumWidth(480)
        splitter.addWidget(left)

        result_panel = self._build_result_group()
        result_panel.setMinimumWidth(420)
        splitter.addWidget(result_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 700])
        root.addWidget(splitter, 1)

    def _build_source_group(self) -> QWidget:
        card = SectionCard(self._translator.translate("translate.tr_sec_source"))
        col = QVBoxLayout()
        col.setSpacing(_m.SPACING_SM)
        row = QHBoxLayout()
        row.setSpacing(_m.SPACING_SM)
        self._btn_load_file = PushButton(self._translator.translate("translate.btn_load_srt"))
        self._btn_load_file.clicked.connect(self._on_load_file_clicked)
        self._btn_pull_editor = PushButton(self._translator.translate("translate.btn_from_editor"))
        self._btn_pull_editor.clicked.connect(self._on_pull_editor_clicked)
        row.addWidget(self._btn_load_file, 1)
        row.addWidget(self._btn_pull_editor, 1)
        col.addLayout(row)
        # Nhãn trạng thái nằm dòng riêng để không bị chèn/cắt trong cột hẹp.
        self._lbl_source = QLabel(self._translator.translate("translate.status_no_source"))
        self._lbl_source.setStyleSheet("font-weight: bold;")
        self._lbl_source.setWordWrap(True)
        col.addWidget(self._lbl_source)
        card.add_layout(col)
        return card

    def _build_ai_config_group(self) -> QWidget:
        from PySide6.QtWidgets import QFormLayout as _QFL
        card = SectionCard(self._translator.translate("translate.tr_sec_ai"))
        form = QFormLayout()
        form.setSpacing(_m.SPACING_SM)
        # Khi cột hẹp, cho phép field xuống dòng dưới nhãn + nở rộng theo cột.
        form.setRowWrapPolicy(_QFL.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(_QFL.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # [v3.23.677] Khoá API + nhà cung cấp AI chuyển sang Cài đặt → "Khoá API và AI" (dùng chung với Gemini TTS
        # và Rút gọn câu). Ở đây chỉ còn một dòng tóm tắt + nút mở Cài đặt.
        from subtitles_extractor.presentation.ai_access_store import (
            shared_ai_access_store,
        )
        from subtitles_extractor.presentation.widgets.ai_access_panel import (
            AiAccessSummary,
        )

        self._ai_store = shared_ai_access_store()
        self._ai_summary = AiAccessSummary(self._translator, self._ai_store)
        form.addRow(self._translator.translate("ai.summary_label"), self._ai_summary)
        self._ai_form = form

        self._lang_combo = QComboBox()
        self._lang_combo.addItems(_TARGET_LANGUAGES)
        self._lang_combo.setCurrentText("Vietnamese")
        form.addRow(self._translator.translate("translate.tr_target_lang"), self._lang_combo)

        options_row = QVBoxLayout()
        options_row.setSpacing(_m.SPACING_XS)
        self._tags_check = QCheckBox(self._translator.translate("translate.opt_speaker"))
        self._desc_check = QCheckBox(self._translator.translate("translate.opt_keep_sfx"))
        self._desc_check.setChecked(True)
        # [v3.23.623] Gán ai nói / nói với ai từ phụ đề (+ tách giọng từ video cạnh phụ đề).
        # Chấm mù 3 tập × 2 lượt: lỗi nặng 27 → 2 (điểm 163 → 32). Mặc định BẬT.
        self._text_cues_check = QCheckBox(self._translator.translate("translate.opt_text_cues"))
        self._text_cues_check.setChecked(True)
        self._text_cues_check.setToolTip(self._translator.translate("translate.opt_text_cues_tip"))
        # [v3.23.671] Nghe âm thanh lời thoại khi lập bảng nhân vật (thêm người không tên có giọng
        # riêng: nam/nữ đệ tử…). Tập 1 phim mẫu: nữ đệ tử 0/3 → 3/3 lượt. Mặc định BẬT.
        self._voice_roster_check = QCheckBox(self._translator.translate("translate.opt_voice_roster"))
        self._voice_roster_check.setChecked(True)
        self._voice_roster_check.setToolTip(
            self._translator.translate("translate.opt_voice_roster_tip"))
        options_row.addWidget(self._tags_check)
        options_row.addWidget(self._desc_check)
        options_row.addWidget(self._text_cues_check)
        options_row.addWidget(self._voice_roster_check)
        form.addRow(self._translator.translate("translate.tr_options"), self._wrap_layout(options_row))

        # [v3.23.129] Độ phân giải VIDEO khi phân tích ngữ cảnh & Visual Cues.
        self._media_res_combo = QComboBox()
        self._media_res_combo.addItem(self._translator.translate("translate.mediares_low"), "low")
        self._media_res_combo.addItem(self._translator.translate("translate.mediares_medium"), "medium")
        self._media_res_combo.addItem(self._translator.translate("translate.mediares_high"), "high")
        self._media_res_combo.setToolTip(
            "Độ phân giải video gửi lên Gemini khi PHÂN TÍCH ngữ cảnh & Visual Cues.\n"
            "Cao = nhìn rõ mặt/biểu cảm/khẩu hình nhất (nhận diện người nói & cảm xúc\n"
            "chính xác hơn) nhưng tốn token & chậm hơn. Chỉ áp dụng Gemini 3.x.\n"
            "Lưu ý: khi DỊCH có đính video, hệ thống luôn dùng mức Thấp để tiết kiệm."
        )
        try:
            _cur_res = self._view_model.get_analysis_media_resolution()
            _idx = self._media_res_combo.findData(_cur_res)
            if _idx >= 0:
                self._media_res_combo.setCurrentIndex(_idx)
        except (AttributeError, RuntimeError):
            pass
        self._media_res_combo.currentIndexChanged.connect(
            self._on_media_resolution_changed
        )
        form.addRow(self._translator.translate("translate.tr_media_res"), self._media_res_combo)
        # [v3.23.140] Mức Thinking cho PHÂN TÍCH ngữ cảnh toàn cục (roster/tóm tắt/cues).
        self._analysis_think_combo = QComboBox()
        self._analysis_think_combo.addItem(self._translator.translate("translate.athink_low"), "low")
        self._analysis_think_combo.addItem(self._translator.translate("translate.athink_medium"), "medium")
        self._analysis_think_combo.addItem(self._translator.translate("translate.athink_high"), "high")
        self._analysis_think_combo.setToolTip(
            "Mức suy luận (thinking) khi PHÂN TÍCH ngữ cảnh toàn cục: bảng nhân vật,\n"
            "tóm tắt, Visual Cues, giới tính/vai vế người nói. Phân tích chạy MỘT LẦN\n"
            "nhưng quyết định chất lượng TOÀN BỘ bản dịch phía sau, nên nên để 'Trung\n"
            "bình' trở lên. 'Cao' hợp phim nhiều nhân vật/cốt truyện phức tạp. Chỉ áp\n"
            "dụng Gemini 3.x."
        )
        try:
            _cur_think = self._view_model.get_analysis_thinking_level()
            _idx_t = self._analysis_think_combo.findData(_cur_think)
            if _idx_t >= 0:
                self._analysis_think_combo.setCurrentIndex(_idx_t)
        except (AttributeError, RuntimeError):
            pass
        self._analysis_think_combo.currentIndexChanged.connect(
            self._on_analysis_thinking_changed
        )
        form.addRow(self._translator.translate("translate.tr_analysis_think"), self._analysis_think_combo)
        # [v3.23.612] Cách gọi tên kèm vai vế: thuần Việt ("anh Kiếm") hay Hán-Việt
        # ("Kiếm ca") — dùng cho cả khâu phân tích ngữ cảnh lẫn khâu dịch.
        self._native_address_check = QCheckBox(
            self._translator.translate("translate.native_address")
        )
        self._native_address_check.setToolTip(
            "Phim hiện đại, đời thường: tiếng gọi tên kèm vai vế viết như người Việt\n"
            "nói — '劍哥' → 'anh Kiếm', '洪大嫂' → 'chị Hồng', '林先生' → 'anh Lâm'\n"
            "hoặc 'ông Lâm', '大嫂' → 'chị dâu' — thay cho 'Kiếm ca', 'Hồng đại tẩu',\n"
            "'Lâm tiên sinh'. Từ vai vế theo đúng cách người nói gọi người đó trong\n"
            "bảng nhân vật (cảnh sát gọi 'cô' thì là 'cô Chân'). Áp cho phân tích ngữ\n"
            "cảnh, mọi giai đoạn dịch, và đổi NGAY bảng nhân vật + bản dịch đang có khi\n"
            "bật (không tốn hạn mức). Phim cổ trang, kiếm hiệp thì để tắt."
        )
        with contextlib.suppress(AttributeError, RuntimeError):
            self._native_address_check.setChecked(self._view_model.get_native_address())
        self._native_address_check.toggled.connect(self._on_native_address_toggled)
        form.addRow(
            self._translator.translate("translate.tr_address_style"),
            self._native_address_check,
        )
        # [v3.23.703] Bật/tắt lượt khớp giọng đọc (tự rút gọn dòng dài hơn khung TTS).
        self._fit_speech_check = QCheckBox(self._translator.translate("translate.fit_for_speech"))
        self._fit_speech_check.setToolTip(self._translator.translate("translate.fit_for_speech_tip"))
        self._fit_speech_check.setChecked(True)
        with contextlib.suppress(AttributeError, RuntimeError):
            self._fit_speech_check.setChecked(self._view_model.get_fit_for_speech())
        self._fit_speech_check.toggled.connect(self._on_fit_for_speech_toggled)
        form.addRow(
            self._translator.translate("translate.tr_fit_for_speech"),
            self._fit_speech_check,
        )
        # [v3.23.149] Số batch dịch chạy SONG SONG mỗi giai đoạn (che độ trễ model).
        self._parallel_combo = QComboBox()
        self._parallel_combo.addItem(self._translator.translate("translate.par_1"), 1)
        self._parallel_combo.addItem(self._translator.translate("translate.par_2"), 2)
        self._parallel_combo.addItem(self._translator.translate("translate.par_3"), 3)
        self._parallel_combo.addItem(self._translator.translate("translate.par_4"), 4)
        self._parallel_combo.setToolTip(
            "Số batch dịch chạy SONG SONG trong mỗi giai đoạn. Mỗi request chờ model\n"
            "10-30 giây; chạy song song che độ trễ đó → nhanh 2-3x. Quota vẫn an toàn:\n"
            "mỗi request đặt chỗ RPM/TPM trước khi gửi nên tổng lưu lượng luôn được\n"
            "điều tiết đúng (tự hạ mức theo RPM của model). Lịch sử dịch neo theo ĐỢT\n"
            "đã hoàn tất để giữ mạch xưng hô/giọng điệu. Batch đính video luôn tuần tự."
        )
        try:
            _cur_par = self._view_model.get_translation_parallel_batches()
            _idx_p = self._parallel_combo.findData(_cur_par)
            if _idx_p >= 0:
                self._parallel_combo.setCurrentIndex(_idx_p)
        except (AttributeError, RuntimeError):
            pass
        self._parallel_combo.currentIndexChanged.connect(self._on_parallel_changed)
        form.addRow(self._translator.translate("translate.tr_parallel"), self._parallel_combo)
        # Nút tải danh sách model từ API
        self._btn_refresh_models = PushButton(self._translator.translate("translate.btn_load_models"))
        self._btn_refresh_models.setToolTip(self._translator.translate("translate.load_models_tip"))
        self._btn_refresh_models.clicked.connect(self._on_refresh_models_clicked)
        form.addRow(self._btn_refresh_models)
        # [v3.23.122] Nút sửa/thêm hạn mức quota của model (khi Google đổi giới hạn).
        self._btn_quota = PushButton(self._translator.translate("translate.btn_quota"))
        self._btn_quota.setToolTip(
            "Xem/sửa giới hạn RPM·TPM·RPD của từng model và thêm model mới.\n"
            "Dùng khi Google thay đổi hạn mức trong tương lai."
        )
        self._btn_quota.clicked.connect(self._on_edit_quota_clicked)
        form.addRow(self._btn_quota)
        card.add_layout(form)
        return card

    def _build_context_group(self) -> QWidget:
        from PySide6.QtWidgets import QTextEdit
        card = SectionCard(
            self._translator.translate("translate.ctx_global_group"),
            collapsible=True, collapsed=True, translator=self._translator,
        )
        form = QFormLayout()
        form.setSpacing(_m.SPACING_SM)

        self._source_lang_edit = QLineEdit()
        self._source_lang_edit.setPlaceholderText(self._translator.translate("translate.ctx_source_lang_ph"))
        form.addRow(self._translator.translate("translate.ctx_source_lang"), self._source_lang_edit)

        # Characters: QTextEdit nhiều dòng vì AI trả "Tên (vai trò)\n..." cho từng nhân vật
        self._characters_edit = QTextEdit()
        self._characters_edit.setPlaceholderText(
            self._translator.translate("translate.ctx_characters_ph")
        )
        self._characters_edit.setMaximumHeight(100)
        self._characters_edit.setAcceptRichText(False)
        form.addRow(self._translator.translate("translate.ctx_characters"), self._characters_edit)

        # Overview: QTextEdit nhiều dòng
        self._overview_edit = QTextEdit()
        self._overview_edit.setPlaceholderText(self._translator.translate("translate.ctx_overview_ph"))
        self._overview_edit.setMaximumHeight(120)
        self._overview_edit.setAcceptRichText(False)
        form.addRow(self._translator.translate("translate.ctx_overview"), self._overview_edit)

        # [v3.23.23] Bảng thuật ngữ & viết tắt (do phân tích sinh) để dịch nhất quán.
        self._glossary_edit = QTextEdit()
        self._glossary_edit.setPlaceholderText(
            "Bảng thuật ngữ & viết tắt (mỗi dòng 'gốc => bản dịch'). "
            "Tự điền khi phân tích; có thể chỉnh tay."
        )
        self._glossary_edit.setMaximumHeight(100)
        self._glossary_edit.setAcceptRichText(False)
        # [v3.23.58] Cập nhật trạng thái nút "Kiểm tra thuật ngữ" khi glossary thay đổi
        # (vd người dùng gõ glossary sau khi đã dịch xong).
        self._glossary_edit.textChanged.connect(self._update_action_states)
        form.addRow(self._translator.translate("translate.ctx_glossary"), self._glossary_edit)

        # [v3.23.35] Ô xem/sửa kết quả Visual Cues (ai nói/nói với ai) — điền sau khi
        # phân tích ngữ cảnh có bật phân tích hình ảnh. Người dùng có thể chỉnh tay.
        self._visual_cues_edit = QTextEdit()
        self._visual_cues_edit.setPlaceholderText(
            self._translator.translate("translate.ctx_visual_ph")
        )
        self._visual_cues_edit.setMaximumHeight(100)
        self._visual_cues_edit.setAcceptRichText(False)
        form.addRow(self._translator.translate("translate.ctx_visual"), self._visual_cues_edit)
        # [v3.23.654] Bảng sửa cues (thay khối JSON) + số dòng nghi (chấm không tốn request).
        cue_row = QHBoxLayout()
        self._btn_cue_table = PushButton(self._translator.translate("translate.btn_cue_table"))
        self._btn_cue_table.setToolTip(self._translator.translate("translate.btn_cue_table_tip"))
        self._btn_cue_table.clicked.connect(lambda _checked=False: self._open_cue_table())
        cue_row.addWidget(self._btn_cue_table)
        # [v3.23.655] Hỏi lại AI chỉ các dòng nghi (cần API key + video).
        self._btn_cue_recheck = PushButton(self._translator.translate("translate.btn_cue_recheck"))
        self._btn_cue_recheck.setToolTip(self._translator.translate("translate.btn_cue_recheck_tip"))
        self._btn_cue_recheck.setEnabled(False)
        self._btn_cue_recheck.clicked.connect(lambda _checked=False: self._on_cue_recheck_clicked())
        cue_row.addWidget(self._btn_cue_recheck)
        self._cue_issue_label = QLabel("")
        cue_row.addWidget(self._cue_issue_label, 1)
        form.addRow("", cue_row)

        # Hàng nút phân tích AI + model chọn + xoá cache
        btn_row = QHBoxLayout()

        # Model selector cho phân tích
        self._analyze_model_combo = QComboBox()
        self._analyze_model_combo.setEditable(True)
        self._analyze_model_combo.addItems(_DEFAULT_MODELS)
        from subtitles_extractor.presentation.pages.translate_widgets import apply_model_tooltips

        apply_model_tooltips(self._analyze_model_combo)     # [v3.23.724] hạn mức miễn phí / chỉ trả phí
        # [v3.23.621] Chấm mù trên phim mẫu (tập 1-3, đính video): phân tích bằng 3.5-flash
        # nhận đúng người nói ~100% dòng (3.1-flash-lite: 89-99%) → lỗi nặng của bản dịch
        # 5 → 2. Hạn mức chỉ 20 request/ngày/key nên hết hạn mức thì TỰ lùi về
        # 3.1-flash-lite (RPD 500) — xem AnalyzeContextWorker.
        self._analyze_model_combo.setCurrentText(DEFAULT_ANALYSIS_MODEL)
        self._analyze_model_combo.setToolTip(
            "Model AI dùng để phân tích toàn bộ phụ đề (nhân vật, xưng hô, người nói)\n"
            "gemini-3.8-flash: khuyến nghị — nhận người nói trong video ~100%, nhanh nhất;\n"
            "hết hạn mức ngày (20/key) thì tự chuyển sang gemini-3.1-flash-lite (500/key)."
        )
        self._analyze_model_combo.setMinimumWidth(200)

        # [v3.23.638] Model dự phòng do người dùng chọn (áp cho mọi lời gọi Gemini khi model quá tải
        # hoặc hết hạn mức ngày): Tự động / Không chuyển model / một model cụ thể.
        self._fallback_model_combo = QComboBox()
        self._fallback_model_combo.setMinimumWidth(200)
        self._fill_fallback_combo(_DEFAULT_MODELS, "")
        self._fallback_model_combo.setToolTip(self._translator.translate("translate.fallback_tip"))
        self._fallback_model_combo.currentIndexChanged.connect(self._on_fallback_model_changed)
        # [v3.23.667] Bật/tắt hẳn việc chuyển sang model dự phòng (tắt = chỉ chờ và thử lại trên
        # model đang chọn — ở mọi khâu: dịch, phân tích, tự kiểm tra).
        self._fallback_enabled_check = QCheckBox(self._translator.translate("translate.fallback_enable"))
        self._fallback_enabled_check.setChecked(True)
        self._fallback_enabled_check.setToolTip(self._translator.translate("translate.fallback_enable_tip"))
        self._fallback_enabled_check.toggled.connect(self._on_fallback_enabled_toggled)

        # [v3.23.640] Model TỰ KIỂM TRA phân tích (bước vá nhân vật/xưng hô/người nói sau phân tích và
        # nút "Tự kiểm tra & tối ưu"). Trước đây cố định gemini-3.1-flash-lite, không có ô chọn nào.
        self._audit_model_combo = QComboBox()
        self._audit_model_combo.setMinimumWidth(200)
        self._fill_audit_combo(_DEFAULT_MODELS, "")
        self._audit_model_combo.setToolTip(self._translator.translate("translate.audit_tip"))
        self._audit_model_combo.currentIndexChanged.connect(self._on_audit_model_changed)

        self._btn_analyze = PushButton(self._translator.translate("translate.btn_analyze"))
        self._btn_analyze.setToolTip(
            "Gemini phân tích TOÀN BỘ phụ đề để tự động phát hiện:\n"
            "• Ngôn ngữ gốc\n"
            "• Danh sách đầy đủ nhân vật + vai trò\n"
            "• Tóm tắt cốt truyện / bối cảnh đầy đủ"
        )
        self._btn_analyze.clicked.connect(self._on_analyze_context_clicked)
        # [v3.23.601] Phân tích khôi phục từ phiên cũ chưa từng được hỏi lại người nói.
        self._btn_complete_speakers = PushButton(
            self._translator.translate("translate.btn_complete_speakers")
        )
        self._btn_complete_speakers.setToolTip(
            self._translator.translate("translate.tip_complete_speakers")
        )
        self._btn_complete_speakers.setEnabled(False)
        self._btn_complete_speakers.clicked.connect(self._on_complete_speakers_clicked)
        self._visual_cues_edit.textChanged.connect(self._update_speaker_button)
        # [v3.23.652] Sửa tay bảng nhân vật / tóm tắt / thuật ngữ / gợi ý hình ảnh từng KHÔNG được lưu vào phiên dịch
        # (tắt app là mất). Tự lưu sau 1,5 s ngừng gõ.
        self._context_save_timer = QTimer(self)
        self._context_save_timer.setSingleShot(True)
        self._context_save_timer.setInterval(1500)
        self._context_save_timer.timeout.connect(self._save_context_edits)
        self._context_edit_path = ""
        for _edit in (self._characters_edit, self._overview_edit, self._glossary_edit, self._visual_cues_edit):
            _edit.textChanged.connect(self._on_context_text_changed)
        # [v3.23.624] Tự đối chiếu phân tích hiện có với phụ đề rồi vá khiếm khuyết.
        self._btn_refine_analysis = PushButton(
            self._translator.translate("translate.btn_refine_analysis")
        )
        self._btn_refine_analysis.setToolTip(
            self._translator.translate("translate.tip_refine_analysis")
        )
        self._btn_refine_analysis.setEnabled(False)
        self._btn_refine_analysis.clicked.connect(self._on_refine_analysis_clicked)
        self._btn_clear_cache = PushButton(self._translator.translate("translate.btn_clear_cache"))
        self._btn_clear_cache.setToolTip(self._translator.translate("translate.tip_clear_cache"))
        self._btn_clear_cache.clicked.connect(self._on_clear_cache_clicked)
        self._btn_clear_cloud = PushButton(self._translator.translate("translate.btn_clear_cloud"))
        self._btn_clear_cloud.setToolTip(
            "Xoá các đoạn video đã tải lên Gemini cloud của video này để giải phóng "
            "dung lượng. Lần phân tích/dịch sau sẽ tải lên lại nếu cần."
        )
        self._btn_clear_cloud.clicked.connect(self._on_clear_cloud_clicked)
        # [v3.23.57] Quản lý bộ nhớ dịch phim bộ (xem/xoá theo từng bộ).
        self._btn_manage_tm = PushButton(self._translator.translate("translate.btn_series_memory"))
        self._btn_manage_tm.setToolTip(
            "Xem và xoá bộ nhớ dịch (câu đã dịch + thuật ngữ) tích luỹ theo từng phim bộ"
        )
        self._btn_manage_tm.clicked.connect(self._on_manage_tm_clicked)
        # [v3.23.484] Xuất dữ liệu chuẩn đoán ngữ cảnh toàn cục (JSON) để soi chất lượng phân
        # tích: roster có bịa tên không, glossary trùng, cues lệch dòng, tóm tắt đủ dày chưa.
        self._btn_export_ctx_diag = PushButton("🩺 Xuất chuẩn đoán ngữ cảnh")
        self._btn_export_ctx_diag.setToolTip(
            "Xuất file JSON chuẩn đoán chất lượng NGỮ CẢNH TOÀN CỤC:\n"
            "• Bảng nhân vật: có tên nào KHÔNG xuất hiện trong phụ đề (nghi AI bịa)?\n"
            "• Glossary: thuật ngữ trùng lặp gây dịch không nhất quán?\n"
            "• Visual cues: có trỏ sai chỉ số dòng không?\n"
            "• Tóm tắt: đủ dày so với thời lượng phim chưa?\n\n"
            "File KHÔNG chứa toàn văn phụ đề (chỉ số liệu + trích đoạn ngắn) — tiện gửi đi "
            "phân tích để cải thiện chất lượng dịch."
        )
        self._btn_export_ctx_diag.clicked.connect(self._on_export_context_diagnostics)
        # [v3.23.601] Hai hàng: một hàng 7 nút vượt bề rộng cột trái (1500-1920 px) nên cả
        # thẻ phải cuộn ngang và nút cuối bị che. Hàng 1 = phân tích, hàng 2 = công cụ.
        btn_row.addWidget(self._analyze_model_combo)
        btn_row.addWidget(self._btn_analyze)
        btn_row.addWidget(self._btn_complete_speakers)
        btn_row.addStretch(1)
        form.addRow(self._wrap_layout(btn_row))
        fallback_row = QHBoxLayout()
        fallback_row.addWidget(self._fallback_enabled_check)
        fallback_row.addWidget(self._fallback_model_combo, 1)
        form.addRow(
            self._translator.translate("translate.fallback_label"), self._wrap_layout(fallback_row),
        )
        form.addRow(self._translator.translate("translate.audit_label"), self._audit_model_combo)
        refine_row = QHBoxLayout()
        refine_row.addWidget(self._btn_refine_analysis)
        refine_row.addStretch(1)
        form.addRow(self._wrap_layout(refine_row))
        tools_row = QHBoxLayout()
        tools_row.addWidget(self._btn_export_ctx_diag)
        tools_row.addWidget(self._btn_clear_cache)
        tools_row.addWidget(self._btn_clear_cloud)
        tools_row.addWidget(self._btn_manage_tm)
        tools_row.addStretch(1)
        form.addRow(self._wrap_layout(tools_row))
        card.add_layout(form)
        return card

    def _build_video_context_group(self) -> QWidget:
        """Nhóm tuỳ chọn đính video làm ngữ cảnh dịch (tự cắt nếu phim dài)."""
        card = SectionCard(
            self._translator.translate("translate.vid_group"),
            collapsible=True, collapsed=True, translator=self._translator,
        )
        v = QVBoxLayout()
        v.setSpacing(_m.SPACING_SM)

        self._attach_video_check = QCheckBox(
            self._translator.translate("translate.vid_attach")
        )
        self._attach_video_check.setToolTip(
            "Khi bật: phân tích ngữ cảnh sẽ LUÔN dùng video (gộp mọi đoạn).\n"
            "Riêng các giai đoạn dịch bên dưới là tuỳ chọn (đính video tốn thêm token).\n"
            "Đo thật 3 tập, chấm mù: dịch KHÔNG đính video nhầm người nói nhiều hơn (lỗi nặng 13 so với 7-8); "
            "đính video chỉ ở 'Dịch thô' cho kết quả ngang đính cả 3 giai đoạn (7 so với 8) mà gửi ít video hơn."
        )
        self._attach_video_check.toggled.connect(self._on_attach_video_toggled)
        v.addWidget(self._attach_video_check)

        pick_row = QHBoxLayout()
        self._btn_pick_video = PushButton(self._translator.translate("translate.btn_choose_video"))
        self._btn_pick_video.clicked.connect(self._on_pick_video_clicked)
        self._btn_pick_video.setEnabled(False)
        self._video_path_label = QLabel(self._translator.translate("translate.no_video"))
        self._video_path_label.setWordWrap(True)
        pick_row.addWidget(self._btn_pick_video)
        pick_row.addWidget(self._video_path_label, 1)
        v.addLayout(pick_row)

        # Chọn giai đoạn dịch áp dụng video (phân tích thì tự động khi đã bật).
        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel(self._translator.translate("translate.vid_for_stage")))
        self._video_stage_literal = QCheckBox(self._translator.translate("translate.stage_raw"))
        self._video_stage_style = QCheckBox(self._translator.translate("translate.stage_style"))
        self._video_stage_localize = QCheckBox(self._translator.translate("translate.stage_localize"))
        for cb in (self._video_stage_literal, self._video_stage_style, self._video_stage_localize):
            cb.setEnabled(False)
            stage_row.addWidget(cb)
        stage_row.addStretch(1)
        v.addLayout(stage_row)

        # [v3.23.648] Chỉ gửi khoảng video của từng lô (± lề) thay vì cả đoạn ~15 phút.
        self._clip_video_check = QCheckBox(self._translator.translate("translate.vid_clip"))
        self._clip_video_check.setToolTip(self._translator.translate("translate.vid_clip_tip"))
        with contextlib.suppress(AttributeError, RuntimeError):
            self._clip_video_check.setChecked(self._view_model.get_clip_video_to_batch())
        self._clip_video_check.toggled.connect(self._on_clip_video_toggled)
        self._clip_video_check.setEnabled(False)       # chỉ có nghĩa khi đã bật đính video
        v.addWidget(self._clip_video_check)

        # [v3.23.31] Visual Cues (Vision Director): quét video xác định ai nói/nói với
        # ai từng dòng để dịch xưng hô đúng vai vế. Tốn token nên là tuỳ chọn riêng.
        self._visual_cues_check = QCheckBox(
            self._translator.translate("translate.opt_analyze_visual")
        )
        self._visual_cues_check.setToolTip(
            "Khi bật: lúc bấm 'Phân tích ngữ cảnh', AI quét video MỘT LẦN (cùng model "
            "với phân tích) để xác định người nói và người nghe của TỪNG dòng, giúp dịch "
            "chọn đại từ xưng hô đúng vai vế. Kết quả hiện ở ô 'Gợi ý hình ảnh' để bạn "
            "xem/chỉnh trước khi dịch.\n"
            "Lưu ý: tốn thêm token (mỗi ~150 dòng một lần gọi) và cần có video."
        )
        self._visual_cues_check.setEnabled(False)
        v.addWidget(self._visual_cues_check)

        # [v3.23.448] Sinh tag cảm xúc [cười]/[thở dài]/[hắng giọng] từ NGỮ CẢNH VIDEO (hình +
        # âm thanh) → giọng đọc TTS biểu cảm. Tag tự lọc khỏi phụ đề (chỉ vào TTS). Cần video.
        self._emotion_cues_check = QCheckBox("Sinh tag cảm xúc cho TTS (cười/thở dài, cần video)")
        self._emotion_cues_check.setChecked(False)
        self._emotion_cues_check.setToolTip(
            "Thử nghiệm (cần đính video + dùng VieNeu v3 Turbo cho TTS): khi HÌNH ẢNH/ÂM "
            "THANH trong video cho thấy rõ nhân vật cười/thở dài/hắng giọng, AI chèn tag "
            "[cười]/[thở dài]/[hắng giọng] để giọng đọc phát biểu cảm THẬT.\n"
            "Tag KHÔNG hiển thị trong phụ đề (phần mềm tự lọc) — chỉ vào audio TTS.\n"
            "Chỉ dùng khi thấy/nghe rõ trong video, không suy đoán từ lời thoại."
        )
        self._emotion_cues_check.setEnabled(False)
        v.addWidget(self._emotion_cues_check)
        # Khôi phục + lưu vào settings_service (container đọc lúc tạo translator).
        try:
            _tr_cfg = self._container.settings_service.current.translation
            self._emotion_cues_check.setChecked(
                bool(getattr(_tr_cfg, "translation_emotion_cues", False))
            )
        except AttributeError:
            pass
        self._emotion_cues_check.toggled.connect(self._on_emotion_cues_toggled)

        self._video_path: str = ""
        card.add_layout(v)
        return card

    def _build_stages_group(self) -> QWidget:
        card = SectionCard(self._translator.translate("translate.tr_sec_stages"))
        grid = QGridLayout()
        grid.setSpacing(_m.SPACING_SM)

        # [v3.23.40] Lấy giá trị mặc định lô/ngữ cảnh từ Cài đặt (tab "Video ngữ cảnh &
        # Dịch") để người dùng chỉnh một nơi áp cho mọi giai đoạn. LITERAL dùng đúng giá
        # trị Cài đặt; STYLE/LOCALIZE cần ngữ cảnh nhiều hơn nên lô nhỏ hơn một chút.
        try:
            tr = self._container.settings_service.current.translation
            base_batch = tr.default_batch_size
            base_ctx = tr.default_context_size
        except AttributeError:
            base_batch, base_ctx = 50, 10
        refine_batch = max(10, int(base_batch * 0.8))
        refine_ctx = base_ctx + 2

        self._stage_preprocess = _StagePanel(
            self._translator.translate("translate.stage1_title"),
            TranslationStageKind.PREPROCESS,
            translator=self._translator,
            default_enabled=False,
            default_model="gemini-3.1-flash-lite",   # RPD 500
            default_temp=0.0,
            default_batch=min(80, base_batch + 30),
            default_ctx=max(0, base_ctx - 4),
            show_retime=True,
        )
        self._stage_literal = _StagePanel(
            self._translator.translate("translate.stage2_title"),
            TranslationStageKind.LITERAL,
            translator=self._translator,
            default_enabled=True,
            default_model="gemini-3.1-flash-lite",   # RPD 500 — khuyến nghị mặc định
            default_temp=0.1,
            default_batch=base_batch,
            default_ctx=base_ctx,
            # [v3.23.620] Đo trên phim mẫu (tập 1+2, 3.1-flash-lite): thinking "low" sửa phần
            # lớn lỗi đảo vai/xưng hô (bẫy 0-7/14 → 11/14), đổi lại ~+4k token suy nghĩ/tập.
            default_thinking=True,
        )
        self._stage_style = _StagePanel(
            self._translator.translate("translate.stage3_title"),
            TranslationStageKind.STYLE,
            translator=self._translator,
            default_enabled=False,
            default_model=DEFAULT_REFINE_MODEL,      # chất lượng cao hơn cho STYLE
            default_temp=0.35,
            default_batch=refine_batch,
            default_ctx=refine_ctx,
            show_style=True,
        )
        self._stage_localize = _StagePanel(
            self._translator.translate("translate.stage4_title"),
            TranslationStageKind.LOCALIZE,
            translator=self._translator,
            default_enabled=False,
            default_model=DEFAULT_REFINE_MODEL,      # chất lượng cao hơn cho LOCALIZE
            default_temp=0.25,
            default_batch=refine_batch,
            default_ctx=refine_ctx,
            show_locale=True,
        )

        grid.addWidget(self._stage_preprocess, 0, 0)
        grid.addWidget(self._stage_literal, 1, 0)
        grid.addWidget(self._stage_style, 2, 0)
        grid.addWidget(self._stage_localize, 3, 0)
        grid.setColumnStretch(0, 1)
        card.add_layout(grid)
        return card

    def _build_actions_group(self) -> QWidget:
        card = SectionCard(self._translator.translate("translate.tr_sec_exec"))
        layout = QVBoxLayout()
        layout.setSpacing(_m.SPACING_SM)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._stage_label = QLabel(self._translator.translate("translate.status_ready_short"))
        layout.addWidget(self._stage_label)

        button_row = QHBoxLayout()
        # [v3.7 U1] Dùng PrimaryPushButton (màu nhấn theo theme Fluent, tự thích
        # nghi sáng/tối) thay cho QPushButton + stylesheet xanh hardcode.
        # [v3.23.332] Hàng đợi dịch cả bộ.
        self._batch_items: list = []
        self._batch_index: int = 0
        self._batch_failures: list[str] = []
        self._batch_done: int = 0
        self._batch_cancelled: bool = False
        self._batch_quota_stopped: bool = False
        self._batch_pending_run: bool = False
        self._btn_run = PrimaryPushButton(self._translator.translate("translate.btn_translate"))
        self._btn_run.clicked.connect(self._on_run_clicked)
        # [v3.23.332] Dịch cả bộ. KHÁC ba khâu kia: dịch gọi dịch vụ có HẠN MỨC NGÀY,
        # nên phải ước lượng request trước và dừng êm khi hết hạn mức.
        self._btn_batch = PushButton(self._translator.translate("translate.btn_translate_batch2"))
        self._btn_batch.setToolTip(
            "Quét thư mục phim bộ, dịch lần lượt các tập đã trích xuất.\n"
            "Hết hạn mức ngày thì dừng êm — chạy lại hôm sau sẽ tự bỏ qua tập đã dịch."
        )
        self._btn_batch.clicked.connect(self._on_batch_translate_clicked)
        self._btn_cancel = PushButton(self._translator.translate("translate.btn_cancel2"))
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        self._btn_cancel.setEnabled(False)
        button_row.addWidget(self._btn_run, 2)
        button_row.addWidget(self._btn_batch, 1)
        button_row.addWidget(self._btn_cancel, 1)
        layout.addLayout(button_row)
        card.add_layout(layout)
        return card

    def _build_result_group(self) -> QWidget:
        from PySide6.QtWidgets import QSplitter, QTextBrowser
        card = SectionCard(self._translator.translate("translate.result_group"))
        layout = QVBoxLayout()
        layout.setSpacing(_m.SPACING_SM)

        # ── Bảng kết quả ─────────────────────────────────────────────────
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["#", self._translator.translate("translate.col_time"), self._translator.translate("translate.col_original"), self._translator.translate("translate.col_translated")])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setWordWrap(True)
        self._table.setAlternatingRowColors(True)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)
        self._table.verticalHeader().setDefaultSectionSize(56)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        # ── Panel chi tiết (bên dưới bảng) ───────────────────────────────
        detail_frame = QFrame()
        detail_frame.setFrameShape(QFrame.Shape.StyledPanel)
        detail_layout = QHBoxLayout(detail_frame)
        detail_layout.setContentsMargins(_m.SPACING_XS, _m.SPACING_XS, _m.SPACING_XS, _m.SPACING_XS)
        detail_layout.setSpacing(_m.SPACING_XS)

        src_box = SectionCard(self._translator.translate("translate.result_orig_full"))
        self._detail_source = QTextBrowser()
        self._detail_source.setReadOnly(True)
        self._detail_source.setPlaceholderText(self._translator.translate("translate.result_orig_ph"))
        src_box.add_widget(self._detail_source)

        tgt_box = SectionCard(self._translator.translate("translate.result_trans_full"))
        self._detail_translation = QTextBrowser()
        self._detail_translation.setReadOnly(True)
        self._detail_translation.setPlaceholderText(self._translator.translate("translate.result_trans_ph"))
        tgt_box.add_widget(self._detail_translation)

        detail_layout.addWidget(src_box, 1)
        detail_layout.addWidget(tgt_box, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._table)
        splitter.addWidget(detail_frame)
        # Tỉ lệ ban đầu: bảng ~70%, detail panel ~30% — tránh bảng bị dẹt
        # setStretchFactor chỉ ảnh hưởng khi resize; setSizes đặt kích thước ban đầu
        splitter.setSizes([420, 160])
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self._splitter = splitter   # giữ ref để _on_result_ready có thể reset
        layout.addWidget(splitter, 1)

        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        # Auto re-size rows khi cột thay đổi chiều rộng (word-wrap cần biết col width)
        self._table.horizontalHeader().sectionResized.connect(self._on_column_resized)

        # ── Hàng nút action ───────────────────────────────────────────────
        # [v3.23.674] Hai nhóm nút tự xuống dòng: (1) làm việc với bản dịch, (2) xuất tệp. Trước đây hàng cố định
        # bị cắt chữ ở màn hình 1366×768 ("Quản lý giai đoạ", "Xuất chẩn đoán d").
        from subtitles_extractor.presentation.widgets.flow_layout import flow_row

        action_holder, action_row = flow_row()
        self._btn_copy_selection = PushButton(self._translator.translate("translate.btn_copy"))
        self._btn_copy_selection.setToolTip(self._translator.translate("translate.tip_copy"))
        self._btn_copy_selection.clicked.connect(self._on_copy_selection_clicked)
        self._btn_copy_selection.setEnabled(False)
        self._btn_send_to_editor = PushButton(self._translator.translate("translate.btn_send_editor"))
        self._btn_send_to_editor.setToolTip(
            "Gửi toàn bộ bản dịch sang trang Chỉnh sửa để tiếp tục chỉnh sửa thủ công"
        )
        self._btn_send_to_editor.clicked.connect(self._on_send_to_editor_clicked)
        self._btn_send_to_editor.setEnabled(False)
        # [v3.23.47] So sánh các giai đoạn dịch (chỉ bật khi có ≥2 giai đoạn).
        self._btn_compare_stages = PushButton(self._translator.translate("translate.btn_compare"))
        self._btn_compare_stages.setToolTip(
            "Xem song song bản dịch qua từng giai đoạn (thô / tinh chỉnh / bản địa hoá)"
        )
        self._btn_compare_stages.clicked.connect(self._on_compare_stages_clicked)
        self._btn_compare_stages.setEnabled(False)
        # [v3.23.608] Xem / sửa / xoá / dùng lại kết quả TỪNG giai đoạn đã lưu.
        self._btn_stage_cache = PushButton(
            self._translator.translate("translate.btn_stage_cache")
        )
        self._btn_stage_cache.setToolTip(
            self._translator.translate("translate.tip_stage_cache")
        )
        self._btn_stage_cache.clicked.connect(self._on_stage_cache_clicked)
        self._btn_stage_cache.setEnabled(False)
        # [v3.23.54] Kiểm tra nhất quán thuật ngữ giữa glossary và bản dịch.
        self._btn_check_glossary = PushButton(self._translator.translate("translate.btn_check_glossary"))
        self._btn_check_glossary.setToolTip(
            "Đối chiếu bảng thuật ngữ với bản dịch để tìm dòng dịch sai/thiếu thuật ngữ"
        )
        self._btn_check_glossary.clicked.connect(self._on_check_glossary_clicked)
        self._btn_check_glossary.setEnabled(False)
        action_row.addWidget(self._btn_copy_selection)
        action_row.addWidget(self._btn_send_to_editor)
        action_row.addWidget(self._btn_compare_stages)
        action_row.addWidget(self._btn_check_glossary)
        action_row.addWidget(self._btn_stage_cache)
        export_holder, export_row = flow_row()
        self._btn_export_srt = PushButton(self._translator.translate("translate.btn_export_srt"))
        self._btn_export_srt.clicked.connect(lambda: self._on_export_clicked(SubtitleFormat.SRT))
        self._btn_export_ass = PushButton(self._translator.translate("translate.btn_export_ass"))
        self._btn_export_ass.clicked.connect(lambda: self._on_export_clicked(SubtitleFormat.ASS))
        export_row.addWidget(self._btn_export_srt)
        export_row.addWidget(self._btn_export_ass)
        self._btn_export_bilingual = PushButton(self._translator.translate("translate.btn_export_bi"))
        self._btn_export_bilingual.setToolTip(
            "Xuất phụ đề SRT có CẢ nguyên văn (gốc) lẫn bản dịch tiếng Việt trên mỗi câu "
            "(gốc ở trên, dịch ở dưới) — tiện cho người xem/người học."
        )
        self._btn_export_bilingual.clicked.connect(self._on_export_bilingual_clicked)
        export_row.addWidget(self._btn_export_bilingual)
        self._btn_export_diag = PushButton(self._translator.translate("translate.btn_export_diag"))
        self._btn_export_diag.setToolTip(
            "Xuất toàn bộ chi tiết quá trình dịch (phụ đề gốc, kết quả, từng giai đoạn, "
            "phân tích ngữ cảnh: nhân vật/tóm tắt/glossary/visual cues) ra một file JSON "
            "để phân tích, khắc phục lỗi và cải thiện chất lượng dịch."
        )
        self._btn_export_diag.clicked.connect(self._on_export_diagnostics_clicked)
        export_row.addWidget(self._btn_export_diag)
        layout.addWidget(action_holder)
        layout.addWidget(export_holder)
        card.add_layout(layout)
        return card

    @staticmethod
    def _wrap_layout(inner: QHBoxLayout) -> QWidget:
        wrapper = QWidget()
        wrapper.setLayout(inner)
        return wrapper


__all__ = ["TranslateUiMixin"]
