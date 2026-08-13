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

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from subtitles_extractor.presentation.fluent_compat import PrimaryPushButton, PushButton, ToolButton

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.presentation.utils.time_format import seconds_to_display
from subtitles_extractor.presentation.view_models.translate_page_view_model import (
    TranslatePageViewModel,
)
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets
from subtitles_extractor.presentation.theme import feedback as _feedback
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme.styles import caption_style
from subtitles_extractor.presentation.widgets.section_card import SectionCard
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name

logger = logging.getLogger(__name__)

# Danh sách ngôn ngữ đích phổ biến (tên đầy đủ — dùng trực tiếp trong prompt).
_TARGET_LANGUAGES = [
    "Vietnamese", "English", "Chinese (Simplified)", "Chinese (Traditional)",
    "Japanese", "Korean", "Thai", "French", "German", "Spanish",
    "Portuguese", "Russian", "Indonesian", "Italian", "Arabic",
]

# Map tên ngôn ngữ hiển thị → mã ISO ngắn để đặt tên file (<tên>.<mã>.srt).
_LANGUAGE_CODES = {
    "Vietnamese": "vi", "English": "en", "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh", "Japanese": "ja", "Korean": "ko",
    "Thai": "th", "French": "fr", "German": "de", "Spanish": "es",
    "Portuguese": "pt", "Russian": "ru", "Indonesian": "id",
    "Italian": "it", "Arabic": "ar",
}

# Danh sách model CHỈ gồm các model đang còn quota MIỄN PHÍ (theo bảng rate-limit thực
# tế của tài khoản, tháng 6/2026). Các model limit 0/0 (Gemini 2 Flash, 2 Flash Lite,
# 2.5 Pro, 3.1 Pro) KHÔNG còn free nên KHÔNG đưa vào.
# Xếp theo RPD giảm dần (RPD cao = dịch được nhiều/ngày, ưu tiên cho bản free).
#   gemini-3.1-flash-lite : RPM 15 · TPM 250K · RPD 500  — KHUYẾN NGHỊ (RPD cao nhất)
#   gemini-3.5-flash       : RPM  5 · TPM 250K · RPD 20  — chất lượng cao nhất
#   gemini-3-flash          : RPM  5 · TPM 250K · RPD 20  — thế hệ 3, cân bằng
#   gemini-2.5-flash        : RPM  5 · TPM 250K · RPD 20  — ổn định, cân bằng
#   gemini-2.5-flash-lite   : RPM 10 · TPM 250K · RPD 20  — tiết kiệm nhất
_DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",     # GA mới nhất — nhanh/rẻ, tối ưu cho lô LỚN (high-volume)
    "gemini-3.1-flash-lite",     # GA — RPD cao, dự phòng cho bản free lô lớn
    "gemini-3.6-flash",          # GA mới nhất — token-efficient, rẻ hơn 3.5-flash, chất lượng cao
    "gemini-3.5-flash",          # GA — chất lượng cao (agentic/coding)
    "gemini-2.5-flash",          # đời cũ — dự phòng
    "gemini-2.5-flash-lite",     # đời cũ — dự phòng tiết kiệm
]

_STYLE_PRESETS = [
    "Trung tính", "Kiếm hiệp", "Tiên hiệp", "Huyền huyễn", "Khoa học viễn tưởng",
    "Tài liệu", "Thiếu nhi", "Cổ trang", "Hiện đại", "Hài", "Trinh thám",
]

_SETTINGS_ORG = "SubtitlesExtractor"
_SETTINGS_APP = "TranslatePage"

# [v3.7 U5] Trên ngưỡng này, không resize từng dòng (tránh đứng UI), dùng chiều
# cao mặc định ~2 dòng cho mỗi hàng.
# [v3.23.131] Ngưỡng số dòng mà còn gọi resizeRowsToContents(): trên ngưỡng này việc đo
# chiều cao từng dòng word-wrap rất chậm (đứng UI vài giây với phim ~1000+ dòng) → dùng
# chiều cao mặc định cho mượt. Hạ từ 1500 -> 300 vì phim tài liệu/phim bộ thường > 800 dòng.
_RESIZE_ROWS_THRESHOLD = 300
_DEFAULT_ROW_HEIGHT_PX = 44


class _StagePanel(SectionCard):
    """Khối cấu hình một giai đoạn dịch (bật/tắt + model + tham số)."""

    def __init__(
        self,
        title: str,
        kind: TranslationStageKind,
        *,
        translator,
        default_enabled: bool,
        default_model: str,
        default_temp: float,
        default_batch: int,
        default_ctx: int,
        show_style: bool = False,
        show_locale: bool = False,
        show_retime: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self._translator = translator  # [v3.23.371] để dịch chuỗi trong panel giai đoạn
        self._kind = kind
        self._show_style = show_style
        self._show_locale = show_locale
        self._show_retime = show_retime

        self.enable_check = QCheckBox(self._translator.translate("translate.stage_enable"))
        self.enable_check.setChecked(default_enabled)
        self.add_widget(self.enable_check)

        self._content = QWidget()
        form = QFormLayout(self._content)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(_m.SPACING_XS)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(_DEFAULT_MODELS)
        self.model_combo.setCurrentText(default_model)
        form.addRow("Model AI:", self.model_combo)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(default_temp)
        form.addRow(self._translator.translate("translate.tr_temp"), self.temp_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 500)
        self.batch_spin.setValue(default_batch)
        form.addRow(self._translator.translate("translate.tr_batch"), self.batch_spin)

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(0, 100)
        self.ctx_spin.setValue(default_ctx)
        form.addRow(self._translator.translate("translate.tr_ctx"), self.ctx_spin)

        self.style_combo: QComboBox | None = None
        if show_style:
            self.style_combo = QComboBox()
            self.style_combo.setEditable(True)
            self.style_combo.addItems(_STYLE_PRESETS)
            self.style_combo.setCurrentText("Trung tính")
            form.addRow(self._translator.translate("translate.tr_style"), self.style_combo)

        self.locale_edit: QLineEdit | None = None
        if show_locale:
            self.locale_edit = QLineEdit()
            self.locale_edit.setPlaceholderText(self._translator.translate("translate.stage_note_ph"))
            form.addRow(self._translator.translate("translate.tr_locale_note"), self.locale_edit)

        self.retime_check: QCheckBox | None = None
        if show_retime:
            self.retime_check = QCheckBox(self._translator.translate("translate.stage_allow_calib"))
            form.addRow(self.retime_check)

        # ── Thinking (Gemini 3.x / 2.5.x) ────────────────────────────────
        self.thinking_check = QCheckBox(self._translator.translate("translate.stage_thinking"))
        self.thinking_check.setToolTip(
            "Model suy nghĩ trước khi trả lời → cải thiện chất lượng dịch\n"
            "Đặc biệt hữu ích cho STYLE và LOCALIZE (phức tạp về ngữ nghĩa)\n"
            "Hỗ trợ: gemini-3.6-flash, gemini-3.5-flash-lite, gemini-3.5-flash, gemini-2.5-flash"
        )
        form.addRow(self.thinking_check)

        self.thinking_budget_spin = QSpinBox()
        self.thinking_budget_spin.setRange(-1, 32768)
        self.thinking_budget_spin.setValue(-1)
        self.thinking_budget_spin.setSpecialValueText("-1  (Dynamic — model tự quyết)")
        self.thinking_budget_spin.setToolTip(
            "-1: Dynamic (model tự chọn, khuyến nghị)\n"
            "0: Tắt thinking\n"
            "1–32768: Số token suy nghĩ tối đa"
        )
        self.thinking_budget_spin.setEnabled(False)
        form.addRow("Thinking Budget (token):", self.thinking_budget_spin)
        self.thinking_check.toggled.connect(self.thinking_budget_spin.setEnabled)

        # [v3.23.125] Mức Thinking cho Gemini 3.x (thinking_level). Trước đây luôn "low"
        # → giờ cho chọn để tăng chất lượng dịch khi cần (cao hơn = sát ngữ cảnh hơn).
        self.thinking_level_combo = QComboBox()
        self.thinking_level_combo.addItem(self._translator.translate("translate.think_low"), "low")
        self.thinking_level_combo.addItem(self._translator.translate("translate.think_medium"), "medium")
        self.thinking_level_combo.addItem(self._translator.translate("translate.think_high"), "high")
        self.thinking_level_combo.setToolTip(
            "Mức suy nghĩ cho Gemini 3.x (3.6-flash, 3.5-flash-lite, 3.5-flash, 3.1-flash-lite…).\n"
            "Cao hơn = dịch sát ngữ cảnh/xưng hô tốt hơn nhưng chậm & tốn token hơn.\n"
            "Chỉ áp dụng Gemini 3.x; Gemini 2.5 dùng Thinking Budget ở trên."
        )
        self.thinking_level_combo.setEnabled(False)
        form.addRow(self._translator.translate("translate.tr_thinking"), self.thinking_level_combo)
        self.thinking_check.toggled.connect(self.thinking_level_combo.setEnabled)

        self.add_widget(self._content)
        self._content.setEnabled(self.enable_check.isChecked())
        self.enable_check.toggled.connect(self._content.setEnabled)

    def set_model_items(self, models: list[str]) -> None:
        """Cập nhật danh sách model trong combo (giữ lựa chọn hiện tại nếu còn trong list)."""
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        elif models:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)

    def is_enabled(self) -> bool:
        return self.enable_check.isChecked()

    def build_config(self) -> TranslationStageConfig:
        """Tạo ``TranslationStageConfig`` bất biến từ trạng thái widget hiện tại."""
        return TranslationStageConfig(
            kind=self._kind,
            model_name=self.model_combo.currentText().strip() or _DEFAULT_MODELS[0],
            temperature=self.temp_spin.value(),
            batch_size=self.batch_spin.value(),
            context_size=self.ctx_spin.value(),
            style_name=(self.style_combo.currentText().strip() if self.style_combo else "Trung tính"),
            locale_notes=(self.locale_edit.text().strip() if self.locale_edit else ""),
            allow_retime=(self.retime_check.isChecked() if self.retime_check else False),
            enable_thinking=self.thinking_check.isChecked(),
            thinking_budget=self.thinking_budget_spin.value(),
            thinking_level=self.thinking_level_combo.currentData() or "low",
        )

    def save_state(self, settings: QSettings, prefix: str) -> None:
        """Lưu toàn bộ trạng thái widget của giai đoạn vào QSettings."""
        settings.setValue(f"{prefix}/enabled", self.enable_check.isChecked())
        settings.setValue(f"{prefix}/model", self.model_combo.currentText())
        settings.setValue(f"{prefix}/temp", self.temp_spin.value())
        settings.setValue(f"{prefix}/batch", self.batch_spin.value())
        settings.setValue(f"{prefix}/ctx", self.ctx_spin.value())
        if self.style_combo is not None:
            settings.setValue(f"{prefix}/style", self.style_combo.currentText())
        if self.locale_edit is not None:
            settings.setValue(f"{prefix}/locale", self.locale_edit.text())
        if self.retime_check is not None:
            settings.setValue(f"{prefix}/retime", self.retime_check.isChecked())
        settings.setValue(f"{prefix}/thinking", self.thinking_check.isChecked())
        settings.setValue(f"{prefix}/thinking_budget", self.thinking_budget_spin.value())
        settings.setValue(
            f"{prefix}/thinking_level", self.thinking_level_combo.currentData()
        )

    def restore_state(self, settings: QSettings, prefix: str) -> None:
        """Khôi phục trạng thái widget từ QSettings (giữ default nếu chưa có)."""
        if settings.contains(f"{prefix}/enabled"):
            self.enable_check.setChecked(settings.value(f"{prefix}/enabled", type=bool))
        model = settings.value(f"{prefix}/model", "", type=str)
        if model:
            self.model_combo.setCurrentText(model)
        if settings.contains(f"{prefix}/temp"):
            self.temp_spin.setValue(settings.value(f"{prefix}/temp", type=float))
        if settings.contains(f"{prefix}/batch"):
            self.batch_spin.setValue(settings.value(f"{prefix}/batch", type=int))
        if settings.contains(f"{prefix}/ctx"):
            self.ctx_spin.setValue(settings.value(f"{prefix}/ctx", type=int))
        if self.style_combo is not None and settings.contains(f"{prefix}/style"):
            self.style_combo.setCurrentText(settings.value(f"{prefix}/style", type=str))
        if self.locale_edit is not None and settings.contains(f"{prefix}/locale"):
            self.locale_edit.setText(settings.value(f"{prefix}/locale", type=str))
        if self.retime_check is not None and settings.contains(f"{prefix}/retime"):
            self.retime_check.setChecked(settings.value(f"{prefix}/retime", type=bool))
        if settings.contains(f"{prefix}/thinking"):
            self.thinking_check.setChecked(settings.value(f"{prefix}/thinking", type=bool))
        if settings.contains(f"{prefix}/thinking_budget"):
            self.thinking_budget_spin.setValue(settings.value(f"{prefix}/thinking_budget", type=int))
        if settings.contains(f"{prefix}/thinking_level"):
            saved_level = settings.value(f"{prefix}/thinking_level", "low", type=str)
            idx = self.thinking_level_combo.findData(saved_level)
            if idx >= 0:
                self.thinking_level_combo.setCurrentIndex(idx)


def parse_quota_rows(
    rows: list[tuple[str, str, str, str]],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """[v3.23.122] Chuyển dữ liệu bảng quota (model, rpm, tpm, rpd) thành dict hợp lệ.

    HÀM THUẦN (dễ test). Bỏ qua dòng thiếu tên model; gom lỗi (số không hợp lệ / âm)
    vào danh sách trả về để UI hiển thị. Tên model chuẩn hoá về chữ thường.

    Returns:
        (limits_dict, errors) — limits_dict: {model: {rpm,tpm,rpd}}; errors: mô tả lỗi.
    """
    limits: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    for model_raw, rpm_raw, tpm_raw, rpd_raw in rows:
        model = (model_raw or "").strip().lower()
        if not model:
            continue
        try:
            rpm, tpm, rpd = int(rpm_raw), int(tpm_raw), int(rpd_raw)
        except (ValueError, TypeError):
            errors.append(f"'{model_raw}': RPM/TPM/RPD phải là số nguyên.")
            continue
        if rpm <= 0 or tpm <= 0 or rpd <= 0:
            errors.append(f"'{model_raw}': các giới hạn phải > 0.")
            continue
        limits[model] = {"rpm": rpm, "tpm": tpm, "rpd": rpd}
    return limits, errors


class TranslatePage(QWidget):
    """Trang dịch phụ đề đa giai đoạn bằng AI."""

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

        # API Key row: ô nhập + nút mắt hiện/ẩn
        api_row = QHBoxLayout()
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText(self._translator.translate("translate.api_key_ph"))
        self._api_key_edit.returnPressed.connect(self._on_run_clicked)
        self._btn_eye = ToolButton()
        self._btn_eye.setIcon(self.style().standardIcon(
            self.style().StandardPixmap.SP_FileDialogDetailedView))
        self._btn_eye.setToolTip(self._translator.translate("translate.api_key_toggle_tip"))
        set_accessible_name(self._btn_eye, self._translator.translate("translate.tr_acc_eye"))
        self._btn_eye.setCheckable(True)
        self._btn_eye.toggled.connect(self._on_eye_toggled)
        # [v3.23.121] Nút quản lý NHIỀU API key (xoay key khi hết quota ngày).
        self._extra_keys: list[str] = []
        self._btn_multi_key = ToolButton()
        self._btn_multi_key.setText("🔑")
        self._btn_multi_key.setToolTip(
            "Quản lý nhiều API key — khi một key hết quota ngày, app tự xoay sang key kế."
        )
        set_accessible_name(self._btn_multi_key, self._translator.translate("translate.tr_acc_multikey"))
        self._btn_multi_key.clicked.connect(self._on_manage_keys_clicked)
        api_row.addWidget(self._api_key_edit)
        api_row.addWidget(self._btn_eye)
        api_row.addWidget(self._btn_multi_key)
        form.addRow("API Key Gemini:", self._wrap_layout(api_row))

        self._lang_combo = QComboBox()
        self._lang_combo.addItems(_TARGET_LANGUAGES)
        self._lang_combo.setCurrentText("Vietnamese")
        form.addRow(self._translator.translate("translate.tr_target_lang"), self._lang_combo)

        self._retry_spin = QSpinBox()
        self._retry_spin.setRange(1, 20)
        self._retry_spin.setValue(5)
        form.addRow(self._translator.translate("translate.tr_retry"), self._retry_spin)

        options_row = QVBoxLayout()
        options_row.setSpacing(_m.SPACING_XS)
        self._tags_check = QCheckBox(self._translator.translate("translate.opt_speaker"))
        self._desc_check = QCheckBox(self._translator.translate("translate.opt_keep_sfx"))
        self._desc_check.setChecked(True)
        options_row.addWidget(self._tags_check)
        options_row.addWidget(self._desc_check)
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
        self._btn_refresh_models.setToolTip(
            "Kết nối Gemini API và lấy danh sách tất cả model có sẵn\n"
            "để điền vào các ô lựa chọn model bên dưới."
        )
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

        # Hàng nút phân tích AI + model chọn + xoá cache
        btn_row = QHBoxLayout()

        # Model selector cho phân tích
        self._analyze_model_combo = QComboBox()
        self._analyze_model_combo.setEditable(True)
        self._analyze_model_combo.addItems(_DEFAULT_MODELS)
        self._analyze_model_combo.setCurrentText("gemini-3.1-flash-lite")  # RPD 500
        self._analyze_model_combo.setToolTip(
            "Model AI dùng để phân tích toàn bộ phụ đề\n"
            "gemini-3.1-flash-lite: khuyến nghị (RPD 500/ngày, nhanh, rẻ)"
        )
        self._analyze_model_combo.setMinimumWidth(200)

        self._btn_analyze = PushButton(self._translator.translate("translate.btn_analyze"))
        self._btn_analyze.setToolTip(
            "Gemini phân tích TOÀN BỘ phụ đề để tự động phát hiện:\n"
            "• Ngôn ngữ gốc\n"
            "• Danh sách đầy đủ nhân vật + vai trò\n"
            "• Tóm tắt cốt truyện / bối cảnh đầy đủ"
        )
        self._btn_analyze.clicked.connect(self._on_analyze_context_clicked)
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
        btn_row.addWidget(self._analyze_model_combo)
        btn_row.addWidget(self._btn_analyze)
        btn_row.addWidget(self._btn_clear_cache)
        btn_row.addWidget(self._btn_clear_cloud)
        btn_row.addWidget(self._btn_manage_tm)
        btn_row.addStretch(1)
        form.addRow(self._wrap_layout(btn_row))
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
            "Riêng các giai đoạn dịch bên dưới là tuỳ chọn (đính video tốn thêm token)."
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

        self._video_path: str = ""
        card.add_layout(v)
        return card

    def _on_attach_video_toggled(self, checked: bool) -> None:
        self._btn_pick_video.setEnabled(checked)
        for cb in (self._video_stage_literal, self._video_stage_style, self._video_stage_localize):
            cb.setEnabled(checked)
        # Visual Cues cũng cần video → chỉ cho bật khi đã bật ngữ cảnh video.
        self._visual_cues_check.setEnabled(checked)
        # [v3.23.120] Mặc định BẬT phân tích hình ảnh khi đính video (theo yêu cầu Toan);
        # người dùng vẫn có thể bỏ chọn nếu muốn tiết kiệm token.
        self._visual_cues_check.setChecked(checked)

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
        )
        self._stage_style = _StagePanel(
            self._translator.translate("translate.stage3_title"),
            TranslationStageKind.STYLE,
            translator=self._translator,
            default_enabled=False,
            default_model="gemini-3.5-flash",        # chất lượng cao hơn cho STYLE
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
            default_model="gemini-3.5-flash",        # chất lượng cao hơn cho LOCALIZE
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
        action_row = QHBoxLayout()
        self._btn_copy_selection = PushButton(self._translator.translate("translate.btn_copy"))
        self._btn_copy_selection.setToolTip(self._translator.translate("translate.tip_copy"))
        self._btn_copy_selection.clicked.connect(self._on_copy_selection_clicked)
        self._btn_copy_selection.setEnabled(False)
        self._btn_send_to_editor = PushButton(self._translator.translate("translate.btn_send_editor"))
        self._btn_send_to_editor.setToolTip(
            "Gửi toàn bộ bản dịch sang trang Biên tập để tiếp tục chỉnh sửa thủ công"
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
        action_row.addStretch(1)
        self._btn_export_srt = PushButton(self._translator.translate("translate.btn_export_srt"))
        self._btn_export_srt.clicked.connect(lambda: self._on_export_clicked(SubtitleFormat.SRT))
        self._btn_export_ass = PushButton(self._translator.translate("translate.btn_export_ass"))
        self._btn_export_ass.clicked.connect(lambda: self._on_export_clicked(SubtitleFormat.ASS))
        action_row.addWidget(self._btn_export_srt)
        action_row.addWidget(self._btn_export_ass)
        self._btn_export_bilingual = PushButton(self._translator.translate("translate.btn_export_bi"))
        self._btn_export_bilingual.setToolTip(
            "Xuất phụ đề SRT có CẢ nguyên văn (gốc) lẫn bản dịch tiếng Việt trên mỗi câu "
            "(gốc ở trên, dịch ở dưới) — tiện cho người xem/người học."
        )
        self._btn_export_bilingual.clicked.connect(self._on_export_bilingual_clicked)
        action_row.addWidget(self._btn_export_bilingual)
        self._btn_export_diag = PushButton(self._translator.translate("translate.btn_export_diag"))
        self._btn_export_diag.setToolTip(
            "Xuất toàn bộ chi tiết quá trình dịch (phụ đề gốc, kết quả, từng giai đoạn, "
            "phân tích ngữ cảnh: nhân vật/tóm tắt/glossary/visual cues) ra một file JSON "
            "để phân tích, khắc phục lỗi và cải thiện chất lượng dịch."
        )
        self._btn_export_diag.clicked.connect(self._on_export_diagnostics_clicked)
        action_row.addWidget(self._btn_export_diag)
        layout.addLayout(action_row)
        card.add_layout(layout)
        return card

    @staticmethod
    def _wrap_layout(inner: QHBoxLayout) -> QWidget:
        wrapper = QWidget()
        wrapper.setLayout(inner)
        return wrapper

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
        vm.resume_detected.connect(self._on_resume_detected)   # [Q3]
        vm.analyze_context_ready.connect(self._on_analyze_context_ready)
        vm.analysis_restored.connect(self._on_analysis_restored)  # [v3.23.45] nạp lại
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
        self._api_key_edit.setText(s.value("api_key", "", type=str))
        # [v3.23.121] Khôi phục các API key dự phòng (mỗi dòng một key).
        extra_raw = s.value("api_keys_extra", "", type=str)
        self._extra_keys = [k.strip() for k in extra_raw.splitlines() if k.strip()]
        self._refresh_multi_key_badge()
        saved_lang = s.value("target_lang", "Vietnamese", type=str)
        if self._lang_combo.findText(saved_lang) >= 0:
            self._lang_combo.setCurrentText(saved_lang)
        # [v3.7 U3] Khôi phục đầy đủ: retry, ngữ cảnh, tuỳ chọn, cấu hình từng giai đoạn.
        if s.contains("retry_count"):
            self._retry_spin.setValue(s.value("retry_count", 5, type=int))
        self._source_lang_edit.setText(s.value("source_lang", "", type=str))
        saved_analyze_model = s.value("analyze_model", "gemini-3.1-flash-lite", type=str)
        self._analyze_model_combo.setCurrentText(saved_analyze_model)
        if s.contains("enable_tags"):
            self._tags_check.setChecked(s.value("enable_tags", False, type=bool))
        if s.contains("include_desc"):
            self._desc_check.setChecked(s.value("include_desc", True, type=bool))
        self._stage_preprocess.restore_state(s, "stage_preprocess")
        self._stage_literal.restore_state(s, "stage_literal")
        self._stage_style.restore_state(s, "stage_style")
        self._stage_localize.restore_state(s, "stage_localize")

    def _persist_inputs(self) -> None:
        s = self._settings
        s.setValue("api_key", self._api_key_edit.text().strip())
        s.setValue("api_keys_extra", "\n".join(self._extra_keys))
        s.setValue("target_lang", self._lang_combo.currentText())
        s.setValue("retry_count", self._retry_spin.value())
        s.setValue("source_lang", self._source_lang_edit.text().strip())
        s.setValue("analyze_model", self._analyze_model_combo.currentText())
        s.setValue("enable_tags", self._tags_check.isChecked())
        s.setValue("include_desc", self._desc_check.isChecked())
        self._stage_preprocess.save_state(s, "stage_preprocess")
        self._stage_literal.save_state(s, "stage_literal")
        self._stage_style.save_state(s, "stage_style")
        self._stage_localize.save_state(s, "stage_localize")
        s.sync()

    # ── Public API cho MainWindow ────────────────────────────────────────
    def load_events(self, events: list[SubtitleEvent]) -> None:
        """Nạp phụ đề từ bên ngoài (vd trang Biên tập đẩy sang)."""
        # [#2] Nạp phụ đề mới = dự án mới → tẩy sạch TOÀN BỘ ngữ cảnh phim cũ.
        self._reset_context_and_video()
        self._view_model.set_source_events(events)

    def _reset_context_and_video(self) -> None:
        """[#2 v3.18] "Tẩy não" toàn bộ ngữ cảnh khi chuyển sang phụ đề/dự án mới.

        Chống rò rỉ ngữ cảnh (context leakage) khiến AI dịch bị ảo giác: xoá Bảng
        nhân vật, Tóm tắt cốt truyện, Ngôn ngữ gốc, và gỡ Video ngữ cảnh (đường dẫn
        + nhãn + bỏ tick ô đính video). Gọi mỗi khi nạp phụ đề mới (từ File, từ trang
        Biên tập, hay khi đổi dự án).
        """
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
        # Gỡ video ngữ cảnh của phim cũ.
        self._video_path = ""
        if getattr(self, "_video_path_label", None) is not None:
            self._video_path_label.setText(self._translator.translate("translate.no_video"))
        if getattr(self, "_attach_video_check", None) is not None:
            self._attach_video_check.blockSignals(True)
            self._attach_video_check.setChecked(False)
            self._attach_video_check.blockSignals(False)

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
        self._characters_edit.clear()
        self._overview_edit.clear()
        self._glossary_edit.clear()

    # ── Slot xử lý sự kiện UI ─────────────────────────────────────────────
    def clear_context_fields(self) -> None:
        """Xoá trắng các ô Ngữ cảnh (Bảng nhân vật + Tóm tắt bối cảnh).

        Dùng khi chuyển sang dự án/phụ đề khác để chống rò rỉ ngữ cảnh giữa các phim.
        """
        self._characters_edit.clear()
        self._overview_edit.clear()
        self._glossary_edit.clear()

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

    def _on_load_file_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Chọn tệp phụ đề nguồn", "", "Phụ đề (*.srt *.ass)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path_str:
            # [#2] Nạp phụ đề mới = dự án mới → tẩy sạch toàn bộ ngữ cảnh phim trước
            # (Bảng nhân vật / Tóm tắt / Ngôn ngữ gốc / Video) để không lẫn sang phim
            # hiện tại.
            self._reset_context_and_video()
            sub_path = Path(path_str)
            self._view_model.load_source_from_file(sub_path)
            # [Smart Context Auto-Detect] Tự dò video cùng tên nằm cạnh tệp phụ đề
            # để nạp thẳng làm Ngữ cảnh Video — đỡ thao tác chọn video thủ công.
            from subtitles_extractor.application.services.sibling_video_finder import (
                find_sibling_video,
            )

            sibling_video = find_sibling_video(sub_path)
            if sibling_video is not None:
                self._video_path = str(sibling_video)
                self._video_path_label.setText(sibling_video.name)
                if getattr(self, "_attach_video_check", None) is not None:
                    self._attach_video_check.setChecked(True)

    def _on_pull_editor_clicked(self) -> None:
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
        self._view_model.set_source_events(events)

        # [#11 v3.18] Auto-Attach: tự bốc luôn VIDEO đang mở ở trang Biên tập sang
        # trang Dịch (đường dẫn + nhãn + tự tick ô đính video làm ngữ cảnh).
        attached_name = ""
        if self._editor_video_provider is not None:
            editor_video = self._editor_video_provider()
            if editor_video:
                self._video_path = str(editor_video)
                attached_name = Path(editor_video).name
                if getattr(self, "_video_path_label", None) is not None:
                    self._video_path_label.setText(attached_name)
                if getattr(self, "_attach_video_check", None) is not None:
                    self._attach_video_check.setChecked(True)

        message = f"Đã lấy {len(events)} dòng từ trang Biên tập."
        if attached_name:
            message += f"\nĐã tự đính video ngữ cảnh: {attached_name}"
        self._show_info(self._translator.translate("common.success"), message)

    def _on_run_clicked(self) -> None:
        # [v3.23.44] Validate sớm tại UI để báo lỗi rõ ràng (thay vì lẫn lộn ở tầng dưới).
        # [v3.23.121] Gộp tất cả API key (chính + dự phòng) -> adapter tự xoay khi hết quota.
        api_key = self._combined_keys_text()
        if not api_key:
            self._show_warning(
                "Thiếu API Key",
                "Hãy dán API Key Gemini vào ô 'API Key' trước khi dịch.\n"
                "Lấy key miễn phí tại: https://aistudio.google.com/apikey",
            )
            self._api_key_edit.setFocus()
            return
        stages = self._collect_enabled_stages()
        if not stages:
            self._show_warning(
                "Chưa bật giai đoạn dịch nào",
                "Hãy bật ít nhất một giai đoạn (vd 'Giai đoạn 2: Dịch thô sát nghĩa') "
                "bằng ô 'Bật giai đoạn này'.",
            )
            return

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
            return

        context = TranslationContext(
            target_lang=self._lang_combo.currentText().strip() or "Vietnamese",
            source_lang=self._source_lang_edit.text().strip(),
            overview=self._overview_edit.toPlainText().strip(),
            characters=self._characters_edit.toPlainText().strip(),
            glossary=self._glossary_edit.toPlainText().strip(),
            visual_cues=self._visual_cues_edit.toPlainText().strip(),
            enable_tags=self._tags_check.isChecked(),
            include_desc=self._desc_check.isChecked(),
        )
        self._view_model.start_translation(
            api_key=api_key,
            retry_count=self._retry_spin.value(),
            stages=stages,
            context=context,
            video_path=self._selected_video_path(),
            attach_video_stages=self._selected_attach_video_stages(),
            enable_visual_cues=self._visual_cues_check.isChecked(),
        )

    def _collect_enabled_stages(self) -> list[TranslationStageConfig]:
        panels = (
            self._stage_preprocess,
            self._stage_literal,
            self._stage_style,
            self._stage_localize,
        )
        return [panel.build_config() for panel in panels if panel.is_enabled()]

    def _on_eye_toggled(self, checked: bool) -> None:
        """Hiện/ẩn API key."""
        mode = QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        self._api_key_edit.setEchoMode(mode)

    # ── Nhiều API key ────────────────────────────────────────────────────
    def _all_keys(self) -> list[str]:
        """Key hiệu lực: key chính + key dự phòng, bỏ trùng/rỗng."""
        out: list[str] = []
        seen: set[str] = set()
        for key in [self._api_key_edit.text().strip(), *self._extra_keys]:
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def _combined_keys_text(self) -> str:
        """Chuỗi tất cả key (nối xuống dòng) để truyền xuống adapter (adapter tự tách)."""
        return "\n".join(self._all_keys())

    @staticmethod
    def _masked_key(key: str) -> str:
        """Che API key, chỉ chừa 4 ký tự cuối để người dùng nhận diện."""
        k = (key or "").strip()
        return f"…{k[-4:]}" if len(k) >= 4 else "…"

    def _keys_quota_status_text(self) -> str:
        """[v3.23.124] Mô tả tình trạng quota/ngày từng key cho model phân tích đang chọn."""
        keys = self._all_keys()
        if not keys:
            return "Chưa có API key nào."
        model = (
            self._analyze_model_combo.currentText().strip() or "gemini-3.1-flash-lite"
        )
        try:
            stats = self._view_model.quota_status_for_keys(keys, model)
        except (AttributeError, KeyError, RuntimeError):
            return ""
        parts = []
        for key, st in zip(keys, stats):
            mark = "🟢" if st["remaining"] > 0 else "🔴"
            parts.append(
                f"{mark} {self._masked_key(key)}: {st['used']}/{st['limit']} req"
            )
        return f"Quota hôm nay ({model}):  " + "   ".join(parts)

    def _refresh_multi_key_badge(self) -> None:
        total = len(self._all_keys())
        self._btn_multi_key.setText("🔑" if total <= 1 else f"🔑{total}")

    def _on_manage_keys_clicked(self) -> None:
        """Mở hộp thoại nhập NHIỀU API key (mỗi dòng một key)."""
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QPlainTextEdit,
            QVBoxLayout,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle(self._translator.translate("translate.dlg_apikey_title"))
        dlg.resize(520, 320)
        lay = QVBoxLayout(dlg)
        info = QLabel(
            "Mỗi dòng một API key. Khi một key hết quota ngày, app tự xoay sang key kế "
            "tiếp còn hạn mức.\nKey ở DÒNG ĐẦU là key chính."
        )
        info.setWordWrap(True)
        lay.addWidget(info)
        editor = QPlainTextEdit()
        editor.setPlaceholderText("AIza...key1\nAIza...key2\nAIza...key3")
        editor.setPlainText("\n".join(self._all_keys()))
        lay.addWidget(editor, 1)

        # [v3.23.124] Hiển thị tình trạng quota request/ngày của từng key (cho model phân
        # tích đang chọn) để người dùng biết key nào sắp/đã hết.
        status = QLabel(self._keys_quota_status_text())
        status.setWordWrap(True)
        status.setStyleSheet(caption_style())
        lay.addWidget(status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        keys: list[str] = []
        seen: set[str] = set()
        for line in editor.toPlainText().splitlines():
            k = line.strip()
            if k and k not in seen:
                seen.add(k)
                keys.append(k)
        # Dòng đầu -> ô key chính; phần còn lại -> key dự phòng.
        self._api_key_edit.setText(keys[0] if keys else "")
        self._extra_keys = keys[1:]
        self._refresh_multi_key_badge()
        self._persist_inputs()
        if len(keys) > 1:
            self._show_success(
                "Đã lưu API key",
                f"Đang dùng {len(keys)} key — sẽ tự xoay khi một key hết quota ngày.",
            )

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

    def _on_edit_quota_clicked(self) -> None:
        """[v3.23.122] Mở hộp thoại xem/sửa/thêm hạn mức quota (RPM·TPM·RPD) của model."""
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QDialog,
            QDialogButtonBox,
            QHBoxLayout,
            QHeaderView,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )
        cfg = self._view_model.get_quota_config()
        custom: dict = cfg.get("custom", {})
        defaults: dict = cfg.get("defaults", {})

        dlg = QDialog(self)
        dlg.setWindowTitle(self._translator.translate("translate.dlg_quota_title"))
        dlg.resize(640, 460)
        lay = QVBoxLayout(dlg)
        info = QLabel(
            "Đặt giới hạn cho TÊN MODEL cụ thể (vd 'gemini-3.5-flash'). Giá trị ở đây sẽ "
            "GHI ĐÈ mặc định.\n"
            "RPM = request/phút · TPM = token/phút · RPD = request/ngày."
        )
        info.setWordWrap(True)
        lay.addWidget(info)
        if defaults:
            ref = ", ".join(
                f"{k}: {v['rpm']}/{v['tpm']}/{v['rpd']}" for k, v in defaults.items()
            )
            ref_lbl = QLabel(self._translator.translate("translate.tr_prefix_default").replace("{ref}", str(ref)))
            ref_lbl.setWordWrap(True)
            ref_lbl.setStyleSheet(caption_style())
            lay.addWidget(ref_lbl)

        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["Model", "RPM", "TPM", "RPD"])
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        def _add_row(
            name: str = "", rpm: int = 5, tpm: int = 250000, rpd: int = 20
        ) -> None:
            r = table.rowCount()
            table.insertRow(r)
            for col, val in enumerate([name, str(rpm), str(tpm), str(rpd)]):
                table.setItem(r, col, QTableWidgetItem(val))

        for name, lim in custom.items():
            _add_row(name, lim["rpm"], lim["tpm"], lim["rpd"])
        if table.rowCount() == 0:
            _add_row("gemini-3.5-flash")
        lay.addWidget(table, 1)

        btn_row = QHBoxLayout()
        btn_add = PushButton(self._translator.translate("translate.btn_add_model"))
        btn_add.clicked.connect(lambda: _add_row())
        btn_del = PushButton(self._translator.translate("translate.btn_remove_row"))
        btn_del.clicked.connect(
            lambda: (
                table.removeRow(table.currentRow())
                if table.currentRow() >= 0 else None
            )
        )
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_del)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        rows: list[tuple[str, str, str, str]] = []
        for r in range(table.rowCount()):
            cells = [table.item(r, c) for c in range(4)]
            rows.append(tuple((cell.text() if cell else "") for cell in cells))  # type: ignore[arg-type]
        limits, errors = parse_quota_rows(rows)
        if errors:
            self._show_warning(self._translator.translate("translate.warn_invalid_title"), "\n".join(errors[:6]))
            return
        self._view_model.save_quota_config(limits)
        self._show_success(
            "Đã lưu hạn mức quota",
            f"Đã áp {len(limits)} model. Áp dụng ngay cho các lần gọi tiếp theo.",
        )

    def _on_export_clicked(self, output_format: SubtitleFormat) -> None:
        if not self._view_model.has_result:
            self._show_warning(self._translator.translate("translate.warn_no_translation_title"), self._translator.translate("translate.warn_translate_first_srt"))
            return
        suffix = output_format.value
        # Đặt tên mặc định theo quy ước: <tên video>.<mã ngôn ngữ>.<srt/ass>.
        default_name = f"translated.{suffix}"
        if self._source_video_path:
            try:
                from subtitles_extractor.domain.value_objects.output_naming import (
                    SubtitleFormat as NamingFormat,
                    translated_subtitle_path,
                )

                lang_code = _LANGUAGE_CODES.get(
                    self._lang_combo.currentText(), "vi"
                )
                default_name = str(translated_subtitle_path(
                    self._source_video_path, lang_code, NamingFormat.from_str(suffix)
                ))
            except (ValueError, OSError, KeyError):
                pass
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Lưu bản dịch", default_name, f"Phụ đề (*.{suffix})",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path_str:
            ok = self._view_model.export_translation(Path(path_str), output_format)
            if ok:
                saved_path = Path(path_str)
                self._show_success(
                    "Xuất thành công",
                    f"Đã lưu bản dịch vào:\n{saved_path}",
                )
                self._reveal_in_explorer(saved_path)

    def _on_export_bilingual_clicked(self) -> None:
        """[v3.23.116] Xuất SRT song ngữ (gốc + dịch trên mỗi câu)."""
        if not self._view_model.has_result:
            self._show_warning(self._translator.translate("translate.warn_no_translation_title"), self._translator.translate("translate.warn_translate_first_bi"))
            return
        default_name = "bilingual.srt"
        if self._source_video_path:
            try:
                default_name = (
                    str(Path(self._source_video_path).with_suffix(""))
                    + ".bilingual.srt"
                )
            except (ValueError, OSError):
                pass
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Lưu phụ đề song ngữ", default_name, "Phụ đề (*.srt)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path_str:
            return
        ok = self._view_model.export_bilingual(Path(path_str), SubtitleFormat.SRT)
        if ok:
            saved_path = Path(path_str)
            self._show_success(
                "Xuất song ngữ thành công",
                f"Đã lưu phụ đề song ngữ vào:\n{saved_path}",
            )
            self._reveal_in_explorer(saved_path)

    def _on_export_diagnostics_clicked(self) -> None:
        """Xuất toàn bộ chi tiết quá trình dịch ra file JSON để phân tích chất lượng."""
        if not self._view_model.has_result:
            self._show_warning(self._translator.translate("translate.warn_no_translation_title"), self._translator.translate("translate.warn_translate_first_diag"))
            return
        default_name = "translation_diagnostics.json"
        if self._source_video_path:
            try:
                default_name = (
                    str(Path(self._source_video_path).with_suffix(""))
                    + ".translation_debug.json"
                )
            except (ValueError, OSError):
                pass
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Xuất chẩn đoán dịch", default_name, "JSON (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path_str:
            return
        try:
            saved = self._view_model.export_translation_diagnostics(Path(path_str))
        except OSError as exc:
            self._show_warning(
                "Không xuất được", f"Lỗi khi ghi file chẩn đoán:\n{exc}"
            )
            return
        self._show_success(
            "Xuất chẩn đoán thành công",
            f"Đã lưu chi tiết quá trình dịch vào:\n{saved}",
        )
        self._reveal_in_explorer(saved)

    def _reveal_in_explorer(self, path: "Path") -> None:
        """[v3.23.51] Mở thư mục chứa tệp vừa lưu (đa nền tảng, không chặn UI)."""
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            folder = path.parent if path.is_file() or path.suffix else path
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except (OSError, RuntimeError) as exc:
            logger.debug("Không mở được thư mục chứa tệp: %s", exc)

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
            video_path = self._source_video_path or self._video_path
            if video_path:
                target_lang = self._lang_combo.currentText().strip() or "Vietnamese"
                if self._view_model.try_restore_saved_analysis(video_path, target_lang):
                    self._stage_label.setText(
                        "Đã nạp lại phân tích ngữ cảnh đã lưu cho video này."
                    )
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
            self._overview_edit.setPlainText(ctx.overview)
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
        # [v3.23.332] Cả bộ: vẫn để luồng lưu/đẩy kết quả chạy bình thường, nhưng sau
        # đó chuyển sang tập kế thay vì dừng chờ người dùng.
        if getattr(self, "_batch_items", None):
            # [v3.23.366] SỬA BUG: batch dịch xong trước đây chỉ đổ ra bảng, KHÔNG ghi
            # tệp → khâu TTS hàng loạt thấy "chưa dịch". Nay ghi .translate.<lang>.srt
            # cạnh video TRƯỚC khi sang tập kế, để nối được TTS → Xuất bản hàng loạt.
            self._persist_batch_translation()
            QTimer.singleShot(0, self._advance_batch_translate)
        self._table.setUpdatesEnabled(False)
        self._table.setRowCount(len(translated_events))
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
        # [v3.7 U5] resizeRowsToContents() trên hàng nghìn dòng word-wrap rất chậm
        # (O(n) layout, có thể đứng UI vài giây). Chỉ làm khít từng dòng cho bảng
        # nhỏ; bảng lớn dùng chiều cao mặc định 2 dòng để giữ UI mượt.
        if len(translated_events) <= _RESIZE_ROWS_THRESHOLD:
            self._table.resizeRowsToContents()
        else:
            self._table.verticalHeader().setDefaultSectionSize(_DEFAULT_ROW_HEIGHT_PX)
        self._table.setUpdatesEnabled(True)
        # Đảm bảo bảng chiếm phần lớn splitter sau mỗi lần dịch xong
        if hasattr(self, "_splitter"):
            self._splitter.setSizes([420, 160])
        self._update_action_states()
        elapsed = self._view_model.last_elapsed_sec
        elapsed_str = f" ({elapsed:.0f}s)" if elapsed < 60 else f" ({int(elapsed//60)}p{int(elapsed%60):02d}s)"
        stage_count = self._view_model.last_stage_count
        self._show_success(
            "Dịch hoàn tất",
            f"Đã dịch {len(translated_events)} dòng qua {stage_count} giai đoạn{elapsed_str}.",
        )


    def _persist_batch_translation(self) -> None:
        """[v3.23.366] Ghi ``.translate.<lang>.srt`` cho tập vừa dịch (chế độ hàng loạt).

        Dùng đúng quy ước tên như khi xuất tay (``_LANGUAGE_CODES`` + ``translated_
        subtitle_path``) để khâu TTS hàng loạt tìm thấy. Không ném lỗi ra ngoài.
        """
        items = getattr(self, "_batch_items", None)
        if not items:
            return
        index = min(self._batch_index, len(items) - 1)
        video_path = items[index].video_path
        try:
            from subtitles_extractor.domain.value_objects.output_naming import (
                SubtitleFormat as NamingFormat,
                translated_subtitle_path,
            )

            lang_code = _LANGUAGE_CODES.get(self._lang_combo.currentText(), "vi")
            out_path = translated_subtitle_path(
                video_path, lang_code, NamingFormat.SRT
            )
            if self._view_model.export_translation(out_path, SubtitleFormat.SRT):
                logger.info("Hàng loạt: đã ghi bản dịch → %s", out_path.name)
        except (ValueError, OSError, KeyError) as exc:
            logger.warning(
                "Hàng loạt: không ghi được bản dịch cho %s: %s", video_path.name, exc
            )

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
        if getattr(self, "_batch_items", None):
            self._batch_cancelled = True
            remaining = len(self._batch_items) - self._batch_index - 1
            self._stage_label.setText(self._translator.translate("translate.tr_cancelling").replace("{n}", str(max(0, remaining))))
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

    def _on_batch_translate_clicked(self) -> None:
        """Quét thư mục phim bộ rồi dịch lần lượt các tập đã trích xuất."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from subtitles_extractor.application.services.batch_translate_plan import (
            build_translate_plan,
            estimate_requests,
            find_episode_videos,
            quota_warning,
            summarise_translate_plan,
        )

        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa các tập", "")
        if not folder:
            return

        videos = find_episode_videos(Path(folder))
        if not videos:
            QMessageBox.information(
                self, self._translator.translate("translate.no_folder_title"),
                self._translator.translate("translate.batch_no_source_body"),
            )
            return

        # [v3.23.365] BÙ tệp .original.srt từ CSDL cho các tập đã trích nhưng thiếu tệp
        # (vd đã trích ở bản cũ chỉ lưu CSDL) — để KHỎI phải trích xuất lại.
        self._backfill_original_from_db(videos)

        plan = build_translate_plan(videos, skip_existing=True)
        runnable = [item for item in plan if item.will_run]
        summary = summarise_translate_plan(plan)
        if not runnable:
            QMessageBox.information(
                self, self._translator.translate("translate.no_episode_title"),
                self._translator.translate("translate.batch_missing_body").replace("{summary}", summary),
            )
            return

        batch_size = self.batch_spin.value() if hasattr(self, "batch_spin") else 40
        requests = estimate_requests(plan, batch_size=batch_size)
        warning = quota_warning(requests, self._current_daily_limit())

        message = (
            f"Tìm thấy {len(videos)} tập.\n{summary}\n\n"
            f"Ước tính khoảng {requests} request API (lô {batch_size} câu)."
        )
        if warning:
            message += f"\n\n⚠️ {warning}"
        message += f"\n\nDịch {len(runnable)} tập?"

        answer = QMessageBox.question(
            self, self._translator.translate("translate.batch_confirm_title"), message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._batch_items = runnable
        self._batch_index = 0
        self._batch_failures = []
        self._batch_cancelled = False
        self._batch_quota_stopped = False
        self._run_next_batch_translate()

    def _current_daily_limit(self) -> int | None:
        """Hạn mức ngày (RPD) của mô hình đang chọn, nếu tra được."""
        try:
            from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
                _match_free_tier_limit,
            )

            combo = getattr(self, "model_combo", None)
            name = combo.currentText().strip() if combo is not None else ""
            if not name:
                return None
            return _match_free_tier_limit(name).rpd
        except Exception as exc:  # noqa: BLE001 — không tra được thì bỏ cảnh báo
            logger.debug("Không tra được hạn mức ngày: %s", exc)
            return None

    def _run_next_batch_translate(self) -> None:
        """Nạp phụ đề gốc của tập kế tiếp; dịch chạy khi nạp xong."""
        if not getattr(self, "_batch_items", None):
            return
        if self._batch_cancelled or self._batch_index >= len(self._batch_items):
            self._finish_batch_translate()
            return

        item = self._batch_items[self._batch_index]
        total = len(self._batch_items)
        self._stage_label.setText(
            f"Cả bộ {self._batch_index + 1}/{total}: {item.video_path.name}"
        )
        self._batch_pending_run = True
        self._view_model.load_source_from_file(item.source_path)

    def _on_source_changed_for_batch(self, count: int) -> None:
        """Nạp xong trong lúc chạy hàng loạt -> bắt đầu dịch tập này."""
        if not getattr(self, "_batch_pending_run", False):
            return
        self._batch_pending_run = False
        if count <= 0:
            index = min(self._batch_index, len(self._batch_items) - 1)
            name = self._batch_items[index].video_path.name
            self._advance_batch_translate(failed=name)
            return
        QTimer.singleShot(0, self._on_run_clicked)

    def _on_error_for_batch(self, message: str) -> bool:
        """Xử lý lỗi khi đang chạy hàng loạt.

        ĐIỂM KHÁC BIỆT của khâu Dịch: hết hạn mức NGÀY thì chạy tiếp cũng vô ích —
        phải dừng cả hàng đợi thay vì đốt thêm lỗi cho từng tập còn lại.

        Returns:
            ``True`` nếu đã xử lý (đang chạy hàng loạt).
        """
        if not getattr(self, "_batch_items", None):
            return False
        index = min(self._batch_index, len(self._batch_items) - 1)
        name = self._batch_items[index].video_path.name

        lowered = message.lower()
        if any(k in lowered for k in ("hạn mức", "quota", "rpd", "resource_exhausted")):
            logger.warning("Cả bộ: hết hạn mức tại tập %s — dừng hàng đợi.", name)
            self._batch_quota_stopped = True
            self._batch_cancelled = True
            self._advance_batch_translate(failed=name)
            return True

        logger.warning("Cả bộ: tập %s lỗi — %s", name, message)
        self._advance_batch_translate(failed=name)
        return True

    def _advance_batch_translate(self, *, failed: str | None = None) -> bool:
        """Chuyển sang tập kế. Trả ``True`` nếu đang chạy hàng loạt."""
        if not getattr(self, "_batch_items", None):
            return False
        if failed:
            self._batch_failures.append(failed)
        if self._batch_cancelled:
            self._finish_batch_translate()
            return True
        self._batch_index += 1
        QTimer.singleShot(0, self._run_next_batch_translate)
        return True

    def _finish_batch_translate(self) -> None:
        """Báo tổng kết và dọn trạng thái hàng đợi."""
        from PySide6.QtWidgets import QMessageBox

        total = len(self._batch_items)
        failures = list(self._batch_failures)
        cancelled = self._batch_cancelled
        quota_stopped = getattr(self, "_batch_quota_stopped", False)
        processed = self._batch_index
        done = max(0, processed - len(failures))

        self._batch_items = []
        self._batch_index = 0
        self._batch_failures = []
        self._batch_cancelled = False
        self._batch_quota_stopped = False
        self._batch_pending_run = False

        remaining = max(0, total - processed)
        if quota_stopped:
            message = (
                f"HẾT HẠN MỨC NGÀY. Đã dịch xong {done}/{total} tập.\n"
                f"Còn {remaining} tập chưa dịch — chạy lại vào ngày mai, hệ thống sẽ "
                "TỰ BỎ QUA các tập đã xong nên không tốn thêm hạn mức."
            )
        elif cancelled:
            message = (
                f"Đã huỷ. Dịch xong {done}/{total} tập, bỏ {remaining} tập còn lại."
            )
        else:
            message = f"Đã dịch xong {done}/{total} tập."
        if failures:
            message += "\n\nThất bại:\n• " + "\n• ".join(failures[:10])

        self._stage_label.setText(message.splitlines()[0])
        QMessageBox.information(self, self._translator.translate("translate.batch_done_title"), message)

    def _on_cancelled(self) -> None:
        self._progress.setValue(0)
        self._stage_label.setText(self._translator.translate("translate.status_cancelled"))
        self._show_info(self._translator.translate("translate.toast_cancelled_title"), self._translator.translate("translate.toast_cancelled_body"))

    def _on_progress_changed(self, percent: int, stage_label: str) -> None:
        self._progress.setValue(percent)
        self._stage_label.setText(f"{stage_label} — {percent}%")

    def _on_busy_changed(self, busy: bool) -> None:
        if busy:
            # Reset tiến độ ngay khi bắt đầu để không hiện 100% của lần dịch cũ
            self._progress.setValue(0)
            self._stage_label.setText(self._translator.translate("translate.status_starting"))
        self._btn_run.setEnabled(not busy)
        self._btn_cancel.setEnabled(busy)
        self._btn_load_file.setEnabled(not busy)
        self._btn_pull_editor.setEnabled(not busy)
        self._btn_analyze.setEnabled(not busy)
        self._btn_clear_cache.setEnabled(not busy)
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

    def _on_refresh_models_clicked(self) -> None:
        """[fix B4/U2] Validate API key + disable button + hiển thị trạng thái."""
        api_key = self._api_key_edit.text().strip()
        if not api_key:
            self._show_warning(self._translator.translate("translate.warn_no_apikey_title"), self._translator.translate("translate.warn_apikey_models"))
            return
        self._btn_refresh_models.setEnabled(False)
        self._btn_refresh_models.setText(self._translator.translate("translate.status_loading_model"))
        self._view_model.fetch_available_models(api_key)

    def _on_analyze_context_clicked(self) -> None:
        api_key = self._combined_keys_text()
        if not api_key:
            self._show_warning(self._translator.translate("translate.warn_no_apikey_title"), self._translator.translate("translate.warn_apikey_analyze"))
            return
        model_name = self._analyze_model_combo.currentText().strip() or "gemini-3.1-flash-lite"
        self._view_model.start_context_analysis(
            api_key=api_key,
            target_lang=self._lang_combo.currentText() or "Vietnamese",
            model_name=model_name,
            video_path=self._selected_video_path(),
            enable_visual_cues=self._visual_cues_check.isChecked(),
        )

    def _on_models_ready(self, models: list) -> None:
        """[fix U2] Khôi phục button + cập nhật combo model."""
        # Luôn khôi phục button dù models có hay không
        self._btn_refresh_models.setEnabled(True)
        self._btn_refresh_models.setText(self._translator.translate("translate.btn_load_models"))
        if not models:
            return
        for panel in (self._stage_preprocess, self._stage_literal,
                      self._stage_style, self._stage_localize):
            panel.set_model_items(models)
        # Cập nhật cả combo model phân tích ngữ cảnh
        current_analyze = self._analyze_model_combo.currentText()
        self._analyze_model_combo.blockSignals(True)
        self._analyze_model_combo.clear()
        self._analyze_model_combo.addItems(models)
        idx = self._analyze_model_combo.findText(current_analyze)
        self._analyze_model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._analyze_model_combo.blockSignals(False)

        self._show_success(
            "Danh sách model cập nhật",
            f"Đã tải {len(models)} model từ Gemini API.",
        )

    def _on_column_resized(self, _col: int, _old: int, _new: int) -> None:
        """Resize chiều cao hàng khi cột bị kéo — word-wrap cần biết width thực tế."""
        row_count = self._table.rowCount()
        if row_count == 0 or row_count > _RESIZE_ROWS_THRESHOLD:
            return
        # Tạm thời ngắt signal để tránh đệ quy
        self._table.horizontalHeader().sectionResized.disconnect(self._on_column_resized)
        try:
            self._table.resizeRowsToContents()
        finally:
            self._table.horizontalHeader().sectionResized.connect(self._on_column_resized)

    def _on_table_selection_changed(self) -> None:
        """Cập nhật detail panel khi chọn dòng trong bảng."""
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if not rows:
            self._detail_source.clear()
            self._detail_translation.clear()
            return
        # Chỉ hiển thị khi chọn 1 dòng; nhiều dòng → hiển thị số lượng
        if len(rows) == 1:
            row = rows[0]
            src_item = self._table.item(row, 2)
            tgt_item = self._table.item(row, 3)
            time_item = self._table.item(row, 1)
            header = f"[{time_item.text() if time_item else ''}]"
            self._detail_source.setPlainText(
                f"{header}\n{src_item.text() if src_item else ''}"
            )
            self._detail_translation.setPlainText(
                f"{header}\n{tgt_item.text() if tgt_item else ''}"
            )
        else:
            self._detail_source.setPlainText(f"Đã chọn {len(rows)} dòng.")
            self._detail_translation.setPlainText(
                "\n".join(
                    (self._table.item(r, 3) or QTableWidgetItem("")).text()
                    for r in rows
                )
            )
        self._update_action_states()

    def _on_send_to_editor_clicked(self) -> None:
        """Gửi toàn bộ bản dịch sang trang Biên tập."""
        ok = self._view_model.push_to_editor()
        if ok:
            self._show_success(
                "Đã gửi sang Biên tập",
                f"Đã chuyển {len(self._view_model.translated_events)} dòng. "
                "Hãy chuyển sang trang Biên tập để chỉnh sửa.",
            )

    def _on_manage_tm_clicked(self) -> None:
        """[v3.23.57] Mở dialog quản lý bộ nhớ dịch phim bộ (xem danh sách + xoá)."""
        from PySide6.QtWidgets import QDialog, QMessageBox

        def _refresh(table) -> None:
            series = self._view_model.list_memory_series()
            table.setRowCount(len(series))
            for row, (key, count) in enumerate(series):
                table.setItem(row, 0, QTableWidgetItem(key))
                table.setItem(row, 1, QTableWidgetItem(str(count)))
            return series

        dialog = QDialog(self)
        dialog.setWindowTitle(self._translator.translate("translate.dlg_memory_title"))
        dialog.resize(640, 460)
        v = QVBoxLayout(dialog)
        v.addWidget(QLabel(
            "Bộ nhớ dịch tích luỹ theo từng phim bộ (gom theo thư mục chứa video). "
            "Chọn một bộ rồi nhấn Xoá nếu muốn dịch lại từ đầu cho bộ đó."
        ))
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Phim bộ (thư mục)", "Số câu đã nhớ"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        series = _refresh(table)
        table.resizeColumnsToContents()
        v.addWidget(table, 1)

        if not series:
            v.addWidget(QLabel(self._translator.translate("translate.mem_empty")))

        btn_row = QHBoxLayout()
        btn_delete = PushButton(self._translator.translate("translate.btn_remove_selected"))
        btn_close = PushButton(self._translator.translate("common.close"))

        def _on_delete() -> None:
            rows = table.selectionModel().selectedRows()
            if not rows:
                return
            key_item = table.item(rows[0].row(), 0)
            if key_item is None:
                return
            series_key = key_item.text()
            confirm = QMessageBox.question(
                dialog, self._translator.translate("translate.mem_delete_title"),
                self._translator.translate("translate.mem_delete_body").replace("{series}", series_key),
            )
            if confirm == QMessageBox.StandardButton.Yes:
                if self._view_model.clear_memory_series(series_key):
                    _refresh(table)

        btn_delete.clicked.connect(_on_delete)
        btn_close.clicked.connect(dialog.accept)
        btn_row.addWidget(btn_delete)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        v.addLayout(btn_row)
        dialog.exec()

    def _on_check_glossary_clicked(self) -> None:
        """[v3.23.54] Kiểm tra nhất quán thuật ngữ và hiển thị danh sách vi phạm."""
        glossary = self._glossary_edit.toPlainText().strip()
        if not glossary:
            self._show_info(
                self._translator.translate("translate.toast_no_glossary_title"),
                self._translator.translate("translate.toast_no_glossary_body"),
            )
            return
        violations = self._view_model.check_glossary_consistency(glossary)
        if not violations:
            self._show_success(
                self._translator.translate("translate.glossary_check_title"),
                "Không phát hiện dòng nào dịch sai/thiếu thuật ngữ theo bảng đã cung cấp.",
            )
            return
        from PySide6.QtWidgets import QDialog

        dialog = QDialog(self)
        dialog.setWindowTitle(self._translator.translate("translate.tr_dlg_review").replace("{n}", str(len(violations))))
        dialog.resize(820, 480)
        v = QVBoxLayout(dialog)
        v.addWidget(QLabel(
            "Các dòng dưới đây có thuật ngữ gốc nhưng bản dịch chuẩn không xuất hiện. "
            "Hãy kiểm tra (có thể là biến thể hợp lệ hoặc cần sửa cho nhất quán)."
        ))
        table = QTableWidget(len(violations), 4)
        table.setHorizontalHeaderLabels([
            self._translator.translate("translate.col_line"),
            self._translator.translate("translate.gl_term"),
            self._translator.translate("translate.gl_standard"),
            self._translator.translate("translate.gl_sentence"),
        ])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setWordWrap(True)
        for row, vio in enumerate(violations):
            table.setItem(row, 0, QTableWidgetItem(str(vio.line_index)))
            table.setItem(row, 1, QTableWidgetItem(vio.source_term))
            table.setItem(row, 2, QTableWidgetItem(vio.expected_target))
            table.setItem(row, 3, QTableWidgetItem(vio.translated_text))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(table, 1)
        close_btn = PushButton(self._translator.translate("common.close"))
        close_btn.clicked.connect(dialog.accept)
        v.addWidget(close_btn)
        dialog.exec()

    def _on_compare_stages_clicked(self) -> None:
        """[v3.23.47] Mở cửa sổ so sánh bản dịch qua từng giai đoạn."""
        from PySide6.QtWidgets import QDialog
        comparison = self._view_model.stage_comparison()
        if not comparison:
            self._show_info(
                self._translator.translate("translate.toast_no_compare_title"),
                self._translator.translate("translate.toast_no_compare_body"),
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self._translator.translate("translate.dlg_compare_title"))
        dialog.resize(900, 560)
        v = QVBoxLayout(dialog)
        v.addWidget(QLabel(
            "So sánh bản dịch qua từng giai đoạn. Các ô khác biệt giúp thấy mỗi giai "
            "đoạn đã tinh chỉnh thế nào."
        ))

        stage_names = [name for name, _ in comparison]
        columns = ["#", "Bản gốc", *stage_names]
        source_texts = [ev.text for ev in self._view_model.source_events]
        row_count = max(len(texts) for _, texts in comparison)

        table = QTableWidget(row_count, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setWordWrap(True)
        table.setAlternatingRowColors(True)
        for row in range(row_count):
            table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            src = source_texts[row] if row < len(source_texts) else ""
            table.setItem(row, 1, QTableWidgetItem(src))
            prev_text = None
            for col, (_, texts) in enumerate(comparison, start=2):
                text = texts[row] if row < len(texts) else ""
                item = QTableWidgetItem(text)
                # Tô sáng ô khác với giai đoạn liền trước để dễ thấy thay đổi.
                if prev_text is not None and text != prev_text:
                    item.setBackground(Qt.GlobalColor.yellow)
                table.setItem(row, col, item)
                prev_text = text
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(table, 1)

        close_btn = PushButton(self._translator.translate("common.close"))
        close_btn.clicked.connect(dialog.accept)
        v.addWidget(close_btn)
        dialog.exec()

    def _fill_context_fields(self, result) -> None:
        """Điền kết quả phân tích vào các ô ngữ cảnh (dùng chung cho phân tích & nạp lại)."""
        if result.source_lang:
            self._source_lang_edit.setText(result.source_lang)
        if result.characters:
            self._characters_edit.setPlainText(result.characters)
        if result.overview:
            self._overview_edit.setPlainText(result.overview)
        if getattr(result, "glossary", ""):
            self._glossary_edit.setPlainText(result.glossary)
        if getattr(result, "visual_cues", ""):
            self._visual_cues_edit.setPlainText(result.visual_cues)

    def _on_analyze_context_ready(self, result) -> None:
        """Điền kết quả phân tích toàn diện vào các trường ngữ cảnh."""
        self._fill_context_fields(result)
        char_count = (
            len([ln for ln in result.characters.splitlines() if ln.strip()])
            if result.characters else 0
        )
        self._show_success(
            "Phân tích hoàn tất",
            f"Ngôn ngữ gốc: '{result.source_lang}' · {char_count} nhân vật · "
            f"Tóm tắt {len(result.overview)} ký tự. Kiểm tra và chỉnh sửa trước khi dịch.",
        )

    def _on_analysis_restored(self, result) -> None:
        """[v3.23.45/48] Điền lại ngữ cảnh + kết quả giai đoạn đã lưu khi mở lại video."""
        self._fill_context_fields(result)
        has_cues = bool(getattr(result, "visual_cues", ""))
        n_stages = len(self._view_model.stage_comparison())
        extra = (
            f" Khôi phục {n_stages} giai đoạn dịch đã lưu — có thể So sánh ngay."
            if n_stages >= 2 else ""
        )
        self._show_info(
            self._translator.translate("translate.toast_restored_title"),
            self._translator.translate("translate.toast_restored_body").replace(
                "{cues}",
                self._translator.translate("translate.toast_restored_cues")
                if has_cues
                else "",
            )
            + extra,
        )
        self._update_action_states()

    def _on_clear_cache_clicked(self) -> None:
        # [v3.23.112] Xoá checkpoint = mất khả năng tiếp tục bản dịch dở -> xác nhận.
        confirm = QMessageBox.question(
            self, self._translator.translate("translate.cache_delete_title"),
            self._translator.translate("translate.cache_delete_body"),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        stages = self._collect_enabled_stages()
        context = TranslationContext(
            target_lang=self._lang_combo.currentText().strip() or "Vietnamese"
        )
        self._view_model.clear_translation_checkpoint(
            self._api_key_edit.text().strip(),
            self._retry_spin.value(),
            stages, context,
        )

    def _on_clear_cloud_clicked(self) -> None:
        video_path = self._source_video_path
        if not video_path:
            self._show_warning(
                "Chưa có video",
                "Chưa xác định được video nguồn để xoá file cloud. Hãy mở video và "
                "phân tích ngữ cảnh trước.",
            )
            return
        api_key = self._api_key_edit.text().strip()
        if not api_key:
            self._show_warning(self._translator.translate("translate.warn_no_apikey_title"), self._translator.translate("translate.warn_apikey_cloud"))
            return
        # [v3.23.112] Xoá file cloud không hoàn tác (phải upload lại) -> xác nhận.
        confirm = QMessageBox.question(
            self, self._translator.translate("translate.cloud_delete_title"),
            self._translator.translate("translate.cloud_delete_body"),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._view_model.delete_cloud_files_for_video(video_path, api_key)

    def _update_action_states(self) -> None:
        has_source = self._view_model.has_source
        has_result = self._view_model.has_result
        busy = self._view_model.is_busy
        has_sel = bool(self._table.selectedItems())
        self._btn_run.setEnabled(has_source and not busy)
        self._btn_analyze.setEnabled(has_source and not busy)
        self._btn_export_srt.setEnabled(has_result and not busy)
        self._btn_export_ass.setEnabled(has_result and not busy)
        self._btn_export_bilingual.setEnabled(has_result and not busy)
        self._btn_copy_selection.setEnabled(has_result and has_sel and not busy)
        self._btn_send_to_editor.setEnabled(has_result and not busy)
        # [v3.23.47] Chỉ bật So sánh khi có ≥2 giai đoạn được lưu.
        has_stages = bool(self._view_model.stage_comparison())
        self._btn_compare_stages.setEnabled(has_stages and not busy)
        # [v3.23.54] Kiểm tra thuật ngữ: bật khi có bản dịch + có glossary.
        has_glossary = bool(self._glossary_edit.toPlainText().strip())
        self._btn_check_glossary.setEnabled(has_result and has_glossary and not busy)

    def _on_status_message(self, message: str) -> None:
        self._stage_label.setText(message)

    def _on_analyze_error(self, message: str) -> None:
        """[fix B6] Phân biệt lỗi phân tích ngữ cảnh với lỗi dịch."""
        self._show_warning(self._translator.translate("translate.warn_analyze_error_title"), message)
        self._stage_label.setText(self._translator.translate("translate.analyze_failed"))

    def _on_resume_detected(self, stage_name: str) -> None:
        """[Q3] Thông báo khi checkpoint được tìm thấy và resume bắt đầu."""
        self._show_info(
            self._translator.translate("translate.toast_resume_title"),
            self._translator.translate("translate.toast_resume_body").replace("{stage}", stage_name),
        )

    def _on_copy_selection_clicked(self) -> None:
        """[U1] Sao chép bản dịch của các dòng đang chọn vào clipboard."""
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if not rows:
            return
        lines: list[str] = []
        for row in rows:
            item = self._table.item(row, 3)  # cột Bản dịch
            if item is not None:
                lines.append(item.text())
        if lines:
            QApplication.clipboard().setText("\n".join(lines))
            self._show_info(self._translator.translate("translate.toast_copied_title"), self._translator.translate("translate.toast_copied_body").replace("{n}", str(len(lines))))

    def _on_table_context_menu(self, pos) -> None:
        """Context menu chuột phải trên bảng kết quả."""
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if not rows:
            return
        menu = QMenu(self)

        # [v3.23.46] Dịch lại các dòng đã chọn (chuyên biệt cho trang Dịch).
        retranslate_action = QAction(f"🔄 Dịch lại {len(rows)} dòng đã chọn", self)
        retranslate_action.triggered.connect(lambda: self._on_retranslate_rows(rows))
        menu.addAction(retranslate_action)

        # [v3.23.46] Sửa bản dịch tại chỗ (chỉ khi chọn đúng 1 dòng).
        if len(rows) == 1:
            edit_action = QAction("✏️ Sửa bản dịch dòng này…", self)
            edit_action.triggered.connect(lambda: self._on_edit_translation_row(rows[0]))
            menu.addAction(edit_action)
        menu.addSeparator()

        copy_action = QAction(f"📋 Sao chép bản dịch ({len(rows)} dòng)", self)
        copy_action.triggered.connect(self._on_copy_selection_clicked)
        copy_src_action = QAction(f"📄 Sao chép bản gốc ({len(rows)} dòng)", self)
        copy_src_action.triggered.connect(lambda: self._copy_column(rows, col=2))
        menu.addAction(copy_action)
        menu.addAction(copy_src_action)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _on_retranslate_rows(self, rows: list[int]) -> None:
        """[v3.23.46] Dịch lại các dòng đã chọn bằng cấu hình hiện tại trên trang."""
        api_key = self._combined_keys_text()
        if not api_key:
            self._show_warning(
                "Thiếu API Key", "Hãy dán API Key Gemini trước khi dịch lại."
            )
            return
        stages = self._collect_enabled_stages()
        if not stages:
            self._show_warning(
                "Chưa bật giai đoạn dịch", "Hãy bật ít nhất một giai đoạn để dịch lại."
            )
            return
        context = TranslationContext(
            target_lang=self._lang_combo.currentText().strip() or "Vietnamese",
            source_lang=self._source_lang_edit.text().strip(),
            overview=self._overview_edit.toPlainText().strip(),
            characters=self._characters_edit.toPlainText().strip(),
            glossary=self._glossary_edit.toPlainText().strip(),
            visual_cues=self._visual_cues_edit.toPlainText().strip(),
            enable_tags=self._tags_check.isChecked(),
            include_desc=self._desc_check.isChecked(),
        )
        self._view_model.retranslate_lines(
            api_key=api_key, retry_count=self._retry_spin.value(),
            line_indices=rows, stages=stages, context=context,
        )

    def _on_edit_translation_row(self, row: int) -> None:
        """[v3.23.46] Mở hộp thoại sửa bản dịch của một dòng, lưu tại chỗ."""
        current_item = self._table.item(row, 3)
        current = current_item.text() if current_item else ""
        new_text, ok = QInputDialog.getMultiLineText(
            self, "Sửa bản dịch", f"Bản dịch dòng #{row + 1}:", current
        )
        if not ok:
            return
        new_text = new_text.strip()
        if self._view_model.update_translation_text(row, new_text):
            # Cập nhật ngay ô trên bảng + ô chi tiết.
            self._table.setItem(row, 3, QTableWidgetItem(new_text))
            self._detail_translation.setPlainText(new_text)
            self._show_success(self._translator.translate("translate.toast_saved_title"), self._translator.translate("translate.toast_line_updated").replace("{n}", str(row + 1)))

    def _copy_column(self, rows: list[int], col: int) -> None:
        lines = [
            (self._table.item(r, col) or QTableWidgetItem("")).text()
            for r in rows
        ]
        QApplication.clipboard().setText("\n".join(lines))
        label = "bản gốc" if col == 2 else "bản dịch"
        self._show_info(self._translator.translate("translate.toast_copied_title"), self._translator.translate("translate.toast_copied_label_body").replace("{n}", str(len(lines))).replace("{label}", label))

    def _show_success(self, title: str, content: str) -> None:
        _feedback.show_success(self, title, content)

    def _show_info(self, title: str, content: str) -> None:
        _feedback.show_info(self, title, content)

    def _show_warning(self, title: str, content: str) -> None:
        _feedback.show_warning(self, title, content)


__all__ = ["TranslatePage"]
