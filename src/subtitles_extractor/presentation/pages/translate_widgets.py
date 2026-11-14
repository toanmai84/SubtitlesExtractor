"""[v3.23.694] Thành phần dùng riêng của trang Dịch: ô cấu hình từng giai đoạn, nạp danh sách model, đọc bảng hạn mức.

Tách nguyên văn khỏi ``translate_page.py`` (U6) — hành vi không đổi.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.presentation.pages.translate_constants import (
    _DEFAULT_MODELS,
    _STYLE_PRESETS,
    EDITS_ONLY_DEFAULT,
)
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.widgets.section_card import SectionCard


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
        default_thinking: bool = False,
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
        apply_model_tooltips(self.model_combo)
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

        # [v3.23.681] Tinh chỉnh / bản địa hoá: AI chỉ trả dòng cần sửa (ít token, ít viết lại vô cớ).
        self.edits_only_check: QCheckBox | None = None
        if show_style or show_locale:
            self.edits_only_check = QCheckBox(self._translator.translate("translate.stage_edits_only"))
            self.edits_only_check.setToolTip(self._translator.translate("translate.stage_edits_only_tip"))
            self.edits_only_check.setChecked(EDITS_ONLY_DEFAULT)
            form.addRow(self.edits_only_check)

        self.retime_check: QCheckBox | None = None
        if show_retime:
            self.retime_check = QCheckBox(self._translator.translate("translate.stage_allow_calib"))
            form.addRow(self.retime_check)

        # ── Thinking (Gemini 3.x / 2.5.x) ────────────────────────────────
        self.thinking_check = QCheckBox(self._translator.translate("translate.stage_thinking"))
        self.thinking_check.setToolTip(
            "Model suy nghĩ trước khi trả lời → cải thiện chất lượng dịch\n"
            "Đặc biệt hữu ích cho STYLE và LOCALIZE (phức tạp về ngữ nghĩa)\n"
            "Hỗ trợ: gemini-3.8-flash, gemini-3.6-flash, "
            "gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-2.5-flash\n"
            "Bỏ chọn: suy nghĩ ở mức THẤP NHẤT model cho phép (Gemini 3 và 2.5 Pro "
            "không tắt hẳn được) — nhanh và ít token nhất."
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
            "Mức suy nghĩ cho Gemini 3.x (3.8-flash, 3.6-flash, 3.5-flash-lite, 3.1-flash-lite…).\n"
            "Cao hơn = dịch sát ngữ cảnh/xưng hô tốt hơn nhưng chậm & tốn token hơn.\n"
            "Chỉ áp dụng Gemini 3.x; Gemini 2.5 dùng Thinking Budget ở trên."
        )
        self.thinking_level_combo.setEnabled(False)
        form.addRow(self._translator.translate("translate.tr_thinking"), self.thinking_level_combo)
        self.thinking_check.toggled.connect(self.thinking_level_combo.setEnabled)
        self.thinking_check.setChecked(default_thinking)

        self.add_widget(self._content)
        self._content.setEnabled(self.enable_check.isChecked())
        self.enable_check.toggled.connect(self._content.setEnabled)

    def set_model_items(self, models: list[str]) -> None:
        """Cập nhật danh sách model trong combo, GIỮ model đang chọn."""
        replace_model_items(self.model_combo, models)

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
            edits_only=bool(self.edits_only_check is not None and self.edits_only_check.isChecked()),
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
        if self.edits_only_check is not None:
            settings.setValue(f"{prefix}/edits_only", self.edits_only_check.isChecked())
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
        if self.edits_only_check is not None and settings.contains(f"{prefix}/edits_only"):
            self.edits_only_check.setChecked(settings.value(f"{prefix}/edits_only", type=bool))
        if settings.contains(f"{prefix}/thinking"):
            self.thinking_check.setChecked(settings.value(f"{prefix}/thinking", type=bool))
        if settings.contains(f"{prefix}/thinking_budget"):
            self.thinking_budget_spin.setValue(settings.value(f"{prefix}/thinking_budget", type=int))
        if settings.contains(f"{prefix}/thinking_level"):
            saved_level = settings.value(f"{prefix}/thinking_level", "low", type=str)
            idx = self.thinking_level_combo.findData(saved_level)
            if idx >= 0:
                self.thinking_level_combo.setCurrentIndex(idx)


def apply_model_tooltips(combo: QComboBox) -> None:
    """[v3.23.724] Chú thích từng model trong ô chọn: hạn mức bản miễn phí, hoặc "chỉ dùng với key trả phí"."""
    from PySide6.QtCore import Qt

    from subtitles_extractor.infrastructure.translation.gemini_model_catalog import describe

    for index in range(combo.count()):
        combo.setItemData(index, describe(combo.itemText(index)), Qt.ItemDataRole.ToolTipRole)


def replace_model_items(combo: QComboBox, models: list[str]) -> None:
    """Thay danh sách model của một ô chọn model mà KHÔNG đổi model đang chọn.

    [v3.23.609] Trước đây model đang chọn không có trong danh sách API trả về (model gõ
    tay, bản preview, model API ẩn khỏi danh sách miễn phí) bị đổi ngầm sang model đầu
    danh sách — đổi luôn hạn mức dùng và làm bộ nhớ đệm giai đoạn coi như cấu hình mới.

    Args:
        combo: Ô chọn model (``setEditable(True)``).
        models: Danh sách model mới.
    """
    current = combo.currentText().strip()
    combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems(models)
        apply_model_tooltips(combo)
        if current:
            index = combo.findText(current)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setEditText(current)
        elif models:
            combo.setCurrentIndex(0)
    finally:
        combo.blockSignals(False)


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


__all__ = ['_StagePanel', 'replace_model_items', 'parse_quota_rows']
