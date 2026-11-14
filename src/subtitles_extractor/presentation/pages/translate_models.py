"""[v3.23.694] Chọn model trên trang Dịch: tải danh sách, model dự phòng, model tự kiểm tra, nhà cung cấp tuỳ chỉnh.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/translate_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import json
import logging

from PySide6.QtCore import Qt

from subtitles_extractor.infrastructure.translation.gemini_model_catalog import describe
from subtitles_extractor.presentation.pages.translate_constants import NO_FALLBACK_CHOICE
from subtitles_extractor.presentation.pages.translate_widgets import replace_model_items

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")



class TranslateModelsMixin:
    """Chọn model trên trang Dịch: tải danh sách, model dự phòng, model tự kiểm tra, nhà cung cấp tuỳ chỉnh."""

    def _on_refresh_models_clicked(self) -> None:
        """[fix B4/U2] Validate API key + disable button + hiển thị trạng thái."""
        if self._warn_custom_incomplete():
            return
        # [v3.23.698] Gemini: gửi MỌI key (bộ dịch thử lần lượt) — trước đây chỉ key đầu, key đó hỏng/bị cách ly thì
        # "Tải danh sách model" báo lỗi dù các key khác vẫn tốt.
        api_key = self._combined_keys_text() if self._custom_mode() else "\n".join(self._all_keys())
        if not api_key:
            self._show_warning(self._translator.translate("translate.warn_no_apikey_title"), self._translator.translate("translate.warn_apikey_models"))
            return
        self._btn_refresh_models.setEnabled(False)
        self._btn_refresh_models.setText(self._translator.translate("translate.status_loading_model"))
        self._models_request_custom = self._custom_mode()   # [v3.23.698] danh sách này thuộc nhà cung cấp nào
        self._view_model.fetch_available_models(api_key)

    def _fallback_selection(self) -> str:
        """Model dự phòng đang chọn trong ô chọn: ``""`` tự động hoặc tên model."""
        data = self._fallback_model_combo.currentData()
        return str(data) if data is not None else ""

    def _fallback_choice(self) -> str:
        """Lựa chọn dự phòng có hiệu lực: ``"-"`` khi đã tắt, còn lại như ô chọn."""
        check = getattr(self, "_fallback_enabled_check", None)
        if check is not None and not check.isChecked():
            return NO_FALLBACK_CHOICE
        return self._fallback_selection()

    def _on_fallback_enabled_toggled(self, checked: bool) -> None:
        """[v3.23.667] Bật/tắt chuyển model dự phòng."""
        self._fallback_model_combo.setEnabled(checked)
        self._view_model.set_fallback_model(self._fallback_choice())
        self._persist_inputs()

    def _fill_fallback_combo(self, models: list[str], selected: str) -> None:
        """[v3.23.638] Dựng lại danh sách model dự phòng và chọn ``selected`` (giữ cả model gõ tay
        chưa có trong danh sách API), rồi báo cho view-model."""
        combo = self._fallback_model_combo
        names = [m for m in dict.fromkeys(models) if m]
        if selected == NO_FALLBACK_CHOICE:
            selected = ""
        if selected and selected not in names:
            names.append(selected)
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(self._translator.translate("translate.fallback_auto"), "")
            for name in names:
                combo.addItem(name, name)
                combo.setItemData(combo.count() - 1, describe(name), Qt.ItemDataRole.ToolTipRole)
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)
        self._view_model.set_fallback_model(self._fallback_choice())

    def _audit_choice(self) -> str:
        """Model tự kiểm tra đang chọn: ``""`` = tự động (mặc định rẻ), hoặc tên model."""
        data = self._audit_model_combo.currentData()
        return str(data) if data is not None else ""

    def _fill_audit_combo(self, models: list[str], selected: str) -> None:
        """[v3.23.640] Dựng lại danh sách model tự kiểm tra (giữ cả model gõ tay) rồi báo view-model."""
        combo = self._audit_model_combo
        names = [m for m in dict.fromkeys(models) if m]
        if selected and selected not in names:
            names.append(selected)
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(self._translator.translate("translate.audit_auto"), "")
            for name in names:
                combo.addItem(name, name)
                combo.setItemData(combo.count() - 1, describe(name), Qt.ItemDataRole.ToolTipRole)
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)
        self._view_model.set_audit_model(self._audit_choice())

    def _on_audit_model_changed(self, _index: int) -> None:
        self._view_model.set_audit_model(self._audit_choice())
        self._persist_inputs()

    def _on_fallback_model_changed(self, _index: int) -> None:
        self._view_model.set_fallback_model(self._fallback_choice())
        self._persist_inputs()

    def _on_models_ready(self, models: list) -> None:
        """[fix U2] Khôi phục button + cập nhật combo model."""
        # Luôn khôi phục button dù models có hay không
        self._btn_refresh_models.setEnabled(True)
        self._btn_refresh_models.setText(self._translator.translate("translate.btn_load_models"))
        if not models:
            return
        requested_custom = getattr(self, "_models_request_custom", None)
        if requested_custom is not None and requested_custom != self._custom_mode():
            # [v3.23.698] Đổi nhà cung cấp trong lúc đang tải: danh sách của nhà cung cấp cũ không được đổ vào ô chọn.
            logger.info("Bỏ danh sách model đến muộn của nhà cung cấp đã đổi (%d model).", len(models))
            return
        for panel in (self._stage_preprocess, self._stage_literal,
                      self._stage_style, self._stage_localize):
            panel.set_model_items(models)
        # Cập nhật cả combo model phân tích ngữ cảnh (giữ model đang chọn).
        replace_model_items(self._analyze_model_combo, models)
        self._fill_fallback_combo(models, self._fallback_selection())
        self._fill_audit_combo(models, self._audit_choice())

        self._show_success(
            "Danh sách model cập nhật",
            f"Đã tải {len(models)} model từ "
            f"{'nhà cung cấp tuỳ chỉnh' if self._custom_mode() else 'Gemini API'}.",
        )

    def _custom_mode(self) -> bool:
        """Người dùng đang chọn nhà cung cấp tuỳ chỉnh."""
        store = getattr(self, "_ai_store", None)
        return store is not None and store.is_custom()

    def _custom_provider_ready(self) -> bool:
        """Đang chọn "Tuỳ chỉnh" VÀ đủ địa chỉ + key (đã áp xuống bộ dịch)."""
        return self._custom_mode() and self._view_model.custom_provider_active()

    def _sync_custom_provider(self) -> None:
        """Áp lựa chọn nhà cung cấp xuống view-model / container và cập nhật giao diện."""
        custom = self._custom_mode()
        was_custom = getattr(self, "_synced_custom", None)
        self._synced_custom = custom
        switched = was_custom is not None and was_custom != custom
        if switched:
            self._swap_provider_models(bool(was_custom), custom)
        self._view_model.set_custom_provider(custom, self._ai_store.custom_url(), self._ai_store.custom_key())
        # Đính video cần Gemini xem video (nhà cung cấp tuỳ chỉnh chỉ nhận chữ/ảnh).
        attach = getattr(self, "_attach_video_check", None)
        if attach is not None:
            if custom and attach.isChecked():
                # Bỏ tick qua tín hiệu thường để các ô phụ thuộc video (Visual Cues quét video, tag
                # cảm xúc, giai đoạn đính video) cũng tắt theo — không còn ô "tick mà xám".
                self._attach_video_before_custom = True
                attach.setChecked(False)
            attach.setEnabled(not custom)
            # [v3.23.700] Quay lại Gemini: bật lại ô đính video đã bị tự bỏ tick (nếu video của phim vẫn còn).
            if switched and not custom and getattr(self, "_attach_video_before_custom", False):
                self._attach_video_before_custom = False
                if getattr(self, "_video_path", ""):
                    attach.setChecked(True)
        button = getattr(self, "_btn_refresh_models", None)
        if button is not None:
            button.setToolTip(self._translator.translate(
                "translate.load_models_custom_tip" if custom else "translate.load_models_tip"))

    # ── [v3.23.700] Model đã chọn nhớ RIÊNG cho từng nhà cung cấp ──
    def _model_combos(self) -> list:
        return [panel.model_combo for panel in (self._stage_preprocess, self._stage_literal,
                                                 self._stage_style, self._stage_localize)]

    def _model_snapshot(self) -> dict:
        return {
            "stages": [combo.currentText().strip() for combo in self._model_combos()],
            "analyze": self._analyze_model_combo.currentText().strip(),
            "fallback": self._fallback_selection(),
            "audit": self._audit_choice(),
        }

    def _apply_model_snapshot(self, snap: dict) -> None:
        def pick(combo, name: str) -> None:
            if not name:
                return
            index = combo.findText(name)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setEditText(name)

        for combo, name in zip(self._model_combos(), snap.get("stages") or [], strict=False):
            pick(combo, str(name or ""))
        pick(self._analyze_model_combo, str(snap.get("analyze") or ""))
        for combo, fill, key in ((self._fallback_model_combo, self._fill_fallback_combo, "fallback"),
                                 (self._audit_model_combo, self._fill_audit_combo, "audit")):
            items = [str(combo.itemData(i)) for i in range(combo.count()) if combo.itemData(i)]
            fill(items, str(snap.get(key) or ""))

    def _swap_provider_models(self, was_custom: bool, custom: bool) -> None:
        """Đổi nhà cung cấp: cất model đang chọn cho nhà cung cấp cũ, khôi phục model đã cất của nhà cung cấp mới.

        Trước đây ô chọn giữ nguyên tên model của nhà cung cấp cũ: chuyển sang tuỳ chỉnh vẫn gửi ``gemini-…`` (404 ở
        đa số nhà cung cấp); quay lại Gemini vẫn gửi tên model tuỳ chỉnh (vd ``gpt-4o-mini``) cho Gemini.
        """

        def key(is_custom: bool) -> str:
            return "models_by_provider/" + ("custom" if is_custom else "gemini")

        settings = self._settings
        try:
            settings.setValue(key(was_custom), json.dumps(self._model_snapshot(), ensure_ascii=False))
            raw = settings.value(key(custom), "", type=str) or ""
            snap = json.loads(raw) if raw else None
        except (TypeError, ValueError) as exc:
            logger.warning("Không đọc/ghi được model theo nhà cung cấp: %s", exc)
            return
        if isinstance(snap, dict):
            self._apply_model_snapshot(snap)
            logger.info("Đổi nhà cung cấp → khôi phục model đã chọn trước đó cho %s.",
                        "nhà cung cấp tuỳ chỉnh" if custom else "Gemini")
        persist = getattr(self, "_persist_inputs", None)
        if callable(persist):
            persist()

    def _warn_custom_incomplete(self) -> bool:
        """Đang chọn "Tuỳ chỉnh" mà thiếu địa chỉ/key → cảnh báo, trả True (caller dừng)."""
        if not self._custom_mode() or self._custom_provider_ready():
            return False
        self._show_warning(
            self._translator.translate("translate.warn_custom_incomplete_t"),
            self._translator.translate("translate.warn_custom_incomplete_b"),
        )
        self._open_ai_settings()
        return True

    def shorten_model(self) -> str:
        """[v3.23.666] Model cho bước rút gọn câu ở trang Lồng tiếng.

        Gemini: model rẻ cố định như trước. Nhà cung cấp tuỳ chỉnh: model phân tích đang chọn — tên
        Gemini cố định có thể không có ở nhà cung cấp (404).
        """
        if self._custom_mode():
            chosen = self._analyze_model_combo.currentText().strip()
            if chosen:
                return chosen
        return "gemini-3.1-flash-lite"


__all__ = ["TranslateModelsMixin"]
