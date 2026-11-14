"""[v3.23.694] Phân tích ngữ cảnh trên trang Dịch: chạy/phân tích lại, điền kết quả, bổ sung Visual Cues, hoàn thiện người nói.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/translate_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import logging

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ANALYSIS_STOP_DAILY_QUOTA,
)

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")



class TranslateAnalysisMixin:
    """Phân tích ngữ cảnh trên trang Dịch: chạy/phân tích lại, điền kết quả, bổ sung Visual Cues, hoàn thiện người nói."""

    def _on_analyze_context_clicked(self) -> None:
        self._sync_text_cues()
        if self._warn_custom_incomplete():
            return
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
            session_path=self._session_video_path(),
        )

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
            self.refresh_cue_issues()  # [v3.23.654]

    def _on_analyze_context_ready(self, result) -> None:
        """Điền kết quả phân tích toàn diện vào các trường ngữ cảnh."""
        new_cues = getattr(result, "visual_cues", "") or ""
        if new_cues:
            # [v3.23.653] Phân tích lại không được đè người nói bạn đã chọn (vd sửa ở trang TTS).
            merged = self._with_user_cues(new_cues)
            if merged != new_cues:
                import dataclasses

                try:
                    result = dataclasses.replace(result, visual_cues=merged)
                except (TypeError, ValueError):
                    pass
        self._fill_context_fields(result)
        char_count = (
            len([ln for ln in result.characters.splitlines() if ln.strip()])
            if result.characters else 0
        )
        # [v3.23.585] Lượt phân tích dừng giữa chừng (hết hạn mức ngày, Gemini quá tải)
        # từng hiện "Phân tích hoàn tất" như một lượt đủ.
        incomplete = self._view_model.incomplete_analysis_message(result)
        if getattr(self, "_batch_analyzing_index", -1) >= 0 and self._batch_analysis_done(
            True, quota=getattr(result, "stop_reason", "") == ANALYSIS_STOP_DAILY_QUOTA
        ):
            return  # [v3.23.669] dịch cả bộ: dịch tiếp ngay, không bật thông báo cho từng tập
        if incomplete:
            self._show_warning("Phân tích CHƯA đủ", incomplete)
            return
        # [v3.23.600] Báo độ phủ người nói ngay (lồng tiếng cần đủ mọi dòng có lời).
        speakers = self._view_model.speaker_coverage_message(
            getattr(result, "visual_cues", "") or ""
        )
        self._show_success(
            "Phân tích hoàn tất",
            f"Ngôn ngữ gốc: '{result.source_lang}' · {char_count} nhân vật · "
            f"Tóm tắt {len(result.overview)} ký tự. "
            + (f"{speakers} " if speakers else "")
            + "Kiểm tra và chỉnh sửa trước khi dịch."
            + (
                f"\n{self._view_model.last_token_report}"
                if self._view_model.last_token_report
                else ""
            ),
        )

    def _on_visual_cues_topped_up(self, visual_cues: str) -> None:
        """[v3.23.559] Hiện Visual Cues đã được quét bổ sung khi dịch vào ô cues.

        Để lần dịch lại/xuất chuẩn đoán kế tiếp dùng bản đã gộp chứ không phải bản thiếu.
        [v3.23.600] Kèm độ phủ người nói ở dòng trạng thái.

        Args:
            visual_cues: JSON Visual Cues đã gộp.
        """
        base = getattr(self, "_cues_run_base", None)
        current = self._visual_cues_edit.toPlainText()
        user_edited = base is not None and current.strip() != base.strip()
        if user_edited:
            from subtitles_extractor.application.services.visual_cue_serializer import (
                merge_cues_three_way,
            )

            visual_cues = merge_cues_three_way(base, current, visual_cues)  # [v3.23.655] giữ sửa tay trong lúc chạy
            logger.info("Gộp Visual Cues của lượt chạy với sửa tay trong lúc chạy (giữ sửa tay).")
        else:
            visual_cues = self._with_user_cues(visual_cues)  # [v3.23.653]
        self._cues_run_base = visual_cues        # lượt gộp kế tiếp của cùng lượt chạy lấy bản này làm gốc
        self._visual_cues_edit.setPlainText(visual_cues)
        if user_edited:
            # setPlainText xoá cờ "đã sửa" → đánh dấu lại để bản ĐÃ GỘP được lưu đè bản chưa gộp view model vừa lưu.
            self._visual_cues_edit.document().setModified(True)
            self._context_edit_path = self._context_edit_path or self._session_video_path()
            self._flush_context_save()
        self.refresh_cue_issues()  # [v3.23.654]
        speakers = self._view_model.speaker_coverage_message(visual_cues)
        if speakers:
            logger.info("Sau khi dịch — %s", speakers)

    def _on_analysis_restored(self, result) -> None:
        """[v3.23.45/48] Điền lại ngữ cảnh + kết quả giai đoạn đã lưu khi mở lại video."""
        self._fill_context_fields(result)
        has_cues = bool(getattr(result, "visual_cues", ""))
        n_stages = self._view_model.stage_output_count
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
        # [v3.23.578] Phân tích cũ không có quy tắc xưng hô → khuyên phân tích lại thay vì
        # để thông báo "không cần phân tích lại" ở trên là lời cuối.
        # [v3.23.601] Phân tích cũ còn dòng thiếu người nói → chỉ cách bổ sung riêng các
        # dòng đó (khâu phân tích/dịch mới tự hỏi lại; phiên khôi phục thì không).
        missing_speakers = self._view_model.missing_speaker_count(
            getattr(result, "visual_cues", "") or ""
        )
        if missing_speakers:
            self._show_warning(
                self._translator.translate("translate.toast_speakers_title"),
                self._translator.translate(
                    "translate.toast_restored_speakers_missing"
                ).replace("{count}", str(missing_speakers)),
            )
        stale_count = self._view_model.roster_without_address_rules(
            getattr(result, "characters", "") or ""
        )
        if stale_count:
            self._show_warning(
                self._translator.translate("translate.toast_restored_no_address_title"),
                self._translator.translate(
                    "translate.toast_restored_no_address_body"
                ).replace("{count}", str(stale_count)),
            )
        self._update_action_states()

    def _on_refine_analysis_clicked(self) -> None:
        """[v3.23.624] Tự kiểm tra & vá phân tích hiện có (cần API key, không cần video)."""
        if not self._visual_cues_valid():
            return
        if self._warn_custom_incomplete():
            return
        api_key = self._combined_keys_text()
        if not api_key:
            self._show_warning(
                self._translator.translate("translate.warn_no_apikey_title"),
                self._translator.translate("translate.warn_apikey_analyze"),
            )
            return
        self._view_model.start_analysis_refinement(
            api_key=api_key,
            target_lang=self._lang_combo.currentText() or "Vietnamese",
            model_name=(
                self._analyze_model_combo.currentText().strip() or "gemini-3.1-flash-lite"
            ),
            source_lang=self._source_lang_edit.text().strip(),
            characters=self._characters_edit.toPlainText().strip(),
            overview=self._overview_edit.toPlainText().strip(),
            glossary=self._glossary_edit.toPlainText().strip(),
            visual_cues=self._visual_cues_edit.toPlainText().strip(),
            session_path=self._session_video_path(),
        )

    def _update_speaker_button(self) -> None:
        """[v3.23.601] Nút "Bổ sung người nói": bật khi còn dòng thiếu người nói."""
        button = getattr(self, "_btn_complete_speakers", None)
        if button is None:
            return
        missing = self._view_model.missing_speaker_count(
            self._visual_cues_edit.toPlainText().strip()
        )
        button.setText(
            self._translator.translate("translate.btn_complete_speakers_n").replace(
                "{count}", str(missing)
            )
            if missing
            else self._translator.translate("translate.btn_complete_speakers")
        )
        button.setToolTip(
            self._translator.translate(
                "translate.tip_complete_speakers"
                if missing
                else "translate.tip_complete_speakers_done"
            )
        )
        button.setEnabled(bool(missing) and not self._view_model.is_busy)

    def _on_complete_speakers_clicked(self) -> None:
        """[v3.23.601] Hỏi lại người nói các dòng còn thiếu (cần API key + video)."""
        if not self._visual_cues_valid():
            return
        if self._warn_custom_incomplete():
            return
        api_key = self._combined_keys_text()
        if not api_key:
            self._show_warning(
                self._translator.translate("translate.warn_no_apikey_title"),
                self._translator.translate("translate.warn_apikey_analyze"),
            )
            return
        video_path = self._selected_video_path()
        if not video_path:
            self._show_warning(
                self._translator.translate("translate.toast_speakers_title"),
                self._translator.translate("translate.tip_complete_speakers_need_video"),
            )
            return
        self._view_model.start_speaker_completion(
            api_key=api_key,
            model_name=(
                self._analyze_model_combo.currentText().strip() or "gemini-3.1-flash-lite"
            ),
            video_path=video_path,
            visual_cues=self._visual_cues_edit.toPlainText().strip(),
            characters=self._characters_edit.toPlainText().strip(),
            target_lang=self._lang_combo.currentText() or "Vietnamese",
            session_path=self._session_video_path(),
        )

    def _on_speaker_completion_done(self, message: str) -> None:
        """[v3.23.601] Báo kết quả bổ sung người nói (cues cập nhật qua tín hiệu riêng)."""
        logger.info("Bổ sung người nói — %s", message)
        self._show_success(
            self._translator.translate("translate.toast_speakers_title"), message
        )
        self._update_speaker_button()

    def _on_analyze_error(self, message: str) -> None:
        """[fix B6] Phân biệt lỗi phân tích ngữ cảnh với lỗi dịch."""
        if getattr(self, "_batch_analyzing_index", -1) >= 0 and self._batch_analysis_done(
            False, message
        ):
            return  # [v3.23.669] dịch cả bộ: không chặn hàng đợi bằng hộp cảnh báo
        self._show_warning(self._translator.translate("translate.warn_analyze_error_title"), message)
        self._stage_label.setText(self._translator.translate("translate.analyze_failed"))


__all__ = ["TranslateAnalysisMixin"]
