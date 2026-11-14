"""[v3.23.691] Trang Dịch — Visual Cues (người nói / người nghe / nhãn cảnh từng câu): soát lại, bảng sửa tay, áp lựa chọn của người dùng.

Tách nguyên văn khỏi ``translate_page.py`` (U6, như ``translate_batch`` ở 3.23.689). Lớp trộn chỉ chứa phương thức; widget,
tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")


class TranslateCuesMixin:
    """Visual Cues (người nói / người nghe / nhãn cảnh từng câu): soát lại, bảng sửa tay, áp lựa chọn của người dùng."""

    def _cue_source_rows(self) -> list[tuple[float, str]]:
        return [
            (float(event.start_sec), str(getattr(event, "text", "") or ""))
            for event in self._view_model.source_events
        ]

    def _on_cue_recheck_clicked(self) -> bool:
        """[v3.23.655] Hỏi lại AI chỉ các dòng Visual Cues bị chấm nghi."""
        if not self._visual_cues_valid():
            return False
        if self._warn_custom_incomplete():
            return False
        api_key = self._combined_keys_text()
        if not api_key:
            self._show_warning(
                self._translator.translate("translate.warn_no_apikey_title"),
                self._translator.translate("translate.warn_apikey_analyze"),
            )
            return False
        video_path = self._selected_video_path()
        if not video_path:
            self._show_warning(
                self._translator.translate("translate.toast_speakers_title"),
                self._translator.translate("translate.tip_complete_speakers_need_video"),
            )
            return False
        self._flush_context_save()
        return self._view_model.start_cue_recheck(
            api_key=api_key,
            model_name=self._analyze_model_combo.currentText().strip() or "gemini-3.1-flash-lite",
            video_path=video_path,
            visual_cues=self._visual_cues_edit.toPlainText().strip(),
            characters=self._characters_edit.toPlainText().strip(),
            target_lang=self._lang_combo.currentText() or "Vietnamese",
            session_path=self._session_video_path(),
        )

    def refresh_cue_issues(self) -> int:
        """[v3.23.654] Đếm dòng Visual Cues đáng nghi (hiện cạnh nút "Bảng gợi ý"); -1 nếu chưa có cues/JSON hỏng."""
        from subtitles_extractor.application.services.cue_review import review_cues
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
            visual_cues_syntax_error,
        )

        raw = self._visual_cues_edit.toPlainText()
        rows = self._cue_source_rows()
        if not raw.strip() or not rows or visual_cues_syntax_error(raw) is not None:
            self._cue_issue_label.setText("")
            self._btn_cue_recheck.setEnabled(False)
            return -1
        issues = review_cues(deserialize_visual_cues(raw), [text for _start, text in rows],
                             self._characters_edit.toPlainText())
        self._cue_issue_label.setText(
            self._translator.translate("translate.cue_issue_count").replace("{count}", str(len(issues)))
            if issues else self._translator.translate("translate.cue_issue_none")
        )
        self._btn_cue_recheck.setEnabled(bool(issues) and not self._view_model.is_busy)
        return len(issues)

    def _open_cue_table(self) -> bool:
        """[v3.23.654] Mở bảng sửa Visual Cues; OK thì ghi lại ô cues và lưu phiên."""
        from subtitles_extractor.presentation.widgets.cue_table_dialog import (
            CueTableDialog,
        )

        if self._view_model.is_busy:
            self._show_warning(
                self._translator.translate("translate.cue_table_title"),
                self._translator.translate("translate.cue_table_busy"),
            )
            return False
        if not self._visual_cues_valid():
            return False
        rows = self._cue_source_rows()
        if not rows:
            self._show_warning(
                self._translator.translate("translate.cue_table_title"),
                self._translator.translate("translate.cue_table_no_source"),
            )
            return False
        translated = [str(getattr(e, "text", "") or "") for e in getattr(self._view_model, "translated_events", []) or []]
        dialog = CueTableDialog(
            self, self._visual_cues_edit.toPlainText(), rows, translated,
            self._characters_edit.toPlainText(), self._translator.translate,
        )
        try:
            accepted = self._exec_cue_dialog(dialog)
            changed = dialog.changed_lines
            merged = dialog.merged_into(self._visual_cues_edit.toPlainText()) if accepted and changed else ""
        finally:
            dialog.deleteLater()  # bảng ~33 nghìn ô: không giữ lại sau khi đóng
        if not accepted or not changed:
            return False
        self._flush_context_save()
        self._visual_cues_edit.setPlainText(merged)
        self._visual_cues_edit.document().setModified(True)
        self._context_edit_path = self._session_video_path()
        self._flush_context_save()
        self.refresh_cue_issues()
        logger.info("Sửa Visual Cues bằng bảng: %d dòng.", len(changed))
        return True

    @staticmethod
    def _exec_cue_dialog(dialog) -> bool:
        from PySide6.QtWidgets import QDialog

        return dialog.exec() == QDialog.DialogCode.Accepted

    def _with_user_cues(self, new_cues: str) -> str:
        """[v3.23.653] ``new_cues`` (kết quả tự động) mà giữ nguyên cue người dùng chọn đang có trong ô."""
        from subtitles_extractor.application.services.visual_cue_serializer import (
            keep_user_cues_json,
            visual_cues_syntax_error,
        )

        current = self._visual_cues_edit.toPlainText()
        if visual_cues_syntax_error(current) is not None:
            return new_cues
        return keep_user_cues_json(new_cues, current)

    def apply_user_speakers(self, speakers: dict[int, str]) -> int:
        """[v3.23.653] Như :meth:`apply_user_cues`, chỉ người nói."""
        return self.apply_user_cues({line_no: (speaker, None) for line_no, speaker in speakers.items()})

    def apply_user_cues(self, corrections: dict) -> int:
        """[v3.23.653] Ghi người nói / [v3.23.654] "nói với ai" người dùng chọn (``id`` câu gốc → (người nói, người
        nghe), ``None`` = giữ nguyên) vào Visual Cues rồi lưu phiên.

        Không tự dịch lại: lần bấm "Dịch" sau, bộ nhớ đệm nhận ra các dòng đổi người nói và chỉ dịch lại lô chứa chúng.

        Returns:
            Số dòng đã đổi; ``-1`` nếu ô cues đang hỏng JSON; ``-2`` nếu đang dịch/phân tích (chưa ghi gì).
        """
        from subtitles_extractor.application.services.visual_cue_serializer import (
            apply_user_cues,
            visual_cues_syntax_error,
        )

        if self._view_model.is_busy:
            return -2
        raw = self._visual_cues_edit.toPlainText()
        if visual_cues_syntax_error(raw) is not None:
            return -1
        new_raw, changed = apply_user_cues(raw, corrections)
        if not changed:
            return 0
        self._flush_context_save()  # sửa tay đang chờ của ô khác được lưu trước
        self._visual_cues_edit.setPlainText(new_raw)
        self._visual_cues_edit.document().setModified(True)
        self._context_edit_path = self._session_video_path()
        self._flush_context_save()
        logger.info("Đồng bộ người nói/người nghe từ trang TTS vào Visual Cues: %d dòng.", len(changed))
        self.refresh_cue_issues()
        return len(changed)

    def _visual_cues_valid(self) -> bool:
        """[v3.23.652] Ô Visual Cues sửa tay phải là JSON hợp lệ trước khi dịch/phân tích (nếu hỏng: báo vị trí lỗi)."""
        from subtitles_extractor.application.services.visual_cue_serializer import (
            visual_cues_syntax_error,
        )

        error = visual_cues_syntax_error(self._visual_cues_edit.toPlainText())
        if error is None:
            return True
        self._show_warning(
            self._translator.translate("translate.warn_cues_invalid_t"),
            self._translator.translate("translate.warn_cues_invalid_b").replace("{error}", error),
        )
        return False


__all__ = ["TranslateCuesMixin"]
