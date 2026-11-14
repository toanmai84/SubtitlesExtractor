"""[v3.23.691] Trang Dịch — Xuất phụ đề (SRT/ASS, song ngữ), xuất chẩn đoán dịch / phân tích ngữ cảnh, mở thư mục chứa tệp.

Tách nguyên văn khỏi ``translate_page.py`` (U6, như ``translate_batch`` ở 3.23.689). Lớp trộn chỉ chứa phương thức; widget,
tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import QFileDialog

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationContext
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.presentation.pages.translate_constants import _LANGUAGE_CODES

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")


class TranslateExportMixin:
    """Xuất phụ đề (SRT/ASS, song ngữ), xuất chẩn đoán dịch / phân tích ngữ cảnh, mở thư mục chứa tệp."""

    def _on_export_clicked(self, output_format: SubtitleFormat) -> None:
        if not self._view_model.has_result:
            self._show_warning(self._translator.translate("translate.warn_no_translation_title"), self._translator.translate("translate.warn_translate_first_srt"))
            return
        suffix = output_format.value
        # Đặt tên mặc định theo quy ước: <tên video>.<mã ngôn ngữ>.<srt/ass>.
        default_name = f"translated.{suffix}"
        session_video = self._session_video_path()
        if session_video:
            try:
                from subtitles_extractor.domain.value_objects.output_naming import (
                    SubtitleFormat as NamingFormat,
                )
                from subtitles_extractor.domain.value_objects.output_naming import (
                    translated_subtitle_path,
                )

                lang_code = _LANGUAGE_CODES.get(
                    self._lang_combo.currentText(), "vi"
                )
                default_name = str(translated_subtitle_path(
                    session_video, lang_code, NamingFormat.from_str(suffix)
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
        session_video = self._session_video_path()
        if session_video:
            try:
                default_name = (
                    str(Path(session_video).with_suffix("")) + ".bilingual.srt"
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
        # [v3.23.428] Tên nhất quán có timestamp → không ghi đè lần xuất trước.
        # [v3.23.582] Hộp thoại mở ở thư mục gốc của ứng dụng (trước đây: cạnh file video,
        # khác chỗ với chuẩn đoán phân tích và nhật ký), và KHÔNG tự mở thư mục sau khi lưu.
        from subtitles_extractor.application.services.diagnostic_naming import (
            build_diagnostic_filename,
        )
        from subtitles_extractor.presentation.utils.export_location import (
            export_dialog_path,
        )
        from subtitles_extractor.presentation.utils.safe_dialogs import save_file

        path_str = save_file(
            self,
            "Xuất chẩn đoán dịch",
            export_dialog_path(
                build_diagnostic_filename(
                    self._session_video_path() or None, "translation_debug", "json"
                )
            ),
            "JSON (*.json)",
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

    def _reveal_in_explorer(self, path: "Path") -> None:
        """[v3.23.51] Mở thư mục chứa tệp vừa lưu (đa nền tảng, không chặn UI)."""
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            folder = path.parent if path.is_file() or path.suffix else path
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except (OSError, RuntimeError) as exc:
            logger.debug("Không mở được thư mục chứa tệp: %s", exc)

    def _on_export_context_diagnostics(self) -> None:
        """[v3.23.484] Xuất JSON chuẩn đoán chất lượng ngữ cảnh toàn cục.

        Báo cáo KHÔNG chứa toàn văn phụ đề (chỉ số liệu + trích đoạn ≤280 ký tự) nên an toàn
        để gửi đi phân tích. Dùng để soi: roster có bịa tên không, glossary trùng, visual cues
        lệch chỉ số dòng, tóm tắt đủ dày so với thời lượng phim.
        """
        import json

        from subtitles_extractor import __version__ as _app_version
        from subtitles_extractor.application.services.context_diagnostics import (
            build_context_diagnostics,
        )

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
        if not any((context.overview, context.characters, context.glossary, context.visual_cues)):
            self._show_warning(
                "Chưa có ngữ cảnh",
                "Chưa có dữ liệu ngữ cảnh để chuẩn đoán. Hãy bấm 'Phân tích ngữ cảnh' "
                "(hoặc nhập tay bảng nhân vật/tóm tắt) trước.",
            )
            return

        video_path = self._session_video_path()
        source_events = self._view_model.source_events
        # Thời lượng suy từ mốc kết thúc DÒNG CUỐI (trang dịch không giữ metadata video) —
        # đủ chính xác để đánh giá tóm tắt có đủ dày so với độ dài phim hay không.
        duration_sec = 0.0
        for event in source_events:
            interval = getattr(event, "interval", None)
            end_sec = float(getattr(interval, "end_sec", 0.0) or 0.0)
            duration_sec = max(duration_sec, end_sec)
        diagnostics = build_context_diagnostics(
            context=context,
            source_events=source_events,
            video_path=video_path,
            duration_sec=duration_sec,
            app_version=_app_version,
            name_readings=self._view_model.name_readings,
            analysis_settings={
                "analyze_model": self._analyze_model_combo.currentText().strip(),
                "media_resolution": str(self._media_res_combo.currentData() or ""),
                "thinking_level": str(self._analysis_think_combo.currentData() or ""),
                "native_address": bool(
                    getattr(self, "_native_address_check", None)
                    and self._native_address_check.isChecked()
                ),
                "attach_video": bool(
                    getattr(self, "_attach_video_check", None)
                    and self._attach_video_check.isChecked()
                ),
            },
        )

        # [v3.23.582] Cùng cách xử lý với nút xuất chuẩn đoán dịch và nhật ký: hộp thoại
        # luôn mở ở thư mục gốc của ứng dụng, tên theo quy ước chung (có timestamp).
        from subtitles_extractor.application.services.diagnostic_naming import (
            build_diagnostic_filename,
        )
        from subtitles_extractor.presentation.utils.export_location import (
            export_dialog_path,
        )
        from subtitles_extractor.presentation.utils.safe_dialogs import save_file

        path = save_file(
            self,
            "Lưu chuẩn đoán ngữ cảnh",
            export_dialog_path(
                build_diagnostic_filename(video_path or None, "context_diagnostics", "json")
            ),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self._show_warning("Không lưu được", f"Lỗi ghi file: {exc}")
            return
        counts = diagnostics.get("counts", {})
        self._show_info(
            "Đã xuất chuẩn đoán ngữ cảnh",
            f"Đã lưu: {path}\n\n"
            f"Phát hiện: {counts.get('errors', 0)} lỗi, {counts.get('warnings', 0)} cảnh báo.\n"
            "File chỉ chứa số liệu + trích đoạn ngắn (không có toàn văn phụ đề).",
        )


__all__ = ["TranslateExportMixin"]
