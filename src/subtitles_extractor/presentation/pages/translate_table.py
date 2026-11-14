"""[v3.23.694] Bảng kết quả dịch: chọn dòng, sao chép, menu chuột phải, dịch lại/sửa dòng, đánh dấu dòng đã sửa, gửi sang Chỉnh sửa.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/translate_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu, QTableWidgetItem

from subtitles_extractor.presentation.pages.translate_constants import (
    _RESIZE_ROWS_THRESHOLD,
)

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")



class TranslateTableMixin:
    """Bảng kết quả dịch: chọn dòng, sao chép, menu chuột phải, dịch lại/sửa dòng, đánh dấu dòng đã sửa, gửi sang Chỉnh sửa."""

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

        # [v3.23.609] Đang dịch thì bản dịch trên bảng sắp bị thay bằng kết quả mới — sửa
        # hay dịch lại dòng lúc này đều mất (trước đây vẫn cho làm, rồi mất không báo).
        busy = self._view_model.is_busy
        # [v3.23.46] Dịch lại các dòng đã chọn (chuyên biệt cho trang Dịch).
        retranslate_action = QAction(f"🔄 Dịch lại {len(rows)} dòng đã chọn", self)
        retranslate_action.triggered.connect(lambda: self._on_retranslate_rows(rows))
        retranslate_action.setEnabled(not busy)
        menu.addAction(retranslate_action)

        # [v3.23.46] Sửa bản dịch tại chỗ (chỉ khi chọn đúng 1 dòng).
        if len(rows) == 1:
            edit_action = QAction("✏️ Sửa bản dịch dòng này…", self)
            edit_action.triggered.connect(lambda: self._on_edit_translation_row(rows[0]))
            edit_action.setEnabled(not busy)
            menu.addAction(edit_action)
        # [v3.23.609] Dòng sửa tay được giữ khi dịch lại — cho cách trả về bản máy dịch.
        edited = sorted(set(rows) & self._view_model.final_edit_positions())
        if edited:
            discard_action = QAction(
                f"↩ Bỏ sửa tay ({len(edited)} dòng) — dùng bản máy dịch", self
            )
            discard_action.triggered.connect(
                lambda: self._view_model.discard_final_edits(edited)
            )
            discard_action.setEnabled(not busy)
            menu.addAction(discard_action)
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
        if not self._visual_cues_valid():
            return
        if self._warn_custom_incomplete():
            return
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
        self._view_model.retranslate_lines(
            api_key=api_key, retry_count=None,
            line_indices=rows, stages=stages,
            context=self._current_translation_context(),
        )

    def _on_edit_translation_row(self, row: int) -> None:
        """[v3.23.46] Mở hộp thoại sửa bản dịch của một dòng, lưu tại chỗ."""
        if self._view_model.is_busy:
            return
        current_item = self._table.item(row, 3)
        current = current_item.text() if current_item else ""
        new_text, ok = QInputDialog.getMultiLineText(
            self, "Sửa bản dịch", f"Bản dịch dòng #{row + 1}:", current
        )
        if not ok:
            return
        new_text = new_text.strip()
        if self._view_model.update_translation_text(
            row, new_text, target_lang=self._lang_combo.currentText().strip()
        ):
            # Cập nhật ngay ô trên bảng + ô chi tiết.
            self._table.setItem(row, 3, QTableWidgetItem(new_text))
            self._mark_edited_row(row, row in self._view_model.final_edit_positions())
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

    def _mark_edited_row(self, row: int, edited: bool) -> None:
        """[v3.23.609] Đánh dấu dòng đã sửa tay trên bảng kết quả (giữ khi dịch lại)."""
        number = self._table.item(row, 0)
        translation = self._table.item(row, 3)
        if number is None or translation is None:
            return
        number.setText(f"{row + 1} ✏" if edited else str(row + 1))
        translation.setToolTip(
            "Đã sửa tay — được giữ nguyên khi bấm Dịch lại. Muốn dùng lại bản máy dịch: "
            "chuột phải → Bỏ sửa tay."
            if edited
            else ""
        )

    def _on_send_to_editor_clicked(self) -> None:
        """Gửi toàn bộ bản dịch sang trang Biên tập."""
        ok = self._view_model.push_to_editor()
        if ok:
            self._show_success(
                "Đã gửi sang Biên tập",
                f"Đã chuyển {len(self._view_model.translated_events)} dòng. "
                "Hãy chuyển sang trang Biên tập để chỉnh sửa.",
            )


__all__ = ["TranslateTableMixin"]
