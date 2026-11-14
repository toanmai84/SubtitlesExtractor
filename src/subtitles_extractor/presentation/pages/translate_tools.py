"""[v3.23.694] Công cụ phụ của trang Dịch: bộ nhớ dịch, kiểm thuật ngữ, so sánh giai đoạn, xoá bộ nhớ đệm cục bộ và trên đám mây.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/translate_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from subtitles_extractor.presentation.fluent_compat import PushButton

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")



class TranslateToolsMixin:
    """Công cụ phụ của trang Dịch: bộ nhớ dịch, kiểm thuật ngữ, so sánh giai đoạn, xoá bộ nhớ đệm cục bộ và trên đám mây."""

    def _on_manage_tm_clicked(self) -> None:
        """[v3.23.57] Mở dialog quản lý bộ nhớ dịch phim bộ (xem danh sách + xoá)."""
        from PySide6.QtWidgets import QDialog, QMessageBox

        def _refresh(table) -> list:
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

    def _on_clear_cache_clicked(self) -> None:
        # [v3.23.112] Xoá checkpoint = mất khả năng tiếp tục bản dịch dở -> xác nhận.
        confirm = QMessageBox.question(
            self, self._translator.translate("translate.cache_delete_title"),
            self._translator.translate("translate.cache_delete_body"),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._view_model.clear_translation_checkpoint(**self._stage_cache_settings())

    def _on_clear_cloud_clicked(self) -> None:
        video_path = self._session_video_path()
        if not video_path:
            self._show_warning(
                "Chưa có video",
                "Chưa xác định được video nguồn để xoá file cloud. Hãy mở video và "
                "phân tích ngữ cảnh trước.",
            )
            return
        # [v3.23.609] Đủ MỌI key (chính + dự phòng): video được tải bằng key còn hạn mức,
        # và file Gemini chỉ xoá được bằng key (cùng dự án) đã tải nó.
        # [v3.23.666] Tệp cloud nằm ở Gemini: luôn dùng key Google, KHÔNG BAO GIỜ gửi key của nhà
        # cung cấp tuỳ chỉnh sang Google.
        api_key = "\n".join(self._all_keys())
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

    def _on_stage_cache_clicked(self) -> None:
        """[v3.23.608] Mở hộp thoại Quản lý giai đoạn dịch (xem/sửa/xoá/dùng lại)."""
        from subtitles_extractor.presentation.widgets.stage_cache_dialog import (
            StageCacheDialog,
        )

        if self._view_model.is_busy:
            return
        self._persist_inputs()
        session = self._view_model.open_stage_cache(**self._stage_cache_settings())
        if session is None:
            self._show_warning(
                self._translator.translate("translate.stage_cache_no_source_title"),
                self._translator.translate("translate.stage_cache_no_source_body"),
            )
            return
        StageCacheDialog(session, self).exec()
        self._update_action_states()


__all__ = ["TranslateToolsMixin"]
