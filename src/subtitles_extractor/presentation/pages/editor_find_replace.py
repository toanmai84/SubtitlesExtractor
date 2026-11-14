"""[v3.23.693] Lọc, tìm/thay chữ, tự sửa dòng thời gian và gộp các dòng giống nhau trên trang Chỉnh sửa.

Tách nguyên văn khỏi ``editor_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``EditorPage``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.presentation.fluent_compat import (
    CheckBox,
    DoubleSpinBox,
    InfoBar,
    PushButton,
)
from subtitles_extractor.presentation.utils.text_replace import replace_in_text_safe
from subtitles_extractor.presentation.utils.time_format import seconds_to_display
from subtitles_extractor.presentation.widgets.subtitle_table_model import (
    COL_WARN,
    SubtitleFilterProxyModel,
)


class EditorFindReplaceMixin:
    """Lọc, tìm/thay chữ, tự sửa dòng thời gian và gộp các dòng giống nhau trên trang Chỉnh sửa."""

    def _apply_filters(self) -> None:
        query = self._find_edit.text().strip().lower()
        filter_mode = self._filter_combo.currentIndex()

        self._proxy_model.set_filter_params(query, filter_mode)
        # [v3.23.657] Lọc "Nghi ngờ OCR": cột cảnh báo ghi lý do bằng chữ → nới rộng (về 40 px khi bỏ lọc).
        suspect_mode = filter_mode == SubtitleFilterProxyModel.FILTER_SUSPECT
        self._table.setColumnWidth(COL_WARN, 190 if suspect_mode else 40)

        match_cnt = self._proxy_model.rowCount()
        if query or filter_mode > 0: self._lbl_find_count.setText(self._translator.translate("editor.ed_find_count").replace("{n}", str(match_cnt)))
        else: self._lbl_find_count.setText("")

        has_query = bool(query)
        self._replace_one_button.setEnabled(has_query)
        self._replace_all_button.setEnabled(has_query)

    def _find_next(self) -> None: self._find(forward=True)

    def _find_prev(self) -> None: self._find(forward=False)

    def _find(self, forward: bool = True) -> None:
        term = self._find_edit.text().strip().lower()
        if not term: return

        matches = [i for i, text in enumerate(self._table_model._lower_text_cache) if term in text]
        if not matches: return

        cur = self._current_row()
        if forward:
            nxt = next((m for m in matches if m > cur), matches[0])
        else:
            nxt = next((m for m in reversed(matches) if m < cur), matches[-1])

        source_index = self._table_model.index(nxt, 0)
        proxy_index = self._proxy_model.mapFromSource(source_index)

        if proxy_index.isValid():
            self._table.selectRow(proxy_index.row())
            self._table.scrollTo(proxy_index, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _replace_one(self) -> None:
        term = self._find_edit.text().strip()
        repl = self._replace_edit.text()
        if not term:
            return
        row_index = self._current_row()
        if row_index < 0:
            return

        # [v3.6 bugfix RO-1]: Dùng _table_model nhất quán thay vì current_events.
        # Trước đây: cache từ _table_model + event.text từ current_events → hai nguồn
        # khác nhau có thể bất đồng bộ. Sau fix mutation bug, cache luôn sync với
        # _table_model.events, nên dùng cả hai từ cùng một nguồn.
        model_events = self._table_model.events
        cache = self._table_model._lower_text_cache
        if row_index >= len(model_events) or row_index >= len(cache):
            return

        if term.lower() in cache[row_index]:
            new_text = self._replace_in_text_safe(term, repl, model_events[row_index].text, count=1)
            self._view_model.update_text(row_index, new_text)
            self._find_next()

    @staticmethod
    def _replace_in_text_safe(term: str, replacement: str, source: str, count: int) -> str:
        # [v3.20] Logic đã tách sang module thuần ``utils.text_replace`` (testable).
        return replace_in_text_safe(term, replacement, source, count)

    def _replace_all(self) -> None:
        term = self._find_edit.text().strip()
        repl = self._replace_edit.text()
        if not term:
            return

        # [v3.6 bugfix]: Dùng nhất quán từ _table_model thay vì kết hợp
        # ViewModel + table cache.  Trước đây:
        #   events = self._view_model.current_events   ← từ ViewModel
        #   cache  = self._table_model._lower_text_cache ← từ table model
        # Nếu cache stale (vd do bug mutation cũ), tìm kiếm sẽ miss/hit sai.
        # Sau khi fix mutation bug, cache luôn được rebuild qua set_events().
        # Dùng _table_model.events và cache cùng nguồn → luôn nhất quán.
        model_events = self._table_model.events
        cache = self._table_model._lower_text_cache

        # Safety guard: cache và events phải cùng độ dài (đảm bảo không IndexError).
        if len(cache) != len(model_events):
            # Rebuild cache nếu mất sync (edge case phòng thủ).
            cache = [e.text.lower() for e in model_events]

        term_lower = term.lower()
        replacements: dict[int, str] = {}

        for index, event in enumerate(model_events):
            if term_lower in cache[index]:
                new_text = self._replace_in_text_safe(term, repl, event.text, count=0)
                replacements[index] = new_text

        if replacements:
            self._view_model.batch_replace_text(replacements)
            InfoBar.success(
                self._translator.translate("editor.ib_replace_t"), self._translator.translate("editor.ib_replace_c").replace("{n}", str(len(replacements))),
                parent=self, duration=2000,
            )
        else:
            # [v3.23.115] Trước đây im lặng -> người dùng tưởng nút hỏng. Báo rõ không khớp.
            InfoBar.info(
                self._translator.translate("editor.ib_noreplace_t"), self._translator.translate("editor.ib_noreplace_c").replace("{term}", term),
                parent=self, duration=2500,
            )

    def _on_auto_fix_timeline(self) -> None:
        events = self._view_model.current_events
        if not events:
            return

        # [v3.6 bugfix AF-1]: Đếm đúng điều kiện của auto_fix_timeline:
        # bao gồm cả overlap (gap < 0) VÀ small positive gap (0 < gap < 0.150s).
        # Trước đây chỉ đếm overlap → thông báo số lỗi thấp hơn thực tế.
        error_count = sum(
            1
            for i in range(len(events) - 1)
            if (events[i + 1].start_sec - events[i].end_sec) < 0.150
            and events[i + 1].start_sec >= events[i].start_sec  # bỏ qua fully-inverted
        )
        if error_count == 0:
            InfoBar.info(self._translator.translate("editor.ib_perfect_t"), self._translator.translate("editor.ib_perfect_c"), parent=self, duration=2000)
            return

        reply = QMessageBox.question(
            self, self._translator.translate("editor.dlg_autofix_t"),
            self._translator.translate("editor.dlg_autofix_b").replace("{n}", str(error_count)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes: return

        fixes = self._view_model.auto_fix_timeline()
        if fixes > 0:
            # [v3.23.619] API chung của hai widget video (trước gọi hàm riêng của MPV →
            # AttributeError trên máy không có libmpv, dùng VideoCanvas).
            self._video_widget.schedule_overlay_refresh()
            InfoBar.success(self._translator.translate("editor.ib_autofix_t"), self._translator.translate("editor.ib_autofix_c").replace("{n}", str(fixes)), parent=self, duration=3000)

    def _on_merge_similar_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._translator.translate("editor.ed_win_merge_dup"))
        dlg.resize(400, 200)
        form = QFormLayout(dlg)

        gap_sb = DoubleSpinBox(dlg)
        gap_sb.setRange(0.0, 10.0); gap_sb.setSingleStep(0.1); gap_sb.setValue(0.5); gap_sb.setSuffix(" s")
        sim_sb = DoubleSpinBox(dlg)
        sim_sb.setRange(0.5, 1.0); sim_sb.setSingleStep(0.05); sim_sb.setValue(0.9); sim_sb.setDecimals(2)

        form.addRow(self._translator.translate("editor.ed_row_maxgap"), gap_sb)
        form.addRow(self._translator.translate("editor.ed_row_minsim"), sim_sb)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        # [v3.6 bugfix DLG-4]: deleteLater cho dialog cấu hình. Đọc giá trị
        # trong try, hủy trong finally.
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            gap_value = gap_sb.value()
            sim_value = sim_sb.value()
        finally:
            dlg.deleteLater()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try: groups = self._view_model.find_similar_groups(gap_value, sim_value)
        finally: QApplication.restoreOverrideCursor()

        if not groups:
            InfoBar.info(self._translator.translate("editor.ib_scan_t"), self._translator.translate("editor.ib_scan_nodup"), parent=self)
            return

        preview_dlg = QDialog(self)
        preview_dlg.setWindowTitle(self._translator.translate("editor.dlg_dup_title").replace("{n}", str(len(groups))))
        preview_dlg.resize(600, 400)
        p_lay = QVBoxLayout(preview_dlg)
        p_lay.addWidget(QLabel(self._translator.translate("editor.ed_lbl_merge_prompt"), preview_dlg))

        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
        table = QTableWidget(preview_dlg)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels([self._translator.translate("editor.th_merge"), self._translator.translate("editor.th_time"), self._translator.translate("editor.th_content")])
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.setRowCount(len(groups))

        checkboxes = []
        events = self._view_model.current_events

        for row_idx, grp in enumerate(groups):
            chk = CheckBox(preview_dlg)
            chk.setChecked(True)
            checkboxes.append(chk)
            chk_widget = QWidget()
            layout = QHBoxLayout(chk_widget); layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(chk)
            table.setCellWidget(row_idx, 0, chk_widget)
            first_ev = events[grp[0]]
            table.setItem(row_idx, 1, QTableWidgetItem(self._translator.translate("editor.dup_row").replace("{time}", seconds_to_display(first_ev.start_sec)).replace("{n}", str(len(grp)))))
            best_text = max((events[idx].text for idx in grp), key=len).replace('\n', ' ')
            table.setItem(row_idx, 2, QTableWidgetItem(best_text))

        p_lay.addWidget(table)
        p_bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel, preview_dlg)
        btn_all = PushButton(self._translator.translate("editor.ed_btn_select_all"), preview_dlg)
        btn_all.clicked.connect(lambda: [c.setChecked(True) for c in checkboxes])
        btn_none = PushButton(self._translator.translate("editor.ed_btn_deselect_all"), preview_dlg)
        btn_none.clicked.connect(lambda: [c.setChecked(False) for c in checkboxes])

        p_bb.addButton(btn_all, QDialogButtonBox.ButtonRole.ActionRole)
        p_bb.addButton(btn_none, QDialogButtonBox.ButtonRole.ActionRole)
        p_bb.accepted.connect(preview_dlg.accept)
        p_bb.rejected.connect(preview_dlg.reject)
        p_lay.addWidget(p_bb)

        # [v3.6 bugfix DLG-4]: deleteLater cho preview dialog. Đọc trạng thái
        # checkbox TRONG try (trước khi hủy), apply merge ngoài try.
        uids_before = self._event_uids()
        try:
            if preview_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            groups_to_merge = [grp for i, grp in enumerate(groups) if checkboxes[i].isChecked()]
        finally:
            preview_dlg.deleteLater()

        # [v3.23.650] Hộp thoại modal chạy vòng lặp sự kiện lồng nhau: Re-OCR/nạp phụ đề có thể đổi danh sách trong
        # lúc mở. `groups` là CHỈ SỐ → áp dụng trên danh sách đã đổi sẽ gộp nhầm câu. Danh sách đổi thì huỷ.
        if groups_to_merge and self._event_uids() != uids_before:
            InfoBar.warning(
                self._translator.translate("editor.ib_scan_t"),
                self._translator.translate("editor.ib_list_changed"), parent=self, duration=4000)
            return

        if groups_to_merge:
            applied = self._view_model.apply_merge_groups(groups_to_merge)
            InfoBar.success(self._translator.translate("editor.ib_done_t"), self._translator.translate("editor.ib_merged_c").replace("{n}", str(applied)), parent=self, duration=3000)


__all__ = ["EditorFindReplaceMixin"]
