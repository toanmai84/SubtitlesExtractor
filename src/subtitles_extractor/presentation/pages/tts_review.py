"""[v3.23.695] Duyệt nhanh trên trang Lồng tiếng: phím tắt, đếm và nhảy tới dòng cần duyệt, chọn dòng bất thường.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/tts_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TTSPage``.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt

from subtitles_extractor.presentation.utils.table_selection import select_rows_bulk

logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")



class TTSReviewMixin:
    """Duyệt nhanh trên trang Lồng tiếng: phím tắt, đếm và nhảy tới dòng cần duyệt, chọn dòng bất thường."""

    def _install_review_shortcuts(self) -> None:
        """[v3.23.682] Phím tắt khi bảng dòng giữ focus: Space nghe dòng, Ctrl+R TTS lại dòng chọn, F8 / Shift+F8 dòng
        cần nghe lại kế tiếp / trước đó (đọc sai theo kiểm tra phát âm hoặc audio dài bất thường).

        Dùng ``click()`` của nút sẵn có: nút đang tắt (đang chạy, chưa có audio…) thì phím cũng không làm gì.
        """
        from PySide6.QtGui import QKeySequence, QShortcut

        bindings = (
            (QKeySequence(Qt.Key.Key_Space), self._btn_play_sel.click),
            (QKeySequence("Ctrl+R"), self._btn_resynth.click),
            (QKeySequence(Qt.Key.Key_F8), lambda: self._jump_to_review_line(forward=True)),
            (QKeySequence("Shift+F8"), lambda: self._jump_to_review_line(forward=False)),
        )
        self._review_shortcuts = []
        for sequence, handler in bindings:
            shortcut = QShortcut(sequence, self._table)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(handler)
            self._review_shortcuts.append(shortcut)

    def _speaker_review_rows(self) -> set[int]:
        """[v3.23.682] Hàng có người nói chưa chắc (cùng tiêu chí với nút "Duyệt người nói nhanh")."""
        line_edits = getattr(self, "_line_edits", None)
        if line_edits is None or not self._dubbing_panel.is_dubbing_enabled():
            return set()
        try:
            cells = self._dubbing.line_speaker_cells()
        except (AttributeError, RuntimeError, ValueError):
            return set()
        wanted = {position for position, cell in enumerate(cells) if cell.needs_review}
        if not wanted:
            return set()
        return {
            row for row in range(self._table.rowCount())
            if line_edits._row_position(row) in wanted
        }

    def _schedule_review_count(self, *_args) -> None:
        if hasattr(self, "_review_timer"):
            self._review_timer.start()

    def review_count(self) -> int:
        """[v3.23.682] Câu đọc sai + audio dài bất thường + dòng người nói cần duyệt."""
        speakers = int(getattr(self._line_edits, "_review_count", 0) or 0) if hasattr(self, "_line_edits") else 0
        return len(self._review_event_indices()) + speakers

    def _review_event_indices(self) -> set[int]:
        """[v3.23.682] Dòng nên nghe lại: AI chấm đọc sai + audio dài bất thường."""
        wrong = set(self._pronunciation.wrong_indices()) if hasattr(self, "_pronunciation") else set()
        return wrong | set(self._abnormal_event_indices())

    def _jump_to_review_line(self, forward: bool = True) -> None:
        """[v3.23.682] Chọn dòng cần nghe lại kế tiếp / trước đó (vòng lại), cuộn tới đó."""
        from subtitles_extractor.presentation.pages.editor_page import next_suspect_row

        wanted = self._review_event_indices()
        rows = []
        for row in range(self._table.rowCount()):
            item0 = self._table.item(row, 0)
            value = item0.data(Qt.ItemDataRole.UserRole) if item0 is not None else None
            if value is not None and int(value) in wanted:
                rows.append(row)
        rows = sorted(set(rows) | self._speaker_review_rows())
        target = next_suspect_row(rows, self._table.currentRow(), forward)
        if target is None:
            self._stage_lbl.setText(self._translator.translate("tts.review_none"))
            return
        self._table.selectRow(target)
        self._table.scrollToItem(self._table.item(target, 0))

    def _abnormal_event_indices(self) -> list[int]:
        """[v3.23.459] event_index của các dòng audio dài bất thường."""
        return [
            int(r.event_index)
            for r in self._last_results
            if getattr(r, "abnormal_length", False) and not r.was_skipped
        ]

    def _on_select_abnormal(self) -> None:
        """Chọn tất cả dòng dài bất thường trong bảng (để TTS lại hàng loạt)."""
        abnormal = set(self._abnormal_event_indices())
        if not abnormal:
            self._show_warning("Không có dòng bất thường", "Không tìm thấy dòng nào dài bất thường.")
            return
        # [v3.23.650] Một lần chọn hàng loạt (selectRow từng hàng: 81 s với 1825 dòng).
        rows: list[int] = []
        for row in range(self._table.rowCount()):
            item0 = self._table.item(row, 0)
            if item0 is None:
                continue
            ev = item0.data(Qt.ItemDataRole.UserRole)
            if ev is not None and int(ev) in abnormal:
                rows.append(row)
        select_rows_bulk(self._table, rows)
        first_row = rows[0] if rows else -1
        if first_row >= 0:
            self._table.scrollToItem(self._table.item(first_row, 0))
        self._stage_lbl.setText(f"Đã chọn {len(abnormal)} dòng dài bất thường — bấm '🔄 TTS lại dòng chọn' để làm lại.")


__all__ = ["TTSReviewMixin"]
