"""[v3.23.608] Hộp thoại "Quản lý giai đoạn dịch" và trình xem/sửa một giai đoạn.

* :class:`StageCacheDialog` — bốn giai đoạn: đã lưu chưa, model nào, lúc nào, bao nhiêu
  dòng sửa tay, và lần bấm "Dịch" tới sẽ làm gì với từng giai đoạn (theo cấu hình đang
  chọn). Xem/sửa, dùng làm bản dịch hiện hành, xoá một giai đoạn hoặc xoá tất cả.
* :class:`StageEditorDialog` — bảng dòng của một giai đoạn (bản gốc, đầu vào, nội dung),
  sửa trực tiếp cột nội dung, lọc, hoàn tác về câu model.

Mọi thao tác đi qua :class:`StageCacheSession` — widget không chạm file.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QCloseEvent, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.application.services.stage_cache import (
    STAGE_LABELS,
    StageCacheEntry,
    StageCacheOverview,
    StageEditorRow,
)
from subtitles_extractor.application.services.stage_checkpoint import (
    StageNotCachedError,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationStageKind,
)
from subtitles_extractor.presentation.fluent_compat import PrimaryPushButton, PushButton
from subtitles_extractor.presentation.utils.time_format import seconds_to_display
from subtitles_extractor.presentation.view_models.stage_cache_session import (
    StageCacheSession,
)
from subtitles_extractor.presentation.view_models.stage_cache_texts import (
    describe_next_run,
    describe_stored,
    format_saved_time,
)

logger = logging.getLogger(__name__)

__all__ = ["StageCacheDialog", "StageEditorDialog"]

#: Nền ô đã sửa tay / đang sửa chưa lưu — bán trong suốt để hợp cả nền sáng lẫn tối.
_EDITED_BRUSH = QBrush(QColor(255, 196, 0, 70))
_PENDING_BRUSH = QBrush(QColor(0, 150, 255, 70))

#: Trên ngưỡng này không đo chiều cao từng dòng (chậm với phim ~4000 dòng).
_RESIZE_ROWS_LIMIT = 300

_OVERVIEW_COLUMNS = (
    "Giai đoạn", "Bản đã lưu", "Model", "Sửa tay", "Lưu lúc", "Lần dịch tới",
)
_EDITOR_COLUMNS = ("#", "Thời gian", "Bản gốc", "Đầu vào giai đoạn", "Nội dung giai đoạn")
_COL_CONTENT = 4


def _read_only_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class StageEditorDialog(QDialog):
    """Xem và sửa từng dòng của một giai đoạn đã lưu.

    Args:
        session: Phiên bộ nhớ đệm.
        kind: Giai đoạn cần sửa.
        parent: Widget cha.

    Raises:
        StageNotCachedError: Giai đoạn chưa có trong bộ nhớ đệm.
    """

    def __init__(
        self,
        session: StageCacheSession,
        kind: TranslationStageKind,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._kind = kind
        self._label = STAGE_LABELS[kind]
        self._rows: list[StageEditorRow] = []
        self._pending: dict[int, str] = {}
        self._loading = False
        self.setWindowTitle(f"Giai đoạn {self._label} — xem và sửa")
        self.resize(1200, 720)
        self._build()
        self._reload()

    # ── Dựng giao diện ────────────────────────────────────────────────────
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Nhấp đúp ô ở cột 'Nội dung giai đoạn' để sửa. Dòng sửa tay được tô vàng và "
            "LUÔN được giữ, kể cả khi giai đoạn này bị dịch lại (đổi model, cấu hình hay "
            "bảng nhân vật). Lưu xong, lần bấm 'Dịch' tới chỉ dịch lại các lô có dòng "
            "đổi ở các giai đoạn sau; các giai đoạn khác dùng lại bản đã lưu."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        filter_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Tìm trong bản gốc / đầu vào / nội dung…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        self._only_edited = QCheckBox("Chỉ dòng đã sửa")
        self._only_edited.toggled.connect(self._apply_filter)
        self._only_changed = QCheckBox("Chỉ dòng khác đầu vào")
        self._only_changed.setToolTip(
            "Chỉ hiện các dòng giai đoạn này đã viết khác đầu vào của nó."
        )
        self._only_changed.toggled.connect(self._apply_filter)
        self._counter = QLabel()
        filter_row.addWidget(self._search, 1)
        filter_row.addWidget(self._only_edited)
        filter_row.addWidget(self._only_changed)
        filter_row.addWidget(self._counter)
        layout.addLayout(filter_row)

        self._table = QTableWidget(0, len(_EDITOR_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_EDITOR_COLUMNS))
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._table.setWordWrap(True)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for column in (2, 3, _COL_CONTENT):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, 1)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        self._btn_revert_selected = PushButton("↩ Hoàn tác dòng đã chọn")
        self._btn_revert_selected.setToolTip(
            "Trả các dòng đang chọn về đúng câu model đã dịch (bấm Lưu để ghi)."
        )
        self._btn_revert_selected.clicked.connect(self._on_revert_selected)
        self._btn_revert_all = PushButton("↩ Hoàn tác mọi dòng sửa tay")
        self._btn_revert_all.clicked.connect(self._on_revert_all)
        self._btn_save = PrimaryPushButton("💾 Lưu")
        self._btn_save.clicked.connect(self._on_save)
        btn_close = PushButton("Đóng")
        btn_close.clicked.connect(self.close)
        buttons.addWidget(self._btn_revert_selected)
        buttons.addWidget(self._btn_revert_all)
        buttons.addStretch(1)
        buttons.addWidget(self._btn_save)
        buttons.addWidget(btn_close)
        layout.addLayout(buttons)

    # ── Dữ liệu ───────────────────────────────────────────────────────────
    def _reload(self) -> None:
        self._rows = self._session.editor_rows(self._kind)
        self._pending.clear()
        self._loading = True
        try:
            self._table.setRowCount(0)  # [v3.23.650] xem pages/translate_page._on_result_ready: refill bảng đầy là O(n²)
            self._table.setRowCount(len(self._rows))
            for position, row in enumerate(self._rows):
                self._table.setItem(position, 0, _read_only_item(str(row.index)))
                self._table.setItem(
                    position, 1,
                    _read_only_item(seconds_to_display(row.start_ms / 1000.0)),
                )
                self._table.setItem(position, 2, _read_only_item(row.source_text))
                self._table.setItem(position, 3, _read_only_item(row.input_text))
                content = QTableWidgetItem(row.text)
                self._table.setItem(position, _COL_CONTENT, content)
                self._paint_row(position)
        finally:
            self._loading = False
        if len(self._rows) <= _RESIZE_ROWS_LIMIT:
            self._table.resizeRowsToContents()
        self._apply_filter()
        self._update_buttons()

    def _paint_row(self, position: int) -> None:
        row = self._rows[position]
        item = self._table.item(position, _COL_CONTENT)
        if item is None:
            return
        # Đổi nền/tooltip cũng phát itemChanged — chặn để không xử lý như một lần sửa.
        was_loading, self._loading = self._loading, True
        try:
            if row.index in self._pending:
                item.setBackground(_PENDING_BRUSH)
                item.setToolTip(f"Chưa lưu. Câu model: {row.model_text}")
            elif row.edited:
                item.setBackground(_EDITED_BRUSH)
                item.setToolTip(f"Đã sửa tay. Câu model: {row.model_text}")
            else:
                item.setBackground(QBrush())
                item.setToolTip("")
        finally:
            self._loading = was_loading

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != _COL_CONTENT:
            return
        position = item.row()
        if not 0 <= position < len(self._rows):
            return
        row = self._rows[position]
        new_text = item.text().strip()
        if not new_text:
            # [v3.23.609] Dòng rỗng mất khỏi phụ đề lẫn lồng tiếng và được "giữ" qua mọi
            # lần dịch lại → không nhận; trả ô về câu trước đó.
            restored = self._pending.get(row.index, row.text)
            self._loading = True
            try:
                item.setText(restored)
            finally:
                self._loading = False
            self._status.setText(
                f"Dòng {row.index}: không để trống. Muốn trả về câu model, chọn dòng rồi "
                "bấm 'Hoàn tác dòng đã chọn'."
            )
            return
        if new_text == row.text.strip():
            self._pending.pop(row.index, None)
        else:
            self._pending[row.index] = new_text
        self._paint_row(position)
        self._update_buttons()

    def _apply_filter(self) -> None:
        needle = self._search.text().strip().casefold()
        only_edited = self._only_edited.isChecked()
        only_changed = self._only_changed.isChecked()
        shown = 0
        for position, row in enumerate(self._rows):
            text = self._current_text(position)
            visible = True
            if only_edited and not (row.edited or row.index in self._pending):
                visible = False
            if only_changed and text.strip() == row.input_text.strip():
                visible = False
            if visible and needle:
                haystack = f"{row.source_text}\n{row.input_text}\n{text}".casefold()
                visible = needle in haystack
            self._table.setRowHidden(position, not visible)
            shown += int(visible)
        edited = sum(1 for row in self._rows if row.edited)
        self._counter.setText(
            f"Hiện {shown}/{len(self._rows)} dòng · {edited} dòng sửa tay"
        )

    def _current_text(self, position: int) -> str:
        item = self._table.item(position, _COL_CONTENT)
        return item.text() if item is not None else ""

    def _update_buttons(self) -> None:
        self._btn_save.setEnabled(bool(self._pending))
        self._btn_revert_all.setEnabled(any(row.edited for row in self._rows))
        self._btn_save.setText(
            f"💾 Lưu ({len(self._pending)} dòng)" if self._pending else "💾 Lưu"
        )

    # ── Thao tác ──────────────────────────────────────────────────────────
    def _on_revert_selected(self) -> None:
        positions = sorted({index.row() for index in self._table.selectedIndexes()})
        for position in positions:
            row = self._rows[position]
            item = self._table.item(position, _COL_CONTENT)
            if item is not None and item.text() != row.model_text:
                item.setText(row.model_text)

    def _on_revert_all(self) -> None:
        edited = sum(1 for row in self._rows if row.edited)
        confirm = QMessageBox.question(
            self,
            "Hoàn tác mọi dòng sửa tay?",
            f"Trả {edited} dòng sửa tay của giai đoạn '{self._label}' về đúng câu model "
            "đã dịch. Không thể hoàn tác thao tác này.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            reverted = self._session.discard_edits(self._kind)
        except (OSError, StageNotCachedError) as exc:
            QMessageBox.warning(self, "Không hoàn tác được", str(exc))
            return
        self._reload()
        self._status.setText(f"Đã trả {reverted} dòng về câu model.")

    def _on_save(self) -> bool:
        if not self._pending:
            return True
        try:
            outcome = self._session.save_edits(self._kind, dict(self._pending))
        except (OSError, StageNotCachedError) as exc:
            logger.warning("Không lưu được sửa đổi giai đoạn: %s", exc)
            QMessageBox.warning(self, "Không lưu được", str(exc))
            return False
        self._reload()
        result = outcome.result
        message = (
            f"Đã lưu {len(result.changed)} dòng sửa, "
            f"hoàn tác {len(result.reverted)} dòng."
        )
        if outcome.downstream:
            later = ", ".join(STAGE_LABELS[kind] for kind in outcome.downstream)
            message += (
                f" Lần bấm 'Dịch' tới: {later} chỉ dịch lại các lô có dòng vừa đổi."
            )
        self._status.setText(message)
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — tên của Qt
        """Hỏi lưu khi còn dòng sửa chưa lưu."""
        if not self._pending:
            event.accept()
            return
        choice = QMessageBox.question(
            self,
            "Còn dòng sửa chưa lưu",
            f"Có {len(self._pending)} dòng đã sửa mà chưa lưu. Lưu trước khi đóng?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if choice == QMessageBox.StandardButton.Save and not self._on_save():
            event.ignore()
            return
        event.accept()

    def reject(self) -> None:
        """Phím Esc đi qua :meth:`closeEvent` để còn hỏi lưu."""
        self.close()


class StageCacheDialog(QDialog):
    """Quản lý bộ nhớ đệm từng giai đoạn dịch của phụ đề hiện tại.

    Args:
        session: Phiên bộ nhớ đệm (phụ đề + cấu hình đang chọn).
        parent: Widget cha.
    """

    def __init__(self, session: StageCacheSession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._overview: StageCacheOverview | None = None
        self.setWindowTitle("Quản lý giai đoạn dịch")
        self.resize(1080, 430)
        self._build()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        layout.addWidget(self._intro)
        self._context_warning = QLabel(
            "⚠ Bảng nhân vật/thuật ngữ đã đổi so với lúc lưu — lần dịch tới sẽ dịch lại "
            "mọi giai đoạn để tên riêng và xưng hô nhất quán (dòng sửa tay vẫn được áp "
            "lại)."
        )
        self._context_warning.setWordWrap(True)
        self._context_warning.setStyleSheet("color: #d97706;")
        layout.addWidget(self._context_warning)

        self._table = QTableWidget(0, len(_OVERVIEW_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_OVERVIEW_COLUMNS))
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setWordWrap(True)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        for column in range(len(_OVERVIEW_COLUMNS) - 1):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            len(_OVERVIEW_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
        )
        self._table.itemSelectionChanged.connect(self._update_buttons)
        self._table.cellDoubleClicked.connect(lambda _row, _col: self._on_edit())
        layout.addWidget(self._table, 1)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        self._btn_edit = PrimaryPushButton("✏️ Xem / sửa…")
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_apply = PushButton("📥 Dùng làm bản dịch hiện hành")
        self._btn_apply.setToolTip(
            "Đưa nội dung giai đoạn này vào bảng kết quả ngay (không gọi API, không qua "
            "các bước dọn cuối của lượt dịch)."
        )
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_delete = PushButton("🗑 Xoá giai đoạn…")
        self._btn_delete.setToolTip(
            "Xoá bản lưu của giai đoạn này: lần dịch tới dịch lại nó (vd để dịch lại "
            "bằng model mạnh hơn)."
        )
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_delete_all = PushButton("Xoá tất cả…")
        self._btn_delete_all.clicked.connect(self._on_delete_all)
        btn_close = PushButton("Đóng")
        btn_close.clicked.connect(self.accept)
        for button in (self._btn_edit, self._btn_apply, self._btn_delete):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self._btn_delete_all)
        buttons.addWidget(btn_close)
        layout.addLayout(buttons)

    # ── Dữ liệu ───────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Đọc lại bộ nhớ đệm và vẽ lại bảng (giữ dòng đang chọn)."""
        selected = self._table.currentRow()
        try:
            overview = self._session.overview()
        except OSError as exc:
            logger.warning("Không đọc được bộ nhớ đệm giai đoạn: %s", exc)
            QMessageBox.warning(self, "Không đọc được bộ nhớ đệm", str(exc))
            return
        self._overview = overview
        stored = len(overview.stored_kinds)
        self._intro.setText(
            f"Phụ đề hiện tại: {overview.total_lines} dòng · {stored} giai đoạn đã lưu. "
            "Kết quả từng giai đoạn được GIỮ sau khi dịch: đổi cấu hình hay sửa tay một "
            "giai đoạn thì lần dịch tới chỉ dịch phần thật sự đổi. Cột 'Lần dịch tới' "
            "tính theo cấu hình đang chọn trên trang."
            if overview.exists
            else f"Phụ đề hiện tại ({overview.total_lines} dòng) chưa có bộ nhớ đệm với "
            "ngôn ngữ và Visual Cues đang chọn. Dịch một lần là các giai đoạn được lưu "
            "để xem, sửa và dùng lại."
        )
        self._context_warning.setVisible(overview.context_changed)
        self._table.setRowCount(len(overview.entries))
        for position, entry in enumerate(overview.entries):
            self._fill_row(position, entry)
        self._table.resizeRowsToContents()
        if 0 <= selected < len(overview.entries):
            self._table.selectRow(selected)
        else:
            first_stored = next(
                (i for i, e in enumerate(overview.entries) if e.stored), -1
            )
            if first_stored >= 0:
                self._table.selectRow(first_stored)
        self._btn_delete_all.setEnabled(overview.exists)
        self._update_buttons()

    def _fill_row(self, position: int, entry: StageCacheEntry) -> None:
        name = entry.label + ("" if entry.enabled else " (đang tắt)")
        model = entry.stored_model or "—"
        if entry.partial_model and entry.partial_model != entry.stored_model:
            model += f" · dở: {entry.partial_model}"
        cells = (
            name,
            describe_stored(entry),
            model,
            str(entry.edited_lines) if entry.edited_lines else "—",
            format_saved_time(entry.completed_at),
            describe_next_run(entry),
        )
        for column, text in enumerate(cells):
            item = _read_only_item(text)
            if column == len(cells) - 1:
                item.setToolTip(text)
            self._table.setItem(position, column, item)

    def _selected_entry(self) -> StageCacheEntry | None:
        if self._overview is None:
            return None
        position = self._table.currentRow()
        if not 0 <= position < len(self._overview.entries):
            return None
        return self._overview.entries[position]

    def _update_buttons(self) -> None:
        entry = self._selected_entry()
        stored = entry is not None and entry.stored
        self._btn_edit.setEnabled(stored)
        self._btn_apply.setEnabled(stored)
        self._btn_delete.setEnabled(
            entry is not None
            and (entry.stored or entry.partial_lines > 0 or entry.edited_lines > 0)
        )

    # ── Thao tác ──────────────────────────────────────────────────────────
    def _on_edit(self) -> None:
        entry = self._selected_entry()
        if entry is None or not entry.stored:
            return
        try:
            editor = StageEditorDialog(self._session, entry.kind, self)
        except (OSError, StageNotCachedError) as exc:
            QMessageBox.warning(self, "Không mở được giai đoạn", str(exc))
            self.refresh()
            return
        editor.exec()
        self.refresh()

    def _on_apply(self) -> None:
        entry = self._selected_entry()
        if entry is None or not entry.stored:
            return
        confirm = QMessageBox.question(
            self,
            "Dùng làm bản dịch hiện hành?",
            f"Thay bản dịch trên bảng kết quả bằng nội dung giai đoạn '{entry.label}' "
            f"({entry.line_count} dòng). Các chỉnh sửa trên bảng kết quả chưa có trong "
            "giai đoạn này sẽ bị thay.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            count = self._session.apply_stage(entry.kind)
        except (OSError, StageNotCachedError) as exc:
            QMessageBox.warning(self, "Không dùng được giai đoạn", str(exc))
            return
        self._status.setText(
            f"Đã đưa {count} dòng của giai đoạn '{entry.label}' vào bảng kết quả."
        )

    def _on_delete(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Xoá giai đoạn?")
        box.setText(
            f"Xoá bản lưu của giai đoạn '{entry.label}'? Lần dịch tới sẽ dịch lại giai "
            "đoạn này; các giai đoạn sau chỉ dịch lại các lô có dòng đổi."
        )
        drop_edits: QCheckBox | None = None
        if entry.edited_lines:
            drop_edits = QCheckBox(f"Xoá cả {entry.edited_lines} dòng sửa tay")
            box.setCheckBox(drop_edits)
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        keep_edits = not (drop_edits is not None and drop_edits.isChecked())
        try:
            self._session.delete_stage(entry.kind, keep_edits=keep_edits)
        except OSError as exc:
            QMessageBox.warning(self, "Không xoá được", str(exc))
            return
        self.refresh()
        self._status.setText(f"Đã xoá bản lưu của giai đoạn '{entry.label}'.")

    def _on_delete_all(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Xoá toàn bộ bộ nhớ đệm?",
            "Xoá bản lưu của MỌI giai đoạn (kể cả dòng sửa tay và tiến độ dở) của phụ đề "
            "này. Lần dịch tới sẽ dịch lại từ đầu.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = self._session.delete_all()
        except OSError as exc:
            QMessageBox.warning(self, "Không xoá được", str(exc))
            return
        self.refresh()
        self._status.setText(f"Đã xoá {removed} file bộ nhớ đệm.")
