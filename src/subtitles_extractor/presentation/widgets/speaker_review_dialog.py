"""[v3.23.630] Hộp thoại "Duyệt nhanh người nói": mỗi dòng một phím số, tự sang dòng kế.

Thay cho chuỗi thao tác chuột phải → menu 24 vai → chọn lặp lại hàng trăm lần (log thật: 861
dòng sửa tay trên 3649 dòng). Hộp thoại chỉ vẽ; quyết định nằm ở
:class:`~subtitles_extractor.application.services.speaker_review_session.SpeakerReviewSession`.
Lựa chọn chỉ có hiệu lực khi bấm "Xong" (một lượt áp cho cả bộ), "Huỷ" bỏ hết.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.application.services.speaker_review_flow import MAX_CANDIDATES
from subtitles_extractor.application.services.speaker_review_session import (
    SpeakerReviewSession,
)

_CURRENT_BACKGROUND = QColor(255, 244, 214)
_CHANGED_FOREGROUND = QColor(20, 60, 120)
_CONTEXT_RADIUS = 3
_COLUMNS = ("index", "time", "speaker", "source", "text")


def _format_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


class SpeakerReviewDialog(QDialog):
    """Duyệt tuần tự dòng nghi sai (hoặc mọi dòng): bấm 1–9 chọn người nói, Space giữ nguyên."""

    def __init__(
        self,
        parent: QWidget | None,
        session: SpeakerReviewSession,
        texts: dict[str, str],
        display_name: Callable[[str], str],
        roles: Sequence[tuple[str, str]],
        reason_of: Callable[[int], str] = lambda _position: "",
    ) -> None:
        """Dựng hộp thoại.

        Args:
            parent: Cửa sổ cha.
            session: Phiên duyệt (đã có hàng đợi).
            texts: Chuỗi giao diện (xem ``strings_*.json`` mục ``tts.rev_*``).
            display_name: Tên vai → tên đọc được.
            roles: ``[(tên đọc được, tên vai)]`` để chọn ở ô "Vai khác".
            reason_of: Vị trí dòng → lý do nghi sai (rỗng nếu không nghi).
        """
        super().__init__(parent)
        self._session = session
        self._texts = texts
        self._display = display_name
        self._reason_of = reason_of
        self.setWindowTitle(texts["title"])
        self.resize(1000, 560)

        self.progress_label = QLabel()
        self.reason_label = QLabel()
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet("color:#aa5a00;")
        intro = QLabel(texts["intro"])
        intro.setWordWrap(True)

        self.all_lines_box = QCheckBox(texts["scope_all"])
        self.all_lines_box.toggled.connect(self._on_scope)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([texts[f"col_{name}"] for name in _COLUMNS])
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        for column, width in enumerate((50, 80, 190, 330, 330)):
            self.table.setColumnWidth(column, width)

        self.candidate_row = QHBoxLayout()
        self.candidate_buttons: list[QPushButton] = []
        self.shared_button = QPushButton(texts["btn_shared"])
        self.shared_button.clicked.connect(lambda _checked=False: self.choose_speaker(""))
        self.other_box = QComboBox()
        self.other_box.addItem(texts["other"], None)
        for label, name in roles:
            self.other_box.addItem(label, name)
        self.other_box.activated.connect(self._on_other)
        self.run_box = QCheckBox()

        self.pending_label = QLabel()
        self.keep_button = QPushButton(texts["btn_keep"])
        self.keep_button.clicked.connect(lambda _checked=False: self.keep())
        self.previous_button = QPushButton(texts["btn_previous"])
        self.previous_button.clicked.connect(lambda _checked=False: self.previous())
        self.undo_button = QPushButton(texts["btn_undo"])
        self.undo_button.clicked.connect(lambda _checked=False: self.undo())
        self.done_button = QPushButton(texts["btn_done"])
        self.done_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton(texts["btn_cancel"])
        self.cancel_button.clicked.connect(self.reject)
        for button in (self.done_button, self.cancel_button, self.keep_button,
                       self.previous_button, self.undo_button, self.shared_button):
            button.setAutoDefault(False)
            button.setDefault(False)

        nav = QHBoxLayout()
        for widget in (self.previous_button, self.keep_button, self.undo_button):
            nav.addWidget(widget)
        nav.addStretch()
        nav.addWidget(self.pending_label)
        nav.addWidget(self.done_button)
        nav.addWidget(self.cancel_button)
        other_row = QHBoxLayout()
        other_row.addWidget(self.shared_button)
        other_row.addWidget(self.other_box, 1)

        top = QHBoxLayout()
        top.addWidget(self.progress_label, 1)
        top.addWidget(self.all_lines_box)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(top)
        layout.addWidget(self.reason_label)
        layout.addWidget(self.table, 1)
        layout.addLayout(self.candidate_row)
        layout.addLayout(other_row)
        layout.addWidget(self.run_box)
        layout.addLayout(nav)

        # Phím tắt xử lý ở keyPressEvent: nút/bảng không nhận tiêu điểm nên Space, mũi tên và
        # chữ số luôn tới hộp thoại (nút có tiêu điểm sẽ "ăn" Space).
        for widget in (
            self.table, self.keep_button, self.previous_button, self.undo_button,
            self.done_button, self.cancel_button, self.shared_button, self.run_box,
            self.all_lines_box,
        ):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.other_box.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.refresh()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 — tên Qt
        """1–9 chọn gợi ý, 0 giọng chung, Space/→ giữ nguyên, ← về trước, Ctrl+Z hoàn tác."""
        key, modifiers = event.key(), event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Z:
            self.undo()
        elif modifiers & ~Qt.KeyboardModifier.KeypadModifier:
            super().keyPressEvent(event)
            return
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            self.choose_candidate(key - Qt.Key.Key_1)
        elif key == Qt.Key.Key_0:
            self.choose_speaker("")
        elif key in (Qt.Key.Key_Space, Qt.Key.Key_Right):
            self.keep()
        elif key == Qt.Key.Key_Left:
            self.previous()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    # ── Thao tác (cũng là API cho kiểm thử) ───────────────────────────────────

    def result(self) -> dict[int, str]:
        """``{vị trí: người nói}`` của các dòng đã đổi."""
        return self._session.changes()

    def choose_candidate(self, index: int) -> list[int]:
        """Chọn gợi ý thứ ``index`` (0-based). Trả các dòng đã đổi."""
        names = self._session.candidates()
        if not 0 <= index < len(names):
            return []
        return self.choose_speaker(names[index])

    def choose_speaker(self, speaker: str) -> list[int]:
        """Chọn ``speaker`` cho dòng (hoặc cả đoạn nếu ô "cả đoạn" đang tick), rồi sang dòng kế."""
        whole = self.run_box.isChecked() and self.run_box.isEnabled()
        changed = self._session.choose(speaker, whole_run=whole)
        self.refresh()
        return changed

    def keep(self) -> None:
        """Giữ nguyên người nói của dòng, sang dòng kế."""
        self._session.next()
        self.refresh()

    def previous(self) -> None:
        """Về dòng trước."""
        self._session.previous()
        self.refresh()

    def undo(self) -> None:
        """Hoàn tác lựa chọn gần nhất."""
        self._session.undo()
        self.refresh()

    # ── Vẽ ────────────────────────────────────────────────────────────────────

    def _on_scope(self, checked: bool) -> None:
        self._session.set_scope(checked)
        self.refresh()

    def _on_other(self, index: int) -> None:
        name = self.other_box.itemData(index)
        self.other_box.setCurrentIndex(0)
        self.setFocus()
        if name is not None:
            self.choose_speaker(str(name))

    def refresh(self) -> None:
        """Vẽ lại theo trạng thái phiên."""
        session = self._session
        position = session.position
        current, total = session.progress
        self.progress_label.setText(
            self._texts["progress"].replace("{k}", str(current)).replace("{n}", str(total))
            if position is not None else self._texts["finished"]
        )
        reason = self._reason_of(position) if position is not None else ""
        self.reason_label.setText(reason)
        self.reason_label.setVisible(bool(reason))
        self._fill_context()
        self._fill_candidates()
        run = session.run()
        extra = max(0, len(run) - 1)
        self.run_box.setText(self._texts["run"].replace("{n}", str(extra)))
        self.run_box.setEnabled(extra > 0)
        changes = len(session.changes())
        self.pending_label.setText(self._texts["pending"].replace("{n}", str(changes)))
        active = position is not None
        for widget in (self.keep_button, self.shared_button, self.other_box):
            widget.setEnabled(active)
        self.previous_button.setEnabled(session.progress[0] > 1 or not active)
        self.undo_button.setEnabled(session.can_undo)

    def _fill_context(self) -> None:
        lines = self._session.context(_CONTEXT_RADIUS)
        current = self._session.position
        self.table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            name = self._display(line.speaker) if line.speaker else self._texts["shared"]
            cells = (str(line.position + 1), _format_time(line.start_sec), name,
                     line.source, line.text)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if line.position == current:
                    item.setBackground(_CURRENT_BACKGROUND)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if column == 2 and line.position in self._session.changes():
                    item.setForeground(_CHANGED_FOREGROUND)
                    item.setToolTip(self._texts["changed_tip"])
                self.table.setItem(row, column, item)
        self.table.resizeRowsToContents()

    def _fill_candidates(self) -> None:
        while self.candidate_row.count():
            item = self.candidate_row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.candidate_buttons = []
        current = self._session.position
        speaker = self._session.line(current).speaker if current is not None else ""
        for index, name in enumerate(self._session.candidates()):
            mark = "✓ " if name == speaker else ""
            button = QPushButton(f"{index + 1} · {mark}{self._display(name)}")
            button.setAutoDefault(False)
            button.setToolTip(name)
            button.clicked.connect(lambda _checked=False, n=name: self.choose_speaker(n))
            self.candidate_row.addWidget(button)
            self.candidate_buttons.append(button)
        self.candidate_row.addStretch()


__all__ = ["SpeakerReviewDialog"]
