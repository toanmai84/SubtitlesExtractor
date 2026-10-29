"""[v3.23.654] Bảng sửa Visual Cues (ai nói / nói với ai / bối cảnh) thay cho khối JSON.

Mỗi câu gốc một hàng: #, giờ, câu gốc, bản dịch, người nói, nói với, bối cảnh, nguồn, ⚠ (lý do nghi của
:mod:`cue_review`). Gõ tên Việt hay tên gốc đều được (quy về tên chuẩn trong bảng nhân vật). Thao tác hàng loạt: gán
người nói / người nghe cho các dòng chọn, đảo người nói ↔ người nghe. Dòng sửa được đánh dấu nguồn "người dùng" để các
bước tự động không đè.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from subtitles_extractor.application.services.cue_label_canonicalizer import canonical_label, roster_label_map
from subtitles_extractor.application.services.cue_review import (
    CUE_GENDER,
    CUE_NO_SPEAKER,
    CUE_SAME_PERSON,
    CUE_SELF_CALL,
    CUE_UNKNOWN_NAME,
    CueIssue,
    review_cues,
)
from subtitles_extractor.application.services.roster_format import parse_roster
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import CUE_ORIGIN_USER, VisualCue

COL_NO, COL_TIME, COL_SOURCE, COL_TRANSLATION, COL_SPEAKER, COL_ADDRESSEE, COL_SCENE, COL_ORIGIN, COL_ISSUE = range(9)
_EDITABLE = (COL_SPEAKER, COL_ADDRESSEE, COL_SCENE)


def display_names(characters: str) -> dict[str, str]:
    """``{tên chuẩn: tên hiển thị}`` — phiên âm đầu tiên kèm tên gốc, vd ``"Lâm Dật (林逸)"``."""
    names: dict[str, str] = {}
    for entry in parse_roster(characters or ""):
        canonical = entry.name.strip()
        readings = [part.strip() for part in re.split(r"\s*/\s*", entry.aka or "") if part.strip()]
        names[canonical] = f"{readings[0]} ({canonical})" if readings else canonical
    return names


def _fmt_time(seconds: float) -> str:
    minutes, sec = divmod(max(0.0, float(seconds)), 60)
    return f"{int(minutes):02d}:{sec:05.2f}"


class CueTableDialog(QDialog):
    """Bảng sửa Visual Cues; :meth:`result_json` trả JSON mới sau khi người dùng bấm OK."""

    def __init__(
        self,
        parent,
        cues_raw: str,
        source_rows: Sequence[tuple[float, str]],
        translations: Sequence[str],
        characters: str,
        translate: Callable[[str], str],
    ) -> None:
        super().__init__(parent)
        self._tr = lambda key: translate(f"translate.cue_table_{key}")
        self.setWindowTitle(self._tr("title"))
        self.resize(1200, 720)
        self._characters = characters
        self._source_rows = list(source_rows)
        self._translations = list(translations) if len(translations) == len(source_rows) else []
        self._label_map = roster_label_map(characters)
        self._display = display_names(characters)
        self._original = {cue.line_no: cue for cue in deserialize_visual_cues(cues_raw)}
        self._cues = dict(self._original)
        self._issues: dict[int, list[CueIssue]] = {}
        self._loading = False

        layout = QVBoxLayout(self)
        intro = QLabel(self._tr("intro"))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        filters = QHBoxLayout()
        self._only_issues = QCheckBox(self._tr("only_issues"))
        self._only_issues.toggled.connect(self._apply_filter)
        filters.addWidget(self._only_issues)
        self._search = QLineEdit()
        self._search.setPlaceholderText(self._tr("search_ph"))
        self._search.textChanged.connect(self._apply_filter)
        filters.addWidget(self._search, 1)
        layout.addLayout(filters)

        self._table = QTableWidget(len(self._source_rows), 9)
        self._table.setHorizontalHeaderLabels([
            "#", self._tr("col_time"), self._tr("col_source"), self._tr("col_translation"),
            self._tr("col_speaker"), self._tr("col_addressee"), self._tr("col_scene"), self._tr("col_origin"), "⚠",
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        header = self._table.horizontalHeader()
        for column in (COL_NO, COL_TIME, COL_ORIGIN):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_SOURCE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_TRANSLATION, QHeaderView.ResizeMode.Stretch)
        for column, width in ((COL_SPEAKER, 170), (COL_ADDRESSEE, 170), (COL_SCENE, 150), (COL_ISSUE, 220)):
            self._table.setColumnWidth(column, width)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, 1)

        actions = QHBoxLayout()
        choices = [("", "")] + [(label, shown) for label, shown in self._display.items()]
        self._choice = QComboBox()
        for label, shown in choices:
            self._choice.addItem(shown or self._tr("nobody"), label)
        actions.addWidget(QLabel(self._tr("assign_label")))
        actions.addWidget(self._choice)
        self._btn_speaker = QPushButton(self._tr("btn_set_speaker"))
        self._btn_speaker.clicked.connect(lambda: self.assign(COL_SPEAKER, self._choice.currentData() or ""))
        actions.addWidget(self._btn_speaker)
        self._btn_addressee = QPushButton(self._tr("btn_set_addressee"))
        self._btn_addressee.clicked.connect(lambda: self.assign(COL_ADDRESSEE, self._choice.currentData() or ""))
        actions.addWidget(self._btn_addressee)
        self._btn_swap = QPushButton(self._tr("btn_swap"))
        self._btn_swap.clicked.connect(self.swap_selected)
        actions.addWidget(self._btn_swap)
        self._btn_recheck = QPushButton(self._tr("btn_recheck"))
        self._btn_recheck.clicked.connect(self.recheck)
        actions.addWidget(self._btn_recheck)
        actions.addStretch()
        layout.addLayout(actions)

        self._status = QLabel("")
        layout.addWidget(self._status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._fill()
        self.recheck()

    # ── Dữ liệu ───────────────────────────────────────────────────────────────

    def _shown(self, label: str) -> str:
        return self._display.get(label, label)

    def _canonical(self, typed: str) -> str:
        typed = (typed or "").strip()
        # "Lâm Dật (林逸)" → thử cả phần trong ngoặc.
        match = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", typed)
        for candidate in ([match.group(2), match.group(1)] if match else []) + [typed]:
            if candidate and self._label_map:
                resolved = canonical_label(candidate, self._label_map)
                if resolved != candidate or candidate in self._display:
                    return resolved
        return typed

    def _fill(self) -> None:
        self._loading = True
        self._table.setUpdatesEnabled(False)
        try:
            for row, (start, text) in enumerate(self._source_rows):
                line_no = row + 1
                cue = self._cues.get(line_no) or VisualCue(line_no=line_no)
                values = {
                    COL_NO: str(line_no), COL_TIME: _fmt_time(start), COL_SOURCE: text,
                    COL_TRANSLATION: self._translations[row] if self._translations else "",
                    COL_SPEAKER: self._shown(cue.speaker), COL_ADDRESSEE: self._shown(cue.addressee),
                    COL_SCENE: cue.scene, COL_ORIGIN: self._origin_text(cue), COL_ISSUE: "",
                }
                for column, value in values.items():
                    item = QTableWidgetItem(value)
                    if column not in _EDITABLE:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._table.setItem(row, column, item)
        finally:
            self._table.setUpdatesEnabled(True)
            self._loading = False

    def _origin_text(self, cue: VisualCue) -> str:
        return self._tr("origin_user") if cue.origin == CUE_ORIGIN_USER else ("AI" if cue.speaker or cue.addressee else "")

    def _set_cue(self, line_no: int, **changes: str) -> None:
        cue = self._cues.get(line_no) or VisualCue(line_no=line_no)
        new = replace(cue, **changes)
        if new == cue:
            return
        if cue.origin != CUE_ORIGIN_USER and not cue.auto_label:
            # [v3.23.655] Cất nhãn AI gốc lần đầu thành cue người dùng → chọn lại "Tự động" ở trang TTS trả về được.
            new = replace(new, auto_label=f"{cue.speaker}\x1f{cue.addressee}")
        self._cues[line_no] = replace(new, origin=CUE_ORIGIN_USER)
        row = line_no - 1
        self._loading = True
        try:
            current = self._cues[line_no]
            self._table.item(row, COL_SPEAKER).setText(self._shown(current.speaker))
            self._table.item(row, COL_ADDRESSEE).setText(self._shown(current.addressee))
            self._table.item(row, COL_SCENE).setText(current.scene)
            self._table.item(row, COL_ORIGIN).setText(self._origin_text(current))
        finally:
            self._loading = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() not in _EDITABLE:
            return
        line_no = item.row() + 1
        text = item.text()
        if item.column() == COL_SPEAKER:
            self._set_cue(line_no, speaker=self._canonical(text))
        elif item.column() == COL_ADDRESSEE:
            self._set_cue(line_no, addressee=self._canonical(text))
        else:
            self._set_cue(line_no, scene=text.strip())

    def _selected_lines(self) -> list[int]:
        return sorted({index.row() + 1 for index in self._table.selectionModel().selectedRows()})

    def assign(self, column: int, label: str) -> int:
        """Gán người nói (``COL_SPEAKER``) hoặc người nghe (``COL_ADDRESSEE``) cho các dòng chọn."""
        lines = self._selected_lines()
        for line_no in lines:
            if column == COL_SPEAKER:
                self._set_cue(line_no, speaker=label)
            else:
                self._set_cue(line_no, addressee=label)
        return len(lines)

    def swap_selected(self) -> int:
        """Đảo người nói ↔ người nghe ở các dòng chọn (nhãn bị gán ngược)."""
        lines = self._selected_lines()
        for line_no in lines:
            cue = self._cues.get(line_no) or VisualCue(line_no=line_no)
            self._set_cue(line_no, speaker=cue.addressee, addressee=cue.speaker)
        return len(lines)

    def recheck(self) -> int:
        """Chấm lại dòng nghi theo cues hiện tại; trả số dòng nghi."""
        texts = [text for _start, text in self._source_rows]
        self._issues = review_cues(list(self._cues.values()), texts, self._characters)
        self._loading = True
        try:
            for row in range(self._table.rowCount()):
                issues = self._issues.get(row + 1, [])
                item = self._table.item(row, COL_ISSUE)
                item.setText("; ".join(self.issue_text(issue) for issue in issues))
        finally:
            self._loading = False
        self._only_issues.setText(f"{self._tr('only_issues')} ({len(self._issues)})")
        self._apply_filter()
        return len(self._issues)

    def issue_text(self, issue: CueIssue) -> str:
        key = {
            CUE_SAME_PERSON: "issue_same", CUE_NO_SPEAKER: "issue_no_speaker", CUE_SELF_CALL: "issue_self_call",
            CUE_GENDER: "issue_gender", CUE_UNKNOWN_NAME: "issue_unknown",
        }.get(issue.reason, "issue_same")
        return self._tr(key).replace("{word}", issue.word).replace("{detail}", self._shown(issue.detail))

    def _apply_filter(self) -> None:
        only = self._only_issues.isChecked()
        needle = self._search.text().strip().lower()
        for row in range(self._table.rowCount()):
            hidden = only and (row + 1) not in self._issues
            if not hidden and needle:
                haystack = " ".join(
                    (self._table.item(row, column).text() or "") for column in (COL_SOURCE, COL_TRANSLATION, COL_SPEAKER, COL_ADDRESSEE)
                ).lower()
                hidden = needle not in haystack
            self._table.setRowHidden(row, hidden)

    # ── Kết quả ───────────────────────────────────────────────────────────────

    @property
    def changed_lines(self) -> list[int]:
        return sorted(line for line, cue in self._cues.items() if self._original.get(line) != cue)

    def merged_into(self, current_raw: str) -> str:
        """[v3.23.654] Chỉ áp các dòng SỬA trong bảng lên JSON hiện tại của ô cues (ô có thể đã đổi trong lúc bảng mở —
        vd đồng bộ người nói từ trang TTS), thay vì ghi đè bằng bản lúc mở bảng."""
        cues = {cue.line_no: cue for cue in deserialize_visual_cues(current_raw)}
        for line_no in self.changed_lines:
            cue = self._cues[line_no]
            if cue.speaker or cue.addressee or cue.scene or cue.origin == CUE_ORIGIN_USER:
                cues[line_no] = cue
            else:
                cues.pop(line_no, None)
        return serialize_visual_cues([cues[k] for k in sorted(cues)])

    def result_json(self) -> str:
        """JSON Visual Cues sau khi sửa (cue rỗng hoàn toàn bị bỏ)."""
        kept = [
            cue for _line, cue in sorted(self._cues.items())
            if cue.speaker or cue.addressee or cue.scene or cue.origin == CUE_ORIGIN_USER
        ]
        return serialize_visual_cues(kept)


__all__ = ["CueTableDialog", "display_names"]
