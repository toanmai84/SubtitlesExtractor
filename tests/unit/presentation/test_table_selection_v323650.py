"""[v3.23.650] Chọn hàng loạt bảng trong một lần (81 s → ~0,1 s với 1825 hàng)."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QAbstractItemView, QApplication, QTableWidget, QTableWidgetItem  # noqa: E402

from subtitles_extractor.presentation.utils.table_selection import merge_row_ranges, select_rows_bulk  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_merge_row_ranges_v323650() -> None:
    assert merge_row_ranges([5, 1, 2, 3, 9, 10, 3]) == [(1, 3), (5, 5), (9, 10)]
    assert merge_row_ranges([]) == []


def _table(rows: int) -> QTableWidget:
    table = QTableWidget(rows, 4)
    for r in range(rows):
        for c in range(4):
            table.setItem(r, c, QTableWidgetItem(f"{r}-{c}"))
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    return table


def test_select_rows_bulk_selects_exactly_the_rows_and_clears_previous_v323650() -> None:
    table = _table(50)
    table.selectRow(0)
    select_rows_bulk(table, [3, 4, 5, 20, 99, -1])                 # 99/-1 ngoài phạm vi → bỏ qua
    selected = sorted({i.row() for i in table.selectionModel().selectedRows()})
    assert selected == [3, 4, 5, 20]


def test_select_rows_bulk_is_fast_for_thousands_of_rows_v323650() -> None:
    table = _table(3649)
    table.show()
    _app.processEvents()
    start = time.perf_counter()
    select_rows_bulk(table, range(0, 3649, 2))                      # 1825 hàng rời rạc (ca đo thật 81 s)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"chọn hàng loạt phải nhanh (đo {elapsed:.2f}s)"
    assert len({i.row() for i in table.selectionModel().selectedRows()}) == 1825
