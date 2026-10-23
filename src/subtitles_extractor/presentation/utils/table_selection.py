"""[v3.23.650] Chọn hàng loạt trong QTableWidget/QTableView trong MỘT lần (thay vì ``selectRow`` từng hàng).

``selectRow`` mỗi hàng ở chế độ MultiSelection phát ``itemSelectionChanged`` và dựng lại vùng chọn mỗi lần — bậc hai:
chọn 1825/3649 dòng mất 81 s (giao diện đơ, Windows báo "Not Responding"); một ``QItemSelection`` mất ~0,09 s.
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QItemSelection, QItemSelectionModel
from PySide6.QtWidgets import QAbstractItemView

__all__ = ["merge_row_ranges", "select_rows_bulk"]


def merge_row_ranges(rows: Iterable[int]) -> list[tuple[int, int]]:
    """Gộp các số hàng thành khoảng liền kề ``(đầu, cuối)``, đã sắp xếp."""
    ranges: list[tuple[int, int]] = []
    for row in sorted(set(rows)):
        if ranges and row == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], row)
        else:
            ranges.append((row, row))
    return ranges


def select_rows_bulk(table: QAbstractItemView, rows: Iterable[int]) -> None:
    """Thay vùng chọn hiện tại bằng đúng các hàng ``rows`` (nguyên hàng), một lần duy nhất."""
    model = table.model()
    selection_model = table.selectionModel()
    if model is None or selection_model is None:
        return
    last_column = max(0, model.columnCount() - 1)
    selection = QItemSelection()
    row_count = model.rowCount()
    for top, bottom in merge_row_ranges(r for r in rows if 0 <= r < row_count):
        selection.select(model.index(top, 0), model.index(bottom, last_column))
    selection_model.select(
        selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
    )
