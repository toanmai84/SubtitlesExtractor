"""Mô hình dữ liệu MVC cho bảng Phụ đề.

TỐI ƯU HÓA ĐỘT PHÁ (V3.45 - Synchronization Polish):
    * [PERFORMANCE] Khởi tạo bộ đệm từ khóa (Text Cache). Hàm `.lower()` chỉ được 
      chạy đúng 1 lần khi nạp phụ đề, giúp ProxyModel tìm kiếm trong nháy mắt
      mà không làm đứng ứng dụng kể cả với 10,000 dòng phụ đề.
    * [UX] Bổ sung SEARCH_QUERY_ROLE: Kết nối Query tìm kiếm từ Proxy Model
      thẳng xuống Delegate để bôi vàng từ khóa theo thời gian thực.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.presentation.utils.time_format import (
    seconds_to_display, srt_to_seconds,
)

COL_INDEX, COL_START, COL_END, COL_DUR, COL_TEXT, COL_SCORE, COL_WARN, COL_GAP = range(8)
_LOW_CONFIDENCE_THRESHOLD: float = 0.6

# Role tùy chỉnh dùng để truyền Query tìm kiếm xuống Delegate
SEARCH_QUERY_ROLE = Qt.ItemDataRole.UserRole + 1


class SubtitleTableModel(QAbstractTableModel):
    text_edit_requested = Signal(int, str)
    time_edit_requested = Signal(int, float, float)
    timing_error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.events: list[SubtitleEvent] = []
        
        # Bộ đệm (Cache) cho việc tìm kiếm siêu tốc
        self._lower_text_cache: list[str] = []

        self._brush_gap_err = QBrush(QColor(255, 204, 204))
        self._brush_conf_warn = QBrush(QColor(255, 210, 210))
        self._brush_short_warn = QBrush(QColor(255, 255, 200))
        self._brush_fast_warn = QBrush(QColor(255, 240, 200))
        self._brush_fg_err = QBrush(QColor(180, 30, 30))

        self._font_bold = QFont()
        self._font_bold.setBold(True)
        self._active_row: int = -1

    def set_events(self, new_events: list[SubtitleEvent]) -> None:
        self.beginResetModel()
        self.events = new_events
        # Nạp Cache `.lower()` một lần duy nhất khi Load dữ liệu
        self._lower_text_cache = [e.text.lower() for e in new_events]
        self.endResetModel()

    def set_active_row(self, row: int) -> None:
        if self._active_row == row: return
        old_row = self._active_row
        self._active_row = row
        if 0 <= old_row < len(self.events):
            self.dataChanged.emit(self.index(old_row, 0), self.index(old_row, 7))
        if 0 <= row < len(self.events):
            self.dataChanged.emit(self.index(row, 0), self.index(row, 7))

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return len(self.events)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 8

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return ["#", "Bắt đầu", "Kết thúc", "Thời lượng", "Nội dung", "Điểm", "⚠", "Cách"][section]
        return super().headerData(section, orientation, role)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        default = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if index.column() in (COL_START, COL_END, COL_TEXT):
            return default | Qt.ItemFlag.ItemIsEditable
        return default

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self.events): return None

        row, col = index.row(), index.column()
        event = self.events[row]

        gap_val = 0.0
        if row + 1 < len(self.events):
            gap_val = self.events[row + 1].start_sec - event.end_sec
            if abs(gap_val) < 0.001: gap_val = 0.0

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if col == COL_INDEX: return str(event.index)
            if col == COL_START: return seconds_to_display(event.start_sec)
            if col == COL_END: return seconds_to_display(event.end_sec)
            if col == COL_DUR: return f"{event.duration_sec:.3f}s"
            if col == COL_TEXT: return event.text
            
            conf_float = float(event.confidence)
            if col == COL_SCORE: return f"{conf_float:.2f}"
            if col == COL_WARN:
                w = []
                if conf_float < _LOW_CONFIDENCE_THRESHOLD: w.append("●")
                if event.is_too_short: w.append("⏱")
                if event.is_too_fast: w.append("⚡")
                return " ".join(w)
            if col == COL_GAP:
                return f"⚠ {gap_val:.3f}s" if gap_val < 0 else (f"{gap_val:.3f}s" if row + 1 < len(self.events) else "—")

        elif role == Qt.ItemDataRole.UserRole and col == COL_TEXT:
            return event.text

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (COL_INDEX, COL_START, COL_END, COL_DUR, COL_SCORE, COL_WARN, COL_GAP):
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        elif role == Qt.ItemDataRole.BackgroundRole:
            if gap_val < 0: return self._brush_gap_err
            if float(event.confidence) < _LOW_CONFIDENCE_THRESHOLD: return self._brush_conf_warn
            if event.is_too_short: return self._brush_short_warn
            if event.is_too_fast: return self._brush_fast_warn

        elif role == Qt.ItemDataRole.ForegroundRole:
            if col == COL_GAP and gap_val < 0: return self._brush_fg_err

        elif role == Qt.ItemDataRole.FontRole:
            if row == self._active_row: return self._font_bold

        elif role == Qt.ItemDataRole.ToolTipRole and col == COL_WARN:
            tips = []
            conf_f = float(event.confidence)
            if conf_f < _LOW_CONFIDENCE_THRESHOLD: tips.append(f"Điểm OCR thấp: {conf_f:.2f}")
            if event.is_too_short: tips.append(f"Quá ngắn: {event.duration_sec:.2f}s (< 0.5s)")
            if event.is_too_fast: tips.append(f"CPS cao: {event.cps:.1f} (> 20)")
            return "\n".join(tips) if tips else None

        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole: return False

        row, col = index.row(), index.column()
        event = self.events[row]

        if col == COL_TEXT:
            new_text = str(value)
            if new_text != event.text:
                self.text_edit_requested.emit(row, new_text)
                # Cập nhật cache ngay khi người dùng sửa chữ trên Table
                self._lower_text_cache[row] = new_text.lower()
            return True
        elif col in (COL_START, COL_END):
            try:
                new_val_sec = srt_to_seconds(str(value))
                if col == COL_START: self.time_edit_requested.emit(row, new_val_sec, event.end_sec)
                else: self.time_edit_requested.emit(row, event.start_sec, new_val_sec)
                return True
            except (ValueError, AttributeError):
                self.timing_error_occurred.emit("Mốc thời gian không hợp lệ. Định dạng: H:MM:SS.mmm")
                return False
        return False


class SubtitleFilterProxyModel(QSortFilterProxyModel):
    #: [v3.23.654] Chế độ lọc "Nghi ngờ OCR" (xem ``ocr_suspicion``).
    FILTER_SUSPECT = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_query: str = ""
        self._filter_mode: int = 0
        self._suspects: dict[int, list[str]] | None = None
        #: [v3.23.657] Đổi mã lý do → chữ hiển thị (trang Chỉnh sửa gắn bản dịch; mặc định giữ mã).
        self._reason_label: Callable[[str], str] = str

    def set_reason_labeler(self, labeler: Callable[[str], str]) -> None:
        """[v3.23.657] Hàm ``mã lý do → nhãn`` cho cột cảnh báo/tooltip của bộ lọc "Nghi ngờ OCR"."""
        self._reason_label = labeler

    def suspect_reasons_text(self, source_row: int, separator: str = ", ") -> str:
        """[v3.23.657] Lý do nghi của một dòng (chuỗi rỗng nếu không nghi)."""
        return separator.join(self._reason_label(r) for r in self.suspects().get(source_row, ()))

    def set_filter_params(self, query: str, mode: int) -> None:
        self._search_query = query.strip().lower()
        self._filter_mode = mode
        self._suspects = None  # tính lại theo danh sách dòng hiện tại (đã sửa / tách / gộp)
        self.invalidateFilter()

    def setSourceModel(self, model) -> None:  # noqa: N802 — API Qt
        previous = self.sourceModel()
        if previous is not None:
            for signal, slot in self._suspect_hooks(previous):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        super().setSourceModel(model)
        if model is not None:
            for signal, slot in self._suspect_hooks(model):
                signal.connect(slot)

    def _suspect_hooks(self, model):
        # Danh sách nghi tính theo HÀNG: phải bỏ TRƯỚC khi proxy lọc lại theo dữ liệu mới ("AboutTo…" chạy trước bộ
        # xử lý nội bộ của proxy); sửa chữ (dataChanged) thì lọc lại sau khi bỏ.
        return (
            (model.modelAboutToBeReset, self._drop_suspects),
            (model.rowsAboutToBeInserted, self._drop_suspects),
            (model.rowsAboutToBeRemoved, self._drop_suspects),
            (model.dataChanged, self._on_source_data_changed),
        )

    def _drop_suspects(self, *_args) -> None:
        self._suspects = None

    def _on_source_data_changed(self, _top_left=None, _bottom_right=None, roles=()) -> None:
        # Tô hàng đang phát (BackgroundRole…) đổi mỗi nhịp phát: không đụng chữ thì không tính lại.
        if roles and Qt.ItemDataRole.DisplayRole not in roles and Qt.ItemDataRole.EditRole not in roles:
            return
        self._suspects = None
        if self._filter_mode == self.FILTER_SUSPECT:
            self.invalidateFilter()

    def suspects(self) -> dict[int, list[str]]:
        """[v3.23.654] ``{hàng nguồn: lý do}`` các dòng nghi OCR sai (tính lười, theo danh sách dòng hiện tại)."""
        if self._suspects is None:
            from subtitles_extractor.application.services.ocr_suspicion import ocr_suspicions

            model = self.sourceModel()
            self._suspects = ocr_suspicions(list(getattr(model, "events", []) or []))
        return self._suspects

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == SEARCH_QUERY_ROLE: return self._search_query
        # [v3.23.657] Đang lọc "Nghi ngờ OCR": cột cảnh báo ghi LÝ DO (trước đây chỉ lọc, không biết vì sao dòng bị nghi).
        if (
            self._filter_mode == self.FILTER_SUSPECT and index.isValid() and index.column() == COL_WARN
            and role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole)
        ):
            source_row = self.mapToSource(index).row()
            reasons = self.suspect_reasons_text(source_row, ", " if role == Qt.ItemDataRole.DisplayRole else "\n")
            if reasons:
                base = super().data(index, role)
                if role == Qt.ItemDataRole.DisplayRole:
                    return f"{base} 🔎 {reasons}".strip() if base else f"🔎 {reasons}"
                return f"{reasons}\n{base}" if base else reasons
        return super().data(index, role)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model: SubtitleTableModel = self.sourceModel() # type: ignore
        if not model or source_row >= len(model.events): return False
        
        # Bỏ qua việc gọi event.text.lower() tốn kém, xài thẳng Cache
        if self._search_query and self._search_query not in model._lower_text_cache[source_row]: 
            return False

        if self._filter_mode == 1:
            event = model.events[source_row]
            if float(event.confidence) >= _LOW_CONFIDENCE_THRESHOLD: return False
        elif self._filter_mode == 2:
            event = model.events[source_row]
            if not (event.is_too_short or event.is_too_fast): return False
        elif self._filter_mode == self.FILTER_SUSPECT:
            if source_row not in self.suspects(): return False
            
        return True

__all__ = ["SubtitleTableModel", "SubtitleFilterProxyModel", "COL_INDEX", "COL_START", "COL_END", "COL_DUR", "COL_TEXT", "COL_SCORE", "COL_WARN", "COL_GAP", "SEARCH_QUERY_ROLE"]
