"""[v3.23.598] Sửa tay TỪNG DÒNG trên bảng dòng của trang TTS: người nói (giọng) và lời.

Tách khỏi ``tts_page`` (đã gần 3000 dòng). Bộ điều khiển này lo:

* hiện các dòng phụ đề NGAY khi nạp (trước đây bảng trống tới khi TTS xong) để xem và
  sửa người nói, lời thoại trước khi tổng hợp;
* cột "Người nói · Giọng" (ẩn khi tắt lồng tiếng) và tô các dòng đã sửa mà chưa TTS lại;
* menu người nói (chuột phải, nhấp đúp ô người nói, nút "Đổi người nói"), bỏ sửa tay,
  chọn nhanh các dòng đã sửa để TTS lại;
* nhớ sửa tay theo phụ đề (QSettings) và áp lại khi nạp lại đúng phụ đề đó.

Việc tính toán (người nói sau sửa tay, dấu phụ đề, lưu/đọc) nằm ở
``application.services.dubbing_line_edits``; giọng và vai do
:class:`TTSDubbingCoordinator` cung cấp.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from functools import partial

from PySide6.QtCore import QObject, QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractButton,
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from subtitles_extractor.application.services.dubbing_line_edits import (
    SHARED_VOICE_SPEAKER,
    decode_line_edits,
    remember_line_edits,
)
from subtitles_extractor.presentation.utils.table_selection import select_rows_bulk
from subtitles_extractor.presentation.fluent_compat import PushButton
from subtitles_extractor.presentation.pages.tts_dubbing import (
    SpeakerCell,
    TTSDubbingCoordinator,
)
from subtitles_extractor.presentation.view_models.tts_page_view_model import (
    TTSPageViewModel,
)
from subtitles_extractor.application.services.dubbing_line_structure import suggest_split
from subtitles_extractor.application.services.speaker_review_flow import same_speaker_run
from subtitles_extractor.application.services.speaker_review_session import (
    SpeakerReviewSession,
)
from subtitles_extractor.presentation.widgets.add_line_dialog import (
    AddLineDialog,
    NewLineRequest,
    SplitLineDialog,
)
from subtitles_extractor.presentation.widgets.dubbing_cast_panel import DubbingCastPanel

logger = logging.getLogger(__name__)

#: Cột của bảng dòng trang TTS (chỉ số LOGIC — cột người nói được dời lên cạnh lời).
COLUMN_INDEX = 0
COLUMN_TIME = 1
COLUMN_TEXT = 2
COLUMN_STATUS = 6
COLUMN_SPEAKER = 7
COLUMN_COUNT = 8
#: Khoá QSettings: ``{dấu phụ đề: sửa tay đã mã hoá}`` (JSON).
SETTINGS_LINE_EDITS_KEY = "dubbing_line_edits"
#: Bảng lớn hơn thì bỏ tự giãn chiều cao hàng (đắt với hàng nghìn dòng).
MAX_ROWS_AUTO_HEIGHT = 1500
_SPEAKER_COLUMN_WIDTH = 175
_EDITED_BACKGROUND = QColor(222, 236, 255)
_EDITED_FOREGROUND = QColor(20, 60, 120)
#: [v3.23.599] Chữ ô người nói của dòng cần soát (cam đậm, như dòng "dài bất thường").
_REVIEW_FOREGROUND = QColor(170, 90, 0)
#: Tên vai tự tìm hiện tối đa bao nhiêu chữ trong mục "Tự động (...)" của menu.
_AUTO_LABEL_MAX = 48

#: Hộp hỏi tên vai mới (tách ra để kiểm thử thay được).
ask_text = QInputDialog.getText


def ask_new_line(
    parent, title: str, labels: dict[str, str], start_sec: float, end_sec: float,
    speakers: list[tuple[str, str | None]],
) -> NewLineRequest | None:
    """[v3.23.625] Hộp thoại thêm dòng (tách ra để kiểm thử thay được)."""
    dialog = AddLineDialog(parent, title, labels, start_sec, end_sec, speakers)
    return dialog.request() if dialog.exec() == AddLineDialog.DialogCode.Accepted else None


def ask_line_timing(
    parent, title: str, labels: dict[str, str], start_sec: float, end_sec: float
) -> tuple[float, float] | None:
    """[v3.23.626] Hộp thoại đổi mốc thời gian một dòng (tách ra để kiểm thử thay được)."""
    dialog = AddLineDialog(parent, title, labels, start_sec, end_sec, [], text=None)
    if dialog.exec() != AddLineDialog.DialogCode.Accepted:
        return None
    return dialog.start_box.value(), dialog.end_box.value()


def ask_split(
    parent, title: str, labels: dict[str, str], text_a: str, text_b: str,
    start_sec: float, end_sec: float, split_sec: float,
) -> tuple[str, str, float] | None:
    """[v3.23.626] Hộp thoại tách dòng (tách ra để kiểm thử thay được)."""
    dialog = SplitLineDialog(
        parent, title, labels, text_a, text_b, start_sec, end_sec, split_sec
    )
    if dialog.exec() != SplitLineDialog.DialogCode.Accepted:
        return None
    return dialog.result_values()


def ask_shorten_preview(parent, title: str, intro: str, headers: list[str], lines: list):
    """[v3.23.629] Xem trước câu rút gọn, trả các dòng được tick (tách ra để kiểm thử thay được)."""
    from subtitles_extractor.presentation.widgets.shorten_preview_dialog import (
        ShortenPreviewDialog,
    )

    dialog = ShortenPreviewDialog(parent, title, intro, headers, lines)
    return dialog.accepted_lines() if dialog.exec() == ShortenPreviewDialog.DialogCode.Accepted else []


#: [v3.23.630] Khoá chuỗi giao diện của hộp thoại duyệt nhanh (``tts.rev_<khoá>``).
REVIEW_TEXT_KEYS = (
    "title", "intro", "scope_all", "col_index", "col_time", "col_speaker", "col_source",
    "col_text", "btn_shared", "btn_keep", "btn_previous", "btn_undo", "btn_done",
    "btn_cancel", "other", "progress", "finished", "run", "pending", "shared", "changed_tip",
)


def ask_speaker_review(
    parent, session: SpeakerReviewSession, texts: dict[str, str], display_name,
    roles: list[tuple[str, str]], reason_of,
) -> dict[int, str] | None:
    """[v3.23.630] Hộp thoại duyệt nhanh người nói; trả ``{vị trí: người nói}`` hoặc ``None``
    nếu huỷ (tách ra để kiểm thử thay được)."""
    from subtitles_extractor.presentation.widgets.speaker_review_dialog import (
        SpeakerReviewDialog,
    )

    dialog = SpeakerReviewDialog(parent, session, texts, display_name, roles, reason_of)
    if dialog.exec() != SpeakerReviewDialog.DialogCode.Accepted:
        return None
    return dialog.result()


def confirm_delete(parent, title: str, text: str) -> bool:
    """[v3.23.625] Hỏi xác nhận xoá dòng (tách ra để kiểm thử thay được)."""
    answer = QMessageBox.question(
        parent, title, text,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def format_row_time(seconds: float) -> str:
    """Mốc thời gian một dòng trên bảng: ``mm:ss.mmm``."""
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def read_only_item(
    text: str,
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
) -> QTableWidgetItem:
    """Ô chỉ đọc.

    Ô ``QTableWidgetItem`` mặc định SỬA ĐƯỢC: bảng kết quả trước bản này cho nhấp đúp sửa
    cả cột "#", "Thời gian", "Trạng thái" (chữ đổi trên bảng nhưng không có tác dụng gì).
    """
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


def row_head_item(position: int) -> QTableWidgetItem:
    """Ô "#" của một dòng; giữ VỊ TRÍ dòng nguồn (0-based) ở ``UserRole``."""
    item = read_only_item(
        str(position + 1), Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
    )
    item.setData(Qt.ItemDataRole.UserRole, int(position))
    return item


def editable_text_item(text: str) -> QTableWidgetItem:
    """Ô lời thoại — sửa được bằng nhấp đúp."""
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
    return item


def selected_positions(table: QTableWidget) -> list[int]:
    """Vị trí dòng nguồn của các hàng đang chọn (đọc ``UserRole`` của cột "#")."""
    positions: list[int] = []
    for item in table.selectedItems():
        if item.column() != COLUMN_INDEX:
            continue
        value = item.data(Qt.ItemDataRole.UserRole)
        if value is not None:
            positions.append(int(value))
    return sorted(set(positions))


class TTSLineEditsController(QObject):
    """Sửa người nói / lời thoại từng dòng trên bảng dòng trang TTS."""

    def __init__(
        self,
        table: QTableWidget,
        view_model: TTSPageViewModel,
        dubbing: TTSDubbingCoordinator,
        panel: DubbingCastPanel,
        settings: QSettings,
        translate: Callable[[str], str],
        *,
        commit_edit: Callable[[], None],
        show_status: Callable[[str], None],
        context_buttons: Sequence[QAbstractButton] = (),
        rebuild_table: Callable[[], None] | None = None,
        resynthesize: Callable[[set[int]], None] | None = None,
        shorten: Callable[[list], None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Nối bảng dòng với view model, bộ điều phối lồng tiếng và cài đặt.

        Args:
            table: Bảng dòng (đủ :data:`COLUMN_COUNT` cột).
            view_model: View model trang TTS.
            dubbing: Bộ điều phối lồng tiếng (vai, giọng, sửa người nói).
            panel: Bảng phân vai (bật/tắt lồng tiếng, giọng từng vai).
            settings: QSettings của trang TTS (nhớ sửa tay theo phụ đề).
            translate: Hàm dịch chuỗi giao diện (``"tts.<khoá>"`` → chuỗi).
            commit_edit: Đẩy ô lời đang mở sửa vào dữ liệu trước khi thao tác.
            show_status: Hiện một dòng trạng thái cho người dùng.
            context_buttons: Nút của trang đưa thêm vào menu chuột phải (Nghe, TTS lại).
            rebuild_table: [v3.23.625] Vẽ lại bảng dòng sau khi thêm/xoá dòng (trang có kết quả
                TTS hiện bằng bảng kết quả, chưa TTS thì bằng các dòng nguồn).
            resynthesize: [v3.23.625] TTS lại các dòng ``{vị trí}`` (dòng mới thêm).
            parent: QObject cha.
        """
        super().__init__(parent)
        self._rebuild_table = rebuild_table
        self._resynthesize = resynthesize
        self._shorten = shorten
        self._table = table
        self._view_model = view_model
        self._dubbing = dubbing
        self._panel = panel
        self._settings = settings
        self._translate = translate
        self._commit_edit = commit_edit
        self._show_status = show_status
        self._context_buttons = list(context_buttons)
        self._refresh_pending = False
        self._review_count = 0
        self._build_buttons()
        self._prepare_table()
        view_model.structure_changed.connect(self._on_structure_changed)
        view_model.line_edits_changed.connect(self._save)
        view_model.line_edits_changed.connect(self.schedule_refresh)
        view_model.busy_changed.connect(lambda _busy: self._update_buttons())
        dubbing.line_speakers_changed.connect(self.schedule_refresh)
        panel.enabled_changed.connect(self._on_dubbing_enabled)
        table.itemSelectionChanged.connect(self._update_buttons)

    # ── Public API ────────────────────────────────────────────────────────────

    def button_row(self) -> QVBoxLayout:
        """Các hàng nút sửa tay (đặt dưới bảng dòng)."""
        return self._button_row

    def selected_positions(self) -> list[int]:
        """Vị trí dòng nguồn của các hàng đang chọn."""
        return selected_positions(self._table)

    def on_source_changed(self, count: int) -> None:
        """Phụ đề mới: áp lại sửa tay đã lưu, nạp vai (nếu bật lồng tiếng), hiện các dòng.

        Phải chạy SAU khi trang xoá kết quả cũ (nối tín hiệu sau ``_on_source_changed``).

        Args:
            count: Số dòng vừa nạp.
        """
        speakers, texts = self._restore() if count > 0 else (0, 0)
        if count > 0 and self._panel.is_dubbing_enabled():
            self._dubbing.load_cast()
        self.show_source_rows()
        if speakers or texts:
            self._show_status(
                self._tr("line_edits_restored")
                .replace("{speakers}", str(speakers))
                .replace("{texts}", str(texts))
            )

    def show_source_rows(self) -> None:
        """Hiện các dòng phụ đề đang nạp (chưa tổng hợp) để xem/sửa trước khi TTS."""
        events = self._view_model.source_events
        pending = self._tr("st_pending")
        self._table.setSortingEnabled(False)
        self._table.blockSignals(True)
        self._table.setUpdatesEnabled(False)
        try:
            self._table.clearContents()
            self._table.setRowCount(len(events))
            for position, event in enumerate(events):
                self._table.setItem(position, COLUMN_INDEX, row_head_item(position))
                self._table.setItem(position, COLUMN_TIME, read_only_item(
                    format_row_time(float(getattr(event, "start_sec", 0.0)))
                ))
                self._table.setItem(position, COLUMN_TEXT, editable_text_item(
                    str(getattr(event, "text", "") or "")
                ))
                for column in range(COLUMN_TEXT + 1, COLUMN_STATUS):
                    self._table.setItem(position, column, read_only_item(""))
                self._table.setItem(position, COLUMN_STATUS, read_only_item(pending))
        finally:
            self._table.setUpdatesEnabled(True)
            self._table.blockSignals(False)
        if len(events) <= MAX_ROWS_AUTO_HEIGHT:
            self._table.resizeRowsToContents()
        self.refresh()

    # ── [v3.23.625] Thêm / xoá dòng ───────────────────────────────────────────

    def shorten_lines(self) -> int:
        """Nhờ AI rút gọn các dòng không vừa khung giọng đọc (kết quả xem trước rồi mới áp dụng).

        Returns:
            Số dòng được gửi đi.
        """
        if self._shorten is None or self._view_model.is_busy:
            return 0
        self._commit_edit()
        fits = self._view_model.speech_misfits()
        if not fits:
            self._show_status(self._tr("shorten_none"))
            return 0
        self._shorten(fits)
        return len(fits)

    def apply_shortened(self, lines: list) -> int:
        """Hiện xem trước ``lines`` (``ShortenedLine``) rồi áp dụng các dòng được tick.

        Returns:
            Số dòng đã áp dụng.
        """
        if not lines:
            self._show_status(self._tr("shorten_nothing"))
            return 0
        # [v3.23.650] Yêu cầu rút gọn chạy nền theo VỊ TRÍ dòng lúc gửi; xoá/chèn/gộp/tách dòng hoặc nạp phụ đề khác
        # trong lúc chờ làm vị trí lệch → câu rút gọn ghi đè nhầm dòng. Chỉ giữ dòng mà chữ hiện tại vẫn là `before`.
        events = self._view_model.source_events
        fresh = [
            line for line in lines
            if 0 <= line.position < len(events)
            and str(getattr(events[line.position], "text", "") or "").replace("\n", " ") == line.before
        ]
        if len(fresh) != len(lines):
            logger.info("Rút gọn AI: bỏ %d dòng đã đổi trong lúc chờ.", len(lines) - len(fresh))
        lines = fresh
        if not lines:
            self._show_status(self._tr("shorten_nothing"))
            return 0
        chosen = ask_shorten_preview(
            self._table.window(), self._tr("dlg_shorten_title"), self._tr("dlg_shorten_intro"),
            ["#", self._tr("dlg_shorten_before"), self._tr("dlg_shorten_after"),
             self._tr("dlg_shorten_syl")], lines,
        )
        if not chosen:
            return 0
        for line in chosen:
            self._view_model.update_event_text(line.position, line.after)
        if self._rebuild_table is not None:
            self._rebuild_table()
        self._show_status(self._tr("shorten_done").replace("{count}", str(len(chosen))))
        self._resynthesize_after({line.position for line in chosen})
        return len(chosen)

    def undo(self, _checked: bool = False) -> bool:
        """Hoàn tác thao tác chỉnh dòng gần nhất (chỉ khi chưa có kết quả TTS)."""
        if not self._view_model.can_undo:
            return False
        self._commit_edit()
        done = self._view_model.undo()
        if done:
            self._show_status(self._tr("line_undone"))
        return done

    def redo(self, _checked: bool = False) -> bool:
        """Làm lại thao tác vừa hoàn tác."""
        if not self._view_model.can_redo:
            return False
        done = self._view_model.redo()
        if done:
            self._show_status(self._tr("line_redone"))
        return done

    def edit_timing(self, position: int | None = None, _checked: bool = False) -> bool:
        """Đổi mốc bắt đầu/kết thúc của MỘT dòng (hộp thoại); dòng đó cần TTS lại.

        Returns:
            ``True`` nếu đã đổi.
        """
        if self._view_model.is_busy:
            return False
        self._commit_edit()
        if position is None:
            chosen = self.selected_positions()
            if len(chosen) != 1:
                return False
            position = chosen[0]
        events = self._view_model.source_events
        if not (0 <= position < len(events)):
            return False
        event = events[position]
        picked = ask_line_timing(
            self._table.window(), self._tr("dlg_timing_title"),
            {"start": self._tr("dlg_add_start"), "end": self._tr("dlg_add_end")},
            float(event.start_sec), float(event.end_sec),
        )
        if picked is None:
            return False
        if not self._view_model.set_line_timing(position, *picked):
            self._show_status(self._tr("line_timing_failed"))
            return False
        self._show_status(self._tr("line_timing_done").replace("{n}", str(position + 1)))
        self._resynthesize_after({position})
        return True

    def merge_lines(self, positions: list[int], _checked: bool = False) -> int | None:
        """Gộp các dòng liền nhau đang chọn thành một dòng (cần TTS lại dòng gộp)."""
        if self._view_model.is_busy or len(set(positions)) < 2:
            return None
        self._commit_edit()
        merged = self._view_model.merge_lines(positions)
        if merged is None:
            self._show_status(self._tr("line_merge_failed"))
            return None
        if self._panel.is_dubbing_enabled():
            self._dubbing.reload_cast()
        self._show_status(self._tr("line_merged").replace("{n}", str(merged + 1)))
        self._resynthesize_after({merged})
        return merged

    def split_line(self, position: int | None = None, _checked: bool = False) -> int | None:
        """Tách MỘT dòng thành hai (hộp thoại: lời từng nửa + mốc cắt); cần TTS lại cả hai."""
        if self._view_model.is_busy:
            return None
        self._commit_edit()
        if position is None:
            chosen = self.selected_positions()
            if len(chosen) != 1:
                return None
            position = chosen[0]
        events = self._view_model.source_events
        if not (0 <= position < len(events)):
            return None
        event = events[position]
        start, end = float(event.start_sec), float(event.end_sec)
        first, second, split = suggest_split(str(event.text or ""), start, end)
        picked = ask_split(
            self._table.window(), self._tr("dlg_split_title"),
            {
                "first": self._tr("dlg_split_first"), "second": self._tr("dlg_split_second"),
                "split": self._tr("dlg_split_at"),
            },
            first, second, start, end, split,
        )
        if picked is None:
            return None
        new_position = self._view_model.split_line(position, *picked)
        if new_position is None:
            self._show_status(self._tr("line_split_failed"))
            return None
        if self._panel.is_dubbing_enabled():
            self._dubbing.reload_cast()
        self._show_status(self._tr("line_split").replace("{n}", str(position + 1)))
        self._resynthesize_after({position, new_position})
        return new_position

    def _resynthesize_after(self, positions: set[int]) -> None:
        """Đã có kết quả TTS: tổng hợp lại ``positions`` (hoặc báo phải TTS lại)."""
        if not self._view_model.has_result:
            return
        if self._resynthesize is not None and self._view_model.can_resynthesize:
            self._resynthesize(positions)
        else:
            self._show_status(self._tr("line_added_retts"))

    def add_line(self, after_position: int | None = None, _checked: bool = False) -> int | None:
        """Thêm một dòng mới sau dòng ``after_position`` (mặc định: dòng đang chọn cuối).

        Hộp thoại hỏi mốc, lời thoại, người nói. Đã có kết quả TTS (VieNeu giữ audio từng
        dòng) thì dòng mới được tổng hợp ngay và master ghép lại.

        Returns:
            Vị trí dòng mới, hoặc ``None`` nếu huỷ / không thêm được.
        """
        if self._view_model.is_busy or not self._view_model.has_source:
            return None
        self._commit_edit()
        if after_position is None:
            chosen = self.selected_positions()
            after_position = max(chosen) if chosen else None
        start, end = self._view_model.suggested_line_timing(after_position)
        request = ask_new_line(
            self._table.window(), self._tr("dlg_add_title"),
            {
                "start": self._tr("dlg_add_start"), "end": self._tr("dlg_add_end"),
                "text": self._tr("dlg_add_text"), "speaker": self._tr("dlg_add_speaker"),
            },
            start, end, self._speaker_choices(),
        )
        if request is None:
            return None
        position = self._view_model.insert_line(
            request.start_sec, request.end_sec, request.text
        )
        if position is None:
            self._show_status(self._tr("line_add_failed"))
            return None
        if request.speaker is not None:
            self._dubbing.set_line_speaker([position], request.speaker)
        elif self._panel.is_dubbing_enabled():
            self._dubbing.reload_cast()
        self._show_status(self._tr("line_added").replace("{n}", str(position + 1)))
        if self._view_model.has_result and self._resynthesize is not None:
            if self._view_model.can_resynthesize:
                self._resynthesize({position})
            else:
                self._show_status(self._tr("line_added_retts"))
        return position

    def delete_lines(self, positions: list[int], _checked: bool = False) -> int:
        """Xoá các dòng (có hỏi xác nhận); đã có kết quả TTS thì ghép lại master.

        Returns:
            Số dòng đã xoá.
        """
        if not positions or self._view_model.is_busy:
            return 0
        self._commit_edit()
        if not confirm_delete(
            self._table.window(), self._tr("dlg_delete_title"),
            self._tr("dlg_delete_text").replace("{count}", str(len(set(positions)))),
        ):
            return 0
        had_result = self._view_model.has_result
        removed = self._view_model.delete_lines(positions)
        if not removed:
            self._show_status(self._tr("line_delete_failed"))
            return 0
        if self._panel.is_dubbing_enabled():
            self._dubbing.reload_cast()
        message = self._tr("line_deleted").replace("{count}", str(removed))
        if had_result and not self._view_model.remerge_master():
            message += " " + self._tr("line_deleted_retts")
        self._show_status(message)
        return removed

    @staticmethod
    def _can_merge(positions: list[int]) -> bool:
        """Có ít nhất hai dòng và chúng liền nhau không."""
        chosen = sorted(set(positions))
        return len(chosen) >= 2 and chosen[-1] - chosen[0] == len(chosen) - 1

    def _speaker_choices(self) -> list[tuple[str, str | None]]:
        """Lựa chọn người nói cho hộp thoại thêm dòng."""
        choices: list[tuple[str, str | None]] = [
            (self._tr("line_speaker_auto"), None),
            (self._translate("tts.dub_voice_shared"), SHARED_VOICE_SPEAKER),
        ]
        if self._panel.is_dubbing_enabled():
            for member in self._dubbing.role_choices():
                choices.append((member.display_name, member.name))
        return choices

    def _on_structure_changed(self) -> None:
        if self._rebuild_table is not None:
            self._rebuild_table()
        else:
            self.show_source_rows()

    def schedule_refresh(self) -> None:
        """Gom nhiều thay đổi liền nhau (sửa tay → nạp lại vai) thành một lần vẽ lại."""
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self.refresh)

    def refresh(self) -> None:
        """Vẽ lại cột người nói, tô dòng đã sửa chưa TTS lại, cập nhật nút."""
        self._refresh_pending = False
        enabled = self._panel.is_dubbing_enabled()
        self._table.setColumnHidden(COLUMN_SPEAKER, not enabled)
        rows = self._table.rowCount()
        cells = self._dubbing.line_speaker_cells() if enabled and rows else []
        self._review_count = sum(1 for cell in cells if cell.needs_review)
        edited = (
            self._view_model.edited_positions if self._view_model.has_result
            else frozenset()
        )
        edited_tip = self._tr("line_edited_tip")
        self._table.blockSignals(True)
        self._table.setUpdatesEnabled(False)
        try:
            for row in range(rows):
                position = self._row_position(row)
                if position is None:
                    continue
                if 0 <= position < len(cells):
                    self._paint_speaker(row, cells[position])
                text_item = self._table.item(row, COLUMN_TEXT)
                if text_item is not None:
                    self._paint_edited(text_item, position in edited, edited_tip)
        finally:
            self._table.setUpdatesEnabled(True)
            self._table.blockSignals(False)
        self._update_buttons()

    def build_speaker_menu(self, positions: list[int]) -> QMenu:
        """Menu chọn người nói cho các dòng ``positions``.

        Mục: "Tự động (vai tự tìm)", các vai đang có thoại kèm giọng, nhân vật khác trong
        bảng nhân vật, "Giọng chung", "Vai khác…".
        """
        menu = QMenu(self._tr("menu_speaker"), self._table)
        # Vai phải nạp TRƯỚC khi đọc người nói từng dòng (role_choices nạp nếu cần).
        choices = self._dubbing.role_choices()
        lines = self._dubbing.line_speakers()
        picked = [lines[position] for position in positions if position < len(lines)]
        current = {line.speaker for line in picked}
        any_overridden = any(line.overridden for line in picked)

        autos = sorted({
            self._dubbing.display_name(line.auto_speaker) or "?" for line in picked
        })
        auto_text = ", ".join(autos)
        if len(auto_text) > _AUTO_LABEL_MAX:
            auto_text = auto_text[: _AUTO_LABEL_MAX - 1] + "…"
        self._add_choice(
            menu, self._tr("menu_speaker_auto").replace("{auto}", auto_text),
            positions, None, checked=bool(picked) and not any_overridden,
        )
        menu.addSeparator()
        casting = self._panel.casting()
        roster_menu: QMenu | None = None
        for member in choices:
            if member.line_count:
                voice = self._dubbing.voice_label(casting.get(member.name, ""))
                target, label = menu, f"{member.display_name} · {voice}"
            else:
                if roster_menu is None:
                    roster_menu = menu.addMenu(self._tr("menu_speaker_roster"))
                target, label = roster_menu, member.display_name
            action = self._add_choice(
                target, label, positions, member.name,
                checked=any_overridden and current == {member.name},
            )
            action.setToolTip(member.name)
        menu.addSeparator()
        self._add_choice(
            menu, self._tr("menu_speaker_shared"), positions, SHARED_VOICE_SPEAKER,
            checked=any_overridden and current == {SHARED_VOICE_SPEAKER},
        )
        other = menu.addAction(self._tr("menu_speaker_other"))
        other.triggered.connect(partial(self._ask_speaker, positions))
        menu.setToolTipsVisible(True)
        return menu

    def build_context_menu(self, positions: list[int]) -> QMenu:
        """Menu chuột phải của bảng dòng."""
        menu = QMenu(self._table)
        busy = self._view_model.is_busy
        speaker_menu = self.build_speaker_menu(positions)
        speaker_menu.setEnabled(bool(positions) and not busy)
        menu.addMenu(speaker_menu)
        edit_action = menu.addAction(self._tr("menu_edit_text"))
        edit_action.setEnabled(len(positions) == 1)
        edit_action.triggered.connect(partial(self._edit_text, positions))
        revert_action = menu.addAction(self._tr("menu_revert"))
        revert_action.setEnabled(
            self._view_model.has_line_edits(positions) and not busy
        )
        revert_action.triggered.connect(partial(self.revert, positions))
        run_action = menu.addAction(self._tr("menu_select_run"))
        run_action.setEnabled(len(positions) == 1 and not busy)
        run_action.triggered.connect(partial(self.select_run, positions[0] if positions else 0))
        review_action = menu.addAction(self._tr("menu_quick_review"))
        review_action.setEnabled(bool(positions) and not busy)
        review_action.triggered.connect(
            partial(self.quick_review, min(positions) if positions else None)
        )
        menu.addSeparator()
        add_action = menu.addAction(self._tr("menu_add_line"))
        add_action.setEnabled(not busy)
        add_action.triggered.connect(partial(self.add_line, max(positions) if positions else None))
        timing_action = menu.addAction(self._tr("menu_timing"))
        timing_action.setEnabled(len(positions) == 1 and not busy)
        timing_action.triggered.connect(partial(self.edit_timing, positions[0] if positions else None))
        merge_action = menu.addAction(self._tr("menu_merge_lines"))
        merge_action.setEnabled(self._can_merge(positions) and not busy)
        merge_action.triggered.connect(partial(self.merge_lines, positions))
        split_action = menu.addAction(self._tr("menu_split_line"))
        split_action.setEnabled(len(positions) == 1 and not busy)
        split_action.triggered.connect(partial(self.split_line, positions[0] if positions else None))
        delete_action = menu.addAction(self._tr("menu_delete_lines"))
        delete_action.setEnabled(bool(positions) and not busy)
        delete_action.triggered.connect(partial(self.delete_lines, positions))
        if self._context_buttons:
            menu.addSeparator()
            for button in self._context_buttons:
                action = menu.addAction(button.text())
                action.setEnabled(button.isEnabled())
                action.triggered.connect(button.click)
        return menu

    def apply_speaker(
        self, positions: list[int], speaker: str | None, _checked: bool = False
    ) -> int:
        """Đổi người nói của các dòng và báo cho người dùng.

        Args:
            positions: Vị trí dòng.
            speaker: Tên vai; rỗng = giọng chung; ``None`` = người nói tự tìm.
            _checked: Tham số ``triggered(bool)`` của QAction (bỏ qua).

        Returns:
            Số dòng thay đổi.
        """
        if not positions:
            return 0
        self._commit_edit()
        changed = self._dubbing.set_line_speaker(positions, speaker)
        if changed:
            self._show_status(
                self._tr("line_speaker_done")
                .replace("{count}", str(changed))
                .replace("{speaker}", self._speaker_text(speaker))
            )
        return changed

    def revert(self, positions: list[int], _checked: bool = False) -> int:
        """Bỏ sửa tay của các dòng; trả ô lời về chữ lúc nạp.

        Returns:
            Số dòng có sửa tay bị bỏ.
        """
        if not self._view_model.has_line_edits(positions):
            return 0
        self._commit_edit()
        count = sum(
            1 for position in set(positions)
            if self._view_model.has_line_edits([position])
        )
        reverted = set(self._dubbing.revert_lines(positions))
        events = self._view_model.source_events
        # Đặt chữ KHÔNG chặn tín hiệu: trang đồng bộ kết quả đang hiện qua itemChanged
        # (view model bỏ qua vì chữ đã đúng).
        for row in range(self._table.rowCount()):
            position = self._row_position(row)
            item = self._table.item(row, COLUMN_TEXT)
            if position in reverted and item is not None and position < len(events):
                item.setText(str(getattr(events[position], "text", "") or ""))
        self._show_status(self._tr("line_edits_reverted").replace("{count}", str(count)))
        return count

    def select_edited(self, _checked: bool = False) -> int:
        """Chọn mọi dòng đã sửa sau lần tổng hợp gần nhất (để TTS lại một lượt).

        Returns:
            Số dòng được chọn.
        """
        count = self._select_positions(set(self._view_model.edited_positions))
        if count:
            self._show_status(
                self._tr("line_edits_selected").replace("{count}", str(count))
            )
        return count

    def select_review(self, _checked: bool = False) -> int:
        """[v3.23.599] Chọn mọi dòng cần soát người nói (chưa rõ người nói / nhãn sai).

        Returns:
            Số dòng được chọn.
        """
        count = self._select_positions(set(self._dubbing.review_positions()))
        if count:
            self._show_status(
                self._tr("review_selected").replace("{count}", str(count))
            )
        return count

    def quick_review(self, start_position: int | None = None, _checked: bool = False) -> int:
        """[v3.23.630] Duyệt nhanh người nói bằng phím số (hộp thoại), áp dụng một lượt khi xong.

        Mặc định đi qua các dòng nghi sai (bằng chứng chắc nhất trước); không có dòng nghi sai,
        hoặc bắt đầu từ một dòng cụ thể, thì duyệt mọi dòng từ đó.

        Args:
            start_position: Dòng bắt đầu (``None`` = đầu hàng đợi nghi sai).

        Returns:
            Số dòng đã đổi người nói.
        """
        if self._view_model.is_busy or not self._view_model.has_source:
            return 0
        self._commit_edit()
        choices = self._dubbing.role_choices()   # nạp vai trước khi đọc người nói
        if not self._view_model.dubbing_cast:
            self._show_status(self._tr("quick_review_no_cast"))
            return 0
        lines = self._view_model.speaker_review_lines()
        issues = self._view_model.speaker_issues()
        queue = [item.position for item in self._view_model.speaker_review_queue()]
        session = SpeakerReviewSession(lines, queue)
        if start_position is not None:
            session.start_at(start_position)
        elif not queue:
            self._show_status(self._tr("quick_review_none"))
            session.set_scope(True)
        texts = {key: self._tr(f"rev_{key}") for key in REVIEW_TEXT_KEYS}
        roles = [(member.display_name, member.name) for member in choices]
        changes = ask_speaker_review(
            self._table.window(), session, texts, self._dubbing.display_name, roles,
            lambda position: (
                self._dubbing.issue_text(issues[position]) if position in issues else ""
            ),
        )
        if not changes:
            return 0
        by_speaker: dict[str, list[int]] = {}
        for position, speaker in sorted(changes.items()):
            by_speaker.setdefault(speaker, []).append(position)
        changed = 0
        for speaker, positions in by_speaker.items():
            changed += self.apply_speaker(positions, speaker)
        self._show_status(self._tr("quick_review_done").replace("{count}", str(changed)))
        return changed

    def select_run(self, position: int, _checked: bool = False) -> int:
        """[v3.23.630] Chọn cả đoạn liền kề cùng người nói chứa dòng ``position``.

        Một lời thoại dài tách nhiều dòng phụ đề thường bị gán sai CẢ đoạn: chọn đoạn rồi
        đổi người nói một lần.

        Returns:
            Số dòng được chọn.
        """
        speakers = self._view_model.effective_line_speakers()
        events = self._view_model.source_events
        run = same_speaker_run(
            speakers, [float(e.start_sec) for e in events],
            [float(e.end_sec) for e in events], position,
        )
        count = self._select_positions(set(run))
        if count:
            self._show_status(self._tr("run_selected").replace("{count}", str(count)))
        return count

    def _select_positions(self, positions: set[int]) -> int:
        """Chọn các hàng mang vị trí dòng ``positions``, cuộn tới hàng đầu tiên."""
        if not positions:
            return 0
        # [v3.23.650] Một lần chọn hàng loạt (selectRow từng hàng: 81 s với 1825 dòng).
        rows = [row for row in range(self._table.rowCount()) if self._row_position(row) in positions]
        select_rows_bulk(self._table, rows)
        first_row = rows[0] if rows else -1
        if first_row >= 0:
            self._table.scrollToItem(self._table.item(first_row, COLUMN_INDEX))
        return len(positions)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _tr(self, key: str) -> str:
        return self._translate(f"tts.{key}")

    def _build_buttons(self) -> None:
        # [v3.23.625] Hai hàng: sáu nút một hàng bị cắt chữ ở cửa sổ 1500 px.
        self._button_row = QVBoxLayout()
        first_row, second_row, third_row = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        self._btn_speaker = PushButton(self._tr("btn_speaker"))
        self._btn_speaker.setToolTip(self._tr("btn_speaker_tip"))
        self._btn_speaker.clicked.connect(self._on_speaker_button)
        self._btn_select_edited = PushButton(self._tr("btn_select_edited"))
        self._btn_select_edited.setToolTip(self._tr("btn_select_edited_tip"))
        self._btn_select_edited.clicked.connect(self.select_edited)
        self._btn_revert = PushButton(self._tr("btn_revert"))
        self._btn_revert.setToolTip(self._tr("btn_revert_tip"))
        self._btn_revert.clicked.connect(
            lambda _checked=False: self.revert(self.selected_positions())
        )
        self._btn_review = PushButton(self._tr("btn_review"))
        self._btn_review.setToolTip(self._tr("btn_review_tip"))
        self._btn_review.clicked.connect(self.select_review)
        self._btn_quick_review = PushButton(self._tr("btn_quick_review"))
        self._btn_quick_review.setToolTip(self._tr("btn_quick_review_tip"))
        self._btn_quick_review.clicked.connect(lambda _checked=False: self.quick_review())
        self._btn_add_line = PushButton(self._tr("btn_add_line"))
        self._btn_add_line.setToolTip(self._tr("btn_add_line_tip"))
        self._btn_add_line.clicked.connect(lambda _checked=False: self.add_line())
        self._btn_undo = PushButton(self._tr("btn_undo"))
        self._btn_undo.clicked.connect(lambda _checked=False: self.undo())
        self._btn_redo = PushButton(self._tr("btn_redo"))
        self._btn_redo.clicked.connect(lambda _checked=False: self.redo())
        self._btn_timing = PushButton(self._tr("btn_timing"))
        self._btn_timing.setToolTip(self._tr("btn_timing_tip"))
        self._btn_timing.clicked.connect(lambda _checked=False: self.edit_timing())
        self._btn_merge = PushButton(self._tr("btn_merge_lines"))
        self._btn_merge.setToolTip(self._tr("btn_merge_lines_tip"))
        self._btn_merge.clicked.connect(
            lambda _checked=False: self.merge_lines(self.selected_positions())
        )
        self._btn_split = PushButton(self._tr("btn_split_line"))
        self._btn_split.setToolTip(self._tr("btn_split_line_tip"))
        self._btn_split.clicked.connect(lambda _checked=False: self.split_line())
        self._btn_shorten = PushButton(self._tr("btn_shorten"))
        self._btn_shorten.setToolTip(self._tr("btn_shorten_tip"))
        self._btn_shorten.clicked.connect(lambda _checked=False: self.shorten_lines())
        self._btn_delete_lines = PushButton(self._tr("btn_delete_lines"))
        self._btn_delete_lines.setToolTip(self._tr("btn_delete_lines_tip"))
        self._btn_delete_lines.clicked.connect(
            lambda _checked=False: self.delete_lines(self.selected_positions())
        )
        for row, buttons in (
            (first_row, (self._btn_speaker, self._btn_quick_review, self._btn_review,
                         self._btn_select_edited, self._btn_revert)),
            (second_row, (self._btn_add_line, self._btn_delete_lines, self._btn_undo,
                          self._btn_redo)),
            (third_row, (self._btn_timing, self._btn_merge, self._btn_split, self._btn_shorten)),
        ):
            for button in buttons:
                button.setEnabled(False)
                row.addWidget(button)
            row.addStretch()
            self._button_row.addLayout(row)

    def _prepare_table(self) -> None:
        table = self._table
        if table.columnCount() < COLUMN_COUNT:
            table.setColumnCount(COLUMN_COUNT)
        table.setHorizontalHeaderItem(
            COLUMN_SPEAKER, QTableWidgetItem(self._tr("col_speaker"))
        )
        header = table.horizontalHeader()
        # Cột người nói đứng cạnh lời thoại (thứ tự HIỂN THỊ; chỉ số logic giữ nguyên để
        # mọi chỗ đọc cột Nội dung = 2 không đổi).
        header.moveSection(header.visualIndex(COLUMN_SPEAKER), COLUMN_TEXT)
        table.setColumnWidth(COLUMN_SPEAKER, _SPEAKER_COLUMN_WIDTH)
        table.setColumnHidden(COLUMN_SPEAKER, not self._panel.is_dubbing_enabled())
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._on_context_menu)
        # Ctrl+Z / Ctrl+Y chỉ khi chính bảng giữ focus (ô đang sửa chữ dùng hoàn tác riêng).
        for sequence, handler in (
            (QKeySequence.StandardKey.Undo, self.undo), (QKeySequence.StandardKey.Redo, self.redo),
        ):
            shortcut = QShortcut(QKeySequence(sequence), table)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(handler)
        table.cellDoubleClicked.connect(self._on_cell_double_clicked)

    def _paint_speaker(self, row: int, cell: SpeakerCell) -> None:
        item = self._table.item(row, COLUMN_SPEAKER)
        if item is None:
            item = read_only_item("")
            self._table.setItem(row, COLUMN_SPEAKER, item)
        item.setText(cell.text)
        item.setToolTip(cell.tooltip)
        font = item.font()
        font.setBold(cell.overridden)
        item.setFont(font)
        if cell.needs_review:
            item.setForeground(_REVIEW_FOREGROUND)
        else:
            item.setData(Qt.ItemDataRole.ForegroundRole, None)

    @staticmethod
    def _paint_edited(item: QTableWidgetItem, edited: bool, tooltip: str) -> None:
        if edited:
            item.setBackground(_EDITED_BACKGROUND)
            item.setForeground(_EDITED_FOREGROUND)
            item.setToolTip(tooltip)
        else:
            item.setData(Qt.ItemDataRole.BackgroundRole, None)
            item.setData(Qt.ItemDataRole.ForegroundRole, None)
            item.setToolTip("")

    def _update_buttons(self) -> None:
        busy = self._view_model.is_busy
        positions = self.selected_positions()
        self._btn_speaker.setEnabled(bool(positions) and not busy)
        self._btn_revert.setEnabled(
            self._view_model.has_line_edits(positions) and not busy
        )
        count = (
            len(self._view_model.edited_positions) if self._view_model.has_result else 0
        )
        label = self._tr("btn_select_edited")
        self._btn_select_edited.setText(f"{label} ({count})" if count else label)
        self._btn_select_edited.setEnabled(count > 0 and not busy)
        review = self._review_count
        label = self._tr("btn_review")
        self._btn_review.setText(f"{label} ({review})" if review else label)
        self._btn_review.setEnabled(review > 0)
        self._btn_quick_review.setEnabled(self._view_model.has_source and not busy)
        self._btn_add_line.setEnabled(self._view_model.has_source and not busy)
        self._btn_delete_lines.setEnabled(bool(positions) and not busy)
        self._btn_undo.setEnabled(self._view_model.can_undo)
        self._btn_redo.setEnabled(self._view_model.can_redo)
        tip_key = "btn_undo_tip" if not self._view_model.has_result else "btn_undo_tip_tts"
        self._btn_undo.setToolTip(self._tr(tip_key))
        self._btn_redo.setToolTip(self._tr(tip_key))
        misfits = len(self._view_model.speech_misfits()) if self._shorten is not None else 0
        label = self._tr("btn_shorten")
        self._btn_shorten.setText(f"{label} ({misfits})" if misfits else label)
        self._btn_shorten.setEnabled(misfits > 0 and not busy and self._shorten is not None)
        self._btn_timing.setEnabled(len(positions) == 1 and not busy)
        self._btn_split.setEnabled(len(positions) == 1 and not busy)
        self._btn_merge.setEnabled(self._can_merge(positions) and not busy)

    def _row_position(self, row: int) -> int | None:
        head = self._table.item(row, COLUMN_INDEX)
        value = head.data(Qt.ItemDataRole.UserRole) if head is not None else None
        return int(value) if value is not None else None

    def _add_choice(
        self,
        menu: QMenu,
        label: str,
        positions: list[int],
        speaker: str | None,
        *,
        checked: bool,
    ) -> QAction:
        action = menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(checked)
        action.triggered.connect(partial(self.apply_speaker, positions, speaker))
        return action

    def _speaker_text(self, speaker: str | None) -> str:
        if speaker is None:
            return self._tr("line_speaker_auto")
        if speaker == SHARED_VOICE_SPEAKER:
            return self._translate("tts.dub_voice_shared")
        return self._dubbing.display_name(speaker)

    def _ask_speaker(self, positions: list[int], _checked: bool = False) -> int:
        name, accepted = ask_text(
            self._table.window(), self._tr("dlg_speaker_title"),
            self._tr("dlg_speaker_label"),
        )
        if not accepted or not str(name).strip():
            return 0
        return self.apply_speaker(positions, str(name).strip())

    def _edit_text(self, positions: list[int], _checked: bool = False) -> None:
        for row in range(self._table.rowCount()):
            if self._row_position(row) in positions:
                item = self._table.item(row, COLUMN_TEXT)
                if item is not None:
                    self._table.setCurrentItem(item)
                    self._table.editItem(item)
                return

    def _on_speaker_button(self, _checked: bool = False) -> None:
        positions = self.selected_positions()
        if not positions:
            return
        menu = self.build_speaker_menu(positions)
        menu.exec(self._btn_speaker.mapToGlobal(QPoint(0, self._btn_speaker.height())))

    def _on_context_menu(self, point: QPoint) -> None:
        row = self._table.rowAt(point.y())
        if row < 0:
            return
        if self._row_position(row) not in self.selected_positions():
            self._table.clearSelection()
            self._table.selectRow(row)
        menu = self.build_context_menu(self.selected_positions())
        menu.exec(self._table.viewport().mapToGlobal(point))

    def _on_cell_double_clicked(self, row: int, column: int) -> None:
        if column != COLUMN_SPEAKER or self._view_model.is_busy:
            return
        position = self._row_position(row)
        positions = self.selected_positions()
        if position is None:
            return
        if position not in positions:
            positions = [position]
        rect = self._table.visualItemRect(self._table.item(row, column))
        menu = self.build_speaker_menu(positions)
        menu.exec(self._table.viewport().mapToGlobal(rect.bottomLeft()))

    def _on_dubbing_enabled(self, enabled: bool) -> None:
        if enabled and self._view_model.has_source and not (
            self._view_model.dubbing_prepared
        ):
            self._dubbing.load_cast()
        self.schedule_refresh()

    # ── Lưu / khôi phục ───────────────────────────────────────────────────────

    def _load_store(self) -> dict[str, object]:
        raw = self._settings.value(SETTINGS_LINE_EDITS_KEY, "", type=str)
        try:
            loaded = json.loads(raw) if raw else {}
        except ValueError:
            logger.warning("Sửa tay từng dòng đã lưu bị hỏng — bỏ qua.")
            return {}
        return dict(loaded) if isinstance(loaded, dict) else {}

    def _restore(self) -> tuple[int, int]:
        fingerprint = self._view_model.subtitle_fingerprint
        raw = self._load_store().get(fingerprint) if fingerprint else None
        if raw is None:
            return 0, 0
        restored = self._view_model.restore_line_edits(decode_line_edits(raw))
        logger.info(
            "Khôi phục sửa tay của phụ đề %s: %d dòng người nói, %d dòng lời thoại.",
            fingerprint, *restored,
        )
        return restored

    def _save(self) -> None:
        fingerprint = self._view_model.subtitle_fingerprint
        if not fingerprint:
            return
        store = self._load_store()
        updated = remember_line_edits(store, fingerprint, self._view_model.line_edits())
        if updated != store:
            self._settings.setValue(
                SETTINGS_LINE_EDITS_KEY, json.dumps(updated, ensure_ascii=False)
            )


__all__ = [
    "COLUMN_COUNT",
    "COLUMN_INDEX",
    "COLUMN_SPEAKER",
    "COLUMN_STATUS",
    "COLUMN_TEXT",
    "COLUMN_TIME",
    "SETTINGS_LINE_EDITS_KEY",
    "TTSLineEditsController",
    "editable_text_item",
    "format_row_time",
    "read_only_item",
    "row_head_item",
    "selected_positions",
]
