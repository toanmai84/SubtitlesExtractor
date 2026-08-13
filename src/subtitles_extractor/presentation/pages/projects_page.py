"""Trang Thư viện — quản lý các dự án Auto-Dubbing đã lưu trong cơ sở dữ liệu.

Mỗi dự án định danh bằng hash nội dung video (độc lập tên/thư mục), nên cùng
một video sẽ không bị làm lại từ đầu chỉ vì đổi tên hay di chuyển. Người dùng
có thể mở lại để tiếp tục công việc dở dang, hoặc xoá dự án.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.domain.entities.project_record import ProjectRecord
from subtitles_extractor.domain.ports.project_repository_port import (
    ProjectRepositoryPort,
)
from subtitles_extractor.presentation.theme import feedback as _feedback
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme.styles import caption_style

logger = logging.getLogger(__name__)


class ProjectsPage(QWidget):
    """Liệt kê, mở lại và xoá các dự án Auto-Dubbing đã lưu."""

    def __init__(
        self,
        project_repository: ProjectRepositoryPort,
        translator,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ProjectsPage")
        self._translator = translator
        self._repo = project_repository
        self._open_callback: Callable[[ProjectRecord], None] | None = None
        self._records: list[ProjectRecord] = []
        self._build_ui()
        self.refresh()

    def set_open_callback(self, callback: Callable[[ProjectRecord], None]) -> None:
        """Đăng ký hàm được gọi khi người dùng mở một dự án (nạp vào quy trình)."""
        self._open_callback = callback

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        header = QHBoxLayout()
        title = QLabel(self._translator.translate("projects.title"))
        title.setStyleSheet(f"font-size:{_m.FONT_SIZE_TITLE}px;font-weight:600;")
        header.addWidget(title)
        header.addStretch()
        refresh_btn = QPushButton(self._translator.translate("projects.refresh"))
        refresh_btn.clicked.connect(self.refresh)
        header.addWidget(refresh_btn)
        root.addLayout(header)

        hint = QLabel(
            self._translator.translate("projects.hash_note")
        )
        hint.setStyleSheet(caption_style()); hint.setWordWrap(True)
        root.addWidget(hint)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            [self._translator.translate("projects.h_name"), self._translator.translate("projects.h_stage"), self._translator.translate("projects.h_lang"), self._translator.translate("projects.h_updated"), self._translator.translate("projects.h_hash")]
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.doubleClicked.connect(self._on_open)
        self._table.itemSelectionChanged.connect(self._update_action_buttons)
        root.addWidget(self._table, stretch=1)

        # [v3.23.108] Thông báo khi thư viện rỗng (thay vì bảng trống khó hiểu).
        self._empty_label = QLabel(
            self._translator.translate("projects.empty")
        )
        self._empty_label.setStyleSheet(caption_style())
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.hide()
        root.addWidget(self._empty_label, stretch=1)

        actions = QHBoxLayout()
        actions.addStretch()
        self._open_btn = QPushButton(self._translator.translate("projects.open"))
        self._open_btn.setToolTip(self._translator.translate("projects.open_tip"))
        self._open_btn.clicked.connect(self._on_open)
        actions.addWidget(self._open_btn)
        # [v3.23.317] Dự án đã xuất bản -> mở thẳng phim hoàn chỉnh bằng trình phát
        # mặc định của hệ điều hành, không phải đi tìm trong thư mục.
        self._play_btn = QPushButton(self._translator.translate("projects.play"))
        self._play_btn.setToolTip(self._translator.translate("projects.play_tip"))
        self._play_btn.clicked.connect(self._on_play_published)
        actions.addWidget(self._play_btn)
        self._delete_btn = QPushButton(self._translator.translate("projects.delete"))
        self._delete_btn.setToolTip(self._translator.translate("projects.delete_tip"))
        self._delete_btn.clicked.connect(self._on_delete)
        actions.addWidget(self._delete_btn)
        root.addLayout(actions)
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        """Bật nút Mở/Xoá chỉ khi có dòng được chọn (rõ affordance cho người dùng)."""
        record = self._selected_record()
        has_selection = record is not None
        self._open_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)
        # Nút xem phim chỉ bật khi dự án THỰC SỰ có tệp đã xuất và tệp còn tồn tại.
        self._play_btn.setEnabled(
            has_selection and self._published_file_of(record) is not None
        )

    @staticmethod
    def _published_file_of(record: ProjectRecord | None) -> Path | None:
        """Tệp phim đã xuất bản của dự án, nếu còn tồn tại trên đĩa.

        Args:
            record: Bản ghi dự án (có thể ``None``).

        Returns:
            :class:`~pathlib.Path` tới tệp, hoặc ``None`` nếu chưa xuất bản / tệp đã
            bị xoá hoặc di chuyển.
        """
        if record is None:
            return None
        raw = getattr(record, "published_video_path", "") or ""
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_file() else None

    def _on_play_published(self) -> None:
        """Mở phim đã xuất bản bằng ứng dụng mặc định của hệ điều hành."""
        record = self._selected_record()
        path = self._published_file_of(record)
        if path is None:
            QMessageBox.information(
                self, self._translator.translate("projects.no_film_t"),
                self._translator.translate("projects.no_film_b"),
            )
            return
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        if not opened:
            QMessageBox.warning(
                self, self._translator.translate("projects.cant_open_t"),
                self._translator.translate("projects.cant_open_b").replace("{path}", str(path)),
            )

    def refresh(self) -> None:
        """Nạp lại danh sách dự án từ cơ sở dữ liệu."""
        try:
            self._records = self._repo.list_all()
        except Exception as exc:  # noqa: BLE001 — hiển thị mọi lỗi DB cho người dùng
            logger.error("Không thể đọc danh sách dự án: %s", exc)
            self._records = []
        self._table.setRowCount(len(self._records))
        # [v3.23.108] Ẩn/hiện bảng vs thông báo rỗng theo số dự án.
        is_empty = not self._records
        self._table.setVisible(not is_empty)
        self._empty_label.setVisible(is_empty)
        for row, rec in enumerate(self._records):
            updated = rec.updated_at.replace("T", " ")[:19] if rec.updated_at else ""
            values = [
                rec.video_name or rec.video_path or self._translator.translate("projects.unnamed"),
                rec.stage.label_vi,
                rec.target_lang or "—",
                updated,
                rec.video_hash,
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                if col == 4:
                    item.setForeground(Qt.GlobalColor.gray)
                self._table.setItem(row, col, item)
        logger.debug("Thư viện: hiển thị %d dự án", len(self._records))
        self._update_action_buttons()

    def _selected_record(self) -> ProjectRecord | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._records):
            return self._records[row]
        return None

    def _on_open(self) -> None:
        rec = self._selected_record()
        if rec is None:
            return
        if self._open_callback is not None:
            logger.info("Mở lại dự án %s (%s)", rec.video_name, rec.video_hash)
            self._open_callback(rec)

    def _on_delete(self) -> None:
        rec = self._selected_record()
        if rec is None:
            return
        confirm = QMessageBox.question(
            self,
            self._translator.translate("projects.del_confirm_t"),
            self._translator.translate("projects.del_confirm_b").replace("{name}", str(rec.video_name)),
        )
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                self._repo.delete(rec.video_hash)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.error("Xoá dự án %s thất bại: %s", rec.video_hash, exc)
                _feedback.show_error(
                    self, self._translator.translate("projects.del_err_t"),
                    self._translator.translate("projects.del_err_b").replace("{name}", str(rec.video_name)).replace("{exc}", str(exc)),
                )
                return
            logger.info("Đã xoá dự án %s", rec.video_hash)
            _feedback.show_success(self, self._translator.translate("projects.deleted"), rec.video_name or "")
            self.refresh()


__all__ = ["ProjectsPage"]
