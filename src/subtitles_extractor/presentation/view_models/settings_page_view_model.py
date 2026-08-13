"""ViewModel cho trang "Cài đặt" — quản lý 10 nhóm cấu hình.

CẢI TIẾN ĐỘT PHÁ (V3.15 - The Control Center Polish):
    * [FEATURE] Cache Cleaner: Tích hợp công cụ dọn rác hệ thống (File tạm ASS, NPY Sóng âm).
    * [UX] Đóng gói các tín hiệu chuẩn bị cho Smart Dirty State.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.infrastructure.settings.application_settings import (
    ApplicationSettings,
)

logger = logging.getLogger(__name__)

def _apply_log_level(level_str: str) -> None:
    numeric = getattr(logging, level_str.upper(), None)
    if numeric is None:
        return
    root = logging.getLogger()
    if root.level != numeric:
        root.setLevel(numeric)
        logger.info("Log level đã đổi sang %s.", level_str)

def _apply_font_size(size_pt: int) -> None:
    """Áp dụng font size toàn cục. Guard: chỉ thực hiện khi size_pt hợp lệ."""
    if not isinstance(size_pt, int) or size_pt <= 0:
        return

    app = QApplication.instance()
    if app is None:
        return

    font = app.font()
    clamped = max(6, min(24, size_pt))
    if font.pointSize() != clamped:
        font.setPointSize(clamped)
        app.setFont(font)
        logger.info("Font size đã đổi sang %dpt.", clamped)

def _apply_theme(theme_str: str) -> None:
    """Áp dụng theme cho qfluentwidgets lập tức."""
    try:
        from subtitles_extractor.presentation.fluent_compat import Theme, setTheme
        if theme_str == "dark":
            setTheme(Theme.DARK)
        elif theme_str == "light":
            setTheme(Theme.LIGHT)
        else:
            setTheme(Theme.AUTO)
    except ImportError:
        pass

class SettingsPageViewModel(QObject):
    settings_loaded = Signal(object)
    settings_saved = Signal(object)
    validation_failed = Signal(str)
    ui_changed = Signal(int, bool, bool)
    cache_cleaned = Signal(int, float)
    database_reset = Signal(int)

    def __init__(self, container: ApplicationContainer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._container = container

    def reload(self) -> None:
        self.settings_loaded.emit(self._container.settings_service.current)

    @property
    def user_data_dir(self) -> Path:
        return self._container.user_data_dir

    def save(self, patch: dict[str, Any]) -> None:
        try:
            new_snapshot: ApplicationSettings = self._container.settings_service.update(**patch)
        except ConfigurationError as exc:
            self.validation_failed.emit(str(exc))
            return

        self._apply_and_emit_snapshot(new_snapshot, is_reset=False)

    def reset_to_defaults(self) -> None:
        snapshot = self._container.settings_service.reset_to_defaults()
        self._apply_and_emit_snapshot(snapshot, is_reset=True)

    def _apply_and_emit_snapshot(self, snapshot: ApplicationSettings, is_reset: bool) -> None:
        self._container.apply_settings_changes()
        _apply_log_level(snapshot.advanced.log_level)
        _apply_font_size(snapshot.ui.safe_font_size)
        _apply_theme(snapshot.ui.theme)

        self.ui_changed.emit(
            snapshot.ui.font_size,
            snapshot.ui.show_waveform,
            snapshot.ui.show_ocr_overlay,
        )

        if is_reset:
            self.settings_loaded.emit(snapshot)

        self.settings_saved.emit(snapshot)
        logger.info("Settings đã cập nhật và adapter đã reset — hiệu lực ngay.")

    def clear_temp_cache(self) -> None:
        """[V3.15 FEATURE] Quét và xóa các file rác sinh ra trong quá trình sử dụng."""
        temp_dir = Path(tempfile.gettempdir())
        files_deleted = 0
        bytes_freed = 0.0

        # Mẫu file cần dọn: se_preview_*.ass, *.waveform.npy
        try:
            for file_path in temp_dir.glob("se_preview_*.ass"):
                bytes_freed += file_path.stat().st_size
                file_path.unlink(missing_ok=True)
                files_deleted += 1

            for file_path in temp_dir.glob("*.waveform.npy"):
                bytes_freed += file_path.stat().st_size
                file_path.unlink(missing_ok=True)
                files_deleted += 1

            for file_path in temp_dir.glob("*.tmp.npy"):
                bytes_freed += file_path.stat().st_size
                file_path.unlink(missing_ok=True)
                files_deleted += 1

        except OSError as exc:
            logger.warning("Lỗi khi dọn dẹp Cache: %s", exc)

        mb_freed = bytes_freed / (1024 * 1024)
        logger.info("Dọn dẹp rác hoàn tất: Xóa %d file, giải phóng %.2f MB.", files_deleted, mb_freed)
        self.cache_cleaned.emit(files_deleted, mb_freed)

    def reset_database(self) -> None:
        """[v3.23.88] Dọn sạch database về như mới tạo (xoá dữ liệu mọi bảng).

        Phát tín hiệu :attr:`database_reset` với TỔNG số bảng đã xoá để UI thông báo.
        Lỗi truy cập DB được ghi log và phát tín hiệu với 0 bảng.
        """
        try:
            cleared = self._container.reset_database()
            total_tables = sum(len(tables) for tables in cleared.values())
            logger.info("Dọn database hoàn tất: xoá dữ liệu %d bảng.", total_tables)
        except (OSError, RuntimeError) as exc:
            logger.error("Lỗi khi dọn database: %s", exc)
            total_tables = 0
        self.database_reset.emit(total_tables)

__all__ = ["SettingsPageViewModel"]
