"""Adapter :class:`SettingsRepositoryPort` lưu cấu hình vào tệp JSON.

Mục đích sử dụng:
    * Mặc định dùng :class:`QSettingsRepository` (registry/INI tự động).
    * Khi chạy CLI/daemon hoặc test mà không khởi tạo QApplication, dùng
      :class:`JsonFileRepository` để tránh phụ thuộc PyQt.

Ghi atomic: ghi vào ``<file>.tmp`` rồi rename — đảm bảo không bao giờ
để lại tệp dở dang nếu bị ngắt giữa chừng.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonFileRepository:
    """Hiện thực :class:`SettingsRepositoryPort` dựa trên một tệp JSON.

    Args:
        file_path: Đường dẫn tệp lưu trữ.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._lock = threading.RLock()
        self._cache: dict[str, Any] = self._read_from_disk()
        self._dirty = False

    # ── Port API ────────────────────────────────────────────────────────

    def load(self, key: str, default: Any) -> Any:
        with self._lock:
            return self._cache.get(key, default)

    def save(self, key: str, value: Any) -> None:
        with self._lock:
            if self._cache.get(key) != value:
                self._cache[key] = value
                self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._write_to_disk(self._cache)
            self._dirty = False
            logger.info("Đã ghi cấu hình xuống %s.", self._file_path)

    def reset(self) -> None:
        with self._lock:
            self._cache.clear()
            self._dirty = True

    # ── I/O ─────────────────────────────────────────────────────────────

    def _read_from_disk(self) -> dict[str, Any]:
        if not self._file_path.exists():
            return {}
        try:
            with self._file_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Không đọc được %s: %s — quay về cấu hình rỗng.",
                self._file_path,
                exc,
            )
            return {}

    def _write_to_disk(self, payload: dict[str, Any]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        # Ghi nguyên tử: tệp tạm cùng thư mục để rename đảm bảo cùng FS.
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=self._file_path.stem + "_",
            suffix=".tmp",
            dir=str(self._file_path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._file_path)
        except OSError:
            # Cố gắng xoá tệp tạm nếu replace thất bại.
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


__all__ = ["JsonFileRepository"]
