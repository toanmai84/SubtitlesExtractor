"""Adapter :class:`SettingsRepositoryPort` dựa trên :class:`QSettings`.

QSettings tự lo việc dùng đúng backend cho từng OS:
    * Windows  → registry
    * macOS    → property list
    * Linux    → file INI trong ``~/.config``

Adapter này lazy-import PySide6 để cho phép phần còn lại của tầng
infrastructure chạy được trên môi trường không có PyQt (ví dụ CI test).
"""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PySide6.QtCore import QSettings

logger = logging.getLogger(__name__)


class QSettingsRepository:
    """Hiện thực :class:`SettingsRepositoryPort` dùng :class:`QSettings`.

    QSettings không xử lý tốt cấu trúc dict lồng nhau, nên adapter
    serialize value sang JSON-string trước khi lưu, deserialize khi đọc.
    Làm vậy round-trip ổn định cho mọi kiểu cơ bản (dict, list, bool…).

    Args:
        organization: Tên tổ chức cho QSettings.
        application:  Tên ứng dụng cho QSettings.
        qsettings:    Inject sẵn instance — dùng cho test.
    """

    def __init__(
        self,
        organization: str = "SubtitlesExtractor",
        application: str = "PaddleOCR",
        qsettings: QSettings | None = None,
    ) -> None:
        self._lock = threading.RLock()
        if qsettings is not None:
            self._qs = qsettings
        else:
            from PySide6.QtCore import QSettings as _QSettings

            self._qs = _QSettings(organization, application)

    # ── Port API ────────────────────────────────────────────────────────

    def load(self, key: str, default: Any) -> Any:
        with self._lock:
            raw = self._qs.value(key, None)
        if raw is None:
            return default
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Giá trị legacy không phải JSON — trả nguyên dạng.
                return raw
        return raw

    def save(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._qs.setValue(key, encoded)

    def flush(self) -> None:
        with self._lock:
            self._qs.sync()
        logger.debug("Đã sync QSettings xuống storage.")

    def reset(self) -> None:
        with self._lock:
            self._qs.clear()
            self._qs.sync()
        logger.info("Đã xoá toàn bộ cấu hình QSettings.")


__all__ = ["QSettingsRepository"]
