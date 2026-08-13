"""Service quản lý vòng đời :class:`ApplicationSettings`.

Cải tiến v2.7+:
    * ``current`` không còn deep-copy — pydantic models được sử dụng như
      immutable snapshot. Caller KHÔNG được mutate trực tiếp; nếu cần
      đổi → gọi ``update()``. Tiết kiệm 100-1000× CPU cho hot path.
    * ``_persist`` debounce 300ms — nếu user đổi nhiều giá trị nhanh,
      chỉ ghi đĩa 1 lần ở cuối.

Tách trách nhiệm:
    * :class:`ApplicationSettings` — schema + validation (pydantic).
    * :class:`SettingsRepositoryPort` — I/O bền vững (QSettings/JSON file…).
    * :class:`SettingsService` — orchestration: load từ repo, validate,
      cache, save lại khi thay đổi.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pydantic import ValidationError

from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.domain.ports.settings_repository_port import (
    SettingsRepositoryPort,
)
from subtitles_extractor.infrastructure.settings.application_settings import (
    ApplicationSettings,
)

logger = logging.getLogger(__name__)

_PERSISTENCE_KEY = "application_settings"
_DEBOUNCE_INTERVAL_SEC: float = 0.3


class SettingsService:
    """Quản lý đọc/ghi cấu hình ứng dụng qua kho lưu trữ.

    Args:
        repository: Adapter hiện thực :class:`SettingsRepositoryPort`.

    Note:
        ``current`` trả về reference (không copy). Caller phải coi snapshot
        là **read-only**. Để thay đổi, dùng ``update()``.
    """

    def __init__(self, repository: SettingsRepositoryPort) -> None:
        self._repo = repository
        self._lock = threading.RLock()
        self._cache = self._load_initial()
        self._persist_timer: threading.Timer | None = None

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def current(self) -> ApplicationSettings:
        """Trả về snapshot hiện tại (read-only).

        Pydantic model là frozen-like — nên reference an toàn cho các
        caller chỉ đọc. Nếu caller cần mutate → ``update()`` thay vì
        copy + mutate.
        """
        with self._lock:
            return self._cache

    def update(self, **patch: Any) -> ApplicationSettings:
        """Áp dụng partial update và persist (debounced).

        Args:
            **patch: Cặp ``key=value`` ở cấp gốc (ví dụ ``ocr=...``).

        Returns:
            Snapshot mới sau khi áp dụng.

        Raises:
            ConfigurationError: Khi giá trị mới không qua validation.
        """
        with self._lock:
            try:
                base = self._cache.model_dump(mode="python")
                for group_name, group_patch in patch.items():
                    if isinstance(group_patch, dict) and isinstance(
                        base.get(group_name), dict
                    ):
                        base[group_name].update(group_patch)
                    else:
                        base[group_name] = group_patch
                validated = ApplicationSettings.model_validate(base)
            except ValidationError as exc:
                raise ConfigurationError(
                    f"Cấu hình mới không hợp lệ: {exc}"
                ) from exc
            self._cache = validated
            self._schedule_persist()
            logger.info("Đã cập nhật cấu hình ứng dụng (%d nhóm).", len(patch))
            return validated

    def reset_to_defaults(self) -> ApplicationSettings:
        """Khôi phục về giá trị mặc định và persist."""
        with self._lock:
            self._cache = ApplicationSettings()
            self._repo.reset()
            self._persist_now()
            logger.info("Đã khôi phục cấu hình về mặc định.")
            return self._cache

    def flush(self) -> None:
        """Cưỡng chế ghi disk ngay (cancel debounce timer).

        Gọi từ shutdown handler để không mất dữ liệu cuối.
        """
        with self._lock:
            if self._persist_timer is not None:
                self._persist_timer.cancel()
                self._persist_timer = None
            self._persist_now()

    # ── Helpers nội bộ ──────────────────────────────────────────────────

    def _load_initial(self) -> ApplicationSettings:
        """Đọc cấu hình từ repo, fallback default nếu thiếu/lỗi."""
        raw = self._repo.load(_PERSISTENCE_KEY, default=None)
        if raw is None:
            logger.info("Chưa có cấu hình lưu trữ — dùng mặc định.")
            return ApplicationSettings()
        try:
            return ApplicationSettings.model_validate(raw)
        except ValidationError as exc:
            logger.warning(
                "Cấu hình lưu trữ không hợp lệ — quay về mặc định. Chi tiết: %s",
                exc,
            )
            return ApplicationSettings()

    def _schedule_persist(self) -> None:
        """Lên lịch ghi đĩa sau ``_DEBOUNCE_INTERVAL_SEC`` giây.

        Nếu có lệnh khác đến trong khoảng debounce, timer cũ bị huỷ và
        đếm lại từ đầu — chỉ ghi đúng 1 lần ở cuối chuỗi thay đổi.
        """
        if self._persist_timer is not None:
            self._persist_timer.cancel()
        self._persist_timer = threading.Timer(
            _DEBOUNCE_INTERVAL_SEC, self._persist_now_with_lock
        )
        self._persist_timer.daemon = True
        self._persist_timer.start()

    def _persist_now_with_lock(self) -> None:
        """Persist từ background thread của Timer — phải re-acquire lock."""
        with self._lock:
            self._persist_now()
            self._persist_timer = None

    def _persist_now(self) -> None:
        """Ghi snapshot hiện tại xuống repo và flush. Caller phải giữ lock."""
        self._repo.save(_PERSISTENCE_KEY, self._cache.model_dump(mode="json"))
        self._repo.flush()


__all__ = ["SettingsService"]
