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
        raw = self._migrate(raw)
        try:
            stored = ApplicationSettings.model_validate(raw)
        except ValidationError as exc:
            logger.warning(
                "Cấu hình lưu trữ không hợp lệ — quay về mặc định. Chi tiết: %s",
                exc,
            )
            return ApplicationSettings()
        return self._apply_env_overrides(stored)

    @staticmethod
    def _migrate(raw: object) -> object:
        """[v3.23.654] Nâng cấu hình lưu từ bản cũ.

        * Lọc chữ trắng: bản ≤ 3.23.653 mặc định TẮT và luôn ghi ``white_text_gate=false`` vào tệp, nên không phân biệt
          được "chưa từng chọn" với "đã tắt". Áp mặc định mới (BẬT) đúng MỘT lần, đánh dấu ``white_text_gate_v654``;
          sau đó lựa chọn của bạn trong Cài đặt được giữ nguyên.
        """
        if not isinstance(raw, dict):
            return raw
        frame = raw.get("frame")
        if isinstance(frame, dict) and "white_text_gate_v654" not in frame:
            frame = dict(frame)
            if frame.get("white_text_gate") is False:
                logger.info("Bật lọc chữ trắng theo mặc định mới (một lần; tắt được trong Cài đặt).")
            frame["white_text_gate"] = True
            frame["white_text_gate_v654"] = True
            raw = {**raw, "frame": frame}
        # [v3.23.661] Mặc định mới đo thật trên phim mẫu (bước 0,12 s + model small). Chỉ đổi khi cấu hình đang giữ ĐÚNG
        # mặc định cũ (người dùng chưa tự chọn), đúng một lần.
        frame = raw.get("frame")
        if isinstance(frame, dict) and "defaults_v661" not in frame:
            frame = dict(frame)
            if abs(float(frame.get("sample_step_sec", 0.04)) - 0.04) < 1e-9:
                frame["sample_step_sec"] = 0.12
                logger.info("Bước lấy mẫu OCR: 0,04 s → 0,12 s (mặc định mới, một lần; đổi được trong Cài đặt).")
            frame["defaults_v661"] = True
            raw = {**raw, "frame": frame}
        ocr = raw.get("ocr")
        if isinstance(ocr, dict) and "defaults_v661" not in ocr:
            ocr = dict(ocr)
            if ocr.get("version", "PP-OCRv6_medium") == "PP-OCRv6_medium" and ocr.get(
                "detection_model", "PP-OCRv6_medium_det"
            ) == "PP-OCRv6_medium_det" and ocr.get("recognition_model", "PP-OCRv6_medium_rec") == "PP-OCRv6_medium_rec":
                ocr.update(version="PP-OCRv6_small", detection_model="PP-OCRv6_small_det",
                           recognition_model="PP-OCRv6_small_rec")
                logger.info("Model OCR: PP-OCRv6 medium → small (mặc định mới, một lần; đổi được trong Cài đặt).")
            ocr["defaults_v661"] = True
            raw = {**raw, "ocr": ocr}
        return raw

    @staticmethod
    def _apply_env_overrides(stored: ApplicationSettings) -> ApplicationSettings:
        """[v3.23.619] Biến môi trường ``SE_*`` ghi đè cấu hình đã lưu.

        README hướng dẫn ``export SE_HW_DEVICE=cpu`` / ``SE_OCR_SCORE_THRESHOLD=...``,
        nhưng trước đây giá trị trong ``config.json`` luôn thắng (biến môi trường chỉ có
        tác dụng ở lần chạy đầu, khi chưa có tệp). Nay đọc biến môi trường của từng nhóm
        (theo ``env_prefix``) và đè lên; giá trị sai kiểu bị bỏ qua kèm cảnh báo.
        """
        from pydantic_settings import EnvSettingsSource

        data = stored.model_dump(mode="python")
        changed = False
        for group_name, field in ApplicationSettings.model_fields.items():
            group_cls = field.annotation
            try:
                env_values = EnvSettingsSource(group_cls)()
            except (ValueError, TypeError) as exc:
                logger.warning("Bỏ qua biến môi trường nhóm %s: %s", group_name, exc)
                continue
            for key, value in env_values.items():
                trial = {**data[group_name], key: value}
                try:
                    group_cls.model_validate(trial)
                except ValidationError:
                    logger.warning(
                        "Bỏ qua biến môi trường %s.%s=%r (giá trị không hợp lệ).",
                        group_name, key, value,
                    )
                    continue
                data[group_name] = trial
                changed = True
                logger.info("Biến môi trường ghi đè cấu hình: %s.%s.", group_name, key)
        return ApplicationSettings.model_validate(data) if changed else stored

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
