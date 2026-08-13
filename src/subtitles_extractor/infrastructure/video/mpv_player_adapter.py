"""Adapter :class:`VideoPlayerPort` dùng python-mpv embed vào Qt widget.

Cơ chế MPV Fallback:
    Nếu thông số ``hwdec`` (Hardware Decode) hoặc ``vo`` (Video Output)
    không tương thích với máy người dùng gây lỗi khởi tạo, hệ thống sẽ
    tự động hạ cấp về chế độ an toàn nhất (``hwdec='no'``, ``vo='gpu'``)
    để đảm bảo không crash app.

Mọi method công khai bắt cụ thể ``(RuntimeError)`` và các exception từ mpv:
    * ``RuntimeError`` — bao gồm các lỗi runtime.
    * ``AttributeError`` — khi mpv-python đã release hoặc API khác version.
    * ``ValueError``/``TypeError`` — argument sai.
    * ``SystemError`` — bắt các lỗi C-core (Error -12) chưa được handle.
    * ``mpv.MPVError`` / ``mpv.ShutdownError`` — lỗi trực tiếp từ libmpv.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtWidgets import QApplication
from subtitles_extractor.presentation.fluent_compat import InfoBar, InfoBarPosition

_LOAD_WAIT_TIMEOUT_SEC: float = 6.0
_SPEED_MIN: float = 0.25
_SPEED_MAX: float = 4.0
_VOLUME_MIN: int = 0
_VOLUME_MAX: int = 100


def _get_mpv_errors() -> tuple[type[Exception], ...]:
    """Lấy các class exception từ mpv an toàn. Bổ sung SystemError chống Crash lõi C."""
    errors: list[type[Exception]] = [RuntimeError, AttributeError, ValueError, TypeError, SystemError]
    try:
        import mpv  # type: ignore[import-not-found]
        if hasattr(mpv, "MPVError"): errors.append(mpv.MPVError)
        if hasattr(mpv, "ShutdownError"): errors.append(mpv.ShutdownError)
    except (ImportError, OSError):
        # [v3.23.261] OSError khi python-mpv cài nhưng thiếu libmpv.
        pass
    return tuple(errors)


class MpvPlayerAdapter:
    def __init__(self, wid: int, *, mpv_options: dict[str, Any] | None = None) -> None:
        try:
            import mpv  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Thư viện python-mpv chưa được cài đặt.") from exc
        except OSError as exc:
            # [v3.23.261] python-mpv cài nhưng thiếu libmpv (DLL/.so) -> OSError.
            raise ImportError(
                "Không tìm thấy libmpv (mpv). Cài mpv hoặc thêm libmpv vào PATH."
            ) from exc

        os.environ.setdefault("HOME", str(Path.home()))

        kwargs = dict(mpv_options or {})
        kwargs["wid"] = str(wid)
        kwargs["log_handler"] = self._make_log_handler(kwargs.get("msg_level", ""))

        self._mpv: Any = self._init_mpv_with_fallback(mpv, kwargs, wid)
        self._loaded_path: Path | None = None

    @staticmethod
    def _init_mpv_with_fallback(mpv_module: Any, kwargs: dict[str, Any], wid: int) -> Any:
        try:
            instance = mpv_module.MPV(**kwargs)
            logger.info("Mpv player khởi tạo THÀNH CÔNG: wid={}, hwdec={}, vo={}.", wid, kwargs.get("hwdec", "?"), kwargs.get("vo", "?"))
            return instance
        except _get_mpv_errors() as exc:
            logger.warning("MPV Fallback kích hoạt — config gốc thất bại ({}). Lùi về chế độ CPU an toàn.", exc)
        except OSError as exc:
            logger.warning("Lỗi OS khi khởi tạo MPV: {}", exc)

        safe_kwargs = kwargs.copy()
        safe_kwargs["hwdec"] = "no"
        safe_kwargs["vo"] = "gpu"
        try:
            instance = mpv_module.MPV(**safe_kwargs)
        except _get_mpv_errors() as fallback_exc:
            raise RuntimeError(f"Không thể khởi tạo MPV Engine: {fallback_exc}.") from fallback_exc
        except OSError as fallback_exc:
            raise RuntimeError(f"Không thể khởi tạo MPV Engine: {fallback_exc}.") from fallback_exc

        logger.info("Mpv player khởi tạo an toàn (Fallback) THÀNH CÔNG.")
        main_window = QApplication.activeWindow()
        if main_window is not None:
            InfoBar.warning(
                title="Cảnh báo phần cứng",
                content="Cấu hình Giải mã Video không tương thích. Đã tự động chuyển về chế độ CPU an toàn.",
                parent=main_window, position=InfoBarPosition.TOP_RIGHT, duration=5000,
            )
        return instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded_path is not None and not bool(self._mpv_get("idle-active", default=True))

    @property
    def is_playing(self) -> bool:
        return not bool(self._mpv_get("pause", default=True))

    @property
    def position_sec(self) -> float:
        value = self._mpv_get("time-pos", default=0.0)
        return float(value) if value is not None else 0.0

    @property
    def duration_sec(self) -> float:
        value = self._mpv_get("duration", default=0.0)
        return float(value) if value is not None else 0.0

    @property
    def video_width(self) -> int:
        value = self._mpv_get("width", default=0)
        return int(value) if value else 0

    @property
    def video_height(self) -> int:
        value = self._mpv_get("height", default=0)
        return int(value) if value else 0

    @property
    def hwdec_current(self) -> str:
        value = self._mpv_get("hwdec-current", default="no")
        return str(value)

    @property
    def eof_reached(self) -> bool:
        value = self._mpv_get("eof-reached", default=False)
        return bool(value)

    def load(self, video_path: Path) -> None:
        if not video_path.exists(): raise FileNotFoundError(f"Không tìm thấy tệp: {video_path}.")
        if not hasattr(self, "_mpv") or self._mpv is None: return

        try:
            try:
                self._mpv.sub_auto = "no"
                self._mpv.sid = "no"
            except _get_mpv_errors(): pass

            self._mpv.command("loadfile", str(video_path), "replace")
            with contextlib.suppress(TimeoutError): self._mpv.wait_until_playing(timeout=_LOAD_WAIT_TIMEOUT_SEC)

            self._mpv.pause = True
            self._loaded_path = video_path
        except _get_mpv_errors() as exc:
            raise RuntimeError(f"Không nạp được video qua mpv: {exc}.") from exc
        except OSError as exc:
            raise RuntimeError(f"Lỗi IO: {exc}") from exc

    def play(self) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try: self._mpv.pause = False
        except _get_mpv_errors(): pass

    def pause(self) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try: self._mpv.pause = True
        except _get_mpv_errors(): pass

    def toggle_play_pause(self) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try: self._mpv.pause = not self._mpv.pause
        except _get_mpv_errors(): pass

    def seek(self, position_sec: float) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try:
            self._mpv.command("seek", position_sec, "absolute", "exact")
        except _get_mpv_errors() as exc:
            logger.debug("Seek {:.3f}s bị bỏ qua (Mpv C-Core bận): {}.", position_sec, exc)

    def step_frame(self, forward: bool = True) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try: self._mpv.command("frame-step" if forward else "frame-back-step")
        except _get_mpv_errors(): pass

    def set_volume(self, volume: int) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try: self._mpv.volume = max(_VOLUME_MIN, min(_VOLUME_MAX, int(volume)))
        except _get_mpv_errors(): pass

    def set_speed(self, speed: float) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try: self._mpv.speed = max(_SPEED_MIN, min(_SPEED_MAX, float(speed)))
        except _get_mpv_errors(): pass

    def set_mute(self, mute: bool) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try: self._mpv.mute = bool(mute)
        except _get_mpv_errors(): pass

    def take_screenshot(self, output_path: Path) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: raise RuntimeError("Mpv instance không khả dụng.")
        try: self._mpv.command("screenshot-to-file", str(output_path), "video")
        except _get_mpv_errors() as exc: raise RuntimeError(f"Screenshot thất bại: {exc}.") from exc
        except OSError as exc: raise RuntimeError(f"Screenshot I/O thất bại: {exc}") from exc

    def send_command(self, *args: object) -> bool:
        if not hasattr(self, "_mpv") or self._mpv is None: return False
        try:
            self._mpv.command(*args)
            return True
        except _get_mpv_errors() as exc:
            logger.debug("send_command({}) thất bại: {}.", args, exc)
            return False

    def release(self) -> None:
        if hasattr(self, "_mpv") and self._mpv is not None:
            try: self._mpv.terminate()
            except _get_mpv_errors() as exc: logger.debug("Lỗi khi terminate mpv: {}.", exc)
            self._mpv = None

    def observe_property(self, name: str, callback: Callable[..., None]) -> None:
        if not hasattr(self, "_mpv") or self._mpv is None: return
        try: self._mpv.observe_property(name, callback)
        except _get_mpv_errors(): pass

    def _mpv_get(self, name: str, default: Any = None) -> Any:
        if not hasattr(self, "_mpv") or self._mpv is None: return default
        try:
            value = getattr(self._mpv, name.replace("-", "_"), None)
            return value if value is not None else default
        except _get_mpv_errors():
            return default

    @staticmethod
    def _make_log_handler(msg_level: str) -> Callable[[str, str, str], None] | None:
        if not msg_level: return None
        def _handler(loglevel: str, component: str, message: str) -> None:
            logger.debug("[mpv:{}] {}: {}.", component, loglevel, message)
        return _handler

__all__ = ["MpvPlayerAdapter"]
