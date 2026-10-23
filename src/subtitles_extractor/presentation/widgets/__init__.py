"""Custom widgets cho UI (v2.34 — MPV + VideoCanvas fallback).

Kiến trúc video player:
    * ``MpvVideoWidget`` — Native MPV playback + libmpv OSD overlay (default).
    * ``VideoCanvas`` — QLabel frame-based fallback khi MPV không khả dụng.
    * ``create_video_widget()`` — Factory tự động chọn implementation phù hợp
      dựa trên availability của ``python-mpv`` + libmpv DLL.

[v2.34 RESTORE]: VideoCanvas đã được tái tạo (sau khi xoá ở v2.33) để phục
vụ vai trò fallback. API hoàn toàn tương thích với MpvVideoWidget — cùng
signal names (``roi_changed``, ``roi_preview``, ``video_clicked``,
``video_double_clicked``) và method names (``set_committed_roi``,
``set_video_size``, ``enable_roi_drawing``, ``load``, ``player``,
``release_player``, ...) — nên caller có thể swap drop-in.
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING

from loguru import logger

try:
    from subtitles_extractor.presentation.widgets.mpv_video_widget import MpvVideoWidget
    from subtitles_extractor.presentation.widgets.video_canvas import VideoCanvas
    _PYQT6_AVAILABLE = True
except ModuleNotFoundError:
    # Môi trường headless (CI / unit test không có PyQt6) — khai báo stub
    MpvVideoWidget = None  # type: ignore[assignment,misc]
    VideoCanvas = None  # type: ignore[assignment,misc]
    _PYQT6_AVAILABLE = False

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


def _is_mpv_available() -> bool:
    """Kiểm tra ``python-mpv`` + libmpv DLL có thể import/load không.

    Thực hiện 1 lần ở module-load để tránh overhead. Kết quả cache vào
    ``_MPV_AVAILABLE_CACHE``.

    Returns:
        ``True`` nếu cả 2 điều kiện:
            * Có thể ``import mpv``.
            * libmpv shared library (DLL/SO) load được không lỗi ``OSError``.
        Ngược lại ``False``.
    """
    if os.environ.get("SUBTITLES_EXTRACTOR_NO_MPV"):
        logger.warning("SUBTITLES_EXTRACTOR_NO_MPV được đặt — dùng VideoCanvas thay mpv.")
        return False
    try:
        from subtitles_extractor.infrastructure.logging.crash_hooks import mpv_disabled_by_crash

        if mpv_disabled_by_crash():
            logger.warning(
                "Lần chạy trước crash trong mpv (access violation) — tạm dùng VideoCanvas. "
                "Xoá tệp mpv_disabled.flag trong thư mục logs (hoặc đặt SUBTITLES_EXTRACTOR_FORCE_MPV=1) để bật lại mpv."
            )
            return False
    except Exception:  # noqa: BLE001 - lớp bảo vệ không được làm hỏng việc dựng trang
        pass
    try:
        # [v3.23.651] Chỉ `import mpv` (python-mpv nạp libmpv ngay lúc import, thiếu DLL → OSError). KHÔNG dựng thử
        # `mpv.MPV()` rồi terminate() nữa: log crash thật cho thấy `access violation` trong `terminate()`/`_loop` của
        # lần dò này ngay lúc khởi động.
        import mpv  # noqa: F401

        return True
    except ImportError as exc:
        logger.warning(
            "python-mpv không khả dụng ({}) — chuyển sang VideoCanvas fallback.",
            exc,
        )
        return False
    except (OSError, RuntimeError, AttributeError) as exc:
        # [v3.23.619] + RuntimeError/AttributeError: libmpv nạp được nhưng khởi tạo lỗi hoặc
        # lệch phiên bản với python-mpv — trước đây lọt ra ngoài làm hỏng việc dựng trang.
        logger.warning(
            "libmpv DLL/SO không dùng được ({}) — chuyển sang VideoCanvas fallback.",
            exc,
        )
        return False


# Cache kết quả check (compute 1 lần ở module-load).
_MPV_AVAILABLE_CACHE: bool | None = None


def is_mpv_available() -> bool:
    """Trả True nếu MPV khả dụng, ngược lại False. Lazy + cached."""
    global _MPV_AVAILABLE_CACHE
    if _MPV_AVAILABLE_CACHE is None:
        _MPV_AVAILABLE_CACHE = _is_mpv_available()
    return _MPV_AVAILABLE_CACHE


def create_video_widget(
    *,
    mpv_options: dict | None = None,
    parent: QWidget | None = None,
    force_fallback: bool = False,
    translator: object | None = None,
) -> MpvVideoWidget | VideoCanvas:
    """Factory — chọn ``MpvVideoWidget`` hoặc ``VideoCanvas`` tự động.

    Args:
        mpv_options: Tham số cho MPV khi khởi tạo (cho phép pass-through từ
            ``container.build_mpv_player_kwargs()``). Bỏ qua nếu fallback.
        parent: Qt parent widget.
        force_fallback: ``True`` để bypass MPV và ép dùng VideoCanvas (vd
            test/debug).

    Returns:
        Widget có API tương thích — caller chỉ cần biết các method/signal
        chung (``roi_changed``, ``set_committed_roi``, ``load``, ...).

        Để biết widget có MPV thực không, gọi ``widget.player() is not None``.
    """
    if force_fallback or not is_mpv_available():
        logger.info("Tạo VideoCanvas (fallback mode — MPV không khả dụng).")
        return VideoCanvas(mpv_options=mpv_options, parent=parent)

    logger.debug("Tạo MpvVideoWidget (MPV native playback + OSD).")
    return MpvVideoWidget(mpv_options=mpv_options, parent=parent, translator=translator)


__all__ = [
    "MpvVideoWidget",
    "VideoCanvas",
    "create_video_widget",
    "is_mpv_available",
]
