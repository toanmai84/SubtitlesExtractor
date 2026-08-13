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
    try:
        import mpv

        # Thử tạo instance MPV không cần WID — chỉ để probe DLL load.
        # Nếu DLL thiếu, sẽ raise OSError ngay ở constructor.
        probe_instance = mpv.MPV()
        probe_instance.terminate()
        return True
    except ImportError as exc:
        logger.warning(
            "python-mpv không khả dụng ({}) — chuyển sang VideoCanvas fallback.",
            exc,
        )
        return False
    except OSError as exc:
        logger.warning(
            "libmpv DLL/SO không load được ({}) — chuyển sang VideoCanvas fallback.",
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
