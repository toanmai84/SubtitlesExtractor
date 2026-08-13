"""Helper build dict kwargs cho ``mpv.MPV(**)`` từ :class:`MpvSettings`.

Tách thành module riêng để 3 nơi dùng (player, frame sampler,
metadata reader) đều convert đồng nhất.

Quy tắc đặt tên: python-mpv chấp nhận cả ``--option-name`` và
``option_name``; ta dùng dấu gạch dưới khi truyền qua ``**kwargs``.
"""

from __future__ import annotations

from typing import Any

from subtitles_extractor.infrastructure.settings.application_settings import (
    MpvSettings,
)


def build_mpv_kwargs(
    mpv_settings: MpvSettings,
    *,
    role: str = "player",
) -> dict[str, Any]:
    """Build dict option mpv từ ``MpvSettings``.

    Args:
        mpv_settings: Cấu hình mpv lấy từ ``ApplicationSettings.mpv``.
        role:         ``"player"`` (UI embed), ``"sampler"`` (headless OCR),
                      ``"metadata"`` (probe only).
    """
    import sys

    kwargs: dict[str, Any] = {
        "input_default_bindings": False,
        "input_vo_keyboard": False,
        "osc": False,
        "loop_file": "no",
        "hr_seek": "yes",
        "keep_open": "yes",
        # QUAN TRỌNG: Tắt tự động load subtitle file cùng tên video.
        # Nếu không tắt, mpv sẽ load file .srt/.ass trong cùng thư mục
        # và hiển thị đè lên video — gây hiểu nhầm "player load từ file".
        "sub_auto": "no",
        "sid": "no",
    }

    # ── HW decoding ──
    kwargs["hwdec"] = mpv_settings.hwdec_mode
    if mpv_settings.hwdec_codecs:
        kwargs["hwdec_codecs"] = mpv_settings.hwdec_codecs

    # ── Cache & demuxer ──
    kwargs["cache"] = mpv_settings.cache
    kwargs["cache_secs"] = str(mpv_settings.cache_secs)
    kwargs["demuxer_max_bytes"] = f"{mpv_settings.demuxer_max_bytes}MiB"

    # ── Quality profile ──
    if mpv_settings.profile != "default":
        kwargs["profile"] = mpv_settings.profile

    # ── Deinterlace ──
    if mpv_settings.deinterlace:
        kwargs["deinterlace"] = "yes"

    # ── Logging ──
    if mpv_settings.log_level != "no":
        kwargs["msg_level"] = f"all={mpv_settings.log_level}"

    # ── VO + GPU API: khác nhau theo role ──────────────────────────────
    if role == "player":
        # Chọn VO phù hợp cho embed widget:
        # * "libmpv" là safe nhất cho mọi platform (render qua libmpv API).
        # * Nếu user chọn khác "auto" thì dùng theo cài đặt.
        # * "auto" là không hợp lệ cho embed → override về "libmpv".
        vo = mpv_settings.video_output
        if vo in ("auto", ""):
            # Chọn VO phù hợp tự động theo platform.
            if sys.platform == "win32":
                vo = "gpu"          # D3D11 trên Windows
            elif sys.platform == "darwin":
                vo = "libmpv"       # Metal qua libmpv trên macOS
            else:
                vo = "gpu"          # OpenGL/Vulkan trên Linux
        kwargs["vo"] = vo

        if mpv_settings.gpu_api != "auto":
            kwargs["gpu_api"] = mpv_settings.gpu_api
        if mpv_settings.gpu_context not in ("auto", ""):
            kwargs["gpu_context"] = mpv_settings.gpu_context
        kwargs["video_sync"] = mpv_settings.video_sync_mode

    elif role == "sampler":
        # Headless — vo=null để không cần window nhưng vẫn decode.
        kwargs["vo"] = "null"
        kwargs["ao"] = "null"

    elif role == "metadata":
        # Chỉ probe metadata — không decode video, không render.
        kwargs["vid"] = "auto"   # Cần auto để property width/height khả dụng.
        kwargs["aid"] = "no"
        kwargs["vo"] = "null"
        kwargs["ao"] = "null"

    return kwargs


__all__ = ["build_mpv_kwargs"]
