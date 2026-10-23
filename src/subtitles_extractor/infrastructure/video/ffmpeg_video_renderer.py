"""Chạy tiến trình ffmpeg để xuất video hoàn chỉnh, có báo tiến độ và huỷ được.

Tách khỏi :mod:`video_render_command` (nơi chỉ DỰNG lệnh) theo đúng SRP: phần dựng
lệnh là hàm thuần nên kiểm thử được đầy đủ, còn module này chỉ lo chạy tiến trình.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Final

from subtitles_extractor.infrastructure.media import find_ffmpeg, missing_ffmpeg_message
from subtitles_extractor.infrastructure.video.video_encoder_selector import (
    is_hardware_encoder,
    mark_encoder_broken,
    pick_video_encoder,
    software_fallback,
)
from subtitles_extractor.infrastructure.video.subtitle_clip import clip_subtitle_file
from subtitles_extractor.infrastructure.video.video_render_command import (
    AUTO_VIDEO_ENCODER,
    RenderRequest,
    SubtitleMode,
    VideoRenderError,
    build_render_command,
    resolve_modes,
)

logger = logging.getLogger(__name__)

# ffmpeg in tiến độ ra stderr dạng: "frame=  123 fps=... time=00:00:04.92 ..."
_TIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"time=(\d+):(\d{2}):(\d{2})\.(\d{1,3})"
)

# Chu kỳ kiểm tra yêu cầu huỷ (giây).
_CANCEL_POLL_SEC: Final[float] = 0.2

ProgressCallback = Callable[[float], None]
"""Nhận tỉ lệ hoàn thành 0.0–1.0."""

CancelCallback = Callable[[], bool]
"""Trả ``True`` khi người dùng yêu cầu huỷ."""


def _parse_progress_seconds(line: str) -> float | None:
    """Đọc mốc thời gian đã xử lý từ một dòng log ffmpeg.

    Args:
        line: Một dòng stderr của ffmpeg.

    Returns:
        Số giây đã xử lý, hoặc ``None`` nếu dòng không chứa tiến độ.
    """
    match = _TIME_PATTERN.search(line)
    if match is None:
        return None
    hours, minutes, seconds, fraction = match.groups()
    # Phần thập phân có thể 1–3 chữ số ("9" = 0.9s, "92" = 0.92s, "920" = 0.920s).
    fractional = int(fraction) / (10 ** len(fraction))
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + fractional


def _no_window_kwargs() -> dict[str, object]:
    """Tham số ẩn cửa sổ console trên Windows (không ảnh hưởng nền tảng khác)."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def render_video(
    request: RenderRequest,
    *,
    total_duration_sec: float = 0.0,
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelCallback | None = None,
) -> Path:
    """Xuất video theo ``request``.

    Args:
        request: Yêu cầu xuất (xem :class:`RenderRequest`).
        total_duration_sec: Thời lượng video để quy đổi tiến độ. ``0`` = không báo %.
        on_progress: Callback nhận tiến độ 0.0–1.0.
        is_cancelled: Callback trả ``True`` để yêu cầu dừng.

    Returns:
        Đường dẫn tệp đã xuất.

    Raises:
        VideoRenderError: Khi thiếu ffmpeg, tham số sai, bị huỷ, hoặc ffmpeg lỗi.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise VideoRenderError(missing_ffmpeg_message(feature="Xuất video kèm phụ đề"))

    # [v3.23.630] Xuất thử một đoạn: cắt phụ đề về mốc 0 (hình/tiếng bắt đầu từ 0 sau -ss).
    if request.is_clip:
        temp_dir = Path(tempfile.mkdtemp(prefix="subext_clip_"))
        try:
            if request.subtitle_path is not None and resolve_modes(request)[0] is not SubtitleMode.NONE:
                request = replace(request, subtitle_path=clip_subtitle_file(
                    request.subtitle_path, temp_dir,
                    request.clip_start_sec, request.clip_duration_sec,
                ))
            return _render_resolved(
                ffmpeg, request, request.clip_duration_sec, on_progress, is_cancelled
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return _render_resolved(ffmpeg, request, total_duration_sec, on_progress, is_cancelled)


def _render_resolved(
    ffmpeg: str,
    request: RenderRequest,
    total_duration_sec: float,
    on_progress: ProgressCallback | None,
    is_cancelled: CancelCallback | None,
) -> Path:
    """Dò bộ mã hoá, chạy lệnh xuất; bộ mã hoá GPU lỗi thì xuất lại bằng CPU."""

    # [v3.23.618] "Tự động (GPU nếu có)": dò bộ mã hoá dùng được thật trên máy này.
    if request.video_encoder == AUTO_VIDEO_ENCODER:
        request = replace(request, video_encoder=pick_video_encoder(ffmpeg, "h264"))
    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Xuất video (%s): %s -> %s",
        request.mode.value,
        request.video_path.name,
        request.output_path.name,
    )
    try:
        return _run_render(
            build_render_command(ffmpeg, request), request,
            total_duration_sec, on_progress, is_cancelled,
        )
    except VideoRenderError:
        cancelled = is_cancelled is not None and is_cancelled()
        encoder = request.video_encoder
        uses_encoder = f"-c:v {encoder}" in " ".join(build_render_command(ffmpeg, request))
        if cancelled or not is_hardware_encoder(encoder) or not uses_encoder:
            raise
        # Bộ mã hoá GPU lỗi khi chạy thật (hết phiên NVENC, thiếu VRAM, driver cũ…).
        fallback = software_fallback(encoder)
        logger.warning("Bộ mã hoá %s lỗi — xuất lại bằng %s.", encoder, fallback)
        mark_encoder_broken(ffmpeg, encoder)
        request = replace(request, video_encoder=fallback)
        return _run_render(
            build_render_command(ffmpeg, request), request,
            total_duration_sec, on_progress, is_cancelled,
        )


def _run_render(
    command: list[str],
    request: RenderRequest,
    total_duration_sec: float,
    on_progress: ProgressCallback | None,
    is_cancelled: CancelCallback | None,
) -> Path:
    """Chạy một lệnh xuất ffmpeg, báo tiến độ, hỗ trợ huỷ."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **_no_window_kwargs(),  # type: ignore[arg-type]
    )

    error_tail: list[str] = []
    try:
        assert process.stderr is not None  # noqa: S101 — đã set stderr=PIPE
        for line in process.stderr:
            if is_cancelled is not None and is_cancelled():
                process.kill()
                process.wait(timeout=5)
                request.output_path.unlink(missing_ok=True)
                raise VideoRenderError("Đã huỷ xuất video theo yêu cầu.")

            # Giữ lại vài dòng cuối để báo lỗi có ngữ cảnh.
            stripped = line.strip()
            if stripped:
                error_tail.append(stripped)
                if len(error_tail) > 10:
                    error_tail.pop(0)

            if on_progress is not None and total_duration_sec > 0:
                processed = _parse_progress_seconds(line)
                if processed is not None:
                    on_progress(min(1.0, processed / total_duration_sec))
    finally:
        process.stderr.close() if process.stderr else None

    return_code = process.wait()
    if return_code != 0:
        detail = " | ".join(error_tail[-3:]) or "không có thông tin"
        request.output_path.unlink(missing_ok=True)
        raise VideoRenderError(f"ffmpeg lỗi (mã {return_code}): {detail}")

    if not request.output_path.is_file():
        raise VideoRenderError("ffmpeg báo thành công nhưng không thấy tệp đích.")

    if on_progress is not None:
        on_progress(1.0)
    logger.info("Xuất video xong: %s", request.output_path)
    return request.output_path


__all__ = ["CancelCallback", "ProgressCallback", "render_video"]
