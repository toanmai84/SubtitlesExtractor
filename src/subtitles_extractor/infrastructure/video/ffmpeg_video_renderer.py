"""Chạy tiến trình ffmpeg để xuất video hoàn chỉnh, có báo tiến độ và huỷ được.

Tách khỏi :mod:`video_render_command` (nơi chỉ DỰNG lệnh) theo đúng SRP: phần dựng
lệnh là hàm thuần nên kiểm thử được đầy đủ, còn module này chỉ lo chạy tiến trình.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from subtitles_extractor.infrastructure.media import find_ffmpeg, missing_ffmpeg_message
from subtitles_extractor.infrastructure.video.video_render_command import (
    RenderRequest,
    VideoRenderError,
    build_render_command,
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

    command = build_render_command(ffmpeg, request)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Xuất video (%s): %s -> %s",
        request.mode.value,
        request.video_path.name,
        request.output_path.name,
    )

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
