"""[v3.23.630] Cắt phụ đề SRT về một đoạn rồi dời mốc về 0 — cho xuất thử một đoạn video.

Xuất thử dùng ``-ss`` trước ``-i`` (nhanh, không đọc cả phim) nên hình/tiếng bắt đầu từ mốc 0;
phụ đề phải được dời tương ứng, nếu không bộ lọc ``subtitles=`` (hay track phụ đề) lệch mốc.
Hàm thuần (chuỗi → chuỗi) + một hàm ghi tệp tạm.
"""

from __future__ import annotations

import re
from pathlib import Path

from subtitles_extractor.infrastructure.video.video_render_command import VideoRenderError

_TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)
#: Cue trống 0,1 giây — ffmpeg không nhận tệp SRT rỗng, nên đoạn không có lời vẫn có tệp hợp lệ.
_PLACEHOLDER = "1\n00:00:00,000 --> 00:00:00,100\n \n"


def _seconds(parts: tuple[str, ...]) -> float:
    """``(giờ, phút, giây, phần nghìn)`` dạng chuỗi → số giây."""
    hours, minutes, seconds, millis = parts
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000


def _stamp(value: float) -> str:
    total_ms = max(0, round(value * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def clip_srt(text: str, start_sec: float, duration_sec: float) -> str:
    """Giữ các cue giao với ``[start, start+duration]``, dời về mốc 0, cắt biên, đánh số lại.

    Args:
        text: Nội dung SRT.
        start_sec: Mốc bắt đầu đoạn.
        duration_sec: Độ dài đoạn.

    Returns:
        Nội dung SRT của đoạn (luôn có ít nhất một cue).
    """
    end_sec = start_sec + duration_sec
    kept: list[tuple[float, float, str]] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.replace("﻿", "").strip()):
        lines = block.strip().splitlines()
        stamp_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if stamp_index is None:
            continue
        match = _TIME_RE.search(lines[stamp_index])
        if match is None:
            continue
        begin = _seconds(match.group(1, 2, 3, 4))
        finish = _seconds(match.group(5, 6, 7, 8))
        if finish <= start_sec or begin >= end_sec:
            continue
        body = "\n".join(lines[stamp_index + 1:]).strip()
        if not body:
            continue
        kept.append((max(begin, start_sec) - start_sec, min(finish, end_sec) - start_sec, body))
    if not kept:
        return _PLACEHOLDER
    return "\n".join(
        f"{number}\n{_stamp(begin)} --> {_stamp(finish)}\n{body}\n"
        for number, (begin, finish, body) in enumerate(kept, 1)
    )


def clip_subtitle_file(
    source: Path, destination_dir: Path, start_sec: float, duration_sec: float
) -> Path:
    """Ghi bản cắt của ``source`` vào ``destination_dir`` và trả đường dẫn.

    Raises:
        VideoRenderError: Phụ đề không phải ``.srt`` (xuất thử đoạn chỉ hỗ trợ SRT) hoặc
            không đọc được.
    """
    if source.suffix.lower() != ".srt":
        raise VideoRenderError(
            "Xuất thử một đoạn chỉ hỗ trợ phụ đề .srt — hãy dùng tệp .srt hoặc tắt “Chỉ xuất "
            "đoạn thử”."
        )
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise VideoRenderError(f"Không đọc được phụ đề {source.name}: {exc}") from exc
    destination = destination_dir / source.name
    destination.write_text(clip_srt(text, start_sec, duration_sec), encoding="utf-8")
    return destination


__all__ = ["clip_srt", "clip_subtitle_file"]
