"""Tiện ích định dạng & phân tích thời gian phụ đề.

Hỗ trợ:
    * SRT format: ``HH:MM:SS,mmm`` (mili-giây).
    * ASS format: ``H:MM:SS.cc`` (centi-giây).
    * Hiển thị ngắn cho UI: ``HH:MM:SS.mmm``.
"""

from __future__ import annotations


def seconds_to_srt(seconds: float) -> str:
    """Convert giây → format SRT ``HH:MM:SS,mmm``."""
    t = max(0.0, float(seconds))
    total_ms = int(round(t * 1000.0))
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_ass(seconds: float) -> str:
    """Convert giây → format ASS ``H:MM:SS.cc``."""
    t = max(0, int(seconds))
    cs = int((max(0.0, seconds) - t) * 100)
    return f"{t // 3600}:{(t % 3600) // 60:02d}:{t % 60:02d}.{cs:02d}"


def seconds_to_display(seconds: float) -> str:
    """Convert giây → format hiển thị UI ``HH:MM:SS.mmm``."""
    t = max(0, int(seconds))
    ms = int((max(0.0, seconds) - t) * 1000)
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}.{ms:03d}"


def srt_to_seconds(text: str) -> float:
    """Parse chuỗi ``HH:MM:SS,mmm`` (hoặc ``H:MM:SS.cc``) → giây.

    Raises:
        ValueError: Nếu chuỗi không phải format thời gian hợp lệ.
    """
    parts = text.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


__all__ = [
    "seconds_to_ass",
    "seconds_to_display",
    "seconds_to_srt",
    "srt_to_seconds",
]
