"""[v3.23.655] Bug #12: phụ đề và audio TTS lệch số dòng ở trang Xuất bản.

Ca thật: phụ đề lấy từ trang Dịch (đủ dòng) nhưng ở trang TTS đã xoá vài dòng → video có chữ mà không có tiếng, không
một lời cảnh báo. Audio TTS luôn đi kèm phụ đề đồng bộ ``<tên audio>.tts[.<lang>].srt`` (đúng các dòng có tiếng), nên so
số dòng hai tệp là đủ biết.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LineMismatch:
    """Phụ đề đang chọn lệch số dòng với phụ đề đồng bộ của audio TTS."""

    subtitle_lines: int
    tts_lines: int
    tts_subtitle: Path

    @property
    def missing_audio(self) -> int:
        """Số dòng có chữ mà không có tiếng (âm = audio nhiều dòng hơn phụ đề)."""
        return self.subtitle_lines - self.tts_lines


def count_subtitle_lines(path: Path) -> int | None:
    """Số dòng phụ đề trong tệp SRT/VTT (đếm ``-->``) hoặc ASS/SSA (đếm ``Dialogue:``); ``None`` nếu không đọc được."""
    try:
        if path.stat().st_size > _MAX_BYTES:
            return None
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    if path.suffix.lower() in {".ass", ".ssa"}:
        return sum(1 for line in text.splitlines() if line.lstrip().startswith("Dialogue:"))
    return sum(1 for line in text.splitlines() if "-->" in line)


def tts_subtitle_for(audio_path: Path) -> Path | None:
    """Phụ đề đồng bộ đi kèm audio TTS (mới nhất nếu có nhiều ngôn ngữ), hoặc ``None``."""
    try:
        candidates = [
            p for p in audio_path.parent.glob(f"{_glob_escape(audio_path.stem)}.tts*.srt") if p.is_file()
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _glob_escape(text: str) -> str:
    return "".join(f"[{ch}]" if ch in "*?[" else ch for ch in text)


def tts_line_mismatch(subtitle_path: str | Path, audio_path: str | Path) -> LineMismatch | None:
    """So số dòng phụ đề đang chọn với phụ đề đồng bộ của audio TTS; ``None`` nếu khớp / không đủ dữ liệu để so."""
    if not subtitle_path or not audio_path:
        return None
    subtitle, audio = Path(subtitle_path), Path(audio_path)
    synced = tts_subtitle_for(audio)
    if synced is None:
        return None
    try:
        if synced.resolve() == subtitle.resolve():
            return None
    except OSError:
        return None
    sub_count, tts_count = count_subtitle_lines(subtitle), count_subtitle_lines(synced)
    if sub_count is None or tts_count is None or sub_count == tts_count:
        return None
    return LineMismatch(sub_count, tts_count, synced)


__all__ = ["LineMismatch", "count_subtitle_lines", "tts_line_mismatch", "tts_subtitle_for"]
