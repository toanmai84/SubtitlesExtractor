"""Adapter :class:`SubtitleExporterPort` cho định dạng Advanced SubStation Alpha.

ASS phong phú hơn SRT (định dạng kiểu chữ, hiệu ứng, vị trí…) — adapter
này dùng style mặc định "Default" nhằm giữ cú pháp tối giản, tương thích
mọi phần mềm phát.

Khác biệt cú pháp đáng chú ý so với SRT:
    * Thời gian: ``H:MM:SS.cc`` (centisecond), ``H`` 1 chữ số.
    * Newline trong text dùng ``\\N``, không phải ``\\n``.
    * Dấu phẩy là ký tự đặc biệt — phải escape khi xuất hiện trong text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.exceptions import SubtitleExportError
from subtitles_extractor.infrastructure.subtitle.atomic_save import (
    atomic_write_text,
)

logger = logging.getLogger(__name__)


_HEADER: str = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


class AssExporter:
    """Hiện thực :class:`SubtitleExporterPort` cho ``.ass``."""

    @property
    def file_extension(self) -> str:
        return ".ass"

    def export(
        self, events: Sequence[SubtitleEvent], output_path: Path
    ) -> Path:
        try:
            content = _HEADER + "".join(
                self._format_event_line(event) for event in events
            )
            atomic_write_text(output_path, content, encoding="utf-8")
        except OSError as exc:
            raise SubtitleExportError(
                f"Không ghi được tệp ASS {output_path}: {exc}."
            ) from exc
        logger.info("Đã ghi %d câu phụ đề ASS → %s.", len(events), output_path)
        return output_path.resolve()

    @staticmethod
    def _format_event_line(event: SubtitleEvent) -> str:
        start = _format_ass_timestamp(event.interval.start_sec)
        end = _format_ass_timestamp(event.interval.end_sec)
        text = _escape_ass_text(event.text)
        return f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"


def _format_ass_timestamp(seconds: float) -> str:
    """Format ``H:MM:SS.cc`` (centisecond)."""
    if seconds < 0:
        seconds = 0.0
    centiseconds_total = int(round(seconds * 100))
    hours, remainder = divmod(centiseconds_total, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cs = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape ký tự đặc biệt ASS và convert HTML tag sang ASS override."""
    sanitized = text.replace("\r\n", "\n").replace("\r", "\n")
    sanitized = sanitized.replace("{", r"\{").replace("}", r"\}")
    sanitized = sanitized.replace("<b>", r"{\b1}").replace("</b>", r"{\b0}")
    sanitized = sanitized.replace("<i>", r"{\i1}").replace("</i>", r"{\i0}")
    sanitized = sanitized.replace("<u>", r"{\u1}").replace("</u>", r"{\u0}")
    sanitized = re.sub(r"<[^>]+>", "", sanitized)
    return sanitized.replace("\n", r"\N")


__all__ = ["AssExporter"]
