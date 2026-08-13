"""Adapter :class:`SubtitleImporterPort` cho định dạng ASS.

Parser tối giản — chỉ đọc các dòng ``Dialogue:`` trong section
``[Events]``, bỏ qua style/script-info. Đủ cho mục đích reload phụ đề
đã xuất.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.infrastructure.subtitle.encoding_detect import (
    read_subtitle_text,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

logger = logging.getLogger(__name__)

# ASS time: ``H:MM:SS.cc`` (centisecond, H 1-2 chữ số).
_ASS_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{2})")
# Format header trong [Events]: liệt kê tên các trường. Dùng để tìm
# chỉ số cột ``Start``, ``End``, ``Text``.
_FORMAT_RE = re.compile(r"^Format:\s*(.+)$", re.IGNORECASE)
_DIALOGUE_RE = re.compile(r"^Dialogue:\s*(.+)$", re.IGNORECASE)


class AssImporter:
    """Hiện thực :class:`SubtitleImporterPort` cho ``.ass``."""

    @property
    def file_extension(self) -> str:
        return ".ass"

    def import_from(self, source_path: Path) -> list[SubtitleEvent]:
        if not source_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp: {source_path}.")
        content = read_subtitle_text(source_path)  # [v3.23.155] dò mã hóa tự động
        events = list(_parse_ass(content))
        logger.info("Đã đọc %d câu phụ đề ASS từ %s.", len(events), source_path.name)
        return events


def _parse_ass(content: str):
    """Generator yield :class:`SubtitleEvent` từ nội dung ASS thô."""
    in_events_section = False
    field_indices: dict[str, int] = {}
    counter = 1
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("["):
            in_events_section = line.lower() == "[events]"
            continue
        if not in_events_section:
            continue

        format_match = _FORMAT_RE.match(line)
        if format_match:
            fields = [name.strip().lower() for name in format_match.group(1).split(",")]
            field_indices = {name: idx for idx, name in enumerate(fields)}
            continue

        dialogue_match = _DIALOGUE_RE.match(line)
        if not dialogue_match:
            continue
        if not field_indices:
            logger.debug("Gặp Dialogue trước Format — bỏ qua: %s", line[:60])
            continue

        # Số phần phải bằng len(fields); cột "Text" luôn là cột cuối, có thể
        # chứa dấu phẩy nên dùng split với maxsplit để giữ nguyên.
        max_splits = len(field_indices) - 1
        parts = [p.strip() for p in dialogue_match.group(1).split(",", maxsplit=max_splits)]
        if len(parts) != len(field_indices):
            continue

        try:
            start_sec = _parse_ass_time(parts[field_indices["start"]])
            end_sec = _parse_ass_time(parts[field_indices["end"]])
            text_raw = parts[field_indices["text"]]
        except (KeyError, ValueError):
            continue

        text = _clean_ass_text(text_raw)
        if not text:
            continue
        try:
            interval = TimeInterval(start_sec=start_sec, end_sec=end_sec)
        except ValueError:
            continue

        yield SubtitleEvent(
            index=counter,
            text=text,
            interval=interval,
            confidence=Confidence(1.0),
        )
        counter += 1


def _parse_ass_time(token: str) -> float:
    match = _ASS_TIME_RE.fullmatch(token.strip())
    if not match:
        raise ValueError(f"Không parse được mốc thời gian ASS: {token!r}.")
    h, m, s, cs = (int(group) for group in match.groups())
    return h * 3600.0 + m * 60.0 + s + cs / 100.0


_BRACE_OPEN_PH = "\x00BO\x00"
_BRACE_CLOSE_PH = "\x00BC\x00"


def _clean_ass_text(raw: str) -> str:
    """Làm sạch text ASS: xoá override block ``{...}``, unescape ``\\{``/``\\}``,
    đổi ``\\N``/``\\n`` thành newline thật.

    Thứ tự quan trọng:
        1. Bảo vệ ``\\{``/``\\}`` bằng placeholder tạm để regex không nhầm.
        2. Xoá ASS override block ``{...}``.
        3. Restore braces từ placeholder.
        4. Convert newline escape.
    """
    # Bước 1: Bảo vệ escaped braces
    protected = raw.replace(r"\{", _BRACE_OPEN_PH).replace(r"\}", _BRACE_CLOSE_PH)
    # Bước 2: Xoá ASS override block (non-greedy để khớp từng cặp đúng)
    no_overrides = re.sub(r"\{[^}]*\}", "", protected)
    # Bước 3: Restore escaped braces
    unescaped = no_overrides.replace(_BRACE_OPEN_PH, "{").replace(_BRACE_CLOSE_PH, "}")
    # Bước 4: ASS / soft newline → real newline
    return unescaped.replace(r"\N", "\n").replace(r"\n", "\n").strip()


__all__ = ["AssImporter"]
