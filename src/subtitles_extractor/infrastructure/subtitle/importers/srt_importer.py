"""Adapter :class:`SubtitleImporterPort` cho định dạng SubRip (.srt).

Parser thuần regex — không phụ thuộc thư viện ngoài. Tolerant với:
    * BOM ở đầu tệp.
    * Newline ``\\r\\n`` lẫn ``\\n``.
    * Số block không tuần tự, hoặc thiếu số block.
    * Dòng trống thừa giữa các block.
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


# [v3.20.3 #5] Vũ khí diệt ký tự TÀNG HÌNH gây sai timing/nhận diện:
# Zero-width space/non-joiner/joiner (200B-200D), BOM/ZWNBSP (FEFF), LRM/RLM
# (200E-200F), word joiner (2060), non-breaking space (00A0) → khoảng trắng thường.
_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]")
_NBSP_RE = re.compile(r"[\u00a0\u202f]")
# [v3.20.3 #5] Chuẩn hoá mọi biến thể mũi tên hỏng về ``-->`` chuẩn:
# ``->``, ``-- >``, ``- ->``, ``—>`` (em dash), ``–>`` (en dash), ``=>``, ``→``,
# với số lượng gạch/khoảng trắng tuỳ ý quanh đầu mũi tên ``>``.
_BROKEN_ARROW_RE = re.compile(r"[\u2012-\u2015\u2192=\-]+\s*>|\u2192")


# Cú pháp ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` (cho phép dấu chấm thay phẩy).
_TIMING_RE = re.compile(
    r"(?P<sh>\d+):(?P<sm>\d+):(?P<ss>\d+)[,.](?P<sms>\d+)\s*-->\s*"
    r"(?P<eh>\d+):(?P<em>\d+):(?P<es>\d+)[,.](?P<ems>\d+)"
)


class SrtImporter:
    """Hiện thực :class:`SubtitleImporterPort` cho ``.srt``."""

    @property
    def file_extension(self) -> str:
        return ".srt"

    def import_from(self, source_path: Path) -> list[SubtitleEvent]:
        if not source_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp: {source_path}.")
        raw = read_subtitle_text(source_path)  # [v3.23.155] dò mã hóa tự động
        events = list(_parse_srt(raw))
        logger.info("Đã đọc %d câu phụ đề từ %s.", len(events), source_path.name)
        return events


def _parse_srt(content: str):
    """Generator yield :class:`SubtitleEvent` từ chuỗi SRT thô.

    [Robust Sequential Parser] Thay vì tách block bằng ``\\n\\s*\\n`` (dễ cắt nhầm
    khi có dòng trống xen kẽ hoặc Zero-width Space), duyệt TUẦN TỰ từng dòng và
    nhận diện block mới dựa trên dòng chứa timecode ``-->``. Chấp nhận file SRT lỗi.
    """
    # Chuẩn hoá newline + diệt SẠCH ký tự tàng hình + chuẩn hoá mũi tên hỏng.
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _INVISIBLE_CHARS_RE.sub("", normalized)
    normalized = _NBSP_RE.sub(" ", normalized)
    normalized = _BROKEN_ARROW_RE.sub("-->", normalized)
    normalized = normalized.strip()
    if not normalized:
        return

    lines = normalized.split("\n")
    total = len(lines)
    counter = 1
    line_index = 0
    while line_index < total:
        timing_match = _TIMING_RE.search(lines[line_index])
        if timing_match is None:
            line_index += 1
            continue

        # Gom các dòng văn bản đến trước timecode kế tiếp.
        text_start = line_index + 1
        cursor = text_start
        while cursor < total and _TIMING_RE.search(lines[cursor]) is None:
            cursor += 1

        text_lines = lines[text_start:cursor]
        # Dòng cuối có thể là số thứ tự của block kế (đứng ngay trước timecode) → bỏ.
        if text_lines and text_lines[-1].strip().isdigit():
            text_lines = text_lines[:-1]
        text = "\n".join(text_lines).strip()
        line_index = cursor

        if not text:
            continue
        try:
            interval = TimeInterval(
                start_sec=_to_seconds(
                    int(timing_match["sh"]), int(timing_match["sm"]),
                    int(timing_match["ss"]), _parse_milliseconds(timing_match["sms"]),
                ),
                end_sec=_to_seconds(
                    int(timing_match["eh"]), int(timing_match["em"]),
                    int(timing_match["es"]), _parse_milliseconds(timing_match["ems"]),
                ),
            )
        except ValueError:
            logger.warning("Block có timing không hợp lệ — bỏ qua: %r", text[:80])
            continue
        yield SubtitleEvent(
            index=counter, text=text, interval=interval, confidence=Confidence(1.0)
        )
        counter += 1


def _parse_milliseconds(fractional_digits: str) -> int:
    """Đổi phần thập phân giây trong SRT về mili-giây ĐÚNG độ phân giải.

    [Millisecond Precision] ``00:00:01,5`` nghĩa là 500ms (0.5 giây), KHÔNG phải
    5ms. Đệm phải bằng '0' đến đủ 3 chữ số (``"5"`` → ``"500"``), cắt nếu dư
    (``"1234"`` → ``"123"``).
    """
    if not fractional_digits:
        return 0
    return int(fractional_digits.ljust(3, "0")[:3])


def _to_seconds(h: int, m: int, s: int, ms: int) -> float:
    return h * 3600.0 + m * 60.0 + s + ms / 1000.0


__all__ = ["SrtImporter"]
