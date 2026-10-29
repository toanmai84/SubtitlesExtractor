"""Adapter :class:`SubtitleExporterPort` cho định dạng SubRip (.srt).

Thay vì dùng :mod:`pysubs2` như bản cũ, exporter này tự sinh nội dung —
đỡ một dependency và đảm bảo format chính xác (encoding, newline, ms).
Format SRT đơn giản::

    1
    00:00:01,234 --> 00:00:03,456
    Câu phụ đề thứ nhất

    2
    00:00:04,000 --> 00:00:05,500
    Câu phụ đề thứ hai
    Có thể nhiều dòng

Quy tắc thời gian: ``HH:MM:SS,mmm`` (dấu phẩy giữa giây và ms — đặc trưng
của SRT, không phải dấu chấm như ASS).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.exceptions import SubtitleExportError
from subtitles_extractor.infrastructure.tts.emotion_cues import strip_emotion_cues
from subtitles_extractor.application.services.bidi_marks import mark_rtl_lines
from subtitles_extractor.application.services.subtitle_line_wrapper import (
    wrap_subtitle_line,
)
from subtitles_extractor.infrastructure.subtitle.atomic_save import (
    atomic_write_text,
)

logger = logging.getLogger(__name__)


class SrtExporter:
    """Hiện thực :class:`SubtitleExporterPort` cho ``.srt``."""

    @property
    def file_extension(self) -> str:
        return ".srt"

    def export(
        self, events: Sequence[SubtitleEvent], output_path: Path
    ) -> Path:
        try:
            content = self._build_content(events)
            atomic_write_text(output_path, content, encoding="utf-8-sig")  # BOM giúp VLC/TV/Aegisub nhận đúng tiếng Việt/Trung
        except OSError as exc:
            raise SubtitleExportError(
                f"Không ghi được tệp SRT {output_path}: {exc}."
            ) from exc
        logger.info("Đã ghi %d câu phụ đề SRT → %s.", len(events), output_path)
        return output_path.resolve()

    @staticmethod
    def _build_content(
        events: Sequence[SubtitleEvent], max_chars_per_line: int = 42
    ) -> str:
        """Tạo nội dung SRT.

        [v3.23.659] Đánh số 1..N theo thứ tự ghi (trước đây dùng ``event.index``: danh sách đã lọc/cắt/ghép hoặc câu
        nhập có index 0/trùng cho số thứ tự nhảy cóc/lặp — nhiều trình phát và trình nhập SRT từ chối). Trang Chỉnh
        sửa luôn đánh lại index liên tục nên số hiển thị không đổi.

        [v3.23.441] Ngắt dòng dài (>``max_chars_per_line``) thành 2 dòng CÂN ĐỐI tại điểm
        tự nhiên (0 = tắt) — tránh player tự wrap tùy tiện giữa cụm từ.
        """
        blocks: list[str] = []
        for number, event in enumerate(events, start=1):
            start = _format_srt_timestamp(event.interval.start_ms())
            end = _format_srt_timestamp(event.interval.end_ms())
            text = _normalize_text(event.text)
            if max_chars_per_line > 0:
                text = wrap_subtitle_line(text, max_chars_per_line)
            # [v3.23.610] Dòng chữ phải-sang-trái: bọc dấu định hướng để dấu câu đúng chỗ.
            text = mark_rtl_lines(text)
            blocks.append(f"{number}\n{start} --> {end}\n{text}\n")
        return "\n".join(blocks)


def _format_srt_timestamp(milliseconds: int) -> str:
    """Format ``HH:MM:SS,mmm``."""
    milliseconds = max(milliseconds, 0)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


# Placeholder tạm thời để bảo vệ escaped braces \{ \} khỏi regex ASS block.
_BRACE_OPEN_PH = "\x00BRACE_OPEN\x00"
_BRACE_CLOSE_PH = "\x00BRACE_CLOSE\x00"


def _normalize_text(text: str) -> str:
    """Chuẩn hoá text phụ đề trước khi ghi ra SRT.

    Các bước:
        1. Bảo vệ escaped braces ``\\{``/``\\}`` bằng placeholder tạm.
        2. Xoá ASS override block ``{...}`` (ví dụ ``{\\an8}``, ``{\\b1}``).
        3. Restore escaped braces về ``{``/``}``.
        4. Xoá HTML tag cơ bản (``<b>``, ``<i>``, ``<font>``…).
        5. Chuyển ASS newline ``\\N``/``\\n`` → newline thật.
        6. Bỏ khoảng trắng thừa 2 đầu.
    """
    # Bước 1: Bảo vệ escaped braces
    cleaned = text.replace(r"\{", _BRACE_OPEN_PH).replace(r"\}", _BRACE_CLOSE_PH)
    # Bước 2: Strip ASS override block
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
    # Bước 3: Restore escaped braces
    cleaned = cleaned.replace(_BRACE_OPEN_PH, "{").replace(_BRACE_CLOSE_PH, "}")
    # Bước 4: Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # [v3.23.447] Bỏ tag cảm xúc VieNeu ([cười]/[thở dài]/[hắng giọng]) khỏi PHỤ ĐỀ — chúng chỉ
    # dành cho giọng đọc TTS, người xem không nên thấy. (TTS giữ tag qua đường prep riêng.)
    cleaned = strip_emotion_cues(cleaned)
    # Bước 5: ASS / soft newline → real newline
    cleaned = cleaned.replace(r"\N", "\n").replace(r"\n", "\n")
    return cleaned.strip()


__all__ = ["SrtExporter"]
