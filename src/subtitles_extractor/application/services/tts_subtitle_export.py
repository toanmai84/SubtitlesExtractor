"""[v3.23.454] Xuất SRT đồng bộ với lời thoại đã TTS (dùng chung cho generate lẫn re-synthesize).

Tách khỏi ``GenerateTTSUseCase`` để trang TTS khi TTS LẠI dòng chọn cũng xuất lại phụ đề khớp nội
dung vừa sửa (trước đây re-TTS chỉ ghi WAV, phụ đề đính kèm giữ nội dung cũ).
"""

from __future__ import annotations

from subtitles_extractor.domain.exceptions import ConfigurationError

import logging
from pathlib import Path

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.domain.value_objects.output_naming import tts_subtitle_path
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import SrtExporter

logger = logging.getLogger(__name__)

#: Khe tối thiểu giữa 2 phụ đề (giây) — phụ đề KHÔNG chồng lấn dù audio có thể chồng tiếng.
_SRT_GAP = 0.04


def export_tts_subtitle(
    results: list[TTSSegmentResult], wav_path: Path, request: TTSRequest
) -> Path | None:
    """Xuất SRT khớp lời thoại đã TTS (mốc thời gian đã điều chỉnh, không chồng lấn).

    Args:
        results: Kết quả TTS (đã gồm nội dung MỚI nếu vừa TTS lại dòng chọn).
        wav_path: Đường dẫn audio output (SRT đặt cạnh: ``<tên>.tts.<lang>.srt``).
        request: Yêu cầu TTS (lấy ``language`` để đặt tên).

    Returns:
        Đường dẫn SRT đã ghi, hoặc None nếu không có dòng nào hợp lệ / lỗi ghi.
    """
    rows: list[tuple[float, float, str]] = []
    for result in results:
        if not (result.text or "").strip():
            continue
        start = result.adjusted_start_sec if result.adjusted_start_sec >= 0.0 else result.start_sec
        end = result.adjusted_end_sec if result.adjusted_end_sec >= 0.0 else result.end_sec
        # File phụ đề giữ tag người nói: ưu tiên subtitle_text (có tag), fallback text.
        subtitle_line = getattr(result, "subtitle_text", "") or result.text
        rows.append((start, end, subtitle_line))

    if not rows:
        return None

    rows.sort(key=lambda t: t[0])
    events: list[SubtitleEvent] = []
    idx_counter = 1
    for i, (start, end, text) in enumerate(rows):
        if i + 1 < len(rows):
            next_start = rows[i + 1][0]
            if end > next_start - _SRT_GAP:
                end = next_start - _SRT_GAP
        if end <= start:
            end = start + 0.1
        try:
            events.append(
                SubtitleEvent(
                    index=idx_counter,
                    text=text,
                    interval=TimeInterval(start_sec=start, end_sec=end),
                )
            )
            idx_counter += 1
        except (ValueError, ConfigurationError) as exc:
            logger.warning("Bỏ qua dòng SRT lỗi timing (%.3f-%.3f): %s", start, end, exc)

    if not events:
        return None

    srt_path = tts_subtitle_path(wav_path, getattr(request, "language", "") or "")
    try:
        SrtExporter().export(events, srt_path)
        logger.info("Đã xuất SRT đồng bộ TTS (%d dòng) → %s", len(events), srt_path)
        return srt_path
    except OSError as exc:
        logger.error("Không xuất được SRT điều chỉnh: %s", exc)
        return None


__all__ = ["export_tts_subtitle"]
