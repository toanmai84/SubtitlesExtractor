"""Use case tổng hợp âm thanh từ danh sách phụ đề."""

from __future__ import annotations

import logging
from pathlib import Path

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    SubtitleTTSPort,
    TTSProgressCallback,
    TTSCancellationCallback,
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import (
    SrtExporter,
)

logger = logging.getLogger(__name__)


class GenerateTTSUseCase:
    """Điều phối quá trình TTS: xác thực → gọi adapter → trả kết quả."""

    def __init__(self, adapter: SubtitleTTSPort) -> None:
        self._adapter = adapter

    def execute(
        self,
        request: TTSRequest,
        output_path: Path,
        progress_cb: TTSProgressCallback | None = None,
        cancel_cb: TTSCancellationCallback | None = None,
    ) -> list[TTSSegmentResult]:
        """Chạy TTS và ghi WAV. Nếu timing thay đổi (elastic), xuất SRT mới.

        Args:
            request:     Cấu hình TTS.
            output_path: Đường dẫn file WAV đầu ra.
            progress_cb: Callback tiến độ.
            cancel_cb:   Callback kiểm tra huỷ.

        Returns:
            Danh sách kết quả từng dòng.
        """
        if not request.events:
            logger.warning("GenerateTTSUseCase: không có events nào.")
            return []

        n_valid = sum(1 for e in request.events if e.text.strip())
        logger.info(
            "GenerateTTS: %d events (%d có text), engine=%s, lang=%s, voice=%s",
            len(request.events), n_valid,
            self._adapter.get_engine_name(), request.language, request.speaker,
        )

        results = self._adapter.generate(
            request=request,
            output_path=output_path,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

        n_ok = sum(1 for r in results if not r.was_skipped)
        n_trunc = sum(1 for r in results if r.was_truncated)
        logger.info(
            "GenerateTTS xong: %d/%d OK, %d bị cắt, WAV → %s",
            n_ok, len(results), n_trunc, output_path,
        )

        # Xuất SRT đã điều chỉnh nếu elastic timing làm thay đổi mốc thời gian
        self._export_adjusted_srt(results, output_path, request)

        return results

    @staticmethod
    def _export_adjusted_srt(
        results: list[TTSSegmentResult], wav_path: Path, request: TTSRequest
    ) -> Path | None:
        """Xuất SRT khớp với lời thoại đã TTS (mốc thời gian đã điều chỉnh).

        Chỉ xuất khi có ít nhất một dòng bị dời thời gian (adjusted_start ≥ 0).
        File SRT đặt cạnh audio: ``<tên>.tts.srt`` (quy ước an toàn, không ghi đè gốc).

        Returns:
            Đường dẫn SRT đã ghi, hoặc None nếu không có gì thay đổi.
        """
        has_adjusted = any(
            r.adjusted_start_sec >= 0.0 and not r.was_skipped for r in results
        )
        if not has_adjusted:
            return None

        events: list[SubtitleEvent] = []
        idx_counter = 1
        rows: list[tuple[float, float, str]] = []
        for r in results:
            # Bỏ qua dòng KHÔNG có text (rỗng thật sự). Giữ dòng mô tả âm thanh
            # (was_skipped nhưng còn text gốc) để phụ đề vẫn hiển thị khớp phim.
            if not r.text.strip():
                continue
            start = r.adjusted_start_sec if r.adjusted_start_sec >= 0.0 else r.start_sec
            end = r.adjusted_end_sec if r.adjusted_end_sec >= 0.0 else r.end_sec
            # [v3.23.77] FILE PHỤ ĐỀ giữ tag người nói: ưu tiên subtitle_text (có tag),
            # fallback text (các dòng skipped vốn đã chứa tag).
            subtitle_line = r.subtitle_text or r.text
            rows.append((start, end, subtitle_line))

        rows.sort(key=lambda t: t[0])
        # Phụ đề KHÔNG được chồng lấn dù audio TTS có thể chồng tiếng: cắt đuôi
        # câu trước về trước mốc bắt đầu câu sau (chừa khe nhỏ 40ms).
        _SRT_GAP = 0.04
        for i, (start, end, text) in enumerate(rows):
            if i + 1 < len(rows):
                next_start = rows[i + 1][0]
                if end > next_start - _SRT_GAP:
                    end = next_start - _SRT_GAP
            if end <= start:
                end = start + 0.1  # đảm bảo end > start cho TimeInterval
            try:
                events.append(SubtitleEvent(
                    index=idx_counter,
                    text=text,
                    interval=TimeInterval(start_sec=start, end_sec=end),
                ))
                idx_counter += 1
            except ValueError as exc:
                logger.warning("Bỏ qua dòng SRT lỗi timing (%.3f-%.3f): %s", start, end, exc)

        if not events:
            return None

        # [3.2] Quy ước đặt tên an toàn: phụ đề đã chỉnh giờ khớp giọng đặt là
        # ``<tên>.tts.<lang>.srt`` để KHÔNG ghi đè phụ đề gốc/bản dịch cùng thư mục.
        from subtitles_extractor.domain.value_objects.output_naming import tts_subtitle_path
        srt_path = tts_subtitle_path(wav_path, getattr(request, "language", "") or "")
        try:
            SrtExporter().export(events, srt_path)
            logger.info("Đã xuất SRT đồng bộ TTS (%d dòng) → %s", len(events), srt_path)
            return srt_path
        except Exception as exc:  # noqa: BLE001 - log mọi lỗi export, không chặn TTS
            logger.error("Không xuất được SRT điều chỉnh: %s", exc)
            return None


__all__ = ["GenerateTTSUseCase"]
