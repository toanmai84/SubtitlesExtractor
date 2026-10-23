"""Use case tổng hợp âm thanh từ danh sách phụ đề."""

from __future__ import annotations

import logging
from pathlib import Path

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    SubtitleTTSPort,
    TTSCancellationCallback,
    TTSProgressCallback,
    TTSRequest,
    TTSSegmentResult,
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
        # [v3.23.454] Dùng service chung để dựng + xuất SRT (DRY — cùng logic với TTS lại
        # dòng chọn). Nhờ vậy phụ đề luôn khớp nội dung mới nhất.
        from subtitles_extractor.application.services.tts_subtitle_export import (
            export_tts_subtitle,
        )
        return export_tts_subtitle(results, wav_path, request)


__all__ = ["GenerateTTSUseCase"]
