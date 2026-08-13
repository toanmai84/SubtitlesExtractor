"""Bộ dịch giả lập (fake) phục vụ kiểm thử tích hợp — không gọi mạng.

``FakeSubtitleTranslator`` cài đặt đầy đủ giao diện :class:`SubtitleTranslatorPort` một
cách TẤT ĐỊNH (deterministic): mỗi giai đoạn dịch thêm một tiền tố nhận biết được vào văn
bản, nhờ đó test có thể kiểm chứng pipeline đã chạy đúng thứ tự giai đoạn và giữ nguyên số
dòng/chỉ số. Có thể nạp sẵn một "từ điển" để mô phỏng việc dùng lại bản dịch nhất quán
(phục vụ test Translation Memory).
"""

from __future__ import annotations

from typing import Any

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    CancellationCallback,
    StageProgressCallback,
    SubtitleContextAnalysis,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
)

__all__ = ["FakeSubtitleTranslator"]


class FakeSubtitleTranslator:
    """Cài đặt :class:`SubtitleTranslatorPort` tất định, không phụ thuộc mạng.

    Args:
        available: Giá trị trả về của :meth:`is_available`.
        dictionary: Ánh xạ ``câu gốc → câu dịch`` để mô phỏng bản dịch nhất quán; câu
            không có trong từ điển sẽ được gắn tiền tố giai đoạn.
        analysis: Kết quả cố định cho :meth:`analyze_global_context` (nếu cần).
    """

    def __init__(
        self,
        *,
        available: bool = True,
        dictionary: dict[str, str] | None = None,
        analysis: SubtitleContextAnalysis | None = None,
    ) -> None:
        self._available = available
        self._dictionary = dictionary or {}
        self._analysis = analysis
        self.translate_stage_calls: list[str] = []

    def is_available(self) -> bool:
        return self._available

    def translate_stage(
        self,
        *,
        stage: TranslationStageConfig,
        context: TranslationContext,
        source_lines: list[TranslationLine],
        input_lines: list[TranslationLine],
        has_prior_translation: bool,
        progress_cb: StageProgressCallback | None = None,
        cancel_cb: CancellationCallback | None = None,
        video_refs: list[Any] | None = None,
        attach_video: bool = False,
    ) -> list[TranslationLine]:
        """Trả về danh sách dòng đã 'dịch' tất định, giữ nguyên index và độ dài."""
        self.translate_stage_calls.append(stage.kind.value)
        if progress_cb is not None:
            progress_cb(1.0)
        prefix = f"[{stage.kind.value}] "
        result: list[TranslationLine] = []
        for line in input_lines:
            translated = self._dictionary.get(line.text)
            new_text = translated if translated is not None else prefix + line.text
            result.append(
                TranslationLine(
                    index=line.index,
                    start_ms=line.start_ms,
                    end_ms=line.end_ms,
                    text=new_text,
                )
            )
        return result

    def analyze_global_context(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        cancel_cb: CancellationCallback | None = None,
        video_refs: list[Any] | None = None,
        with_visual_cues: bool = False,
        prior_context: str = "",
    ) -> SubtitleContextAnalysis:
        if self._analysis is not None:
            return self._analysis
        return SubtitleContextAnalysis(
            source_lang="zh",
            characters="Nhân vật A",
            overview="Tóm tắt thử nghiệm.",
        )
