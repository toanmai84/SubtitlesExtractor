"""Đồ giả dùng chung cho các test "tiếp tục lượt dịch dở" (v3.23.584, v3.23.586).

Chạy THẬT use case dịch + adapter Gemini, chỉ thay bước gọi API từng lô bằng bộ dịch giả
có thể văng lỗi đúng một lần để mô phỏng lượt chạy bị đứt giữa chừng.
"""

from __future__ import annotations

import threading
from dataclasses import replace

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

LITERAL = TranslationStageKind.LITERAL
STYLE = TranslationStageKind.STYLE


def make_line(index: int, text: str, original: str = "") -> TranslationLine:
    """Một dòng dịch tối giản (bản gốc mặc định bằng chính ``text``)."""
    return TranslationLine(
        index=index, start_ms=index * 1000, end_ms=index * 1000 + 900, text=text,
        original_text=original or text,
    )


class FlakyBatches:
    """Thay ``_translate_single_batch``: văng lỗi MỘT lần — ở lượt gọi thứ ``fail_at``
    (đếm từ 0, chạy tuần tự) hoặc ở lô bắt đầu tại ``fail_start`` (chạy song song)."""

    def __init__(self, fail_at: int | None = None, fail_start: int | None = None) -> None:
        self.fail_at = fail_at
        self.fail_start = fail_start
        self.calls: list[int] = []
        self._lock = threading.Lock()

    def __call__(self, *, batch, start_idx, **_kwargs):
        with self._lock:
            if self.fail_at is not None and len(self.calls) == self.fail_at:
                self.fail_at = None
                raise SubtitleTranslationError("Hết hạn mức ngày (giả lập)")
            if self.fail_start is not None and start_idx == self.fail_start:
                self.fail_start = None
                raise SubtitleTranslationError("Mất kết nối (giả lập)")
            self.calls.append(start_idx)
        return [replace(line, text=f"dịch {line.index}") for line in batch]


def build_adapter(batches: FlakyBatches, parallel: int = 1) -> GeminiSubtitleTranslator:
    """Adapter Gemini thật, chỉ thay bước dịch một lô bằng ``batches``."""
    adapter = GeminiSubtitleTranslator(api_key="KEY_A", parallel_batches=parallel)
    adapter._ensure_available = lambda: None  # type: ignore[method-assign]
    adapter._translate_single_batch = batches  # type: ignore[method-assign]
    return adapter


def build_request(
    count: int = 10, *, characters: str = "", glossary: str = ""
) -> TranslateSubtitlesRequest:
    """Yêu cầu dịch ``count`` dòng qua hai giai đoạn (dịch thô, tinh chỉnh), lô 2 dòng."""
    return TranslateSubtitlesRequest(
        events=[
            SubtitleEvent(index=i, text=f"句{i}", interval=TimeInterval(i, i + 0.9))
            for i in range(1, count + 1)
        ],
        stages=[
            TranslationStageConfig(kind=LITERAL, model_name="m", batch_size=2),
            TranslationStageConfig(kind=STYLE, model_name="m", batch_size=2),
        ],
        context=TranslationContext(
            target_lang="Vietnamese", characters=characters, glossary=glossary
        ),
    )
