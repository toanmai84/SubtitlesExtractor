"""[v3.23.92] Phân tích tuần tự tích luỹ: prior_context truyền + chèn vào prompt."""

from __future__ import annotations

from typing import Any

from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    TranslationLine,
)


class _RecordingTranslator:
    """Translator giả ghi lại kwargs để kiểm prior_context được truyền xuống."""

    def __init__(self) -> None:
        self.received: dict[str, Any] = {}

    def analyze_global_context(self, **kwargs: Any) -> SubtitleContextAnalysis:
        self.received = kwargs
        return SubtitleContextAnalysis(source_lang="zh", characters="", overview="")


def test_use_case_forwards_prior_context() -> None:
    tr = _RecordingTranslator()
    uc = AnalyzeSubtitleContextUseCase(tr)
    lines = [TranslationLine(index=1, start_ms=0, end_ms=1000, text="你好")]
    uc.execute(lines, "Vietnamese", prior_context="# Nhân vật:\nTần Chính (秦政)")
    assert tr.received["prior_context"] == "# Nhân vật:\nTần Chính (秦政)"


def test_use_case_default_prior_context_empty() -> None:
    tr = _RecordingTranslator()
    uc = AnalyzeSubtitleContextUseCase(tr)
    lines = [TranslationLine(index=1, start_ms=0, end_ms=1000, text="你好")]
    uc.execute(lines, "Vietnamese")
    assert tr.received["prior_context"] == ""
