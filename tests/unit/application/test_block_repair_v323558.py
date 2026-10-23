"""[v3.23.558] Hồi quy: khối lệch dòng VẮT QUA ranh giới lô + lỗ ngoại lệ ở bước dọn dẹp.

1. Chuẩn đoán thật v3.23.557: khối 3407-3421 nằm ở đuôi lô 3361-3420 — chỉ còn 5 cửa sổ
   trượt, thiếu 1 so với ngưỡng 6, nên lưới lệch khối lúc dịch (chạy theo từng lô) mù hẳn.
   Từ dòng 3409 mỗi câu mang bản dịch của câu KẾ TIẾP, đi thẳng tới bản cuối.
2. Các bước dọn dẹp sau cùng chỉ bắt ``RuntimeError/ValueError/OSError``, nhưng lỗi gọi
   Gemini là ``SubtitleTranslationError`` — một lỗi API đúng lúc dọn dẹp làm hỏng cả lượt
   dịch đã chạy xong mọi giai đoạn.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)

# Câu gốc có độ dài dao động rõ để tương quan độ dài có ý nghĩa.
_SOURCES = [
    "好", "你今天去哪里了", "我", "那个人到底是谁啊", "走", "我们明天一起去吃饭吧",
    "嗯", "他说的都是真的吗", "对", "你为什么不早点告诉我",
]


def _vi(source: str) -> str:
    """Bản dịch 'đúng': độ dài tỉ lệ với câu gốc (khoảng 3 ký tự Việt / 1 chữ Hán)."""
    return "x" * (len(source) * 3)


def _film(count: int = 120) -> tuple[list[TranslationLine], dict[int, str]]:
    sources = [_SOURCES[i % len(_SOURCES)] for i in range(count)]
    lines = [
        TranslationLine(
            index=i + 1, start_ms=i * 1000, end_ms=i * 1000 + 900,
            text=text, original_text=text,
        )
        for i, text in enumerate(sources)
    ]
    good = {line.index: _vi(line.original_text or "") for line in lines}
    return lines, good


def _shifted(good: dict[int, str], start: int, end: int) -> dict[int, str]:
    """Mỗi dòng trong [start, end] mang bản dịch của dòng KẾ TIẾP — đúng kiểu lỗi thật."""
    result = dict(good)
    for index in range(start, end + 1):
        result[index] = good.get(index + 1, "")
    return result


class _FakeTranslator:
    def __init__(self, answer: dict[int, str] | None = None, error: bool = False) -> None:
        self.answer = answer or {}
        self.error = error
        self.calls: list[tuple[int, int]] = []

    def translate_stage(self, *, source_lines, **_kwargs):
        self.calls.append((source_lines[0].index, source_lines[-1].index))
        if self.error:
            raise SubtitleTranslationError("Gọi Gemini thất bại: 503 UNAVAILABLE")
        return [replace(line, text=self.answer[line.index]) for line in source_lines]


def _request(with_literal: bool = True) -> TranslateSubtitlesRequest:
    kind = TranslationStageKind.LITERAL if with_literal else TranslationStageKind.STYLE
    return TranslateSubtitlesRequest(
        events=[],
        stages=[TranslationStageConfig(kind=kind, model_name="m", batch_size=60)],
        context=TranslationContext(target_lang="Vietnamese"),
    )


def _stage(lines: list[TranslationLine], texts: dict[int, str]) -> list[TranslationLine]:
    return [replace(line, text=texts[line.index]) for line in lines]


class TestBlockAcrossBatchBoundary:
    def test_block_straddling_batch_edge_is_repaired(self) -> None:
        """Khối 50-70 vắt qua ranh giới lô 60 — lưới theo lô không thấy, lưới này thấy."""
        lines, good = _film()
        stage = _stage(lines, _shifted(good, 50, 70))
        translator = _FakeTranslator(answer=good)
        use_case = TranslateSubtitlesUseCase(translator=translator)
        result = use_case._repair_misaligned_blocks(
            stage, lines, _request(), None, "Dịch thô"
        )
        assert len(translator.calls) == 1
        assert {line.index: line.text for line in result} == good

    def test_repair_is_rejected_when_retry_is_still_shifted(self) -> None:
        """Dịch lại mà vẫn lệch thì GIỮ NGUYÊN bản cũ."""
        lines, good = _film()
        shifted = _shifted(good, 50, 70)
        stage = _stage(lines, shifted)
        translator = _FakeTranslator(answer=shifted)
        use_case = TranslateSubtitlesUseCase(translator=translator)
        result = use_case._repair_misaligned_blocks(
            stage, lines, _request(), None, "Dịch thô"
        )
        assert result is stage

    def test_clean_stage_costs_nothing(self) -> None:
        lines, good = _film()
        stage = _stage(lines, good)
        translator = _FakeTranslator(answer=good)
        use_case = TranslateSubtitlesUseCase(translator=translator)
        assert use_case._repair_misaligned_blocks(
            stage, lines, _request(), None, "Dịch thô"
        ) is stage
        assert translator.calls == []

    def test_api_failure_keeps_the_stage_output(self) -> None:
        lines, good = _film()
        stage = _stage(lines, _shifted(good, 50, 70))
        use_case = TranslateSubtitlesUseCase(translator=_FakeTranslator(error=True))
        assert use_case._repair_misaligned_blocks(
            stage, lines, _request(), None, "Dịch thô"
        ) is stage

    def test_without_literal_stage_it_only_reports(self) -> None:
        lines, good = _film()
        stage = _stage(lines, _shifted(good, 50, 70))
        translator = _FakeTranslator(answer=good)
        use_case = TranslateSubtitlesUseCase(translator=translator)
        assert use_case._repair_misaligned_blocks(
            stage, lines, _request(with_literal=False), None, "Tinh chỉnh"
        ) is stage
        assert translator.calls == []


class TestCleanupNeverDestroysFinishedWork:
    """Lỗi API ở bước dọn dẹp CUỐI không được làm hỏng lượt dịch đã xong mọi giai đoạn."""

    def test_duplicate_cleanup_survives_api_error(self) -> None:
        lines = [
            TranslationLine(index=1, start_ms=0, end_ms=900, text="各位各位",
                            original_text="各位各位"),
            TranslationLine(index=2, start_ms=1000, end_ms=1900, text="大家稍安勿躁",
                            original_text="大家稍安勿躁"),
        ]
        current = [replace(line, text="Xin mọi người bình tĩnh.") for line in lines]
        use_case = TranslateSubtitlesUseCase(translator=_FakeTranslator(error=True))
        assert use_case._cleanup_duplicate_adjacent(
            current, lines, _request(), None
        ) is current

    def test_residual_cjk_cleanup_survives_api_error(self) -> None:
        lines = [
            TranslationLine(index=1, start_ms=0, end_ms=900, text="你好世界",
                            original_text="你好世界"),
        ]
        current = [replace(lines[0], text="Xin chào 世界")]
        use_case = TranslateSubtitlesUseCase(translator=_FakeTranslator(error=True))
        try:
            result = use_case._cleanup_residual_cjk(current, lines, _request(), None)
        except SubtitleTranslationError:  # pragma: no cover - chỉ chạy khi lỗi tái phát
            pytest.fail("Lỗi API ở bước dọn chữ Hán làm hỏng cả lượt dịch")
        assert [line.index for line in result] == [1]
