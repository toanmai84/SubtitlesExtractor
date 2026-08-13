"""Tests bảo vệ các fix Trang Dịch v3.7.

B1 — enable_tags/include_desc PHẢI có tác dụng (trước đây là cờ chết):
     enable_tags=False → KHÔNG chèn '[speaker:]'; include_desc=False → KHÔNG chèn '(desc)'.
B2 — Huỷ ném TranslationCancelledError (phân biệt với lỗi thật).
B4 — _interruptible_sleep dừng sớm khi cancel_cb trả True.
"""

from __future__ import annotations

import time

import pytest

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationCancelledError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _make_events(texts: list[str]) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i + 1, text=t, interval=TimeInterval(float(i), float(i) + 1.0))
        for i, t in enumerate(texts)
    ]


class _SpeakerDescTranslator:
    """Translator giả: gán speaker='A' và description='cười' cho mọi dòng."""

    def is_available(self) -> bool:
        return True

    def translate_stage(self, *, stage, context, source_lines, input_lines,
                        has_prior_translation, progress_cb=None, cancel_cb=None,
                        video_refs=None, attach_video=False):
        return [
            TranslationLine(
                index=ln.index, start_ms=ln.start_ms, end_ms=ln.end_ms,
                text=ln.text, speaker="A", description="cười",
            )
            for ln in input_lines
        ]


_STAGES = [TranslationStageConfig(TranslationStageKind.LITERAL, "m")]


class TestTagsAndDescGating:
    def _run(self, *, enable_tags: bool, include_desc: bool) -> str:
        uc = TranslateSubtitlesUseCase(_SpeakerDescTranslator())
        ctx = TranslationContext("Vietnamese", enable_tags=enable_tags, include_desc=include_desc)
        resp = uc.execute(TranslateSubtitlesRequest(_make_events(["xin chào"]), _STAGES, ctx))
        return resp.events[0].text

    def test_both_on_includes_both(self) -> None:
        text = self._run(enable_tags=True, include_desc=True)
        assert "[A:]" in text and "(cười)" in text

    def test_tags_off_omits_speaker(self) -> None:
        text = self._run(enable_tags=False, include_desc=True)
        assert "[A:]" not in text and "(cười)" in text

    def test_desc_off_omits_description(self) -> None:
        text = self._run(enable_tags=True, include_desc=False)
        assert "[A:]" in text and "(cười)" not in text

    def test_both_off_plain_text(self) -> None:
        text = self._run(enable_tags=False, include_desc=False)
        assert text == "xin chào"


class TestCancellationErrorType:
    def test_cancel_raises_cancelled_error(self) -> None:
        class _Cancelling:
            def is_available(self):
                return True
            def translate_stage(self, **_kwargs):
                return []

        uc = TranslateSubtitlesUseCase(_Cancelling())
        with pytest.raises(TranslationCancelledError):
            uc.execute(
                TranslateSubtitlesRequest(_make_events(["a"]), _STAGES, TranslationContext("Vietnamese")),
                cancel_cb=lambda: True,
            )

    def test_cancelled_is_subclass_of_translation_error(self) -> None:
        # Tương thích ngược: code cũ bắt SubtitleTranslationError vẫn bắt được huỷ.
        assert issubclass(TranslationCancelledError, SubtitleTranslationError)


class TestInterruptibleSleep:
    def test_returns_true_when_cancelled(self) -> None:
        # cancel_cb luôn True → dừng gần như tức thì, trả True.
        start = time.monotonic()
        cancelled = GeminiSubtitleTranslator._interruptible_sleep(10.0, lambda: True)
        elapsed = time.monotonic() - start
        assert cancelled is True
        assert elapsed < 1.0  # phải thoát rất nhanh, không ngủ hết 10s

    def test_returns_false_when_not_cancelled(self) -> None:
        cancelled = GeminiSubtitleTranslator._interruptible_sleep(0.1, lambda: False)
        assert cancelled is False

    def test_none_cancel_cb_sleeps_normally(self) -> None:
        cancelled = GeminiSubtitleTranslator._interruptible_sleep(0.05, None)
        assert cancelled is False
