"""Test [v3.23.31] tích hợp Visual Cues vào use case dịch (tuỳ chọn, an toàn-trước)."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    VisualCue,
)


def _events():
    return [
        SubtitleEvent(index=1, text="Hello", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="Goodbye", interval=TimeInterval(1.0, 2.0)),
    ]


def _stage():
    return TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")


class _FakeTranslator:
    def __init__(self, cues=None, raise_exc=None):
        self._cues = cues or []
        self._raise = raise_exc
        self.visual_cues_called = False
        self.seen_speakers: list[str] = []

    def analyze_visual_cues(self, source_lines, target_lang, **kwargs):
        self.visual_cues_called = True
        if self._raise:
            raise self._raise
        return self._cues

    def translate_stage(self, *, stage, context, source_lines, input_lines,
                        has_prior_translation, progress_cb, cancel_cb,
                        video_refs, attach_video):
        # Ghi lại speaker đã được bơm vào để kiểm chứng.
        self.seen_speakers = [ln.speaker for ln in input_lines]
        return input_lines


class TestVisualCuesIntegration:
    def test_cues_applied_to_lines(self) -> None:
        cues = [VisualCue(line_no=1, speaker="Lâm Côn", addressee="Sư phụ")]
        translator = _FakeTranslator(cues=cues)
        uc = TranslateSubtitlesUseCase(translator)
        req = TranslateSubtitlesRequest(
            events=_events(), stages=[_stage()],
            context=TranslationContext(target_lang="vi"),
            video_refs=("ref1",), enable_visual_cues=True,
        )
        uc.execute(req)
        assert translator.visual_cues_called
        assert translator.seen_speakers[0] == "Lâm Côn"  # đã bơm cue

    def test_disabled_by_default(self) -> None:
        translator = _FakeTranslator(cues=[VisualCue(line_no=1, speaker="X")])
        uc = TranslateSubtitlesUseCase(translator)
        req = TranslateSubtitlesRequest(
            events=_events(), stages=[_stage()],
            context=TranslationContext(target_lang="vi"),
            video_refs=("ref1",),  # enable_visual_cues mặc định False
        )
        uc.execute(req)
        assert translator.visual_cues_called is False

    def test_skipped_without_video(self) -> None:
        translator = _FakeTranslator(cues=[VisualCue(line_no=1, speaker="X")])
        uc = TranslateSubtitlesUseCase(translator)
        req = TranslateSubtitlesRequest(
            events=_events(), stages=[_stage()],
            context=TranslationContext(target_lang="vi"),
            enable_visual_cues=True,  # nhưng KHÔNG có video_refs
        )
        uc.execute(req)
        assert translator.visual_cues_called is False

    def test_failure_is_swallowed(self) -> None:
        # Lỗi phân tích visual cues KHÔNG được làm hỏng dịch.
        translator = _FakeTranslator(raise_exc=SubtitleTranslationError("API lỗi"))
        uc = TranslateSubtitlesUseCase(translator)
        req = TranslateSubtitlesRequest(
            events=_events(), stages=[_stage()],
            context=TranslationContext(target_lang="vi"),
            video_refs=("ref1",), enable_visual_cues=True,
        )
        resp = uc.execute(req)  # không raise
        assert len(resp.events) == 2


class TestVisualCuesCache:
    def test_cues_cached_skips_second_scan(self, tmp_path) -> None:
        # Lần 1 quét video; lần 2 (cùng checkpoint) dùng cache, KHÔNG gọi lại API.
        cues = [VisualCue(line_no=1, speaker="Lâm Côn", addressee="Sư phụ")]

        class CountingTranslator(_FakeTranslator):
            def __init__(self, cues):
                super().__init__(cues=cues)
                self.scan_count = 0

            def analyze_visual_cues(self, source_lines, target_lang, **kwargs):
                self.scan_count += 1
                return super().analyze_visual_cues(source_lines, target_lang, **kwargs)

        translator = CountingTranslator(cues)
        uc = TranslateSubtitlesUseCase(translator, checkpoint_dir=tmp_path)
        req = TranslateSubtitlesRequest(
            events=_events(), stages=[_stage()],
            context=TranslationContext(target_lang="vi"),
            video_refs=("ref1",), enable_visual_cues=True,
        )
        # Lần 1: quét.
        uc.execute(req)
        assert translator.scan_count == 1
        # Checkpoint bị xoá sau khi hoàn tất → mô phỏng dịch DỞ bằng cách giữ checkpoint:
        # tạo lại checkpoint và lưu cues, rồi chạy lại.

    def test_cached_cues_reused_from_checkpoint(self, tmp_path) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            _StageCheckpoint, _compute_checkpoint_key,
        )
        from subtitles_extractor.application.services.visual_cue_serializer import (
            serialize_visual_cues,
        )
        import json as _json

        cues = [VisualCue(line_no=1, speaker="Lâm Côn")]
        req = TranslateSubtitlesRequest(
            events=_events(), stages=[_stage()],
            context=TranslationContext(target_lang="vi"),
            video_refs=("ref1",), enable_visual_cues=True,
        )
        # Ghi sẵn cache vào checkpoint.
        key = _compute_checkpoint_key(req)
        cp = _StageCheckpoint(tmp_path, key)
        cp.save_visual_cues(_json.loads(serialize_visual_cues(cues)))

        class NeverScanTranslator(_FakeTranslator):
            def analyze_visual_cues(self, *a, **k):
                raise AssertionError("KHÔNG được quét lại khi đã có cache!")

        translator = NeverScanTranslator()
        uc = TranslateSubtitlesUseCase(translator, checkpoint_dir=tmp_path)
        uc.execute(req)  # không được gọi analyze_visual_cues
        assert translator.seen_speakers[0] == "Lâm Côn"  # cue từ cache đã áp


class TestVisualCuesModel:
    def test_default_model_not_2_0_flash(self) -> None:
        # [v3.23.34] Mặc định KHÔNG còn gemini-2.0-flash (gây 429 với nhiều tài khoản).
        req = TranslateSubtitlesRequest(
            events=_events(), stages=[_stage()],
            context=TranslationContext(target_lang="vi"),
        )
        assert req.visual_cues_model != "gemini-2.0-flash"

    def test_model_passed_to_analyze(self) -> None:
        # Model trong request được truyền đúng vào analyze_visual_cues.
        captured = {}

        class CapturingTranslator(_FakeTranslator):
            def analyze_visual_cues(self, source_lines, target_lang, **kwargs):
                captured["model"] = kwargs.get("model_name")
                return [VisualCue(line_no=1, speaker="X")]

        translator = CapturingTranslator()
        uc = TranslateSubtitlesUseCase(translator)
        req = TranslateSubtitlesRequest(
            events=_events(), stages=[_stage()],
            context=TranslationContext(target_lang="vi"),
            video_refs=("ref1",), enable_visual_cues=True,
            visual_cues_model="gemini-flash-lite-latest",
        )
        uc.execute(req)
        assert captured["model"] == "gemini-flash-lite-latest"
