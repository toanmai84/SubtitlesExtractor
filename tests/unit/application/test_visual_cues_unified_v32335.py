"""Test [v3.23.35→37] Visual Cues tích hợp CHUNG với phân tích ngữ cảnh.

[v3.23.37] Cues lấy LUÔN trong analyze_global_context (with_visual_cues=True) — cùng
request, cùng model — thay vì gọi analyze_visual_cues riêng → tiết kiệm quota.
"""

from __future__ import annotations

from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    VisualCue,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _lines():
    return [
        TranslationLine(index=1, start_ms=0, end_ms=1000, text="Hello"),
        TranslationLine(index=2, start_ms=1000, end_ms=2000, text="Goodbye"),
    ]


def _events():
    return [
        SubtitleEvent(index=1, text="Hello", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="Goodbye", interval=TimeInterval(1.0, 2.0)),
    ]


class _AnalyzeTranslator:
    """Translator giả: analyze_global_context nhận with_visual_cues, trả cues kèm."""

    def __init__(self, cues_json="", raise_exc=None):
        self._cues_json = cues_json
        self._raise = raise_exc
        self.model = None
        self.with_cues = None

    def analyze_global_context(
        self, *, source_lines, target_lang, model_name,
        cancel_cb=None, video_refs=None, with_visual_cues=False,
        prior_context="",
    ):
        self.model = model_name
        self.with_cues = with_visual_cues
        if self._raise:
            raise self._raise
        return SubtitleContextAnalysis(
            source_lang="en", characters="A", overview="o", glossary="",
            visual_cues=self._cues_json if with_visual_cues else "",
        )


class TestUnifiedAnalysis:
    def test_cues_inline_same_model(self) -> None:
        translator = _AnalyzeTranslator(cues_json='[{"id":1,"spk":"Lâm Côn"}]')
        uc = AnalyzeSubtitleContextUseCase(translator)
        result = uc.execute(
            _lines(), "vi", model_name="gemini-3.5-flash",
            video_refs=["ref"], enable_visual_cues=True,
        )
        # Cues lấy CHUNG trong analyze_global_context, cùng model.
        assert translator.with_cues is True
        assert translator.model == "gemini-3.5-flash"
        assert result.visual_cues == '[{"id":1,"spk":"Lâm Côn"}]'

    def test_no_cues_without_flag(self) -> None:
        translator = _AnalyzeTranslator(cues_json='[{"id":1}]')
        uc = AnalyzeSubtitleContextUseCase(translator)
        result = uc.execute(_lines(), "vi", video_refs=["ref"])  # không bật
        assert translator.with_cues is False
        assert result.visual_cues == ""

    def test_no_cues_without_video(self) -> None:
        translator = _AnalyzeTranslator(cues_json='[{"id":1}]')
        uc = AnalyzeSubtitleContextUseCase(translator)
        # Bật cues nhưng KHÔNG có video → không yêu cầu cues inline.
        result = uc.execute(_lines(), "vi", enable_visual_cues=True)
        assert translator.with_cues is False
        assert result.visual_cues == ""


class _DictTranslator:
    def __init__(self):
        self.seen_speakers = []
        self.cues_scanned = False

    def analyze_visual_cues(self, *a, **k):
        self.cues_scanned = True
        raise AssertionError("KHÔNG được quét lại khi context đã có visual_cues!")

    def translate_stage(self, *, stage, context, source_lines, input_lines,
                       has_prior_translation, progress_cb, cancel_cb,
                       video_refs, attach_video):
        self.seen_speakers = [ln.speaker for ln in input_lines]
        return input_lines


class TestTranslateReusesContextCues:
    def test_context_cues_reused_no_rescan(self) -> None:
        from subtitles_extractor.application.services.visual_cue_serializer import (
            serialize_visual_cues,
        )
        cues_json = serialize_visual_cues([VisualCue(line_no=1, speaker="Lâm Côn")])
        translator = _DictTranslator()
        uc = TranslateSubtitlesUseCase(translator)
        req = TranslateSubtitlesRequest(
            events=_events(),
            stages=[TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")],
            context=TranslationContext(target_lang="vi", visual_cues=cues_json),
            video_refs=("ref1",), enable_visual_cues=True,
        )
        uc.execute(req)
        assert translator.cues_scanned is False
        assert translator.seen_speakers[0] == "Lâm Côn"
