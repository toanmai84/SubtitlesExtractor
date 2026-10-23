"""[v3.23.623] Gán ai nói / nói với ai CHỈ BẰNG CHỮ (không video) và nối vào đường ống dịch.

Đo thật (tập mẫu 1-2): gemini-3.5-flash-lite đọc cả tập một lượt đúng ~96% người nói trong
6-12 giây — gần bằng phân tích có video (96-100%).
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    VisualCue,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

ROSTER = (
    "叶然 (Diệp Nhiên) [nam, ~20] — chính; xưng hô: với Dao Dao tự xưng 'ta' gọi 'muội'\n"
    "瑶瑶 (Dao Dao) [nữ, ~20] — sư muội; xưng hô: với Diệp Nhiên tự xưng 'muội' gọi 'sư huynh'"
)
LINES = [
    TranslationLine(index=1, start_ms=0, end_ms=900, text="师兄我就先走了"),
    TranslationLine(index=2, start_ms=1000, end_ms=1900, text="你走吧"),
]


class _Translator:
    def __init__(self, fail_models: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_models = fail_models or set()

    def analyze_text_cues(self, lines, target_lang, model_name="", characters="", **_k):
        self.calls.append(model_name)
        if model_name in self.fail_models:
            raise SubtitleTranslationError("hết hạn mức")
        return [VisualCue(1, "瑶瑶", "叶然"), VisualCue(2, "叶然", "瑶瑶")]


def _use_case(translator: Any) -> TranslateSubtitlesUseCase:
    uc = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    uc._translator = translator
    return uc


def _request(**kw: Any) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[], stages=[],
        context=TranslationContext(target_lang="Vietnamese", characters=ROSTER),
        text_cues=True, **kw)


def test_ensure_text_cues_fills_context_and_uses_lite_model() -> None:
    translator = _Translator()
    out = _use_case(translator)._ensure_text_cues(LINES, _request(), None, None, None)
    cues = json.loads(out.context.visual_cues)
    assert [(c["id"], c["spk"], c["to"]) for c in cues] == [(1, "瑶瑶", "叶然"), (2, "叶然", "瑶瑶")]
    assert translator.calls == ["gemini-3.5-flash-lite"]


def test_ensure_text_cues_falls_back_to_stage_model_then_gives_up() -> None:
    translator = _Translator(fail_models={"gemini-3.5-flash-lite"})
    out = _use_case(translator)._ensure_text_cues(
        LINES, _request(visual_cues_model="gemini-3.1-flash-lite"), None, None, None)
    assert out.context.visual_cues and translator.calls == [
        "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    broken = _Translator(fail_models={"gemini-3.5-flash-lite", "m2"})
    req = _request(visual_cues_model="m2")
    assert _use_case(broken)._ensure_text_cues(LINES, req, None, None, None) is req


def test_existing_cues_are_never_overwritten() -> None:
    translator = _Translator()
    req = _request()
    req = replace(req, context=replace(req.context, visual_cues='[{"id":1,"spk":"X","to":"Y"}]'))
    assert _use_case(translator)._ensure_text_cues(LINES, req, None, None, None) is req
    assert translator.calls == []


def test_cues_usable_needs_flag_and_content() -> None:
    uc = TranslateSubtitlesUseCase
    assert not uc._cues_usable(_request())  # cờ bật nhưng chưa có cues
    req = _request()
    with_cues = replace(req, context=replace(req.context, visual_cues="[{}]"))
    assert uc._cues_usable(with_cues)
    assert not uc._cues_usable(replace(with_cues, text_cues=False))


def test_analysis_use_case_labels_from_text_only_without_video() -> None:
    class _Analyzer(_Translator):
        def analyze_global_context(self, **_k):
            return SubtitleContextAnalysis(source_lang="zh", characters=ROSTER, overview="o")

    translator = _Analyzer()
    uc = AnalyzeSubtitleContextUseCase(translator)
    result = uc.execute(LINES, "Vietnamese", model_name="gemini-3.6-flash", text_cues=True)
    assert json.loads(result.visual_cues)[0]["spk"] == "瑶瑶"
    assert translator.calls == ["gemini-3.5-flash-lite"]
    # Mặc định tắt → không gọi.
    translator.calls.clear()
    assert uc.execute(LINES, "Vietnamese", model_name="m").visual_cues == ""
    assert translator.calls == []


def test_payload_carries_pron_on_every_line_but_rule_once() -> None:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    adapter._active_characters = ROSTER
    lines = [
        TranslationLine(index=1, start_ms=0, end_ms=1, text="师兄我就先走了",
                        speaker="瑶瑶", addressee="叶然"),
        TranslationLine(index=2, start_ms=2, end_ms=3, text="我不想走",
                        speaker="瑶瑶", addressee="叶然"),
        TranslationLine(index=3, start_ms=4, end_ms=5, text="你走吧",
                        speaker="叶然", addressee="瑶瑶"),
    ]
    payload = [{"line_no": ln.index, "text": ln.text} for ln in lines]
    adapter._attach_address_rules_once(payload, lines)
    assert [p.get("pron") for p in payload] == ["我=muội", "我=muội", "你=muội"]
    assert "address" in payload[0] and "address" not in payload[1]
    assert "address" in payload[2]
