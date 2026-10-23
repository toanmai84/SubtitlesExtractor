"""[v3.23.448] Test dịch sinh tag cảm xúc từ ngữ cảnh video (gated toggle)."""
from __future__ import annotations
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext, TranslationStageConfig, TranslationStageKind,
)


def _prompt(emotion_cues: bool, kind=TranslationStageKind.LOCALIZE) -> str:
    a = GeminiSubtitleTranslator(api_key="k", emotion_cues=emotion_cues)
    ctx = TranslationContext(source_lang="zh", target_lang="vi")
    stage = TranslationStageConfig(kind=kind, model_name="x")
    return a._system_instruction(stage, ctx, has_prior=False)


def test_rule_present_when_enabled():
    p = _prompt(True)
    assert "[cười] / [thở dài] / [hắng giọng]" in p
    assert "HÌNH ẢNH hoặc ÂM THANH" in p


def test_rule_absent_when_disabled():
    assert "[cười] / [thở dài]" not in _prompt(False)


def test_rule_in_all_translate_stages():
    for kind in (TranslationStageKind.LITERAL, TranslationStageKind.STYLE,
                 TranslationStageKind.LOCALIZE):
        assert "[cười] / [thở dài]" in _prompt(True, kind), f"thiếu ở {kind}"


def test_default_off():
    a = GeminiSubtitleTranslator(api_key="k")
    assert a._emotion_cues is False
