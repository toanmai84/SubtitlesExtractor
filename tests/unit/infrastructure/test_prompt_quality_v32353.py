"""Test [v3.23.53] củng cố prompt: glossary cứng, chống meta-text (best practice LLM)."""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)


def _instr(glossary="", kind=TranslationStageKind.LITERAL):
    tr = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    stage = TranslationStageConfig(kind=kind, model_name="gemini-3.1-flash-lite")
    ctx = TranslationContext(
        target_lang="Vietnamese", glossary=glossary,
        characters="Lâm Hằng", overview="Tiên hiệp",
    )
    return tr._system_instruction(stage, ctx, has_prior=False)


class TestGlossaryEnforcement:
    def test_strong_glossary_directive(self) -> None:
        instr = _instr(glossary="灵力 => linh lực")
        assert "BẮT BUỘC TUYỆT ĐỐI" in instr
        assert "KHÔNG tự thay bằng từ đồng nghĩa" in instr

    def test_no_glossary_no_directive(self) -> None:
        instr = _instr(glossary="")
        assert "BẢNG THUẬT NGỮ" not in instr


class TestAntiMetaText:
    def test_literal_has_anti_meta_rule(self) -> None:
        instr = _instr(kind=TranslationStageKind.LITERAL)
        assert "TUYỆT ĐỐI KHÔNG chèn chú thích" in instr
        assert "không dịch được" in instr.lower()


class TestStructure:
    def test_sections_present(self) -> None:
        instr = _instr(glossary="x => y")
        for section in ("## VAI TRÒ", "## BỐI CẢNH", "## NHIỆM VỤ", "## QUY TẮC"):
            assert section in instr
