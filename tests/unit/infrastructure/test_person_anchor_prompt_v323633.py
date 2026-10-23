# ruff: noqa: RUF001 — chữ Hán/dấu câu fullwidth thuộc câu gốc chép từ phim
"""[v3.23.633] Prompt giữ NGÔI theo chữ gốc + prompt gán người nói kiểm nội dung với nhãn giọng.

Số đo thật (Gemini, tập 1-3 phim mẫu): lỗi đảo ngôi/gọi sai "huynh" trên dòng bẫy
1,67/lượt (12 lượt) → 0/lượt (6 lượt); độ đúng gán người nói tập 1: 86,5% → 93,7%.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    PERSON_ANCHOR_RULE_REFINE,
    GeminiSubtitleTranslator,
    _text_cues_system_instruction,
)


def _prompt(kind: TranslationStageKind) -> str:
    translator = GeminiSubtitleTranslator(api_key="k")
    ctx = TranslationContext(source_lang="zh", target_lang="vi")
    return translator._system_instruction(
        TranslationStageConfig(kind=kind, model_name="x"), ctx, has_prior=True)


def test_literal_prompt_has_no_person_anchor_rule() -> None:
    """8 bảng chấm mù: quy tắc ngôi ở giai đoạn dịch thô làm điểm tệ hơn (76 so với 47) vì ép
    "ngươi" cả khi 你 hướng tới bề trên → KHÔNG áp cho giai đoạn dịch thô."""
    prompt = _prompt(TranslationStageKind.LITERAL)
    assert PERSON_ANCHOR_RULE_REFINE not in prompt
    assert "GIỮ NGUYÊN VAI & NGÔI" not in prompt
    assert "NGÔI THEO CHỮ GỐC LÀ CHUẨN" not in prompt


@pytest.mark.parametrize("kind", [TranslationStageKind.STYLE, TranslationStageKind.LOCALIZE])
def test_refine_prompts_keep_person_and_distrust_labels(kind: TranslationStageKind) -> None:
    prompt = _prompt(kind)
    assert PERSON_ANCHOR_RULE_REFINE in prompt
    assert "GIỮ NGUYÊN VAI & NGÔI" in prompt
    # quy tắc nằm TRONG mục quy tắc, không bị đẩy ra ngoài
    assert prompt.index("## QUY TẮC") < prompt.index(PERSON_ANCHOR_RULE_REFINE)


def test_person_rules_do_not_leak_test_data() -> None:
    """Ví dụ trong prompt là ví dụ chung, không phải câu của phim dùng để đo."""
    for rule in (PERSON_ANCHOR_RULE_REFINE,):
        assert "天纵之资" not in rule and "叶然" not in rule


class TestTextCuesVoiceRules:
    def test_voice_labels_are_strong_but_checked_against_content(self) -> None:
        prompt = _text_cues_system_instruction("Vietnamese", "", True)
        assert "GỘP NHẦM" in prompt
        assert "kiểm nội dung" in prompt
        assert "không tự xưng 你 với chính mình" in prompt

    def test_no_voice_block_without_voice_labels(self) -> None:
        prompt = _text_cues_system_instruction("Vietnamese", "", False)
        assert "<giong_noi>" not in prompt
