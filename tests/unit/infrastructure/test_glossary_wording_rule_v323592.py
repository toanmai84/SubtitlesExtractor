"""[v3.23.592] Prompt phân tích: phần TRƯỚC ngoặc của mục thuật ngữ phải dùng được.

Bản phân tích thật v3.23.591 ghi ``所長 = sở trưởng (trưởng đồn công an)``: đọc âm từng
chữ, còn cách nói người Việt hiểu thì nằm trong ngoặc. Prompt cũ chỉ dặn "phần trước ngoặc
phải dùng được thẳng trong lời thoại" cho tiếng lóng và thành ngữ. Khâu dịch chép phần
trước ngoặc: "Sở trưởng" ở cả 15 dòng của lượt dịch đó.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GLOSSARY_WORDING_RULE,
)
from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

SLANG_RULE_TAIL = "trước ngoặc phải là cụm dùng được thẳng trong lời thoại."


class TestRuleText:
    def test_rule_covers_every_entry_not_only_slang(self) -> None:
        rule = GLOSSARY_WORDING_RULE.format(target_lang="Vietnamese")
        assert "đúng với MỌI mục, kể cả chức danh và thành ngữ" in rule
        assert "đưa nó ra TRƯỚC ngoặc" in rule

    def test_rule_shows_the_real_mistake_and_its_fix(self) -> None:
        rule = GLOSSARY_WORDING_RULE.format(target_lang="Vietnamese")
        assert "SAI: '所長 => sở trưởng (trưởng đồn công an)'" in rule
        assert "ĐÚNG: '所長 => trưởng đồn'" in rule

    def test_rule_still_allows_reading_names_and_everyday_words(self) -> None:
        rule = GLOSSARY_WORDING_RULE.format(target_lang="Vietnamese")
        assert "Đọc âm từng chữ chỉ dành cho tên riêng" in rule
        assert "người nói Vietnamese thật sự dùng" in rule

    def test_rule_is_one_more_bullet_of_the_criteria_list(self) -> None:
        assert GLOSSARY_WORDING_RULE.startswith("\n  - ")
        assert "{" not in GLOSSARY_WORDING_RULE.format(target_lang="x")


def _analysis_instructions(target_lang: str) -> list[str]:
    """Chạy ``analyze_global_context`` thật, trả các chỉ thị hệ thống đã dựng."""
    adapter = bare_analysis_adapter()
    built: list[str] = []

    def _build_config(
        _temperature: float, _schema: Any, system_instruction: str, **_kwargs: Any
    ) -> MagicMock:
        built.append(system_instruction)
        return MagicMock()

    def _gemini(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "source_lang": "zh", "characters": [], "overview": "Tóm tắt.", "glossary": [],
        }

    lines = [TranslationLine(index=1, start_ms=0, end_ms=1, text="所長")]
    with (
        patch.object(adapter, "_build_config", _build_config),
        patch.object(adapter, "_call_gemini", _gemini),
        patch.object(adapter, "_ensure_available", lambda: None),
    ):
        adapter.analyze_global_context(lines, target_lang)
    assert built, "analyze_global_context không dựng cấu hình nào"
    return built


class TestAnalysisPromptWiring:
    @pytest.mark.parametrize("target_lang", ["Vietnamese", "English"])
    def test_rule_is_sent_with_the_analysis_request(self, target_lang: str) -> None:
        rule = GLOSSARY_WORDING_RULE.format(target_lang=target_lang)
        for instruction in _analysis_instructions(target_lang):
            assert rule in instruction

    def test_rule_follows_the_slang_rule_it_extends(self) -> None:
        for instruction in _analysis_instructions("Vietnamese"):
            slang_at = instruction.index(SLANG_RULE_TAIL)
            rule_at = instruction.index("Phần TRƯỚC ngoặc là thứ khâu dịch chép")
            assert slang_at < rule_at
