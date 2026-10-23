# ruff: noqa: RUF001 — chữ Hán trong ví dụ prompt
"""[v3.23.634] Prompt PHÂN TÍCH: đại từ theo thể loại, đúng một phương án, không bịa tên/trộn chữ Hán.

Số đo thật (Gemini, tập 1-3 phim mẫu tiên hiệp, chấm tự động theo đáp án): 12 lượt chỉ chữ
(3.1-flash-lite) lỗi 32 → 21; có video + 3.8-flash (tập 1-2) lỗi 14 → 8.
"""
from __future__ import annotations

from typing import Any

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    ANALYSIS_ADDRESS_REGISTER_RULE,
    GeminiSubtitleTranslator,
)


def _analysis_prompt(monkeypatch: pytest.MonkeyPatch) -> str:
    translator = GeminiSubtitleTranslator(api_key="k")
    seen: list[str] = []

    def fake_call(model: str, prompt: Any, config: Any, parser: Any, *a: Any, **k: Any) -> Any:
        seen.append(str(getattr(config, "system_instruction", "")))
        return parser({"source_lang": "zh", "characters": [], "overview": "x"})

    monkeypatch.setattr(translator, "_call_gemini", fake_call)
    try:
        translator.analyze_global_context(
            [TranslationLine(index=1, text="师父就闭死关", start_ms=0, end_ms=1000)], "Vietnamese")
    except Exception:  # noqa: BLE001 — chỉ cần prompt đã gửi
        pass
    assert seen, "không gọi Gemini"
    return seen[0]


def test_register_rule_is_in_analysis_prompt_before_the_name_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _analysis_prompt(monkeypatch)
    assert ANALYSIS_ADDRESS_REGISTER_RULE in prompt
    assert prompt.index("ĐẠI TỪ THEO THỂ LOẠI") < prompt.index("Sau chữ 'với' PHẢI viết đúng tên")


def test_rule_content() -> None:
    rule = ANALYSIS_ADDRESS_REGISTER_RULE
    assert "cổ phong" in rule and "TUYỆT ĐỐI không dùng 'anh', 'em'" in rule
    assert "phim hiện đại thì ngược lại" in rule          # không áp cổ phong cho phim hiện đại
    assert "ĐÚNG MỘT phương án" in rule and "KHÔNG trộn chữ Hán" in rule
    assert "TUYỆT ĐỐI KHÔNG tự đặt tên riêng" in rule


def test_rule_examples_are_generic_not_from_the_measured_film() -> None:
    for leaked in ("Dao Dao", "Sở Dao", "叶然", "Diệp Nhiên", "Nam Cung", "南宫"):
        assert leaked not in ANALYSIS_ADDRESS_REGISTER_RULE
