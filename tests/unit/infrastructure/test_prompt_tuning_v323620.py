"""[v3.23.620] Tinh chỉnh prompt Phân tích + Dịch thô, đo bằng Gemini thật trên phim mẫu.

Lỗi đo được (tập 1+2, gemini-3.1-flash-lite, cấu hình mặc định):
* 11-19 dòng bị chèn "(…)" bịa ở đầu câu (cảm xúc, "Nội tâm Diệp Nhiên") vì schema mô tả
  trường description là "chú thích ngắn cho dòng" — phụ đề gốc không có tiếng động nào.
* Tắt tag người nói thì prompt bảo "để trống speaker" → model bỏ luôn bước xác định ai nói
  với ai → đảo xưng hô (你 → "Anh", 师兄 → "Sư muội").
* aka trong Roster ghi cụm mô tả ("Sư phụ của Diệp Nhiên") bị chép nguyên vào lời thoại.
* Prompt chứa chuỗi '\\n' dạng chữ và các mốc phiên bản nội bộ "[v3.23.xxx]".
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _TRANSLATE_SCHEMA,
    DESCRIPTION_FIELD_NOTE,
    GeminiSubtitleTranslator,
    _batch_constrained_schema,
)

STAGES = (TranslationStageKind.LITERAL, TranslationStageKind.STYLE, TranslationStageKind.LOCALIZE)


def _prompt(kind: TranslationStageKind, *, enable_tags: bool = False,
            include_desc: bool = True) -> str:
    translator = GeminiSubtitleTranslator(api_key="k")
    ctx = TranslationContext(source_lang="zh", target_lang="vi",
                             enable_tags=enable_tags, include_desc=include_desc)
    return translator._system_instruction(
        TranslationStageConfig(kind=kind, model_name="x"), ctx, has_prior=False)


def test_description_field_note_forbids_emotion() -> None:
    schema = _batch_constrained_schema(
        _TRANSLATE_SCHEMA, [SimpleNamespace(index=1, text="x")])
    note = schema["properties"]["subtitles"]["items"]["properties"]["description"]["description"]
    assert note == DESCRIPTION_FIELD_NOTE
    assert "KHÔNG ghi cảm xúc" in note and "Mặc định để trống" in note


def test_literal_asks_for_speaker_even_without_tags() -> None:
    prompt = _prompt(TranslationStageKind.LITERAL, enable_tags=False)
    assert "để trống trường speaker" not in prompt
    assert "điền trường speaker" in prompt and "KHÔNG hiển thị" in prompt
    assert "我/我们 = NGƯỜI NÓI" in prompt


def test_literal_description_only_real_sound() -> None:
    prompt = _prompt(TranslationStageKind.LITERAL, include_desc=True)
    assert "TUYỆT ĐỐI không ghi cảm xúc" in prompt
    assert "để trống description" in _prompt(TranslationStageKind.LITERAL, include_desc=False)


@pytest.mark.parametrize("kind", STAGES)
def test_prompt_has_no_literal_backslash_n_nor_version_tags(kind: TranslationStageKind) -> None:
    prompt = _prompt(kind)
    # '\\n' dạng chữ chỉ được phép trong câu cấm xuất chuỗi đó (PREPROCESS), không ở đây.
    assert "Hán).\\n" not in prompt
    assert "[v3.23." not in prompt


def test_analysis_prompt_requires_dialogue_form_aka(monkeypatch: pytest.MonkeyPatch) -> None:
    translator = GeminiSubtitleTranslator(api_key="k")
    seen: list[str] = []

    def fake_call(model: str, prompt: Any, config: Any, parser: Any, *a: Any, **k: Any) -> Any:
        seen.append(str(getattr(config, "system_instruction", "")))
        return parser({"source_lang": "zh", "characters": [], "overview": "x"})

    monkeypatch.setattr(translator, "_call_gemini", fake_call)
    lines = [TranslationLine(index=1, text="师父就闭死关", start_ms=0, end_ms=1000)]
    try:
        translator.analyze_global_context(lines, "Vietnamese")
    except Exception:  # noqa: BLE001 — chỉ cần prompt đã gửi
        pass
    assert seen, "không gọi Gemini"
    assert "'Sư phụ của X'" in seen[0] and "Sở Dao / Dao Dao" in seen[0]


@pytest.mark.parametrize("kind", STAGES)
def test_role_procedure_in_every_translate_stage(kind: TranslationStageKind) -> None:
    assert "XÁC ĐỊNH VAI TRƯỚC KHI DỊCH" in _prompt(kind)


@pytest.mark.parametrize("kind", STAGES)
def test_register_rule_v323621(kind: TranslationStageKind) -> None:
    """[v3.23.621] Phim cổ trang không trôi sang 'anh/em', 'mày/tao'."""
    prompt = _prompt(kind)
    assert "VĂN PHONG THEO BỐI CẢNH" in prompt and "'mày/tao'" in prompt
