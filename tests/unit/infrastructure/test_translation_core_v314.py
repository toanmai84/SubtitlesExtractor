"""Unit test cho nâng cấp lõi dịch thuật (v3.14.1, Nhóm 2).

Phủ: Dual Payload (truyền kép gốc + nháp), bảo toàn original_text/addressee qua
merge, và Strict Pronoun Guard trong system prompt.
"""

from __future__ import annotations

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _line(**kwargs) -> TranslationLine:
    base = dict(index=1, start_ms=0, end_ms=1000, text="bản nháp")
    base.update(kwargs)
    return TranslationLine(**base)


class TestDualPayload:
    def test_dual_payload_includes_original(self) -> None:
        line = _line(text="Ta đi đây", original_text="我走了", addressee="Nam đệ tử")
        payload = GeminiSubtitleTranslator._line_to_dual_payload(line)
        assert payload["original"] == "我走了"
        assert payload["text"] == "Ta đi đây"
        assert payload["addressee"] == "Nam đệ tử"
        assert payload["line_no"] == 1

    def test_dual_payload_omits_original_when_same(self) -> None:
        line = _line(text="同", original_text="同")
        payload = GeminiSubtitleTranslator._line_to_dual_payload(line)
        assert "original" not in payload

    def test_plain_payload_has_no_original(self) -> None:
        line = _line(text="x", original_text="y")
        assert "original" not in GeminiSubtitleTranslator._line_to_payload(line)


class TestFieldPreservation:
    def test_merge_preserves_original_and_addressee(self) -> None:
        batch = [_line(text="我走了", original_text="我走了", addressee="Nữ tỳ")]
        ai_items = [{"text": "Ta đi đây", "speaker": "A"}]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, ai_items, is_preprocess=False)
        assert merged[0].text == "Ta đi đây"
        assert merged[0].original_text == "我走了"
        assert merged[0].addressee == "Nữ tỳ"

    def test_merge_preprocess_preserves_fields(self) -> None:
        batch = [_line(text="我赱了", original_text="我赱了", addressee="X")]
        merged = GeminiSubtitleTranslator._merge_ai_items(
            batch, [{"text": "我走了"}], is_preprocess=True
        )
        assert merged[0].addressee == "X"


class TestPronounGuard:
    def _instance(self) -> GeminiSubtitleTranslator:
        return GeminiSubtitleTranslator(api_key="dummy")

    def _ctx(self) -> TranslationContext:
        return TranslationContext(target_lang="Vietnamese", source_lang="zh")

    def test_literal_prompt_has_strict_pronoun_guard(self) -> None:
        stage = TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        prompt = self._instance()._system_instruction(stage, self._ctx(), has_prior=False)
        assert "XƯNG HÔ" in prompt
        assert "TRUNG TÍNH" in prompt
        assert "BIẾT CHẮC" in prompt  # chỉ dùng đại từ giới tính khi biết chắc

    def test_localize_prompt_has_pronoun_guard(self) -> None:
        stage = TranslationStageConfig(kind=TranslationStageKind.LOCALIZE, model_name="m")
        prompt = self._instance()._system_instruction(stage, self._ctx(), has_prior=True)
        assert "XƯNG HÔ" in prompt
        assert "TRUNG TÍNH" in prompt

    def test_prompt_uses_section_headings(self) -> None:
        # [v3.23.26] Best practice Gemini: cấu trúc bằng heading nhất quán.
        stage = TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        prompt = self._instance()._system_instruction(stage, self._ctx(), has_prior=False)
        for section in ("## VAI TRÒ", "## BỐI CẢNH", "## NHIỆM VỤ", "## QUY TẮC", "## ĐỊNH DẠNG"):
            assert section in prompt


class TestThinkingFallback:
    """[Yêu cầu 4] Ưu tiên bật thinking; chỉ fallback khi model thật sự không hỗ trợ."""

    def test_detect_thinking_unsupported(self) -> None:
        f = GeminiSubtitleTranslator._is_thinking_unsupported_error
        assert f(Exception("ThinkingConfig is not supported by this model"))
        assert f(Exception("Unknown field thinking_config"))

    def test_non_thinking_errors_ignored(self) -> None:
        f = GeminiSubtitleTranslator._is_thinking_unsupported_error
        assert not f(Exception("500 internal error"))
        assert not f(Exception("quota exceeded"))  # không nhắc 'thinking'

    def test_call_gemini_strips_thinking_and_retries(self, monkeypatch) -> None:
        from types import SimpleNamespace

        translator = GeminiSubtitleTranslator(api_key="dummy")
        # Giả lập SDK types để _strip_thinking_from_config dựng lại được config.
        translator._types_module = SimpleNamespace(
            GenerateContentConfig=lambda **kw: SimpleNamespace(**kw)
        )
        config = SimpleNamespace(
            temperature=0.2, response_mime_type="application/json",
            response_schema={}, system_instruction="x", thinking_config="THINK",
        )
        calls: list[bool] = []  # mỗi phần tử: config lần gọi có thinking_config?

        class _Resp:
            text = '{"subtitles": []}'
            usage_metadata = None

        def fake_generate(model, contents, config):  # noqa: ANN001
            calls.append(hasattr(config, "thinking_config"))
            if len(calls) == 1:
                raise RuntimeError("ThinkingConfig is not supported by this model")
            return _Resp()

        fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate))
        monkeypatch.setattr(translator, "_get_client", lambda: fake_client)

        data = translator._call_gemini("flash-lite", "prompt", config, lambda d: None)
        assert data == {"subtitles": []}
        assert calls == [True, False]  # lần 1 có thinking, lần 2 đã strip


class TestHallucinationDrop:
    """[#14] AI trả rỗng → giữ nguyên văn bản gốc, không mất câu."""

    def test_empty_ai_text_keeps_original(self) -> None:
        batch = [_line(text="我走了", original_text="我走了")]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, [{"text": ""}], is_preprocess=False)
        assert merged[0].text == "我走了"

    def test_whitespace_ai_text_keeps_original(self) -> None:
        batch = [_line(text="原文")]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, [{"text": "   "}], is_preprocess=False)
        assert merged[0].text == "原文"

    def test_none_ai_text_keeps_original(self) -> None:
        batch = [_line(text="原文")]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, [{"text": None}], is_preprocess=False)
        assert merged[0].text == "原文"

    def test_valid_ai_text_used(self) -> None:
        batch = [_line(text="我走了")]
        merged = GeminiSubtitleTranslator._merge_ai_items(batch, [{"text": "Ta đi đây"}], is_preprocess=False)
        assert merged[0].text == "Ta đi đây"
