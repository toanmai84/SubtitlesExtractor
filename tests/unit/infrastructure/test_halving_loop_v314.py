"""Test chống 'Quả bom chia đôi' (halving vô hạn) — chỉ LITERAL mới halving."""

from __future__ import annotations

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _batch_same_as_origin(n: int) -> list[TranslationLine]:
    # text (input giai đoạn hiện tại) khác original; out của AI sẽ = original (chưa dịch).
    return [
        TranslationLine(index=i + 1, start_ms=i * 1000, end_ms=i * 1000 + 800, text=f"việt {i}", original_text="原文")
        for i in range(n)
    ]


def _make_translator(call_counter: list[int]) -> GeminiSubtitleTranslator:
    t = GeminiSubtitleTranslator(api_key="dummy")

    def fake_call(model_name, prompt, config, validator, cancel_cb=None, video_files=None, est_tokens=0):
        call_counter[0] += 1
        # AI trả 100% giống BẢN GỐC (mô phỏng "không dịch / giữ nguyên").
        return {"subtitles": [{"line_no": i + 1, "text": "原文"} for i in range(3)]}

    t._call_gemini = fake_call  # type: ignore[assignment]
    return t


def _run(translator, batch, is_literal):
    return translator._translate_single_batch(
        batch=batch, source_before=[], source_after=[], history_before=[],
        start_idx=0, is_preprocess=False, is_literal=is_literal,
        config=None, model_name="m", cancel_cb=None,
    )


class TestHalvingLoopGuard:
    def test_style_stage_no_halving(self) -> None:
        # Khâu STYLE/LOCALIZE: AI giữ nguyên 100% là HỢP LỆ → KHÔNG halving (1 lần gọi).
        counter = [0]
        t = _make_translator(counter)
        result = _run(t, _batch_same_as_origin(3), is_literal=False)
        assert counter[0] == 1
        assert len(result) == 3

    def test_literal_stage_triggers_halving(self) -> None:
        # Khâu LITERAL: giống bản gốc = chưa dịch → halving (gọi API > 1 lần).
        counter = [0]
        t = _make_translator(counter)
        _run(t, _batch_same_as_origin(3), is_literal=True)
        assert counter[0] > 1  # đã chia đôi lô
