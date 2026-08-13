"""Test [v3.23.19] ước lượng token video vào quota để điều tiết đúng (chống 429 TPM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


def _adapter() -> GeminiSubtitleTranslator:
    a = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    a._retry_count = 2
    a._quota_manager = None
    return a


class TestVideoTokenEstimate:
    def test_video_tokens_added_to_estimate(self) -> None:
        adapter = _adapter()
        captured = {}

        def fake_call(model, prompt, config, validator, cancel_cb=None,
                      video_files=None, est_tokens=0):
            captured["est"] = est_tokens
            return {"subtitles": [{"line_no": 1, "text": "x"}]}

        batch = [TranslationLine(index=1, start_ms=0, end_ms=1000, text="hi",
                                 original_text="hi")]
        with patch.object(adapter, "_line_to_payload", lambda l, next_start_ms=None: {"line_no": l.index, "text": l.text}), \
             patch.object(adapter, "_call_gemini", fake_call), \
             patch.object(adapter, "_validate_batch", lambda *a: None):
            adapter._translate_single_batch(
                batch=batch, source_before=[], source_after=[], history_before=[],
                start_idx=0, is_preprocess=False, is_literal=True, config=MagicMock(),
                model_name="m", cancel_cb=None, _ctx_size=20, video_files=["h"],
                dual_payload=False, video_token_estimate=107_000,
            )
        assert captured["est"] > 107_000  # text + video

    def test_no_video_estimate_text_only(self) -> None:
        adapter = _adapter()
        captured = {}

        def fake_call(model, prompt, config, validator, cancel_cb=None,
                      video_files=None, est_tokens=0):
            captured["est"] = est_tokens
            return {"subtitles": [{"line_no": 1, "text": "x"}]}

        batch = [TranslationLine(index=1, start_ms=0, end_ms=1000, text="hi",
                                 original_text="hi")]
        with patch.object(adapter, "_line_to_payload", lambda l, next_start_ms=None: {"line_no": l.index, "text": l.text}), \
             patch.object(adapter, "_call_gemini", fake_call), \
             patch.object(adapter, "_validate_batch", lambda *a: None):
            adapter._translate_single_batch(
                batch=batch, source_before=[], source_after=[], history_before=[],
                start_idx=0, is_preprocess=False, is_literal=True, config=MagicMock(),
                model_name="m", cancel_cb=None, _ctx_size=20, video_files=None,
                dual_payload=False, video_token_estimate=0,
            )
        assert captured["est"] < 10_000  # chỉ text, nhỏ
