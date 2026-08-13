"""[v3.23.133] Test CHỐNG DỒN LỆCH khi batch có index lỗ (do lọc dòng nhiễu ♪).

Đi qua validate/merge THẬT (chỉ mock _call_gemini ở tầng thấp). Trước fix: index lỗ +
model trả đúng line_no → bị đánh số lại theo vị trí → dồn lệch. Sau fix: re-index liên
tục [1..N] để gửi model rồi ánh xạ ngược → mỗi dòng nhận ĐÚNG bản dịch của nó.
"""

from __future__ import annotations

import re
from typing import Any

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _line(idx: int, text: str) -> TranslationLine:
    return TranslationLine(
        index=idx, start_ms=idx * 1000, end_ms=idx * 1000 + 900, text=text
    )


class _Ctx:
    source_lang = "en"
    enable_tags = False
    include_desc = False

    def __getattr__(self, _name: str) -> Any:
        return None


def _adapter(monkeypatch: Any) -> GeminiSubtitleTranslator:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    monkeypatch.setattr(adapter, "_ensure_available", lambda: None, raising=False)
    monkeypatch.setattr(
        adapter, "_response_schema_for", lambda *a, **k: {}, raising=False
    )
    monkeypatch.setattr(adapter, "_system_instruction", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(adapter, "_build_config", lambda *a, **k: None, raising=False)

    # Model NGOAN: trả đúng các line_no nhận được trong khối <current_batch>.
    def fake_call(model, prompt, config, validator, **kw):  # type: ignore[no-untyped-def]
        block = re.search(r"<current_batch>(.*?)</current_batch>", prompt, re.S)
        body = block.group(1) if block else prompt
        line_nos = [int(m) for m in re.findall(r'"line_no":\s*(\d+)', body)]
        payload = {"subtitles": [{"line_no": n, "text": f"VI{n}"} for n in line_nos]}
        validator(payload)
        return payload

    monkeypatch.setattr(adapter, "_call_gemini", fake_call, raising=False)
    return adapter


def test_alignment_with_music_noise_lines(monkeypatch: Any) -> None:
    adapter = _adapter(monkeypatch)
    # Dòng 3 và 6 là ♪ (nhiễu) → bị lọc, tạo LỖ HỔNG index.
    lines = [
        _line(1, "Hello"), _line(2, "world"), _line(3, "♪ ♪"),
        _line(4, "foo"), _line(5, "bar"), _line(6, "♪"),
        _line(7, "baz"),
    ]
    stage = TranslationStageConfig(
        kind=TranslationStageKind.LITERAL, model_name="m", batch_size=10, context_size=0,
    )
    out = adapter.translate_stage(
        stage=stage, context=_Ctx(), source_lines=lines, input_lines=lines,
        has_prior_translation=False,
    )
    by = {ln.index: ln.text for ln in out}
    # Dòng thật: mỗi dòng nhận bản dịch CỦA CHÍNH NÓ (không dồn lệch).
    # Sau re-index: [1,2,4,5,7] -> [1,2,3,4,5]; model trả VI1..VI5; ánh xạ ngược:
    #   1->VI1, 2->VI2, 4->VI3, 5->VI4, 7->VI5.
    assert by[1] == "VI1"
    assert by[2] == "VI2"
    assert by[4] == "VI3"
    assert by[5] == "VI4"
    assert by[7] == "VI5"
    # Dòng nhiễu giữ NGUYÊN.
    assert by[3] == "♪ ♪"
    assert by[6] == "♪"
    # Đủ dòng, đúng thứ tự.
    assert [ln.index for ln in out] == [1, 2, 3, 4, 5, 6, 7]
