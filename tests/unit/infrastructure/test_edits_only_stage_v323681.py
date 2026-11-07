"""[v3.23.681] Giai đoạn tinh chỉnh / bản địa hoá "chỉ trả dòng cần sửa".

Trước đây STYLE/LOCALIZE phải chép lại CẢ lô (50 dòng) dù chỉ sửa vài dòng: tốn token đầu ra (chậm nhất trong một
lượt gọi) và dễ "viết lại cho có". Chế độ mới: model chỉ trả mảng ``edits`` gồm các dòng đổi chữ; ứng dụng tự ghép lại
thành lô đủ dòng TRƯỚC mọi phép kiểm sẵn có (căn dòng, vá dòng thiếu, chia đôi lô).
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.application.services.stage_checkpoint import (
    stage_config_signature,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _EDITS_SCHEMA,
    EDITS_ONLY_RULE,
    GeminiSubtitleTranslator,
    _BatchValidationError,
    expand_edits_payload,
    is_edits_schema,
)

ROWS = [
    ("师父", "Sư phụ."),
    ("你回来了", "Người đã về rồi."),
    ("今天天气不错", "Hôm nay trời đẹp ghê."),
    ("走吧", "Đi thôi."),
]


def _lines() -> list[TranslationLine]:
    return [
        TranslationLine(
            index=i + 1, start_ms=i * 1000, end_ms=i * 1000 + 900, text=draft,
            original_text=source, speaker="Diệp Nhiên" if i % 2 else "Sở Dao", description="",
        )
        for i, (source, draft) in enumerate(ROWS)
    ]


class _Ctx:
    source_lang = "zh"
    target_lang = "Vietnamese"
    enable_tags = False
    include_desc = False

    def __getattr__(self, _name: str) -> Any:
        return None


class _EditingModel:
    """Model giả: đọc <current_batch>, chỉ trả các dòng có trong ``fixes`` (theo câu gốc)."""

    def __init__(self, fixes: dict[str, str]) -> None:
        self.fixes = fixes
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model: str, prompt: str, config: Any, validator: Any, **_kw: Any) -> dict[str, Any]:
        block = re.search(r"<current_batch>\n(.*?)\n</current_batch>", prompt, re.S)
        assert block is not None
        items = json.loads(block.group(1))
        self.calls.append({"schema": config.response_schema, "prompt": prompt, "system": config.system})
        if is_edits_schema(config.response_schema):
            payload: dict[str, Any] = {"edits": [
                {"line_no": item["line_no"], "text": self.fixes[item["original"]]}
                for item in items if item.get("original") in self.fixes
            ]}
        else:
            payload = {"subtitles": [
                {"line_no": item["line_no"], "speaker": "", "text": self.fixes.get(item.get("original"), item["text"])}
                for item in items
            ]}
        validator(payload)
        return payload


def _adapter(monkeypatch: pytest.MonkeyPatch, model: _EditingModel) -> GeminiSubtitleTranslator:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    monkeypatch.setattr(adapter, "_ensure_available", lambda: None, raising=False)
    monkeypatch.setattr(adapter, "_response_schema_for", lambda *a, **k: {"properties": {"subtitles": {}}}, raising=False)
    monkeypatch.setattr(adapter, "_system_instruction", lambda *a, **k: "SYS", raising=False)
    monkeypatch.setattr(
        adapter, "_build_config",
        lambda temperature, schema, system, **k: SimpleNamespace(response_schema=schema, system=system),
        raising=False,
    )
    monkeypatch.setattr(adapter, "_call_gemini", model, raising=False)
    return adapter


def _run(adapter: GeminiSubtitleTranslator, *, kind=TranslationStageKind.STYLE, edits_only=True, prior=True):
    lines = _lines()
    return adapter.translate_stage(
        stage=TranslationStageConfig(kind=kind, model_name="m", batch_size=50, context_size=0, edits_only=edits_only),
        context=_Ctx(), source_lines=lines, input_lines=lines, has_prior_translation=prior,
    )


def test_only_edited_lines_change_and_rest_keep_previous_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _EditingModel({"你回来了": "Sư phụ về rồi ạ."})
    adapter = _adapter(monkeypatch, model)
    out = _run(adapter)
    assert [line.text for line in out] == ["Sư phụ.", "Sư phụ về rồi ạ.", "Hôm nay trời đẹp ghê.", "Đi thôi."]
    assert [line.speaker for line in out] == [line.speaker for line in _lines()]     # người nói không bị xoá
    [call] = model.calls
    assert call["schema"] is _EDITS_SCHEMA
    assert call["system"].endswith(EDITS_ONLY_RULE)
    assert "CHỈ đưa vào 'edits'" in call["prompt"]
    assert adapter.edits_stats == {"batches": 1, "lines": 4, "edited": 1}


def test_empty_edits_keeps_whole_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    out = _run(_adapter(monkeypatch, _EditingModel({})), kind=TranslationStageKind.LOCALIZE)
    assert [line.text for line in out] == [draft for _s, draft in ROWS]


def test_off_by_default_and_never_without_prior_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _EditingModel({})
    _run(_adapter(monkeypatch, model), edits_only=False)
    _run(_adapter(monkeypatch, model), prior=False)               # chưa có bản dịch → phải dịch đủ từng dòng
    _run(_adapter(monkeypatch, model), kind=TranslationStageKind.LITERAL)
    assert all(not is_edits_schema(call["schema"]) for call in model.calls)
    assert TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m").edits_only is False


def test_expand_edits_payload() -> None:
    batch = _lines()
    data: dict[str, Any] = {"edits": [{"line_no": 13, "text": "  Trời đẹp quá.  "}, {"line_no": 12, "text": ""}]}
    assert expand_edits_payload(data, batch, start_idx=10) == 1
    assert "edits" not in data
    assert [item["line_no"] for item in data["subtitles"]] == [11, 12, 13, 14]
    assert data["subtitles"][2]["text"] == "Trời đẹp quá."
    assert data["subtitles"][1]["text"] == "Người đã về rồi."              # sửa rỗng = giữ nguyên
    assert data["subtitles"][1]["speaker"] == "Diệp Nhiên"

    # Model đánh số lại từ 1 (hành vi đã biết của bản lite) → KHÔNG gán nhầm dòng; báo lỗi để thử lại.
    with pytest.raises(_BatchValidationError):
        expand_edits_payload({"edits": [{"line_no": 1, "text": "x"}]}, batch, start_idx=10)
    with pytest.raises(_BatchValidationError):
        expand_edits_payload({"edits": "sai"}, batch, start_idx=0)
    single = {"edits": {"line_no": 1, "text": "Thưa sư phụ."}}               # một đối tượng thay vì mảng
    assert expand_edits_payload(single, batch, start_idx=0) == 1


def test_root_array_reply_is_read_as_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model trả mảng ở gốc → bộ chuẩn hoá đổi thành ``subtitles``; ở chế độ edits phải hiểu là danh sách sửa."""

    class _RootArrayModel(_EditingModel):
        def __call__(self, model, prompt, config, validator, **_kw):
            payload: dict[str, Any] = {"subtitles": [{"line_no": 4, "text": "Mình đi thôi."}]}
            validator(payload)
            return payload

    out = _run(_adapter(monkeypatch, _RootArrayModel({})))
    assert [line.text for line in out][-1] == "Mình đi thôi."
    assert [line.text for line in out][:3] == [draft for _s, draft in ROWS[:3]]


def test_stage_signature_unchanged_unless_enabled() -> None:
    base = TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m")
    on = TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m", edits_only=True)
    assert stage_config_signature(base, False) == stage_config_signature(
        TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m", edits_only=False), False,
    )
    assert stage_config_signature(base, False) != stage_config_signature(on, False)
