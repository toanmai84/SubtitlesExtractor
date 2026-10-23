"""[v3.23.620] Lô dịch mặc định (50 dòng) bị Gemini từ chối 400 INVALID_ARGUMENT.

Schema lô gắn CẢ minItems/maxItems lẫn khoảng line_no; Gemini coi là quá phức tạp khi lô
≥ 50 dòng (đo thật trên gemini-3.1/3.5-flash-lite: hai ràng buộc chạy tới 40 dòng, chỉ
số phần tử tới 50, chỉ khoảng line_no tới 200). Lỗi 400 chung chung còn bị nhận nhầm là
"model không hỗ trợ thinking" → tắt thinking, vẫn 400, hỏng cả lượt dịch.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _TRANSLATE_SCHEMA,
    MAX_ITEM_COUNT_CONSTRAINED_BATCH,
    GeminiSubtitleTranslator,
    _batch_constrained_schema,
    relax_schema_constraints,
)
from tests.unit.infrastructure.gemini_call_harness import (
    build_translator,
    install_fake_clock,
)

REAL_400 = (
    "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Request contains an "
    "invalid argument.', 'status': 'INVALID_ARGUMENT'}}"
)


def _batch(n: int) -> list[Any]:
    return [SimpleNamespace(index=i + 1, text="x") for i in range(n)]


def _array(schema: dict) -> dict:
    return schema["properties"]["subtitles"]


@pytest.mark.parametrize("n", [1, 20, MAX_ITEM_COUNT_CONSTRAINED_BATCH])
def test_small_batches_keep_both_constraints(n: int) -> None:
    schema = _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(n))
    assert _array(schema)["minItems"] == n == _array(schema)["maxItems"]
    assert _array(schema)["items"]["properties"]["line_no"]["maximum"] == n


@pytest.mark.parametrize("n", [50, 80, 200])
def test_large_batches_keep_only_line_range(n: int) -> None:
    schema = _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(n))
    assert "minItems" not in _array(schema) and "maxItems" not in _array(schema)
    assert _array(schema)["items"]["properties"]["line_no"]["maximum"] == n


def test_relax_steps_items_then_line_range_then_none() -> None:
    schema = _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(10))
    step1 = relax_schema_constraints(schema)
    assert "minItems" not in _array(step1)
    assert "minimum" in _array(step1)["items"]["properties"]["line_no"]
    step2 = relax_schema_constraints(step1)
    assert "minimum" not in _array(step2)["items"]["properties"]["line_no"]
    assert relax_schema_constraints(step2) is None
    assert "minItems" in _array(schema)  # không sửa tại chỗ


class _SchemaPickyServer:
    """Gemini giả: từ chối 400 khi schema còn minItems (như máy chủ thật với lô lớn)."""

    def __init__(self) -> None:
        self.schemas: list[dict] = []

    def generate_content(self, model: str, contents: list, config: Any) -> Any:
        schema = copy.deepcopy(config.response_schema)
        self.schemas.append(schema)
        if "minItems" in _array(schema):
            raise RuntimeError(REAL_400)
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None, candidates=[])


class _ConfigNS(SimpleNamespace):
    def model_copy(self, update: dict) -> "_ConfigNS":
        return _ConfigNS(**{**self.__dict__, **update})


def test_call_gemini_relaxes_schema_on_generic_400(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = install_fake_clock(monkeypatch)
    server = _SchemaPickyServer()
    translator, _slept = build_translator(monkeypatch, clock, None, server)
    config = _ConfigNS(
        temperature=0.2, response_mime_type="application/json", system_instruction="x",
        response_schema=_batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(10)),
        thinking_config=SimpleNamespace(thinking_level="minimal"),
    )
    result = translator._call_gemini("gemini-3.1-flash-lite", "prompt", config, lambda d: d)
    assert result == {"subtitles": []}
    assert len(server.schemas) == 2
    assert "minItems" not in _array(server.schemas[1])
    # Thinking KHÔNG bị tắt nhầm vì lỗi schema.
    assert not translator._minimal_level_rejected_models()


def test_generic_400_detection() -> None:
    assert GeminiSubtitleTranslator._is_generic_invalid_argument(RuntimeError(REAL_400))
    assert not GeminiSubtitleTranslator._is_generic_invalid_argument(
        RuntimeError("400 INVALID_ARGUMENT. Thinking level MINIMAL is not supported")
    )
