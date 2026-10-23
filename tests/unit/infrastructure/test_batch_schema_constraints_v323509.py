"""[v3.23.509] Ràng buộc số dòng ngay trong SCHEMA của từng lô.

Tài liệu Gemini (Structured outputs) cho phép ``minItems``/``maxItems`` với mảng và
``minimum``/``maximum`` với số nguyên. Ép ở tầng schema mạnh hơn hẳn so với năn nỉ trong prompt —
đúng nguyên tắc "schema hơn chỉ thị" mà dự án đã đúc kết qua các lần sửa trước.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _TRANSLATE_NO_DESC_SCHEMA,
    _TRANSLATE_SCHEMA,
    _batch_constrained_schema,
)


def _batch(*indices: int) -> list[TranslationLine]:
    return [
        TranslationLine(index=i, start_ms=0, end_ms=100, text=f"dòng {i}") for i in indices
    ]


class TestItemCountConstraint:
    def test_item_count_pinned_to_batch_size(self) -> None:
        schema = _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(41, 42, 43))
        array_schema = schema["properties"]["subtitles"]
        assert array_schema["minItems"] == 3
        assert array_schema["maxItems"] == 3

    def test_line_no_range_pinned(self) -> None:
        schema = _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(41, 42, 43))
        line_no = schema["properties"]["subtitles"]["items"]["properties"]["line_no"]
        assert line_no["minimum"] == 41
        assert line_no["maximum"] == 43

    def test_single_line_batch(self) -> None:
        schema = _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(7))
        array_schema = schema["properties"]["subtitles"]
        assert array_schema["minItems"] == array_schema["maxItems"] == 1


class TestDescriptions:
    """Tài liệu khuyến nghị dùng ``description`` để dẫn dắt model."""

    @pytest.mark.parametrize("field", ["line_no", "text"])
    def test_item_fields_documented(self, field: str) -> None:
        schema = _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(1, 2))
        assert schema["properties"]["subtitles"]["items"]["properties"][field]["description"]

    def test_array_documented_with_expected_count(self) -> None:
        schema = _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(1, 2))
        assert "2" in schema["properties"]["subtitles"]["description"]


class TestSafety:
    def test_base_schema_not_mutated(self) -> None:
        _batch_constrained_schema(_TRANSLATE_SCHEMA, _batch(1, 2))
        assert "minItems" not in _TRANSLATE_SCHEMA["properties"]["subtitles"]
        text_field = _TRANSLATE_SCHEMA["properties"]["subtitles"]["items"]["properties"]["text"]
        assert "description" not in text_field

    def test_empty_batch_returns_base_schema(self) -> None:
        assert _batch_constrained_schema(_TRANSLATE_SCHEMA, []) is _TRANSLATE_SCHEMA

    def test_unknown_shape_returns_base_schema(self) -> None:
        other = {"type": "OBJECT", "properties": {"khac": {"type": "STRING"}}}
        assert _batch_constrained_schema(other, _batch(1)) is other

    def test_works_for_no_description_schema(self) -> None:
        schema = _batch_constrained_schema(_TRANSLATE_NO_DESC_SCHEMA, _batch(5, 6))
        assert schema["properties"]["subtitles"]["minItems"] == 2
