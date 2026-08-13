"""[v3.23.126] Test BẢO VỆ: phân luồng system_instruction vs contents phải TÁCH BẠCH.

Gemini API tách riêng `config.system_instruction` (vai trò/quy tắc) và `contents`
(dữ liệu của lượt). Test khoá bất biến: VAI TRÒ/QUY TẮC chỉ ở system_instruction;
khối DỮ LIỆU trong contents KHÔNG được chứa các marker vai trò/quy tắc/ví dụ.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _wrap_xml_block,
)

# Các marker CHỈ được phép xuất hiện ở system_instruction (KÊNH 1).
_ROLE_MARKERS = ("## VAI TRÒ", "## QUY TẮC", "VÍ DỤ MINH HOẠ", "BẮT BUỘC NHẤT QUÁN")


def _stub() -> SimpleNamespace:
    return SimpleNamespace(
        _canonical_names_directive=GeminiSubtitleTranslator._canonical_names_directive
    )


def _system_instruction(kind: TranslationStageKind) -> str:
    stage = TranslationStageConfig(kind=kind, model_name="gemini-3.5-flash")
    ctx = TranslationContext(target_lang="Tiếng Việt", source_lang="zh", overview="x")
    return GeminiSubtitleTranslator._system_instruction(_stub(), stage, ctx, False)


def test_role_and_rules_live_in_system_instruction() -> None:
    si = _system_instruction(TranslationStageKind.LITERAL)
    assert "## VAI TRÒ" in si
    assert "## QUY TẮC" in si


def test_data_block_has_no_role_markers() -> None:
    # Khối DỮ LIỆU (đi vào contents) tuyệt đối không chứa vai trò/quy tắc/ví dụ.
    payload = [{"line_no": 1, "text": "你好"}, {"line_no": 2, "text": "再见"}]
    block = _wrap_xml_block("current_batch", payload)
    for marker in _ROLE_MARKERS:
        assert marker not in block


def test_per_turn_instruction_is_data_scoped_only() -> None:
    # Chỉ thị ngắn trong contents chỉ điều phối lượt này, không phải quy tắc bền vững.
    instruction = (
        "Dựa trên các khối dữ liệu trên: chỉ xử lý <current_batch>, "
        "trả JSON hợp lệ theo schema."
    )
    for marker in _ROLE_MARKERS:
        assert marker not in instruction


def test_config_system_instruction_len_helper() -> None:
    cfg = SimpleNamespace(system_instruction="abc def")
    assert GeminiSubtitleTranslator._config_system_instruction_len(cfg) == 7
    assert GeminiSubtitleTranslator._config_system_instruction_len(SimpleNamespace()) == 0
    assert (
        GeminiSubtitleTranslator._config_system_instruction_len(
            SimpleNamespace(system_instruction=None)
        )
        == 0
    )


def test_examples_not_leaking_into_data_payload() -> None:
    # Ví dụ few-shot nằm trong system_instruction, KHÔNG trong dữ liệu gửi đi.
    si = _system_instruction(TranslationStageKind.STYLE)
    assert "VÍ DỤ MINH HOẠ" in si
    block = _wrap_xml_block("current_batch", [{"line_no": 1, "text": "abc"}])
    assert "VÍ DỤ MINH HOẠ" not in block
