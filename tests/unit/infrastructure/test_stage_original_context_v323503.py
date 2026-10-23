"""[v3.23.503] Mỗi giai đoạn phải có phụ đề GỐC THÔ để tự đối chiếu và sửa lệch của khâu trước.

Hai lỗ hổng trước đây:
1. ``original_text`` chỉ được gieo ở giai đoạn SAU preprocess → nó lưu bản ĐÃ HIỆU ĐÍNH, không phải
   bản gốc thô. PREPROCESS làm lệch thì mọi khâu sau đối chiếu với dữ liệu đã sai.
2. LITERAL không dùng payload kép, dù đầu vào của nó là KẾT QUẢ của PREPROCESS → lệch do PREPROCESS
   gây ra trôi qua mà không khâu nào bắt được.
"""
from __future__ import annotations

import inspect

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationStageKind
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

_STAGE_SOURCE = inspect.getsource(GeminiSubtitleTranslator.translate_stage)
_BATCH_SOURCE = inspect.getsource(GeminiSubtitleTranslator._translate_single_batch)


def test_original_text_seeded_before_preprocess() -> None:
    """Điều kiện gieo KHÔNG được loại trừ preprocess, nếu không sẽ mất bản gốc thô."""
    assert "if not has_prior_translation:" in _STAGE_SOURCE
    assert "if not has_prior_translation and not is_preprocess:" not in _STAGE_SOURCE


def test_dual_payload_covers_all_post_preprocess_stages() -> None:
    """LITERAL, STYLE, LOCALIZE đều phải nhận bản gốc kèm theo."""
    block = _STAGE_SOURCE[_STAGE_SOURCE.index("use_dual_payload = ") :][:400]
    for kind in (
        TranslationStageKind.LITERAL,
        TranslationStageKind.STYLE,
        TranslationStageKind.LOCALIZE,
    ):
        assert f"TranslationStageKind.{kind.name}" in block, kind.name


def test_self_correction_loop_tied_to_dual_payload() -> None:
    """Vòng tự sửa dòng dịch thiếu chạy ở mọi giai đoạn có payload kép."""
    index = _STAGE_SOURCE.index("expansion_indices")
    assert "if use_dual_payload:" in _STAGE_SOURCE[index : index + 400]


def test_instruction_tells_model_to_fix_previous_stage_errors() -> None:
    for phrase in (
        "phụ đề GỐC THÔ",
        "đối chiếu 'text' với 'original'",
        "nội dung của dòng khác",  # cụm liền mạch trong một chuỗi nguồn
        "đừng kế thừa lỗi của khâu trước",
    ):
        assert phrase in _BATCH_SOURCE, phrase


def test_plain_stage_instruction_unchanged() -> None:
    """Giai đoạn không có payload kép vẫn dùng chỉ thị gọn (không thêm nhiễu)."""
    assert "chỉ xử lý <current_batch> và trả JSON" in _BATCH_SOURCE
