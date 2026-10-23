"""[v3.23.153] Test _merge_ai_items GIỮ ``scene`` (cue sống qua các giai đoạn).

Bug: ``_merge_ai_items`` tạo TranslationLine mới nhưng bỏ rơi ``scene`` ở cả hai nhánh
(preprocess lẫn dịch) trong khi giữ speaker/description/addressee/original_text. Visual
cues chỉ được áp MỘT LẦN vào source_lines trước vòng stage, nên scene mất VĨNH VIỄN ngay
sau giai đoạn đầu -> ``_line_to_dual_payload`` của STYLE/LOCALIZE không còn trường "cue"
-> khâu tinh chỉnh giọng điệu mù bối cảnh/cảm xúc (suy giảm chất lượng âm thầm).
"""

from __future__ import annotations

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _bare_translator() -> "GeminiSubtitleTranslator":
    """[v3.23.536] Thực thể TRỐNG để gọi các hàm thuần cần ``self``.

    ``_length_hint``, ``_line_to_payload`` và ``_line_to_dual_payload`` là phương thức thực thể
    (không phải staticmethod); phần thân chỉ dùng vài thuộc tính tuỳ chọn nên dựng bằng ``__new__``
    là đủ, không cần API key.
    """
    instance = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    instance._reading_cps_budget = 0.0
    instance._emotion_cues = False
    return instance


def _line(index: int, text: str) -> TranslationLine:
    return TranslationLine(
        index=index, start_ms=index * 1000, end_ms=index * 1000 + 2000, text=text,
        speaker="Tần Chính", addressee="Nữ tỳ", scene="tức giận, trong đại điện",
        original_text="原文", description="tiếng gió",
    )


def test_merge_keeps_scene_translate_branch() -> None:
    batch = [_line(1, "bản nháp")]
    ai_items = [{"line_no": 1, "text": "bản tinh chỉnh"}]
    merged = GeminiSubtitleTranslator._merge_ai_items(
        batch, ai_items, is_preprocess=False
    )
    assert merged[0].text == "bản tinh chỉnh"
    assert merged[0].scene == "tức giận, trong đại điện"
    assert merged[0].addressee == "Nữ tỳ"
    assert merged[0].speaker == "Tần Chính"
    assert merged[0].original_text == "原文"


def test_merge_keeps_scene_preprocess_branch() -> None:
    batch = [_line(1, "văn bản gốc")]
    ai_items = [{"line_no": 1, "text": "đã tiền xử lý"}]
    merged = GeminiSubtitleTranslator._merge_ai_items(batch, ai_items, is_preprocess=True)
    assert merged[0].scene == "tức giận, trong đại điện"


def test_merge_keeps_scene_when_ai_returns_empty() -> None:
    batch = [_line(1, "giữ nguyên")]
    merged = GeminiSubtitleTranslator._merge_ai_items(
        batch, [{"line_no": 1, "text": ""}], is_preprocess=False
    )
    assert merged[0].text == "giữ nguyên"  # fallback chống nuốt chữ
    assert merged[0].scene == "tức giận, trong đại điện"


def test_scene_survives_into_next_stage_payload() -> None:
    """Bất biến chất lượng: output stage trước vẫn tạo payload có 'cue' cho stage sau."""
    batch = [_line(1, "bản nháp")]
    merged = GeminiSubtitleTranslator._merge_ai_items(
        batch, [{"line_no": 1, "text": "bản dịch"}], is_preprocess=False
    )
    dual = _bare_translator()._line_to_dual_payload(merged[0])
    single = _bare_translator()._line_to_payload(merged[0])
    assert dual.get("cue") == "tức giận, trong đại điện"
    assert single.get("cue") == "tức giận, trong đại điện"
    assert dual.get("addressee") == "Nữ tỳ"
