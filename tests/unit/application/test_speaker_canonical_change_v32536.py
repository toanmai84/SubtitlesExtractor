"""[v3.23.136] Test: "đổi người nói" xét theo TÊN ĐÃ CHUẨN HOÁ, không theo chuỗi thô.

Cùng một người dưới biến thể tên (tên CJK vs phiên âm, hoa/thường, alias roster) phải được
coi là CÙNG người -> chỉ tag nhãn ở dòng đầu, không lặp ở dòng sau.
"""

from __future__ import annotations

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase as UseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
)


def _ctx() -> TranslationContext:
    return TranslationContext(target_lang="Vietnamese", enable_tags=True)


def _line(text: str, speaker: str) -> TranslationLine:
    return TranslationLine(index=1, start_ms=0, end_ms=1, text=text, speaker=speaker)


def test_alias_variants_tagged_once() -> None:
    # Hai biến thể tên cùng quy về "Trương Vĩ".
    canon = {
        UseCase._normalize_name_key("张伟"): "Trương Vĩ",
        UseCase._normalize_name_key("Zhang Wei"): "Trương Vĩ",
    }
    ctx = _ctx()
    text1, last = UseCase._compose_display_text(
        _line("Xin chào", "张伟"), "Xin chào", ctx, "", None, canon
    )
    text2, last = UseCase._compose_display_text(
        _line("Tạm biệt", "Zhang Wei"), "Tạm biệt", ctx, last, None, canon
    )
    assert text1.startswith("[Trương Vĩ:]")
    # Dòng 2 cùng người (qua chuẩn hoá) -> KHÔNG lặp nhãn.
    assert not text2.startswith("[")
    assert text2 == "Tạm biệt"


def test_different_speakers_each_tagged() -> None:
    ctx = _ctx()
    t1, last = UseCase._compose_display_text(
        _line("A", "John"), "A", ctx, "", None, None
    )
    t2, last = UseCase._compose_display_text(
        _line("B", "Mary"), "B", ctx, last, None, None
    )
    assert t1.startswith("[John:]")
    assert t2.startswith("[Mary:]")  # đổi người thật -> vẫn tag


def test_same_raw_speaker_not_repeated() -> None:
    # Hồi quy: cùng chuỗi thô vẫn không lặp (hành vi cũ giữ nguyên).
    ctx = _ctx()
    t1, last = UseCase._compose_display_text(
        _line("A", "John"), "A", ctx, "", None, None
    )
    t2, _ = UseCase._compose_display_text(
        _line("B", "John"), "B", ctx, last, None, None
    )
    assert t1.startswith("[John:]")
    assert not t2.startswith("[")
