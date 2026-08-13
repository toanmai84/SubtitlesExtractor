"""[v3.23.133] Test: dòng chỉ ký hiệu/nhạc (♪), số, dấu câu được GIỮ NGUYÊN (mọi ngôn
ngữ) — không gửi model dịch (tránh model trả rỗng + vá thừa).
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _lines(*texts: str):
    return [
        SimpleNamespace(index=i + 1, text=t) for i, t in enumerate(texts)
    ]


def _ctx(lang: str):
    return SimpleNamespace(source_lang=lang)


def test_symbol_and_music_lines_are_noise_any_language() -> None:
    lines = _lines("♪ ♪", "Hello there", "♫♫", "123", "- Yeah.", "...")
    noise = GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("en"))
    # ♪ ♪(1), ♫♫(3), 123(4), ...(6) là nhiễu; "Hello"(2) & "- Yeah."(5) thì không.
    assert noise == {1, 3, 4, 6}


def test_text_with_music_and_words_is_translated() -> None:
    lines = _lines("♪ La la la ♪", "♪ Đoạn nhạc có lời ♪")
    noise = GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("en"))
    assert noise == set()  # có chữ → vẫn dịch


def test_cjk_noise_still_detected_plus_symbols() -> None:
    # Nguồn CJK: dòng không có ký tự CJK (nhiễu OCR) + dòng ký hiệu đều là nhiễu.
    lines = _lines("中文字幕", "akas", "♪ ♪")
    noise = GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("zh"))
    assert 2 in noise and 3 in noise and 1 not in noise


def test_vietnamese_diacritics_not_noise() -> None:
    lines = _lines("Xin chào các bạn", "Đây là phụ đề")
    noise = GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("en"))
    assert noise == set()
