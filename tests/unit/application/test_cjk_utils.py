"""Tests cho :mod:`cjk_utils` — phân biệt CJK / Latin."""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.cjk_utils import (
    adaptive_min_text_chars,
    cjk_char_count,
    cjk_ratio,
    contains_cjk,
    effective_text_length,
    estimate_min_reading_duration_sec,
    is_cjk_char,
    is_predominantly_cjk,
)


class TestIsCjkChar:
    @pytest.mark.parametrize(
        "char",
        ["你", "好", "世", "界", "性", "別", "雄", "私", "は", "サ", "안"],
    )
    def test_cjk_chars(self, char: str) -> None:
        assert is_cjk_char(char)

    @pytest.mark.parametrize(
        "char",
        ["a", "Z", "1", "ô", "ế", " ", "?", "，"],  # comma fullwidth không phải morpheme
    )
    def test_non_cjk_chars(self, char: str) -> None:
        # Note: comma fullwidth ， (U+FF0C) thuộc CJK punctuation block
        # nhưng không phải morpheme — code hiện tại không list block này.
        assert not is_cjk_char(char)

    def test_multichar_input_returns_false(self) -> None:
        assert not is_cjk_char("好的")
        assert not is_cjk_char("")


class TestContainsCjk:
    def test_pure_cjk(self) -> None:
        assert contains_cjk("你好")

    def test_mixed(self) -> None:
        assert contains_cjk("Hello 世界")

    def test_pure_latin(self) -> None:
        assert not contains_cjk("Hello world")

    def test_vietnamese_diacritics_not_cjk(self) -> None:
        assert not contains_cjk("Tôi yêu Việt Nam")

    def test_empty(self) -> None:
        assert not contains_cjk("")


class TestCjkCharCount:
    def test_pure_cjk(self) -> None:
        assert cjk_char_count("性别") == 2

    def test_mixed(self) -> None:
        assert cjk_char_count("Hello 世界") == 2

    def test_pure_latin(self) -> None:
        assert cjk_char_count("Hello") == 0


class TestCjkRatio:
    def test_pure_cjk(self) -> None:
        assert cjk_ratio("性别") == 1.0

    def test_pure_latin(self) -> None:
        assert cjk_ratio("Hello") == 0.0

    def test_mixed_50_50(self) -> None:
        # "AB你好" — 2 Latin + 2 CJK = 50% CJK.
        assert cjk_ratio("AB你好") == 0.5

    def test_whitespace_ignored(self) -> None:
        # Whitespace không tính vào tổng.
        assert cjk_ratio("性 别") == 1.0

    def test_empty(self) -> None:
        assert cjk_ratio("") == 0.0
        assert cjk_ratio("   ") == 0.0


class TestIsPredominantlyCjk:
    def test_pure_cjk(self) -> None:
        assert is_predominantly_cjk("你好")

    def test_pure_latin(self) -> None:
        assert not is_predominantly_cjk("Hello")

    def test_majority_cjk_passes(self) -> None:
        # "中国GDP" — 2 CJK + 3 Latin = 40% CJK < threshold 50%.
        assert not is_predominantly_cjk("中国GDP")

    def test_strict_majority(self) -> None:
        # "中国GD" — 2 CJK + 2 Latin = 50% (just at threshold).
        assert is_predominantly_cjk("中国GD")

    def test_custom_threshold(self) -> None:
        # threshold=0.3 → 30% CJK đủ.
        assert is_predominantly_cjk("中国GDP", threshold=0.3)


class TestEffectiveTextLength:
    def test_latin(self) -> None:
        assert effective_text_length("Hello") == 5

    def test_cjk_double_width(self) -> None:
        # CJK = 2 đơn vị visual.
        assert effective_text_length("你好") == 4

    def test_mixed(self) -> None:
        # "Hi你好" = 2 (Hi) + 4 (你好) = 6.
        assert effective_text_length("Hi你好") == 6

    def test_whitespace_ignored(self) -> None:
        assert effective_text_length("Hi 你好") == 6


class TestEstimateMinReadingDuration:
    def test_floor_for_empty(self) -> None:
        assert estimate_min_reading_duration_sec("") >= 0.3

    def test_cjk_short_uses_floor(self) -> None:
        # "好" = 1 ký tự / 3.5 char/sec ≈ 0.286s — bị raise lên floor 0.3s.
        assert estimate_min_reading_duration_sec("好") >= 0.3

    def test_cjk_longer(self) -> None:
        # "你好世界" = 4 ký tự / 3.5 ≈ 1.14s.
        duration = estimate_min_reading_duration_sec("你好世界")
        assert 1.0 <= duration <= 1.3

    def test_latin_longer(self) -> None:
        # "Hello world!" = 12 ký tự (incl. space) / 12 char/sec = 1.0s.
        duration = estimate_min_reading_duration_sec("Hello world!")
        # Loose bounds vì space có thể được tính khác.
        assert 0.7 <= duration <= 1.2


class TestAdaptiveMinTextChars:
    def test_cjk_returns_one(self) -> None:
        # CJK 1 ký tự là phụ đề hợp lệ.
        assert adaptive_min_text_chars("好") == 1
        assert adaptive_min_text_chars("性别") == 1

    def test_latin_returns_default(self) -> None:
        assert adaptive_min_text_chars("hi") == 2
        assert adaptive_min_text_chars("Hello") == 2

    def test_custom_latin_min(self) -> None:
        assert adaptive_min_text_chars("hi", latin_min=5) == 5
        # CJK vẫn = 1 bất kể latin_min.
        assert adaptive_min_text_chars("好", latin_min=10) == 1
