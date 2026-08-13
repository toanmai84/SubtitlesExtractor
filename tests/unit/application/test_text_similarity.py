"""Tests cho :mod:`text_similarity` — RapidFuzz-backed similarity.

Bao phủ:
    * Phát hiện CJK / Hangul / Kana.
    * Logic CJK ngắn KHÔNG boost (sửa bug "性别" vs "雄性").
    * Latin/Việt với Jaro-Winkler boost cho prefix match.
    * Length penalty cho chuỗi khác độ dài.
    * Early exit qua score_cutoff.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.text_similarity import (
    clear_caches,
    contains_cjk,
    jaro_winkler_similarity,
    raw_fuzz_ratio,
    text_similarity,
    viterbi_similarity,
    viterbi_similarity_with_cutoff,
)


class TestContainsCjk:
    def test_pure_latin(self) -> None:
        assert not contains_cjk("Hello world")

    def test_pure_cjk(self) -> None:
        assert contains_cjk("性别")

    def test_mixed(self) -> None:
        assert contains_cjk("Hello 世界")

    def test_hangul(self) -> None:
        assert contains_cjk("안녕하세요")

    def test_hiragana(self) -> None:
        assert contains_cjk("こんにちは")

    def test_katakana(self) -> None:
        assert contains_cjk("カタカナ")

    def test_empty_string(self) -> None:
        assert not contains_cjk("")

    def test_vietnamese_diacritics_not_cjk(self) -> None:
        # Tiếng Việt có dấu nhưng không phải CJK.
        assert not contains_cjk("Tôi yêu Việt Nam")


class TestRawFuzzRatio:
    def test_identical(self) -> None:
        assert raw_fuzz_ratio("hello", "hello") == 1.0

    def test_empty_returns_zero(self) -> None:
        assert raw_fuzz_ratio("", "hello") == 0.0
        assert raw_fuzz_ratio("hello", "") == 0.0

    def test_one_char_typo(self) -> None:
        # "Hello" vs "Helo" — Levenshtein distance = 1.
        assert raw_fuzz_ratio("Hello", "Helo") > 0.85

    def test_completely_different(self) -> None:
        assert raw_fuzz_ratio("abc", "xyz") < 0.4

    def test_normalized_to_0_1(self) -> None:
        # Đảm bảo không bị quên chia 100.
        for sim in [
            raw_fuzz_ratio("hello", "world"),
            raw_fuzz_ratio("test", "test"),
            raw_fuzz_ratio("a", "ab"),
        ]:
            assert 0.0 <= sim <= 1.0


class TestJaroWinklerSimilarity:
    def test_identical(self) -> None:
        assert jaro_winkler_similarity("hello", "hello") == 1.0

    def test_prefix_match_boosted(self) -> None:
        """Jaro-Winkler ưu tiên prefix match."""
        # "Tôi yêu" prefix → JW cao hơn typo cuối.
        prefix_match = jaro_winkler_similarity("Tôi yêu Việt", "Tôi yêu Hà Nội")
        # 2 chuỗi share prefix "Tôi yêu " → similarity > 0.5.
        assert prefix_match > 0.5

    def test_no_prefix_match_low(self) -> None:
        # 性别 vs 雄性 — không share prefix → JW gần 0.
        assert jaro_winkler_similarity("性别", "雄性") < 0.5


class TestTextSimilarityCjk:
    """Quan trọng: regression tests cho bug 性别/雄性."""

    def test_cjk_two_chars_share_one_NOT_boosted(self) -> None:
        # Sửa bug v2.17: ratio 0.5 KHÔNG được boost lên 0.85 cho CJK ngắn.
        sim = text_similarity("性别", "雄性")
        assert sim < 0.6, (
            f"CJK 2-char share 1 char không được boost cao, nhận {sim}."
        )

    def test_cjk_identical(self) -> None:
        assert text_similarity("性别", "性别") == 1.0

    def test_latin_short_typo_boosted(self) -> None:
        # Latin ngắn — typo "Helo"/"Hello" coi là tương tự.
        # Length diff 1/5 = 0.2 → penalty (max_len=5) = 0.16.
        # raw_fuzz_ratio = 0.889; sau penalty còn ~0.79 — vẫn cao.
        sim = text_similarity("Helo", "Hello")
        assert sim > 0.7

    def test_cjk_long_string_normal(self) -> None:
        sim = text_similarity("Tôi yêu Việt Nam", "Tôi yêu Việt Nam")
        assert sim == 1.0

    def test_normalize_case_and_whitespace(self) -> None:
        # text_similarity normalize lowercase + strip whitespace.
        assert text_similarity("Hello World", "hello world") >= 0.9
        assert text_similarity("hello  world", "hello world") >= 0.9


class TestViterbiSimilarity:
    def test_no_normalization(self) -> None:
        # Viterbi giữ raw text — case khác phải khác similarity.
        sim_lower = viterbi_similarity("hello", "hello")
        sim_mixed = viterbi_similarity("hello", "HELLO")
        assert sim_lower == 1.0
        assert sim_mixed < 1.0  # Khác case → khác.


class TestViterbiSimilarityWithCutoff:
    def test_below_cutoff_returns_zero(self) -> None:
        # Cutoff 0.95 — 2 chuỗi rõ ràng dưới ngưỡng → trả 0.
        sim = viterbi_similarity_with_cutoff(
            "Hello world", "Goodbye everyone", score_cutoff=0.95
        )
        assert sim == 0.0

    def test_above_cutoff_returns_real_score(self) -> None:
        # Cutoff thấp hơn similarity thực → trả giá trị thật.
        sim = viterbi_similarity_with_cutoff(
            "Hello", "Hello", score_cutoff=0.5
        )
        assert sim == 1.0

    def test_length_diff_early_exit(self) -> None:
        # Length diff lớn → upper bound thấp → trả 0 mà không gọi RapidFuzz.
        sim = viterbi_similarity_with_cutoff(
            "a", "abcdefghijk", score_cutoff=0.9
        )
        assert sim == 0.0

    def test_cjk_with_cutoff(self) -> None:
        # CJK ngắn không boost, "性别" vs "雄性" = 0.5 < cutoff 0.6 → 0.
        sim = viterbi_similarity_with_cutoff(
            "性别", "雄性", score_cutoff=0.6
        )
        assert sim == 0.0

    def test_identical_short_circuit(self) -> None:
        # Identical strings short-circuit về 1.0 không gọi RapidFuzz.
        sim = viterbi_similarity_with_cutoff(
            "test", "test", score_cutoff=0.99
        )
        assert sim == 1.0


class TestClearCaches:
    def test_clear_does_not_crash(self) -> None:
        # Pre-fill caches.
        text_similarity("hello", "world")
        viterbi_similarity("a", "b")
        raw_fuzz_ratio("x", "y")
        jaro_winkler_similarity("p", "q")
        # Clear — không raise.
        clear_caches()
        # Vẫn dùng được sau khi clear.
        assert text_similarity("hello", "hello") == 1.0


class TestSymmetry:
    """Đảm bảo similarity là đối xứng: f(a, b) == f(b, a)."""

    @pytest.mark.parametrize(
        "a,b",
        [
            ("hello", "world"),
            ("性别", "雄性"),
            ("Tôi yêu", "Tôi không"),
            ("a", "abc"),
            ("", "x"),
        ],
    )
    def test_text_similarity_symmetric(self, a: str, b: str) -> None:
        assert text_similarity(a, b) == text_similarity(b, a)

    @pytest.mark.parametrize(
        "a,b",
        [
            ("hello", "world"),
            ("性别", "雄性"),
            ("a", "abc"),
        ],
    )
    def test_viterbi_similarity_symmetric(self, a: str, b: str) -> None:
        assert viterbi_similarity(a, b) == viterbi_similarity(b, a)
