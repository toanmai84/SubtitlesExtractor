"""Unit tests cho các fix v2.24 trong SubtitleBuilder.

Mỗi test ứng với một fix được mô tả trong CHANGELOG.md > [2.24.0].
Mục đích: regression-protection cho Char-Restorer (Yi-Restorer + Suffix
Append), Box-level garbage filter, Enhanced Latin gibberish detector.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_builder import (
    SubtitleBuilder,
    _is_latin_gibberish,
    _is_single_cjk_char,
    _accumulate_confidence_bucket,
)


class TestIsSingleCjkCharHelper:
    """`_is_single_cjk_char` chỉ trả True cho 1 ký tự CJK đơn."""

    @pytest.mark.parametrize(
        "valid_cjk_char",
        ["一", "了", "个", "的", "你", "我", "他"],
    )
    def test_valid_cjk_single_chars(self, valid_cjk_char: str) -> None:
        assert _is_single_cjk_char(valid_cjk_char) is True

    @pytest.mark.parametrize(
        "invalid_input",
        ["", "ab", "12", " ", "一个", "A", "1", ".", "，"],
    )
    def test_invalid_inputs_return_false(self, invalid_input: str) -> None:
        assert _is_single_cjk_char(invalid_input) is False


class TestAccumulateConfidenceBucket:
    """`_accumulate_confidence_bucket` phân loại đúng high vs medium."""

    def test_none_initial_returns_zero_then_increment(self) -> None:
        assert _accumulate_confidence_bucket(None, 0.95) == (1, 0)
        assert _accumulate_confidence_bucket(None, 0.88) == (0, 1)
        assert _accumulate_confidence_bucket(None, 0.50) == (0, 0)

    def test_existing_bucket_increments(self) -> None:
        assert _accumulate_confidence_bucket((2, 1), 0.93) == (3, 1)
        assert _accumulate_confidence_bucket((2, 1), 0.87) == (2, 2)

    def test_boundary_conf_092_counts_as_high(self) -> None:
        """Đúng tại biên 0.92: thuộc high bucket (>= 0.92)."""
        assert _accumulate_confidence_bucket(None, 0.92) == (1, 0)

    def test_boundary_conf_085_counts_as_medium(self) -> None:
        """Đúng tại biên 0.85: thuộc medium bucket (>= 0.85, < 0.92)."""
        assert _accumulate_confidence_bucket(None, 0.85) == (0, 1)


class TestRestoreDroppedYiPrefix:
    """`_restore_dropped_yi_prefix` thực hiện 2 op: yi_insert và suffix_append.

    Heuristic critical (xem docstring v2.24):
      - yi_insert: 1 high HOẶC 2 med.
      - suffix_append: 2 high HOẶC 4 med (chặt hơn).
    """

    def _call_restorer(
        self,
        voted_text: str,
        texts_with_confidences: list[tuple[str, float]],
    ) -> str:
        return SubtitleBuilder._restore_dropped_yi_prefix(
            voted_text, texts_with_confidences
        )

    def test_yi_prefix_inserted_when_one_high_conf_frame_exists(self) -> None:
        """`'起去前厅用饭'` → `'一起去前厅用饭'` nếu có 1 frame conf cao."""
        voted = "起去前厅用饭"
        candidates = [
            (voted, 0.99),
            (voted, 0.99),
            (voted, 0.99),
            ("一起去前厅用饭", 0.95),  # 1 high evidence
        ]
        assert self._call_restorer(voted, candidates) == "一起去前厅用饭"

    def test_yi_insert_middle_position(self) -> None:
        """`'整整盆灵液'` → `'整整一盆灵液'` (insert giữa)."""
        voted = "整整盆灵液"
        candidates = [(voted, 0.99)] * 30 + [("整整一盆灵液", 0.99)]
        assert self._call_restorer(voted, candidates) == "整整一盆灵液"

    def test_yi_insert_skipped_when_evidence_too_weak(self) -> None:
        """1 frame conf < 0.85 → KHÔNG đủ evidence."""
        voted = "起去前厅用饭"
        candidates = [
            (voted, 0.99),
            ("一起去前厅用饭", 0.70),  # conf < 0.85
        ]
        assert self._call_restorer(voted, candidates) == voted

    def test_yi_insert_two_medium_conf_frames_sufficient(self) -> None:
        """2 frames conf >= 0.85 (med bucket) đủ cho yi_insert."""
        voted = "起去前厅用饭"
        candidates = [
            (voted, 0.99),
            ("一起去前厅用饭", 0.88),
            ("一起去前厅用饭", 0.87),
        ]
        assert self._call_restorer(voted, candidates) == "一起去前厅用饭"

    def test_suffix_append_le_when_two_high_conf_frames(self) -> None:
        """`'感觉丹田要撑爆'` → `'感觉丹田要撑爆了'` với 2+ high evidence."""
        voted = "感觉丹田要撑爆"
        candidates = [(voted, 0.99)] * 16 + [("感觉丹田要撑爆了", 0.99)] * 8
        assert self._call_restorer(voted, candidates) == "感觉丹田要撑爆了"

    def test_suffix_append_skipped_when_only_one_high_conf(self) -> None:
        """Chỉ 1 high conf cho suffix_append → CHẶT hơn, KHÔNG apply."""
        voted = "感觉丹田要撑爆"
        candidates = [(voted, 0.99)] * 16 + [("感觉丹田要撑爆了", 0.99)]
        # Chỉ 1 high, không đủ (cần >= 2 high cho suffix_append)
        assert self._call_restorer(voted, candidates) == voted

    def test_suffix_append_skipped_for_garbage_char(self) -> None:
        """`'总归是'` + `'介'` (2 med) chỉ là OCR noise → KHÔNG apply.
        
        Regression test cho false positive: 介 là OCR error của 个 trên box riêng.
        Với chỉ 2 med (< 4 med threshold), không upgrade.
        """
        voted = "总归是"
        candidates = [(voted, 1.00)] * 21 + [("总归是介", 0.86), ("总归是介", 0.86)]
        # 2 med < 4 med threshold cho suffix_append → KHÔNG apply
        assert self._call_restorer(voted, candidates) == voted

    def test_non_cjk_voted_text_skipped(self) -> None:
        """Voted text non-CJK → không apply (vd toàn Latin)."""
        voted = "hello"
        candidates = [(voted, 0.99)] * 10 + [("一hello", 0.99)]
        assert self._call_restorer(voted, candidates) == voted

    def test_voted_text_too_short_skipped(self) -> None:
        """Voted text < 2 chars → không apply (rủi ro false positive cao)."""
        voted = "好"
        candidates = [(voted, 0.99)] * 10 + [("一好", 0.99)]
        assert self._call_restorer(voted, candidates) == voted

    def test_empty_voted_text_returns_empty(self) -> None:
        assert self._call_restorer("", []) == ""


class TestBoxLevelGarbageFilterPatterns:
    """Kiểm tra 2 regex pattern dùng trong `_pre_filter_garbage_boxes` v2.24."""

    def test_digit_only_regex_matches_pure_digits(self) -> None:
        from subtitles_extractor.application.services.subtitle_builder import (
            _DIGIT_ONLY_TOKEN_REGEX,
        )
        assert _DIGIT_ONLY_TOKEN_REGEX.fullmatch("33325535") is not None
        assert _DIGIT_ONLY_TOKEN_REGEX.fullmatch("03330330") is not None
        assert _DIGIT_ONLY_TOKEN_REGEX.fullmatch("123 456") is not None  # space ok
        assert _DIGIT_ONLY_TOKEN_REGEX.fullmatch("123a") is None  # có chữ
        assert _DIGIT_ONLY_TOKEN_REGEX.fullmatch("哈哈") is None  # CJK

    def test_latin_repetitive_regex_matches_repeated_chars(self) -> None:
        from subtitles_extractor.application.services.subtitle_builder import (
            _LATIN_REPETITIVE_REGEX,
        )
        # Phải có >= 3 ký tự lặp liên tiếp (chính xác là regex `[A-Za-z]\1{2,}`,
        # tức 1 ký tự + >= 2 ký tự giống nhau = tổng >= 3).
        assert _LATIN_REPETITIVE_REGEX.fullmatch("Mooooo") is not None  # ooooo
        assert _LATIN_REPETITIVE_REGEX.fullmatch("hooooo") is not None
        assert _LATIN_REPETITIVE_REGEX.fullmatch("aaaa") is not None  # 4 a's
        assert _LATIN_REPETITIVE_REGEX.fullmatch("abc") is None  # no repeat
        assert _LATIN_REPETITIVE_REGEX.fullmatch("HELLO") is None  # ll only 2
        assert _LATIN_REPETITIVE_REGEX.fullmatch("HELLOOO") is not None  # ooo


class TestEnhancedLatinGibberishDetectorV224:
    """Test 2 rule mới trong `_is_latin_gibberish` v2.24."""

    def test_anai_caught_by_pattern_suspicious_low_conf_rule(self) -> None:
        """`'ANAI'` (vowel-heavy, conf 0.54, 6 frames) → drop."""
        assert _is_latin_gibberish("ANAI", confidence=0.54, frame_count=6) is True

    def test_anai_with_higher_frame_count_still_caught(self) -> None:
        """`'ANAI'` với 5-6 frames vẫn drop nếu conf < 0.60.

        Note: early return `frame_count >= 7` ở đầu hàm chặn rule mới cho
        frame_count cao hơn. Đây là intentional — frame_count >= 7 chứng tỏ
        text "ổn định" trên màn hình đủ lâu để giả định là phụ đề thực sự.
        Rule mới chỉ áp dụng trong khoảng 3-6 frames.
        """
        assert _is_latin_gibberish("ANAI", confidence=0.55, frame_count=5) is True
        assert _is_latin_gibberish("ANAI", confidence=0.55, frame_count=6) is True

    def test_anai_high_conf_kept(self) -> None:
        """Confidence >= 0.85 → keep kể cả vowel pattern."""
        assert _is_latin_gibberish("ANAI", confidence=0.90, frame_count=5) is False

    def test_aar_caught_by_short_uppercase_low_conf_rule(self) -> None:
        """`'AAR'` (3 chars all upper, conf 0.44, 2 frames) → drop."""
        assert _is_latin_gibberish("AAR", confidence=0.44, frame_count=2) is True

    def test_aar_with_medium_conf_kept(self) -> None:
        """Conf >= 0.50 → keep ký tự ngắn (có thể là acronym hợp lệ)."""
        assert _is_latin_gibberish("AAR", confidence=0.60, frame_count=2) is False

    def test_whitelist_acronyms_never_dropped(self) -> None:
        """`'OK'`, `'USD'`, `'GPS'` ... → KHÔNG drop kể cả conf thấp."""
        assert _is_latin_gibberish("OK", confidence=0.3, frame_count=1) is False
        assert _is_latin_gibberish("USD", confidence=0.4, frame_count=1) is False
        assert _is_latin_gibberish("GPS", confidence=0.2, frame_count=1) is False

    def test_normal_english_word_not_dropped(self) -> None:
        """Câu Latin bình thường → keep."""
        assert _is_latin_gibberish(
            "Hello World", confidence=0.5, frame_count=3
        ) is False
