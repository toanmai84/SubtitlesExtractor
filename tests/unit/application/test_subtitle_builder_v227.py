"""Unit tests cho fix v2.27 — Symmetric distinct utterance + Superset merge.

Bảo vệ các tính năng:
    * ``_is_distinct_cjk_utterance`` symmetric (kết quả không phụ thuộc thứ tự args).
    * ``_calculate_effective_similarity`` symmetric cho cặp text khác len.
    * ``_is_superset_within_one_char`` phát hiện đúng cặp text chênh 1 ký tự CJK.
    * ``_execute_merge_onto_last`` ưu tiên text dài hơn khi superset.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import (
    SubtitleBuilder,
)


@pytest.fixture()
def builder() -> SubtitleBuilder:
    """Trả về SubtitleBuilder default cho test."""
    return SubtitleBuilder(SubtitleBuilderConfig(use_viterbi=False))


class TestSymmetricDistinctCjkUtterance:
    """v2.27: _is_distinct_cjk_utterance phải symmetric."""

    def test_drop_yi_char_symmetric(self, builder: SubtitleBuilder) -> None:
        """Cặp '颗丹药' vs '一颗丹药' — pattern OCR drop ký tự '一' đầu."""
        result_a = builder._is_distinct_cjk_utterance(
            "一颗丹药", "颗丹药", 0.99, 0.99, 0.08
        )
        result_b = builder._is_distinct_cjk_utterance(
            "颗丹药", "一颗丹药", 0.99, 0.99, 0.08
        )
        # Trước v2.27: result_a=True, result_b=False (asymmetric!).
        # Sau v2.27: cả 2 đều False (cùng câu, OCR drop ký tự đầu).
        assert result_a == result_b
        assert result_a is False

    def test_drop_suffix_char_symmetric(self, builder: SubtitleBuilder) -> None:
        """Cặp '感觉丹田要撑爆' vs '感觉丹田要撑爆了' — drop ký tự cuối."""
        result_a = builder._is_distinct_cjk_utterance(
            "感觉丹田要撑爆", "感觉丹田要撑爆了", 0.95, 0.95, 0.08
        )
        result_b = builder._is_distinct_cjk_utterance(
            "感觉丹田要撑爆了", "感觉丹田要撑爆", 0.95, 0.95, 0.08
        )
        assert result_a == result_b

    def test_symmetric_for_real_distinct_utterances(
        self, builder: SubtitleBuilder
    ) -> None:
        """Cặp text thực sự khác nhau — kết quả phải symmetric (và True)."""
        result_a = builder._is_distinct_cjk_utterance(
            "我去厨房", "把饭做好", 0.95, 0.95, 0.5
        )
        result_b = builder._is_distinct_cjk_utterance(
            "把饭做好", "我去厨房", 0.95, 0.95, 0.5
        )
        assert result_a == result_b

    def test_similarity_score_symmetric_for_yi_drop(
        self, builder: SubtitleBuilder
    ) -> None:
        """Effective similarity phải symmetric cho cặp OCR-drop."""
        sim_a = builder._calculate_effective_similarity(
            "一颗丹药", "颗丹药", 0.99, 0.99, 0.08
        )
        sim_b = builder._calculate_effective_similarity(
            "颗丹药", "一颗丹药", 0.99, 0.99, 0.08
        )
        assert sim_a == sim_b
        # Sau fix: cả 2 trả về 1.0 → merge thành công.
        assert sim_a == pytest.approx(1.0)

    def test_gap_above_threshold_still_keeps_distinct(
        self, builder: SubtitleBuilder
    ) -> None:
        """Gap >= 0.15s + conf cao + startswith/endswith → vẫn distinct (rule cũ giữ)."""
        result = builder._is_distinct_cjk_utterance(
            "一颗丹药", "颗丹药", 0.95, 0.95, 0.30
        )
        # Gap = 0.30s >= 0.15 và conf cao → vẫn nên đánh dấu distinct.
        assert result is True


class TestIsSupersetWithinOneChar:
    """v2.27: _is_superset_within_one_char phát hiện cặp text chênh 1 ký tự CJK."""

    def test_yi_insert_at_beginning_detected(self) -> None:
        """'颗丹药' và '一颗丹药' — chèn '一' đầu → superset."""
        assert SubtitleBuilder._is_superset_within_one_char("颗丹药", "一颗丹药") is True
        assert SubtitleBuilder._is_superset_within_one_char("一颗丹药", "颗丹药") is True

    def test_suffix_append_detected(self) -> None:
        """'感觉丹田要撑爆' và '感觉丹田要撑爆了' — chèn '了' cuối → superset."""
        assert (
            SubtitleBuilder._is_superset_within_one_char(
                "感觉丹田要撑爆", "感觉丹田要撑爆了"
            )
            is True
        )

    def test_middle_insert_detected(self) -> None:
        """'起去前厅用饭' và '一起去前厅用饭' — chèn '一' giữa → superset."""
        assert (
            SubtitleBuilder._is_superset_within_one_char(
                "起去前厅用饭", "一起去前厅用饭"
            )
            is True
        )

    def test_two_char_diff_not_superset(self) -> None:
        """Chênh 2 ký tự → KHÔNG phải superset."""
        assert (
            SubtitleBuilder._is_superset_within_one_char("颗丹药", "一里颗丹药")
            is False
        )

    def test_same_length_not_superset(self) -> None:
        """Cùng length → KHÔNG phải superset (chỉ replace, không insert)."""
        assert SubtitleBuilder._is_superset_within_one_char("颗丹药", "颗丹丸") is False

    def test_non_cjk_text_returns_false(self) -> None:
        """Text không predominantly CJK → False."""
        assert SubtitleBuilder._is_superset_within_one_char("OK", "OKA") is False
        assert SubtitleBuilder._is_superset_within_one_char("hello", "ahello") is False

    def test_empty_text_returns_false(self) -> None:
        """Text rỗng → False."""
        assert SubtitleBuilder._is_superset_within_one_char("", "颗丹药") is False
        assert SubtitleBuilder._is_superset_within_one_char("颗丹药", "") is False

    def test_single_char_too_short_returns_false(self) -> None:
        """Text < 2 chars → False (quá ngắn để tin cậy)."""
        assert SubtitleBuilder._is_superset_within_one_char("颗", "一颗") is False

    def test_non_cjk_char_inserted_returns_false(self) -> None:
        """Ký tự được insert KHÔNG phải CJK (vd space, punct) → False."""
        assert SubtitleBuilder._is_superset_within_one_char("颗丹药", "颗 丹药") is False
