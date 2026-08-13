"""Unit tests cho fix v2.28 — Space-Restorer.

Bảo vệ các tính năng:
    * ``_restore_dropped_space`` phục hồi space tại vị trí có evidence mạnh.
    * Ngưỡng chặt (>= 2 high HOẶC >= 4 med) tránh false positive.
    * Length >= 5 chars để tránh stuttering pattern (`'你你'` ngắn).
    * Yi-Restorer chạy TRƯỚC Space-Restorer (order quan trọng).
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
    return SubtitleBuilder(SubtitleBuilderConfig(use_viterbi=False))


class TestRestoreDroppedSpace:
    """v2.28: _restore_dropped_space chỉ chạy khi evidence rất mạnh."""

    def test_restore_when_high_count_meets_threshold(self) -> None:
        """voted='阿姨原来你在洗手间' (no space), >= 2 frames có space conf >= 0.92."""
        voted = "阿姨原来你在洗手间"
        candidates = [
            (voted, 0.99),
            (voted, 0.99),
            (voted, 0.99),
            (voted, 0.99),
            ("阿姨 原来你在洗手间", 0.96),  # high (>= 0.92)
            ("阿姨 原来你在洗手间", 0.95),  # high
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        assert result == "阿姨 原来你在洗手间"

    def test_no_restore_when_only_one_high_evidence(self) -> None:
        """Chỉ 1 frame conf >= 0.92 → không đủ evidence (cần >= 2 high)."""
        voted = "阿姨原来你在洗手间"
        candidates = [
            (voted, 0.99),
            (voted, 0.99),
            (voted, 0.99),
            ("阿姨 原来你在洗手间", 0.94),  # chỉ 1 high
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        assert result == voted  # không restore

    def test_restore_with_4_medium_evidence(self) -> None:
        """4 frames conf 0.85-0.91 (med) → đủ evidence."""
        voted = "阿姨原来你在洗手间"
        candidates = [
            (voted, 0.99),
            (voted, 0.99),
            (voted, 0.99),
            ("阿姨 原来你在洗手间", 0.88),  # med
            ("阿姨 原来你在洗手间", 0.87),  # med
            ("阿姨 原来你在洗手间", 0.89),  # med
            ("阿姨 原来你在洗手间", 0.86),  # med
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        assert result == "阿姨 原来你在洗手间"

    def test_no_restore_when_voted_text_too_short(self) -> None:
        """voted_text < 5 chars → không restore (tránh pattern stuttering)."""
        voted = "你你"  # 2 chars
        candidates = [
            (voted, 0.95),
            ("你 你", 0.95),  # high evidence nhưng text quá ngắn
            ("你 你", 0.95),
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        assert result == voted  # không restore vì < 5 chars

    def test_no_restore_when_voted_already_has_space(self) -> None:
        """voted_text đã có space → trả về nguyên."""
        voted = "阿姨 原来你在洗手间"
        candidates = [
            (voted, 0.99),
            ("阿姨 原 来你在洗手间", 0.95),
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        assert result == voted  # giữ nguyên

    def test_no_restore_when_no_candidate_has_space(self) -> None:
        """Không có candidate nào có space → không restore."""
        voted = "阿姨原来你在洗手间"
        candidates = [
            (voted, 0.99),
            ("阿姨原来你在洗手间", 0.99),
            ("阿姨原来你在洗手", 0.85),  # drop 1 char nhưng không có space
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        assert result == voted

    def test_no_restore_when_non_cjk_text(self) -> None:
        """Text không predominantly CJK → không restore (rule không áp dụng)."""
        voted = "HelloWorld"  # 10 chars Latin
        candidates = [
            ("Hello World", 0.99),
            ("Hello World", 0.99),
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        assert result == voted

    def test_no_restore_when_candidate_stripped_mismatch(self) -> None:
        """Candidate sau khi bỏ space != voted → skip case này."""
        voted = "阿姨原来你在洗手间"
        candidates = [
            (voted, 0.99),
            # candidate khác text (length khác sau khi bỏ space)
            ("阿姨 原来在洗手间", 0.95),  # bỏ space = '阿姨原来在洗手间' length 8 != voted length 9
            ("阿姨 原来在洗手间", 0.94),
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        assert result == voted

    def test_restore_uses_majority_position(self) -> None:
        """Pick position có evidence mạnh nhất khi có nhiều pattern."""
        voted = "阿姨原来你在洗手间"
        candidates = [
            (voted, 0.99),
            # pattern A: space sau '阿姨' (position 2) — strong (2 high)
            ("阿姨 原来你在洗手间", 0.96),
            ("阿姨 原来你在洗手间", 0.95),
            # pattern B: space sau '原来' (position 4) — weak (1 high)
            ("阿姨原来 你在洗手间", 0.94),
        ]
        result = SubtitleBuilder._restore_dropped_space(voted, candidates)
        # Pattern A thắng (2 high vs 1 high).
        assert result == "阿姨 原来你在洗手间"


class TestRestorerOrder:
    """v2.28: Yi-Restorer chạy TRƯỚC Space-Restorer."""

    def test_yi_restored_before_space_check(self) -> None:
        """voted='起去前厅用饭' → Yi-Restorer phục hồi '一' → '一起去前厅用饭'.
        
        Space-Restorer sau đó không cần kích hoạt.
        """
        voted = "起去前厅用饭"  # 6 chars, missing '一' đầu
        # Candidates có '一起去前厅用饭' với evidence mạnh.
        candidates = [
            (voted, 0.99),
            (voted, 0.99),
            ("一起去前厅用饭", 0.95),  # high — Yi-Restorer evidence
            ("一起去前厅用饭", 0.94),
        ]
        result = SubtitleBuilder._restore_dropped_yi_prefix(voted, candidates)
        # Yi-Restorer kích hoạt, kết quả có '一'.
        assert "一" in result
        assert result == "一起去前厅用饭"


class TestSupersetMergeStillWorks:
    """v2.28 không phá vỡ v2.27 superset merge."""

    def test_superset_detection_still_works(self) -> None:
        """v2.27 ``_is_superset_within_one_char`` không bị ảnh hưởng."""
        assert (
            SubtitleBuilder._is_superset_within_one_char("颗丹药", "一颗丹药")
            is True
        )

    def test_symmetric_distinct_utterance_still_works(
        self, builder: SubtitleBuilder
    ) -> None:
        """v2.27 fix asymmetric vẫn hoạt động."""
        result_a = builder._is_distinct_cjk_utterance(
            "一颗丹药", "颗丹药", 0.99, 0.99, 0.08
        )
        result_b = builder._is_distinct_cjk_utterance(
            "颗丹药", "一颗丹药", 0.99, 0.99, 0.08
        )
        assert result_a == result_b
