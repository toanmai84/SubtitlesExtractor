"""Test [v3.23.63] thang đo khoảng cách (metrics) — hằng số & snap_to_scale."""

from __future__ import annotations

import pytest

from subtitles_extractor.presentation.theme import metrics


class TestScaleConstants:
    def test_multiples_of_four(self) -> None:
        for value in (
            metrics.SPACING_XS, metrics.SPACING_SM, metrics.SPACING_MD,
            metrics.SPACING_LG, metrics.SPACING_XL, metrics.SPACING_XXL,
        ):
            assert value % 4 == 0

    def test_ascending_order(self) -> None:
        scale = [
            metrics.SPACING_NONE, metrics.SPACING_XS, metrics.SPACING_SM,
            metrics.SPACING_MD, metrics.SPACING_LG, metrics.SPACING_XL,
            metrics.SPACING_XXL,
        ]
        assert scale == sorted(scale)
        assert len(set(scale)) == len(scale)  # không trùng


class TestSnapToScale:
    @pytest.mark.parametrize("value,expected", [
        (0, 0), (2, 0), (3, 4), (4, 4), (6, 4), (10, 8),
        (14, 12), (18, 16), (30, 32),
    ])
    def test_snaps_to_nearest(self, value: int, expected: int) -> None:
        assert metrics.snap_to_scale(value) == expected

    def test_above_max_returns_max(self) -> None:
        assert metrics.snap_to_scale(100) == metrics.SPACING_XXL

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            metrics.snap_to_scale(-1)

    def test_tie_prefers_smaller(self) -> None:
        # 6 cách đều 4 và 8 → chọn 4 (bậc nhỏ hơn).
        assert metrics.snap_to_scale(6) == 4
