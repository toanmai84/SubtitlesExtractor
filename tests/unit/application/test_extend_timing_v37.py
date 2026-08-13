"""Tests cho fix v3.7: phục hồi timing biên từ soft-drop frames.

Vấn đề: Stages 2+3 lọc frame confidence-thấp/spatial-outlier ở biên đầu/cuối
phụ đề → start trễ hơn thực tế ~1 sample step, end sớm hơn thực tế ~1 step.
Fix: thu thập soft-drop timestamps và mở rộng group timing sát biên.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    extend_group_timing_from_soft_drops,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import FrameGroup

_STEP = 0.04  # 40ms sample step (25fps lấy mẫu)


def _group(start: float, end: float, text: str = "X") -> FrameGroup:
    return FrameGroup(
        reconstructed_text=text,
        start_timestamp_sec=start,
        end_timestamp_sec=end,
        accumulated_confidence=0.9,
        total_frames_count=3,
    )


class TestExtendGroupTimingFromSoftDrops:
    # ── START mở rộng ────────────────────────────────────────────────────────

    def test_extends_start_when_soft_drop_just_before(self) -> None:
        """Frame bị lọc 1 step trước start → start được kéo về."""
        groups = [_group(5.080, 6.000)]
        soft = [5.040]  # frame bị lọc, 1 step trước 5.080
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        assert result[0].start_timestamp_sec == pytest.approx(5.040, abs=1e-6)
        assert result[0].end_timestamp_sec == pytest.approx(6.000, abs=1e-6)  # không đổi

    def test_extends_start_to_earliest_in_window(self) -> None:
        """2 soft-drop liên tiếp trước start → start kéo về frame sớm nhất."""
        groups = [_group(5.120, 6.000)]
        soft = [5.040, 5.080]  # 2 frame bị lọc
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        # Window tìm [5.120-0.060, 5.120-0.020] = [5.060, 5.100] → chỉ 5.080 vào cửa sổ
        assert result[0].start_timestamp_sec == pytest.approx(5.080, abs=1e-6)

    def test_soft_drop_between_groups_claimed_by_prev_end(self) -> None:
        """Soft-drop giữa 2 group → END của group trước mở rộng trước (fade-out của A).

        Soft-drop 5.240 nằm sát A.end=5.200 và sát B.start=5.280. Vì function
        xử lý tuần tự (A trước B), A.end được mở rộng đến 5.240 trước. Sau đó
        khi B thử mở rộng start, min_allowed=5.240 nên 5.240 không vượt qua được
        (5.240 > 5.240 là False) → B.start giữ nguyên 5.280.
        Đây là hành vi ĐÚNG: soft-drop trong vùng fade-out của A.
        """
        groups = [_group(5.000, 5.200, "A"), _group(5.280, 6.000, "B")]
        soft = [5.240]  # frame giữa A_end=5.200 và B_start=5.280
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        # A nhận soft-drop → A.end mở rộng; B giữ nguyên
        assert result[0].end_timestamp_sec == pytest.approx(5.240, abs=1e-6)
        assert result[1].start_timestamp_sec == pytest.approx(5.280, abs=1e-6)

    def test_next_group_start_extended_when_soft_drop_closer_to_it(self) -> None:
        """Soft-drop sát B.start nhưng NGOÀI tầm A.end → B.start được mở rộng."""
        # A.end = 5.000, B.start = 5.200, soft = 5.160
        # A window end: [5.000+0.020, 5.000+0.060] = [5.020, 5.060] → 5.160 NGOÀI
        # B window start: [5.200-0.060, 5.200-0.020] = [5.140, 5.180] → 5.160 TRONG
        groups = [_group(4.000, 5.000, "A"), _group(5.200, 6.000, "B")]
        soft = [5.160]  # chỉ trong cửa sổ START của B
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        assert result[1].start_timestamp_sec == pytest.approx(5.160, abs=1e-6)
        assert result[0].end_timestamp_sec == pytest.approx(5.000, abs=1e-6)  # A không đổi

    # ── END mở rộng ──────────────────────────────────────────────────────────

    def test_extends_end_when_soft_drop_just_after(self) -> None:
        """Frame bị lọc 1 step sau end → end được kéo ra."""
        groups = [_group(5.000, 5.960)]
        soft = [6.000]  # frame bị lọc, 1 step sau 5.960
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        assert result[0].end_timestamp_sec == pytest.approx(6.000, abs=1e-6)
        assert result[0].start_timestamp_sec == pytest.approx(5.000, abs=1e-6)

    def test_end_not_extended_past_next_group(self) -> None:
        """Không mở rộng end vào vùng của group tiếp theo."""
        groups = [_group(5.000, 5.960, "A"), _group(6.080, 7.000, "B")]
        soft = [6.000, 6.040]  # frame giữa A_end=5.960 và B_start=6.080
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        # A end extended → 6.000, nhưng KHÔNG tới 6.040 (sẽ > 6.080 - margin)
        a_end = result[0].end_timestamp_sec
        b_start = result[1].start_timestamp_sec
        assert a_end < b_start  # không overlap

    def test_end_extended_to_latest_in_window(self) -> None:
        """2 soft-drop liên tiếp sau end → end kéo tới frame muộn nhất."""
        groups = [_group(5.000, 5.920)]
        soft = [5.960, 6.000]  # 2 frame bị lọc
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        # Window [5.920+0.020, 5.920+0.060] = [5.940, 5.980] → 5.960 vào cửa sổ
        assert result[0].end_timestamp_sec == pytest.approx(5.960, abs=1e-6)

    # ── Các trường hợp biên ─────────────────────────────────────────────────

    def test_empty_soft_drops_returns_unchanged(self) -> None:
        groups = [_group(5.0, 6.0)]
        result = extend_group_timing_from_soft_drops(groups, [], _STEP)
        assert result[0].start_timestamp_sec == pytest.approx(5.0)
        assert result[0].end_timestamp_sec == pytest.approx(6.0)

    def test_empty_groups_returns_empty(self) -> None:
        result = extend_group_timing_from_soft_drops([], [5.0], _STEP)
        assert result == []

    def test_soft_drop_too_far_not_extended(self) -> None:
        """Soft-drop cách biên > 1.5 step → KHÔNG mở rộng (không phải biên nhân quả)."""
        groups = [_group(5.000, 6.000)]
        soft = [4.880]  # cách start 120ms = 3 steps → ngoài cửa sổ 0.5-1.5 step
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        assert result[0].start_timestamp_sec == pytest.approx(5.000, abs=1e-6)

    def test_no_double_count_same_timestamp(self) -> None:
        """Cùng timestamp trong soft_drop và group boundary → không bị lỗi."""
        groups = [_group(5.040, 6.000)]
        soft = [5.040]  # trùng start → nằm ngoài cửa sổ [5.040-0.060, 5.040-0.020]
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        assert result[0].start_timestamp_sec == pytest.approx(5.040, abs=1e-6)

    def test_both_sides_extended(self) -> None:
        """Mở rộng CẢ start lẫn end trong cùng 1 group."""
        groups = [_group(5.080, 5.960)]
        soft = [5.040, 6.000]
        result = extend_group_timing_from_soft_drops(groups, soft, _STEP)
        assert result[0].start_timestamp_sec == pytest.approx(5.040, abs=1e-6)
        assert result[0].end_timestamp_sec == pytest.approx(6.000, abs=1e-6)
