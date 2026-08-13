"""Test cho công cụ phân tích trần khả thi (ceiling_analyzer) — hàm thuần."""

from __future__ import annotations

from dataclasses import dataclass

from tools.calibration.ceiling_analyzer import (
    CeilingBreakdown,
    aggregate_breakdowns,
    analyze_ceiling,
)


@dataclass
class _FakeBox:
    text: str


@dataclass
class _FakeFrame:
    timestamp_sec: float
    text_boxes: list[_FakeBox]

    def get_joined_text(self) -> str:
        return "".join(b.text for b in self.text_boxes)


def _frame(ts: float, *texts: str) -> _FakeFrame:
    return _FakeFrame(timestamp_sec=ts, text_boxes=[_FakeBox(t) for t in texts])


class TestCeilingAnalyzer:
    def test_achieved_when_built_matches(self) -> None:
        gt = [(1.0, 2.0, "你好世界")]
        built = [(1.0, 2.0, "你好世界")]
        frames = [_frame(1.5, "你好世界")]
        result = analyze_ceiling(ground_truth=gt, built_events=built, frames=frames)
        assert result.achieved == 1 and result.recoverable == 0
        assert result.input_limited == 0
        assert result.achieved_rate == 1.0 and result.ceiling_rate == 1.0

    def test_recoverable_when_ocr_has_text_but_engine_missed(self) -> None:
        gt = [(1.0, 2.0, "你好世界")]
        built = [(1.0, 2.0, "你好")]  # engine cắt mất 2 ký tự
        frames = [_frame(1.5, "你好世界")]  # nhưng OCR có đủ
        result = analyze_ceiling(ground_truth=gt, built_events=built, frames=frames)
        assert result.recoverable == 1 and result.achieved == 0
        assert result.ceiling_rate == 1.0 and result.engine_gap_rate == 1.0

    def test_input_limited_when_ocr_never_captured(self) -> None:
        gt = [(1.0, 2.0, "一举两得")]
        built = [(1.0, 2.0, "举两得")]
        frames = [_frame(1.5, "举两得")]  # OCR không bao giờ bắt được 一
        result = analyze_ceiling(ground_truth=gt, built_events=built, frames=frames)
        assert result.input_limited == 1 and result.recoverable == 0
        assert result.ceiling_rate == 0.0 and result.input_limited_rate == 1.0

    def test_traditional_as_simplified_counts_as_achieved(self) -> None:
        try:
            import zhconv  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("zhconv không có — bỏ qua kiểm tra phồn/giản")
        gt = [(1.0, 2.0, "性別雌性")]   # phồn thể 別
        built = [(1.0, 2.0, "性别雌性")]  # giản thể 别
        frames = [_frame(1.5, "性别雌性")]
        strict = analyze_ceiling(ground_truth=gt, built_events=built, frames=frames)
        fair = analyze_ceiling(
            ground_truth=gt, built_events=built, frames=frames,
            traditional_as_simplified=True,
        )
        # Strict: khác 別/别 → không đạt; Fair: coi tương đương → đạt.
        assert strict.achieved == 0
        assert fair.achieved == 1

    def test_aggregate_sums_correctly(self) -> None:
        a = CeilingBreakdown(total_cues=10, achieved=8, recoverable=1, input_limited=1)
        b = CeilingBreakdown(total_cues=20, achieved=18, recoverable=1, input_limited=1)
        total = aggregate_breakdowns([a, b])
        assert total.total_cues == 30 and total.achieved == 26
        assert total.recoverable == 2 and total.input_limited == 2
