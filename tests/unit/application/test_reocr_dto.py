"""Test cho các DTO Re-OCR.

Bao phủ:
    * :class:`TimeRange` validation và merge.
    * :func:`_merge_overlapping_ranges` với các edge case.
    * :class:`ReOcrRequest` validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    OcrEngineConfig,
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.dtos.reocr_dto import (
    ReOcrRequest,
    ReOcrResponse,
    TimeRange,
    _merge_overlapping_ranges,
)
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
)


def _make_minimal_request(time_ranges: list[TimeRange]) -> ReOcrRequest:
    """Helper tạo ReOcrRequest tối thiểu cho test."""
    return ReOcrRequest(
        video_path=Path("/tmp/dummy.mp4"),
        time_ranges=time_ranges,
        replace_uids=["uid-1"],
        roi=None,
        sampling=FrameSamplingConfig(),
        ocr=OcrEngineConfig(),
        builder=SubtitleBuilderConfig(),
    )


class TestTimeRange:
    def test_creates_valid_range(self) -> None:
        time_range = TimeRange(start_sec=1.0, end_sec=3.5)
        assert time_range.duration_sec == pytest.approx(2.5)

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ValueError, match="không nhận giá trị âm"):
            TimeRange(start_sec=-0.1, end_sec=1.0)

    def test_rejects_start_after_end(self) -> None:
        with pytest.raises(ValueError, match="không hợp lệ"):
            TimeRange(start_sec=2.0, end_sec=1.0)

    def test_rejects_zero_duration(self) -> None:
        with pytest.raises(ValueError):
            TimeRange(start_sec=1.0, end_sec=1.0)

    def test_overlaps_detects_intersection(self) -> None:
        first = TimeRange(0.0, 2.0)
        second = TimeRange(1.0, 3.0)
        assert first.overlaps(second)
        assert second.overlaps(first)

    def test_overlaps_does_not_count_touching_ranges(self) -> None:
        first = TimeRange(0.0, 1.0)
        second = TimeRange(1.0, 2.0)
        # Tiếp giáp tại 1.0 — không gọi là overlap.
        assert not first.overlaps(second)

    def test_merge_returns_minimum_enclosing_range(self) -> None:
        first = TimeRange(0.0, 2.0)
        second = TimeRange(1.5, 3.0)
        merged = first.merge(second)
        assert merged.start_sec == 0.0
        assert merged.end_sec == 3.0


class TestMergeOverlappingRanges:
    def test_empty_input_returns_empty(self) -> None:
        assert _merge_overlapping_ranges([], 0.0) == []

    def test_no_overlap_no_merge(self) -> None:
        ranges = [TimeRange(0.0, 1.0), TimeRange(5.0, 6.0)]
        merged = _merge_overlapping_ranges(ranges, merge_window_sec=0.0)
        assert len(merged) == 2

    def test_overlapping_merged(self) -> None:
        ranges = [TimeRange(0.0, 2.0), TimeRange(1.0, 3.0)]
        merged = _merge_overlapping_ranges(ranges, merge_window_sec=0.0)
        assert len(merged) == 1
        assert merged[0].start_sec == 0.0
        assert merged[0].end_sec == 3.0

    def test_close_ranges_merged_within_window(self) -> None:
        # Cách nhau 0.3s — merge_window=0.5 → gộp.
        ranges = [TimeRange(0.0, 1.0), TimeRange(1.3, 2.0)]
        merged = _merge_overlapping_ranges(ranges, merge_window_sec=0.5)
        assert len(merged) == 1
        assert merged[0].end_sec == 2.0

    def test_close_ranges_not_merged_outside_window(self) -> None:
        # Cách nhau 0.6s — merge_window=0.5 → không gộp.
        ranges = [TimeRange(0.0, 1.0), TimeRange(1.6, 2.5)]
        merged = _merge_overlapping_ranges(ranges, merge_window_sec=0.5)
        assert len(merged) == 2

    def test_unsorted_input_handled(self) -> None:
        # Input lộn xộn → vẫn ra kết quả đúng.
        ranges = [TimeRange(5.0, 6.0), TimeRange(0.0, 1.0), TimeRange(0.5, 2.0)]
        merged = _merge_overlapping_ranges(ranges, merge_window_sec=0.0)
        assert len(merged) == 2
        assert merged[0].start_sec == 0.0
        assert merged[0].end_sec == 2.0
        assert merged[1].start_sec == 5.0

    def test_chain_of_overlaps_collapsed_to_one(self) -> None:
        ranges = [
            TimeRange(0.0, 2.0),
            TimeRange(1.5, 3.5),
            TimeRange(3.0, 5.0),
        ]
        merged = _merge_overlapping_ranges(ranges, merge_window_sec=0.0)
        assert len(merged) == 1
        assert merged[0].end_sec == 5.0


class TestReOcrRequest:
    def test_valid_request(self) -> None:
        request = _make_minimal_request([TimeRange(0.0, 1.0)])
        assert request.total_duration_sec == pytest.approx(1.0)

    def test_rejects_empty_time_ranges(self) -> None:
        with pytest.raises(ValueError, match="time_ranges"):
            _make_minimal_request([])

    def test_rejects_negative_merge_window(self) -> None:
        with pytest.raises(ValueError, match="merge_window_sec"):
            ReOcrRequest(
                video_path=Path("/tmp/x.mp4"),
                time_ranges=[TimeRange(0.0, 1.0)],
                replace_uids=["u1"],
                roi=None,
                sampling=FrameSamplingConfig(),
                ocr=OcrEngineConfig(),
                builder=SubtitleBuilderConfig(),
                merge_window_sec=-0.5,
            )

    def test_total_duration_uses_merged_ranges(self) -> None:
        # 3 range overlap → total = 1 range gộp.
        request = ReOcrRequest(
            video_path=Path("/tmp/x.mp4"),
            time_ranges=[
                TimeRange(0.0, 2.0),
                TimeRange(1.0, 3.0),
                TimeRange(2.5, 4.0),
            ],
            replace_uids=["u1", "u2", "u3"],
            roi=None,
            sampling=FrameSamplingConfig(),
            ocr=OcrEngineConfig(),
            builder=SubtitleBuilderConfig(),
            merge_window_sec=0.0,
        )
        # Sau merge: [0.0, 4.0] → 4 giây.
        assert request.total_duration_sec == pytest.approx(4.0)


class TestReOcrResponse:
    def test_basic_response(self) -> None:
        response = ReOcrResponse(
            new_events=[],
            replaced_uids=["u1", "u2"],
            elapsed_seconds=1.5,
            frames_processed=42,
            ranges_processed=2,
        )
        assert response.elapsed_seconds == 1.5
        assert response.frames_processed == 42
        assert response.ranges_processed == 2
        assert response.replaced_uids == ["u1", "u2"]
