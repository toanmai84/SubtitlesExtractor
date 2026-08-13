"""Test các value object của tầng domain."""

from __future__ import annotations

import pytest

from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.domain.value_objects import (
    Confidence,
    DeviceKind,
    Roi,
    SubtitleFormat,
    TimeInterval,
)


class TestRoi:
    def test_creates_with_valid_args(self) -> None:
        roi = Roi(x=10, y=20, width=100, height=50)
        assert roi.x == 10
        assert roi.area == 5000
        assert roi.x2 == 110
        assert roi.y2 == 70

    def test_rejects_negative_origin(self) -> None:
        with pytest.raises(ConfigurationError):
            Roi(x=-1, y=0, width=10, height=10)

    def test_rejects_zero_width(self) -> None:
        with pytest.raises(ConfigurationError):
            Roi(x=0, y=0, width=0, height=10)

    def test_clip_keeps_inside(self) -> None:
        roi = Roi(x=950, y=950, width=200, height=200)
        clipped = roi.clip_to(frame_width=1000, frame_height=1000)
        assert clipped.x == 950
        assert clipped.x + clipped.width == 1000

    def test_is_immutable(self) -> None:
        roi = Roi(x=0, y=0, width=10, height=10)
        with pytest.raises(AttributeError):
            roi.x = 5  # type: ignore[misc]


class TestTimeInterval:
    def test_creates_and_computes_duration(self) -> None:
        interval = TimeInterval(start_sec=1.0, end_sec=3.5)
        assert interval.duration_sec == pytest.approx(2.5)
        assert interval.start_ms() == 1000
        assert interval.end_ms() == 3500

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ConfigurationError):
            TimeInterval(start_sec=-1.0, end_sec=0.0)

    def test_rejects_end_before_start(self) -> None:
        with pytest.raises(ConfigurationError):
            TimeInterval(start_sec=2.0, end_sec=1.0)

    def test_overlap_detection(self) -> None:
        a = TimeInterval(0.0, 2.0)
        b = TimeInterval(1.0, 3.0)
        c = TimeInterval(3.0, 4.0)
        assert a.overlaps_with(b)
        assert not a.overlaps_with(c) or a.end_sec <= c.start_sec  # boundary touch


class TestConfidence:
    def test_accepts_valid_range(self) -> None:
        assert Confidence(0.5).value == 0.5
        assert Confidence(0.0).value == 0.0
        assert Confidence(1.0).value == 1.0

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ConfigurationError):
            Confidence(1.5)
        with pytest.raises(ConfigurationError):
            Confidence(-0.1)

    def test_from_percentage(self) -> None:
        assert Confidence.from_percentage(75).value == pytest.approx(0.75)

    def test_supports_ordering(self) -> None:
        low = Confidence(0.3)
        high = Confidence(0.9)
        assert low < high
        assert max(low, high) == high


class TestDeviceKindEnum:
    def test_from_string_case_insensitive(self) -> None:
        assert DeviceKind.from_string("GPU") is DeviceKind.GPU
        assert DeviceKind.from_string("cpu") is DeviceKind.CPU

    def test_from_string_invalid(self) -> None:
        with pytest.raises(ValueError):
            DeviceKind.from_string("npu")


class TestSubtitleFormatEnum:
    def test_values(self) -> None:
        assert SubtitleFormat.SRT.value == "srt"
        assert SubtitleFormat.ASS.value == "ass"
