"""Test các entity của tầng domain."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


class TestVideoMetadata:
    def test_creates_with_valid_args(self) -> None:
        meta = VideoMetadata(
            path=Path("/tmp/video.mp4"),
            width=1920,
            height=1080,
            fps=29.97,
            total_frames=300,
            duration_sec=10.0,
        )
        assert meta.aspect_ratio == pytest.approx(1920 / 1080)
        assert meta.duration_str == "00:00:10"
        assert meta.filename == "video.mp4"

    def test_rejects_zero_fps(self) -> None:
        with pytest.raises(ConfigurationError):
            VideoMetadata(
                path=Path("/tmp/v.mp4"),
                width=10, height=10, fps=0, total_frames=0, duration_sec=0,
            )

    def test_replace_roi_clips_to_frame(self) -> None:
        meta = VideoMetadata(
            path=Path("/tmp/v.mp4"),
            width=1920, height=1080,
            fps=30.0, total_frames=1, duration_sec=0.0,
        )
        oversized = Roi(x=1900, y=1070, width=200, height=200)
        new_meta = meta.replace_roi(oversized)
        assert new_meta.roi is not None
        assert new_meta.roi.x + new_meta.roi.width <= 1920
        assert new_meta.roi.y + new_meta.roi.height <= 1080

    def test_duration_str_formats_hours_correctly(self) -> None:
        meta = VideoMetadata(
            path=Path("/v.mp4"),
            width=1, height=1, fps=1.0,
            total_frames=3725, duration_sec=3725.0,
        )
        assert meta.duration_str == "01:02:05"


class TestOcrFrameResult:
    def test_empty_frame_reports_correctly(self) -> None:
        frame = OcrFrameResult(frame_index=0, timestamp_sec=0.0, text_boxes=[])
        assert frame.is_empty
        assert frame.joined_text == ""
        assert frame.mean_confidence.value == 0.0

    def test_joined_text_orders_by_y_then_x(self) -> None:
        # Box bên dưới (y lớn) đến trước trong list — sau khi sort phải xuống dòng dưới.
        frame = OcrFrameResult(
            frame_index=1,
            timestamp_sec=0.5,
            text_boxes=[
                OcrTextBox(
                    text="Dòng dưới",
                    confidence=Confidence(0.9),
                    polygon=[(0, 100), (200, 100), (200, 130), (0, 130)],
                ),
                OcrTextBox(
                    text="Dòng trên",
                    confidence=Confidence(0.8),
                    polygon=[(0, 10), (200, 10), (200, 40), (0, 40)],
                ),
            ],
        )
        assert frame.joined_text == "Dòng trên\nDòng dưới"

    def test_mean_confidence_average(self) -> None:
        frame = OcrFrameResult(
            frame_index=0,
            timestamp_sec=0.0,
            text_boxes=[
                OcrTextBox(text="A", confidence=Confidence(0.5)),
                OcrTextBox(text="B", confidence=Confidence(0.9)),
            ],
        )
        assert frame.mean_confidence.value == pytest.approx(0.7)


class TestSubtitleEvent:
    def test_basic_event(self) -> None:
        event = SubtitleEvent(
            index=1,
            text="Xin chào",
            interval=TimeInterval(0.0, 2.5),
            confidence=Confidence(0.95),
            frame_count=10,
        )
        assert event.start_sec == 0.0
        assert event.end_sec == 2.5
        assert event.duration_sec == 2.5
        # uid được gán tự động.
        assert event.uid

    def test_uids_are_unique(self) -> None:
        e1 = SubtitleEvent(index=1, text="a", interval=TimeInterval(0, 1))
        e2 = SubtitleEvent(index=2, text="b", interval=TimeInterval(1, 2))
        assert e1.uid != e2.uid
