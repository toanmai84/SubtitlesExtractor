"""Test trích phụ đề NHÚNG (embedded) — nhánh text-based và bitmap-OCR.

Nhánh text dùng MKV thật dựng bằng ffmpeg (bỏ qua nếu môi trường không có ffmpeg).
Nhánh bitmap dùng cổng giả + OCR giả để kiểm đường ảnh → OCR → SubtitleEvent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from subtitles_extractor.application.use_cases.extract_embedded_subtitles import (
    ExtractEmbeddedRequest,
    ExtractEmbeddedSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.ports.embedded_subtitle_port import (
    BitmapSubtitleFrame,
    EmbeddedExtractionResult,
    EmbeddedSubtitleTrack,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.infrastructure.video.ffmpeg_embedded_subtitle_adapter import (
    FfmpegEmbeddedSubtitleAdapter,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _build_mkv_with_srt(tmp_path: Path) -> Path:
    srt = tmp_path / "s.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nXin chào thế giới\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nDòng thứ hai\n",
        encoding="utf-8",
    )
    mkv = tmp_path / "v.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=320x240:d=7",
         "-i", str(srt), "-c:v", "libx264", "-c:s", "srt",
         "-map", "0:v", "-map", "1", str(mkv)],
        check=True, capture_output=True,
    )
    return mkv


@pytest.mark.skipif(not _HAS_FFMPEG, reason="Cần ffmpeg/ffprobe")
class TestTextTrackExtraction:
    def test_list_and_extract_text_track(self, tmp_path: Path) -> None:
        mkv = _build_mkv_with_srt(tmp_path)
        adapter = FfmpegEmbeddedSubtitleAdapter()

        tracks = adapter.list_tracks(mkv)
        assert len(tracks) == 1
        assert tracks[0].codec == "subrip"
        assert tracks[0].is_bitmap is False

        result = adapter.extract_track(mkv, tracks[0])
        assert result.is_bitmap is False
        assert [e.text for e in result.events] == ["Xin chào thế giới", "Dòng thứ hai"]
        assert abs(result.events[0].interval.start_sec - 1.0) < 0.05


class _FakeBitmapPort:
    """Cổng giả trả về 2 ảnh phụ đề bitmap."""

    def __init__(self, image_paths: list[Path]) -> None:
        self._image_paths = image_paths

    def list_tracks(self, video_path: Path):
        return [EmbeddedSubtitleTrack(0, "dvd_subtitle", is_bitmap=True)]

    def extract_track(self, video_path: Path, track):
        frames = [
            BitmapSubtitleFrame(self._image_paths[0], 1.0, 3.0),
            BitmapSubtitleFrame(self._image_paths[1], 4.0, 6.0),
        ]
        return EmbeddedExtractionResult(bitmap_frames=frames, is_bitmap=True)


class _FakeOcrEngine:
    """OCR giả: trả 'HELLO' cho ảnh đầu, 'WORLD' cho ảnh sau."""

    def __init__(self) -> None:
        self.is_initialized = False

    def initialize(self) -> None:
        self.is_initialized = True

    def infer(self, image_rgb, frame_index, timestamp_sec):
        text = "HELLO" if frame_index == 0 else "WORLD"
        box = OcrTextBox(text=text, confidence=Confidence(0.9), polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
        return OcrFrameResult(frame_index=frame_index, timestamp_sec=timestamp_sec, text_boxes=[box])

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
        return [
            self.infer(img, idx, ts)
            for img, idx, ts in zip(images_rgb, frame_indices, timestamps_sec, strict=True)
        ]


class TestBitmapTrackOcr:
    def test_bitmap_track_is_ocred_into_events(self, tmp_path: Path) -> None:
        # Dựng 2 ảnh bất kỳ (OCR giả không thực sự đọc nội dung).
        paths = []
        for i in range(2):
            p = tmp_path / f"sub_{i}.png"
            cv2.imwrite(str(p), np.zeros((40, 120, 3), dtype=np.uint8))
            paths.append(p)

        use_case = ExtractEmbeddedSubtitlesUseCase(_FakeBitmapPort(paths), _FakeOcrEngine())
        track = use_case.list_tracks(tmp_path / "x.mkv")[0]
        response = use_case.execute(ExtractEmbeddedRequest(tmp_path / "x.mkv", track))

        assert response.used_ocr is True
        assert [e.text for e in response.events] == ["HELLO", "WORLD"]
        assert response.events[0].interval.start_sec == 1.0
        assert response.events[1].interval.end_sec == 6.0

    def test_cancellation_stops_early(self, tmp_path: Path) -> None:
        paths = []
        for i in range(2):
            p = tmp_path / f"sub_{i}.png"
            cv2.imwrite(str(p), np.zeros((40, 120, 3), dtype=np.uint8))
            paths.append(p)

        use_case = ExtractEmbeddedSubtitlesUseCase(_FakeBitmapPort(paths), _FakeOcrEngine())
        track = use_case.list_tracks(tmp_path / "x.mkv")[0]
        response = use_case.execute(
            ExtractEmbeddedRequest(tmp_path / "x.mkv", track),
            cancellation_check=lambda: True,  # huỷ ngay
        )
        assert response.events == []


class TestBitmapOcrBatching:
    """[v3.23.94] OCR bitmap nhúng phải dùng infer_batch (theo lô) để tận dụng GPU."""

    def test_uses_infer_batch_with_configured_batch_size(self, tmp_path) -> None:
        import numpy as np

        from subtitles_extractor.application.use_cases.extract_embedded_subtitles import (
            ExtractEmbeddedSubtitlesUseCase,
        )
        from subtitles_extractor.domain.ports.embedded_subtitle_port import (
            BitmapSubtitleFrame,
        )

        class _RecordingOcr:
            def __init__(self) -> None:
                self.is_initialized = True
                self.batch_calls: list[int] = []
                self.infer_calls = 0

            def initialize(self) -> None:
                self.is_initialized = True

            def infer(self, image_rgb, frame_index, timestamp_sec):
                self.infer_calls += 1
                raise AssertionError("Phải dùng infer_batch, không infer từng ảnh")

            def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
                self.batch_calls.append(len(images_rgb))
                return [
                    OcrFrameResult(
                        frame_index=i, timestamp_sec=t,
                        text_boxes=[OcrTextBox(text="x", confidence=Confidence(0.9),
                                               polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])],
                    )
                    for i, t in zip(frame_indices, timestamps_sec, strict=True)
                ]

        # 5 ảnh PNG nhỏ
        frames = []
        for i in range(5):
            p = tmp_path / f"f{i}.png"
            cv2.imwrite(str(p), np.full((8, 8, 3), 255, np.uint8))
            frames.append(
                BitmapSubtitleFrame(image_path=p, start_sec=float(i), end_sec=i + 1.0)
            )

        ocr = _RecordingOcr()
        uc = ExtractEmbeddedSubtitlesUseCase(
            embedded_port=object(), ocr_engine=ocr, ocr_batch_size=2
        )
        events = uc._ocr_bitmap_frames(frames, None, None)

        assert ocr.infer_calls == 0  # KHÔNG gọi infer từng ảnh
        assert ocr.batch_calls == [2, 2, 1]  # 5 ảnh, lô 2 -> 3 lần gọi batch
        assert len(events) == 5

