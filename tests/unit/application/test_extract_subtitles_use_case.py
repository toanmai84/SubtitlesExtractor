"""Test :class:`ExtractSubtitlesUseCase` với mock adapter — kiểm tra orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import (
    SubtitleBuilder,
)
from subtitles_extractor.application.use_cases.extract_subtitles import (
    ExtractSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    SampledFrame,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat


def _make_metadata(tmp_path: Path) -> VideoMetadata:
    fake_video = tmp_path / "fake.mp4"
    fake_video.touch()
    return VideoMetadata(
        path=fake_video,
        width=1920, height=1080,
        fps=30.0, total_frames=300, duration_sec=10.0,
    )


def _make_sampled_frame(idx: int, ts: float) -> SampledFrame:
    return SampledFrame(
        frame_index=idx,
        timestamp_sec=ts,
        image_rgb=np.zeros((50, 200, 3), dtype=np.uint8),
    )


def _make_ocr_result(idx: int, ts: float, text: str) -> OcrFrameResult:
    return OcrFrameResult(
        frame_index=idx,
        timestamp_sec=ts,
        text_boxes=[
            OcrTextBox(
                text=text,
                confidence=Confidence(0.9),
                polygon=[(0, 10), (100, 10), (100, 40), (0, 40)],
            )
        ],
    )


@pytest.fixture
def metadata_reader(tmp_path: Path) -> MagicMock:
    reader = MagicMock()
    reader.read.return_value = _make_metadata(tmp_path)
    return reader


@pytest.fixture
def frame_sampler() -> MagicMock:
    sampler = MagicMock()

    def _iter(*_args, **_kwargs) -> Iterator[SampledFrame]:
        yield _make_sampled_frame(0, 0.0)
        yield _make_sampled_frame(1, 0.1)
        yield _make_sampled_frame(2, 0.2)

    sampler.iter_frames.side_effect = _iter
    return sampler


@pytest.fixture
def ocr_engine() -> MagicMock:
    engine = MagicMock()
    engine.is_initialized = False

    def _infer_batch(images_rgb, frame_indices, timestamps_sec):
        return [
            _make_ocr_result(idx, ts, "Xin chào")
            for idx, ts in zip(frame_indices, timestamps_sec, strict=True)
        ]

    engine.infer_batch.side_effect = _infer_batch
    return engine


@pytest.fixture
def srt_exporter() -> MagicMock:
    exporter = MagicMock()
    exporter.export.side_effect = lambda events, path: path
    return exporter


def test_use_case_runs_full_pipeline(
    metadata_reader: MagicMock,
    frame_sampler: MagicMock,
    ocr_engine: MagicMock,
    srt_exporter: MagicMock,
    tmp_path: Path,
) -> None:
    use_case = ExtractSubtitlesUseCase(
        metadata_reader=metadata_reader,
        frame_sampler=frame_sampler,
        ocr_engine=ocr_engine,
        builder=SubtitleBuilder(SubtitleBuilderConfig(min_duration_sec=0.0)),
        exporters={"srt": srt_exporter},
    )
    metadata = metadata_reader.read.return_value
    request = ExtractSubtitlesRequest(
        video_path=metadata.path,
        output_path=tmp_path / "out.srt",
        output_format=SubtitleFormat.SRT,
        sampling=FrameSamplingConfig(),
        ocr=OcrEngineConfig(batch_size=4),
    )
    response = use_case.execute(request)
    assert response.frames_processed == 3
    assert len(response.events) == 1  # 3 frame cùng câu → gộp
    ocr_engine.initialize.assert_called_once()
    srt_exporter.export.assert_called_once()


def test_use_case_skips_init_if_already_initialized(
    metadata_reader: MagicMock,
    frame_sampler: MagicMock,
    ocr_engine: MagicMock,
    srt_exporter: MagicMock,
    tmp_path: Path,
) -> None:
    ocr_engine.is_initialized = True
    use_case = ExtractSubtitlesUseCase(
        metadata_reader=metadata_reader,
        frame_sampler=frame_sampler,
        ocr_engine=ocr_engine,
        builder=SubtitleBuilder(SubtitleBuilderConfig(min_duration_sec=0.0)),
        exporters={"srt": srt_exporter},
    )
    metadata = metadata_reader.read.return_value
    request = ExtractSubtitlesRequest(
        video_path=metadata.path,
        output_path=tmp_path / "out.srt",
    )
    use_case.execute(request)
    ocr_engine.initialize.assert_not_called()


def test_use_case_raises_for_unknown_format(
    metadata_reader: MagicMock,
    frame_sampler: MagicMock,
    ocr_engine: MagicMock,
    srt_exporter: MagicMock,
    tmp_path: Path,
) -> None:
    use_case = ExtractSubtitlesUseCase(
        metadata_reader=metadata_reader,
        frame_sampler=frame_sampler,
        ocr_engine=ocr_engine,
        builder=SubtitleBuilder(SubtitleBuilderConfig()),
        exporters={"srt": srt_exporter},  # KHÔNG đăng ký ASS
    )
    metadata = metadata_reader.read.return_value
    request = ExtractSubtitlesRequest(
        video_path=metadata.path,
        output_path=tmp_path / "out.ass",
        output_format=SubtitleFormat.ASS,
    )
    with pytest.raises(KeyError):
        use_case.execute(request)
