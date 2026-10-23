"""[v3.23.615] Bộ lấy mẫu khung không được bỏ khung vì sai số làm tròn mốc thời gian.

Video thật 14/10 (phim "天骄女友退婚" tập 1: HEVC 25fps, 4421 khung, time_base 1/12800),
bước lấy mẫu 0.04s (đúng một khung): tệp OCR thô chỉ có 3700 khung — bộ lấy mẫu bỏ 721
khung (16%), đều đặn 1 trong 2-3 khung từ giây 64 trở đi. Mốc kế tiếp được đặt bằng
``mốc khung vừa lấy + 0.04``; ``pts * time_base`` của khung sau có lúc nhỏ hơn mốc đó một
phần tỉ giây, phép so ``<`` bỏ oan khung. Chạy lại bộ lấy mẫu PyAV của bản 3.23.614 trên
chính video đó cho đúng 3700 khung; bộ OpenCV lấy trùng 22 khung và bỏ 22 khung khác.
Đo bằng điểm ảnh vùng phụ đề: 14/15 câu bắt đầu trễ 40ms và 20/22 câu kết thúc sớm thêm
40ms rơi đúng vào khung bị bỏ.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    is_sample_due,
)

av = pytest.importorskip("av")

FPS = 25
TIME_BASE = Fraction(1, 12800)
PTS_PER_FRAME = 512
#: Đủ dài để vượt mốc giây 64, nơi sai số làm tròn bắt đầu làm rơi khung.
FRAME_COUNT = 1760


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Video 25fps, time_base 1/12800 như video thật; mỗi khung một màu khác nhau."""
    path = tmp_path_factory.mktemp("sampler") / "clip.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=FPS)
    stream.width = stream.height = 32
    stream.pix_fmt = "yuv420p"
    stream.time_base = TIME_BASE
    for index in range(FRAME_COUNT):
        image = np.full((32, 32, 3), (index * 37) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = index * PTS_PER_FRAME
        frame.time_base = TIME_BASE
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def _metadata(path: Path):
    from subtitles_extractor.infrastructure.video.pyav_metadata_reader import (
        PyAvMetadataReader,
    )

    return PyAvMetadataReader().read(path)


def _timestamps(sampler, path: Path) -> list[float]:
    config = FrameSamplingConfig(sample_step_sec=1.0 / FPS)
    frames = sampler.iter_frames(_metadata(path), None, config)
    return [frame.timestamp_sec for frame in frames]


def test_rounding_shortfall_still_counts_as_due() -> None:
    previous = float(1601 * PTS_PER_FRAME * TIME_BASE)  # 64.04s
    current = float(1602 * PTS_PER_FRAME * TIME_BASE)  # 64.08s
    assert current < previous + 0.04  # đúng ca của video thật: hụt do làm tròn
    assert is_sample_due(current, previous + 0.04)


def test_frame_before_target_is_not_due() -> None:
    assert not is_sample_due(1.0, 1.04)


def test_pyav_sampler_takes_every_frame(video: Path) -> None:
    from subtitles_extractor.infrastructure.video.decoders import PyAvFrameSampler

    stamps = _timestamps(PyAvFrameSampler(), video)
    assert len(stamps) == FRAME_COUNT
    assert np.diff(stamps).max() < 1.5 / FPS


def test_opencv_sampler_takes_every_frame_once(video: Path) -> None:
    pytest.importorskip("cv2")
    from subtitles_extractor.infrastructure.video.decoders import OpenCvFrameSampler

    stamps = _timestamps(OpenCvFrameSampler(), video)
    gaps = np.diff(stamps)
    assert gaps.min() > 0.5 / FPS  # không lấy trùng một khung
    assert gaps.max() < 1.5 / FPS  # không bỏ khung nào


class _FakeMpv:
    """mpv giả: "exact" seek trả khung có mốc ≤ thời điểm tua (như mpv thật)."""

    def __init__(self) -> None:
        self.frame_shown = -1
        self.seeks: list[float] = []

    def command(self, name: str, target: float, *_flags: str) -> None:
        self.seeks.append(target)
        self.frame_shown = int(np.floor(target * FPS + 1e-12))

    def screenshot_raw(self) -> dict[str, object]:
        value = (self.frame_shown * 37) % 256
        return {"w": 4, "h": 4, "stride": 16, "format": "bgr0",
                "data": bytes([value] * 64)}

    @property
    def playback_time(self) -> float:
        return self.frame_shown / FPS


def test_mpv_sampler_seeks_every_frame_without_drift() -> None:
    from types import SimpleNamespace

    from subtitles_extractor.infrastructure.video.decoders import MpvFrameSampler

    fake = _FakeMpv()
    metadata = SimpleNamespace(duration_sec=70.0)
    config = FrameSamplingConfig(sample_step_sec=1.0 / FPS, phash_distance_threshold=-1)
    frames = list(MpvFrameSampler()._iterate(fake, metadata, None, config))
    shown = [round(frame.timestamp_sec * FPS) for frame in frames]
    assert shown == list(range(len(shown)))
