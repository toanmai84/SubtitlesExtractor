"""Các adapter giải mã video — 4 backend song song."""

from __future__ import annotations

from subtitles_extractor.infrastructure.video.decoders.mpv_frame_sampler import (
    MpvFrameSampler,
)
from subtitles_extractor.infrastructure.video.decoders.opencv_frame_sampler import (
    OpenCvFrameSampler,
)
from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import (
    PyAvFrameSampler,
)
from subtitles_extractor.infrastructure.video.decoders.pynvvideocodec_frame_sampler import (
    PyNvVideoCodecFrameSampler,
)

__all__ = [
    "MpvFrameSampler",
    "OpenCvFrameSampler",
    "PyAvFrameSampler",
    "PyNvVideoCodecFrameSampler",
]
