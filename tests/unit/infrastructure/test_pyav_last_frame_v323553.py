"""[v3.23.553] Bộ giải mã PyAV không được làm rơi frame khi giải mã tuần tự.

Lỗi: mỗi lần ``_decode_next_frame`` tạo generator ``container.decode()`` MỚI rồi bỏ sau
frame đầu. Packet giải ra nhiều frame (packet flush cuối tệp) thì phần còn lại mất theo
generator → frame cuối không bao giờ hiện, trình phát không bao giờ tới EOF. Lỗi bị che
nhiều phiên bản vì môi trường test thiếu PyAV nên cả bộ test bị bỏ qua.
"""

from __future__ import annotations

from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from subtitles_extractor.infrastructure.video.pyav_frame_decoder import (  # noqa: E402
    PyAvFrameDecoder,
)

_FPS = 25
_FRAME_COUNT = 50


@pytest.fixture(scope="module")
def short_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "short.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=_FPS)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for index in range(_FRAME_COUNT):
        image = np.full((48, 64, 3), index * 4 % 256, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def test_sequential_decode_yields_every_frame(short_video: Path) -> None:
    decoder = PyAvFrameDecoder()
    try:
        decoder.open(short_video)
        decoded = 1  # open() đã giải mã frame đầu
        while decoder.next_frame() is not None:
            decoded += 1
        assert decoded == _FRAME_COUNT
        last_pts = (_FRAME_COUNT - 1) / _FPS
        assert decoder.position_sec == pytest.approx(last_pts, abs=1e-6)
    finally:
        decoder.release()


def test_decode_after_seek_reaches_last_frame(short_video: Path) -> None:
    decoder = PyAvFrameDecoder()
    try:
        decoder.open(short_video)
        decoder.seek(decoder.duration_sec - 0.1)
        while decoder.next_frame() is not None:
            pass
        assert decoder.position_sec == pytest.approx((_FRAME_COUNT - 1) / _FPS, abs=1e-6)
    finally:
        decoder.release()
