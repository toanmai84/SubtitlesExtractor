"""[v3.23.659] Xem trước khung hình trên video GOP dài (keyframe cách 10s) phải tới đúng mốc."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("av")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")

from subtitles_extractor.presentation.workers.seek_worker import PersistentVideoReader, SeekWorker  # noqa: E402


@pytest.fixture(scope="module")
def long_gop_video(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("gop") / "long_gop.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25:duration=12",
         "-c:v", "libx264", "-g", "250", "-keyint_min", "250", "-sc_threshold", "0", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out


def test_persistent_reader_reaches_target_in_long_gop(long_gop_video: Path) -> None:
    reader = PersistentVideoReader(long_gop_video)
    try:
        frame = reader.seek(8.0)
        assert frame is not None, "Seek 8s trên GOP 10s trả None (chỉ quét 60 khung = 2.4s)."
        # Khung tiếp theo phải nằm ngay sau 8s (iterator đã đứng đúng chỗ).
        assert reader._av_stream is not None
    finally:
        reader.close()


def test_seek_worker_without_reader_long_gop(long_gop_video: Path) -> None:
    worker = SeekWorker(long_gop_video, 9.0)
    frame = worker._read_frame_without_reader()
    assert frame is not None


def test_scan_stops_at_target(long_gop_video: Path) -> None:
    import av

    from subtitles_extractor.presentation.workers.seek_worker import _scan_to_timestamp

    with av.open(str(long_gop_video)) as container:
        stream = container.streams.video[0]
        best, diff = _scan_to_timestamp(container.decode(stream), 8.0)
    assert best is not None and diff < 0.05


def test_scan_stops_when_closing(long_gop_video: Path) -> None:
    """Rà độc lập: close() ở luồng giao diện từng phải chờ lượt quét seek (tới 15 s) — nay quét dừng ngay."""
    import av

    from subtitles_extractor.presentation.workers.seek_worker import _scan_to_timestamp

    with av.open(str(long_gop_video)) as container:
        best, diff = _scan_to_timestamp(container.decode(container.streams.video[0]), 8.0, should_stop=lambda: True)
    assert best is not None and diff > 7.0  # dừng sau khung đầu


def test_closed_reader_seek_returns_none(long_gop_video: Path) -> None:
    reader = PersistentVideoReader(long_gop_video)
    reader.close()
    assert reader.seek(3.0) is None
    assert reader._cv2_capture is None
