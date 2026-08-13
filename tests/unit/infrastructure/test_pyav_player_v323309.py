"""Test trình phát video PyAV (thay libmpv GPL) — v3.23.309.

Bộ test TỰ SINH video mẫu bằng PyAV nên chạy được ở mọi máy, không cần tệp ngoài.

Chỉ số frame được mã hoá bằng **8 khối đen/trắng lớn** (nhị phân, ngưỡng 128) thay vì
giá trị điểm ảnh, vì codec nén mất dữ liệu làm sai lệch giá trị điểm ảnh tới ±3 —
cách này miễn nhiễm cả nén lẫn làm tròn RGB↔YUV.

Bao phủ:
    * Seek chính xác tới frame (CFR và **VFR** — VFR là yêu cầu bắt buộc của dự án).
    * Bước frame tiến/lùi, khứ hồi về đúng vị trí.
    * Trường hợp biên: đầu video, vượt quá cuối, hết video, release lặp.
    * Adapter: tuân thủ hợp đồng ``VideoPlayerPort``, nhịp phát, tốc độ, tự dừng.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av", reason="Cần PyAV để kiểm thử giải mã video")

from subtitles_extractor.domain.ports.video_player_port import (  # noqa: E402
    VideoPlayerPort,
)
from subtitles_extractor.infrastructure.video.pyav_frame_decoder import (  # noqa: E402
    PyAvFrameDecoder,
)
from subtitles_extractor.infrastructure.video.pyav_player_adapter import (  # noqa: E402
    PyAvPlayerAdapter,
)

_WIDTH = 320
_HEIGHT = 180
_FPS = 25
_FRAME_COUNT = 100
_BLOCK_WIDTH = _WIDTH // 8


def _draw_frame(index: int) -> np.ndarray:
    """Vẽ chỉ số frame dạng nhị phân 8 bit bằng khối đen/trắng lớn."""
    image = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    for bit in range(8):
        if (index >> bit) & 1:
            image[:, bit * _BLOCK_WIDTH : (bit + 1) * _BLOCK_WIDTH, :] = 255
    return image


def _read_frame_index(image: np.ndarray) -> int:
    """Đọc lại chỉ số frame từ ảnh (ngược với :func:`_draw_frame`)."""
    value = 0
    for bit in range(8):
        block = image[
            _HEIGHT // 4 : 3 * _HEIGHT // 4,
            bit * _BLOCK_WIDTH + 8 : (bit + 1) * _BLOCK_WIDTH - 8,
            :,
        ]
        if block.mean() > 128:
            value |= 1 << bit
    return value


@pytest.fixture(scope="module")
def cfr_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Video tốc độ khung hình CỐ ĐỊNH (25 fps)."""
    path = tmp_path_factory.mktemp("video") / "cfr.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=_FPS)
    stream.width, stream.height, stream.pix_fmt = _WIDTH, _HEIGHT, "yuv420p"
    for index in range(_FRAME_COUNT):
        frame = av.VideoFrame.from_ndarray(_draw_frame(index), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture(scope="module")
def vfr_video(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[int, float]]:
    """Video BIẾN THIÊN tốc độ khung hình + bảng mốc thời gian thật của từng frame."""
    path = tmp_path_factory.mktemp("video") / "vfr.mkv"
    time_base = Fraction(1, 1000)
    gaps_ms = [40, 40, 120, 200, 40, 80, 300, 40, 160, 40] * 4

    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=Fraction(1000, 1))
    stream.width, stream.height, stream.pix_fmt = _WIDTH, _HEIGHT, "yuv420p"
    stream.time_base = time_base

    pts = 0
    truth: dict[int, float] = {}
    for index, gap in enumerate(gaps_ms):
        frame = av.VideoFrame.from_ndarray(_draw_frame(index), format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        truth[index] = pts / 1000.0
        for packet in stream.encode(frame):
            container.mux(packet)
        pts += gap
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path, truth


class _FakeClock:
    """Đồng hồ giả để kiểm soát thời gian trong test."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


# ── Decoder: CFR ─────────────────────────────────────────────────────────────
def test_open_reads_metadata(cfr_video: Path) -> None:
    decoder = PyAvFrameDecoder()
    decoder.open(cfr_video)
    try:
        assert decoder.is_loaded
        assert decoder.video_size == (_WIDTH, _HEIGHT)
        assert decoder.fps == pytest.approx(_FPS, abs=0.01)
        assert decoder.duration_sec == pytest.approx(_FRAME_COUNT / _FPS, abs=0.1)
    finally:
        decoder.release()


@pytest.mark.parametrize("seconds", [0.0, 0.04, 0.4, 1.0, 2.04, 3.0, 3.5])
def test_seek_is_frame_accurate(cfr_video: Path, seconds: float) -> None:
    """Seek phải cho ra frame ĐẦU TIÊN có ``PTS >= mốc đích``."""
    decoder = PyAvFrameDecoder()
    decoder.open(cfr_video)
    try:
        image = decoder.seek(seconds)
        assert image is not None
        assert _read_frame_index(image) == round(seconds * _FPS)
        assert decoder.position_sec >= seconds - 1e-6
    finally:
        decoder.release()


def test_step_forward_and_backward_round_trip(cfr_video: Path) -> None:
    """Tiến N frame rồi lùi N frame phải quay về đúng frame ban đầu."""
    decoder = PyAvFrameDecoder()
    decoder.open(cfr_video)
    try:
        decoder.seek(2.0)
        start = _read_frame_index(decoder.current_image())

        forward = []
        for _ in range(5):
            image = decoder.next_frame()
            assert image is not None
            forward.append(_read_frame_index(image))
        assert forward == list(range(start + 1, start + 6))

        for _ in range(5):
            image = decoder.previous_frame()
            assert image is not None
        assert _read_frame_index(decoder.current_image()) == start
    finally:
        decoder.release()


def test_previous_frame_at_start_returns_none(cfr_video: Path) -> None:
    """Ở frame đầu, lùi phải trả ``None`` và KHÔNG làm trôi vị trí."""
    decoder = PyAvFrameDecoder()
    decoder.open(cfr_video)
    try:
        decoder.seek(0.0)
        assert decoder.previous_frame() is None
        assert _read_frame_index(decoder.current_image()) == 0
    finally:
        decoder.release()


def test_seek_beyond_end_is_clamped(cfr_video: Path) -> None:
    decoder = PyAvFrameDecoder()
    decoder.open(cfr_video)
    try:
        decoder.seek(9999.0)
        assert decoder.position_sec <= decoder.duration_sec + 1e-6
        assert decoder.current_image() is not None
    finally:
        decoder.release()


def test_next_frame_returns_none_at_eof(cfr_video: Path) -> None:
    decoder = PyAvFrameDecoder()
    decoder.open(cfr_video)
    try:
        decoder.seek(decoder.duration_sec)
        for _ in range(10):
            if decoder.next_frame() is None:
                break
        else:
            pytest.fail("next_frame không bao giờ trả None ở cuối video")
    finally:
        decoder.release()


def test_release_is_idempotent(cfr_video: Path) -> None:
    decoder = PyAvFrameDecoder()
    decoder.open(cfr_video)
    decoder.release()
    decoder.release()
    assert not decoder.is_loaded


def test_open_missing_file_raises() -> None:
    from subtitles_extractor.domain.exceptions import VideoNotFoundError

    with pytest.raises(VideoNotFoundError):
        PyAvFrameDecoder().open(Path("khong-ton-tai-12345.mp4"))


# ── Decoder: VFR (yêu cầu BẮT BUỘC của dự án) ────────────────────────────────
@pytest.mark.parametrize("seconds", [0.0, 0.15, 0.5, 1.0, 1.5, 2.0])
def test_seek_on_vfr_matches_real_pts(
    vfr_video: tuple[Path, dict[int, float]], seconds: float
) -> None:
    """Trên video VFR, mốc trả về phải khớp PTS THẬT của frame đó.

    Đây là phép thử then chốt: nếu dùng ``frame_idx / fps`` sẽ sai hàng trăm ms.
    """
    path, truth = vfr_video
    decoder = PyAvFrameDecoder()
    decoder.open(path)
    try:
        image = decoder.seek(seconds)
        assert image is not None
        index = _read_frame_index(image)
        assert decoder.position_sec == pytest.approx(truth[index], abs=0.002)
        assert decoder.position_sec >= seconds - 1e-3
    finally:
        decoder.release()


def test_step_frame_on_vfr(vfr_video: tuple[Path, dict[int, float]]) -> None:
    path, _ = vfr_video
    decoder = PyAvFrameDecoder()
    decoder.open(path)
    try:
        decoder.seek(1.0)
        start = _read_frame_index(decoder.current_image())
        for _ in range(4):
            assert decoder.next_frame() is not None
        assert _read_frame_index(decoder.current_image()) == start + 4
        for _ in range(4):
            assert decoder.previous_frame() is not None
        assert _read_frame_index(decoder.current_image()) == start
    finally:
        decoder.release()


# ── Adapter ──────────────────────────────────────────────────────────────────
def test_adapter_satisfies_port(cfr_video: Path) -> None:
    """Adapter phải thoả hợp đồng ``VideoPlayerPort`` (protocol runtime-checkable)."""
    player = PyAvPlayerAdapter()
    try:
        player.load(cfr_video)
        assert isinstance(player, VideoPlayerPort)
    finally:
        player.release()


def test_load_does_not_autoplay(cfr_video: Path) -> None:
    """``load`` KHÔNG được tự phát (đúng hợp đồng port)."""
    received: list[int] = []
    player = PyAvPlayerAdapter(on_frame=lambda img: received.append(_read_frame_index(img)))
    try:
        player.load(cfr_video)
        assert player.is_loaded
        assert not player.is_playing
        assert received == [0]  # chỉ gửi frame đầu để hiển thị
    finally:
        player.release()


def test_playback_advances_at_correct_rate(cfr_video: Path) -> None:
    """Phát 400ms ở 25fps phải ra đúng 10 frame liên tiếp."""
    received: list[int] = []
    clock = _FakeClock()
    player = PyAvPlayerAdapter(
        on_frame=lambda img: received.append(_read_frame_index(img)), clock=clock
    )
    try:
        player.load(cfr_video)
        received.clear()
        player.play()
        for _ in range(10):
            clock.advance(1 / _FPS)
            player.tick()
        assert received == list(range(1, 11))
        assert player.position_sec == pytest.approx(0.4, abs=0.02)
    finally:
        player.release()


def test_speed_doubles_advance_rate(cfr_video: Path) -> None:
    received: list[int] = []
    clock = _FakeClock()
    player = PyAvPlayerAdapter(
        on_frame=lambda img: received.append(_read_frame_index(img)), clock=clock
    )
    try:
        player.load(cfr_video)
        player.set_speed(2.0)
        received.clear()
        player.play()
        for _ in range(10):
            clock.advance(1 / _FPS)
            player.tick()
        assert 18 <= len(received) <= 22  # ~20 frame, cho phép sai số neo thời gian
    finally:
        player.release()


def test_pause_freezes_position(cfr_video: Path) -> None:
    clock = _FakeClock()
    player = PyAvPlayerAdapter(clock=clock)
    try:
        player.load(cfr_video)
        player.play()
        clock.advance(0.2)
        player.tick()
        player.pause()
        frozen = player.position_sec
        for _ in range(5):
            clock.advance(0.1)
            player.tick()
        assert player.position_sec == frozen
    finally:
        player.release()


def test_step_frame_pauses_and_moves_exactly_one(cfr_video: Path) -> None:
    received: list[int] = []
    player = PyAvPlayerAdapter(on_frame=lambda img: received.append(_read_frame_index(img)))
    try:
        player.load(cfr_video)
        player.seek(2.0)
        start = received[-1]
        player.play()
        player.step_frame(forward=True)
        assert not player.is_playing  # bước frame luôn dừng phát
        assert received[-1] == start + 1
        player.step_frame(forward=False)
        assert received[-1] == start
    finally:
        player.release()


def test_playback_stops_at_end(cfr_video: Path) -> None:
    clock = _FakeClock()
    player = PyAvPlayerAdapter(clock=clock)
    try:
        player.load(cfr_video)
        player.seek(player.duration_sec - 0.1)
        player.play()
        for _ in range(30):
            clock.advance(1 / _FPS)
            player.tick()
        assert not player.is_playing
        assert player.eof_reached
    finally:
        player.release()


def test_toggle_play_pause(cfr_video: Path) -> None:
    player = PyAvPlayerAdapter()
    try:
        player.load(cfr_video)
        assert not player.is_playing
        player.toggle_play_pause()
        assert player.is_playing
        player.toggle_play_pause()
        assert not player.is_playing
    finally:
        player.release()


def test_set_volume_and_speed_are_clamped(cfr_video: Path) -> None:
    player = PyAvPlayerAdapter()
    try:
        player.load(cfr_video)
        player.set_volume(500)
        assert player._volume == 100
        player.set_volume(-10)
        assert player._volume == 0
        player.set_speed(99.0)
        assert player._speed == 4.0
        player.set_speed(0.01)
        assert player._speed == 0.25
    finally:
        player.release()


def test_adapter_release_is_idempotent(cfr_video: Path) -> None:
    player = PyAvPlayerAdapter()
    player.load(cfr_video)
    player.release()
    player.release()
    assert not player.is_loaded


def test_frame_callback_error_does_not_break_playback(cfr_video: Path) -> None:
    """Lỗi từ tầng UI không được làm hỏng vòng phát."""

    def broken(_: np.ndarray) -> None:
        raise RuntimeError("lỗi UI giả lập")

    clock = _FakeClock()
    player = PyAvPlayerAdapter(on_frame=broken, clock=clock)
    try:
        player.load(cfr_video)  # không được ném lỗi
        player.play()
        for _ in range(3):
            clock.advance(1 / _FPS)
            player.tick()
        assert player.position_sec > 0
    finally:
        player.release()
