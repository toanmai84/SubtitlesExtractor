"""Test giải mã âm thanh + đồng bộ A/V của trình phát PyAV — v3.23.310.

Tự sinh video CÓ TIẾNG (sóng sin 440Hz) nên chạy được ở mọi máy, không cần tệp ngoài
và KHÔNG cần thiết bị âm thanh thật (dùng thiết bị giả qua :class:`AudioSinkPort`).

Bao phủ:
    * Giải mã + tái lấy mẫu: số mẫu khớp thời lượng, tần số giữ nguyên (kiểm bằng FFT).
    * Đồng bộ: âm thanh làm ĐỒNG HỒ CHỦ, hình bám theo.
    * Seek đồng bộ lại tiếng; đổi tốc độ thì quay về đồng hồ hệ thống.
    * Tụt xa (máy khựng) → nhảy thẳng thay vì tụt lại vĩnh viễn.
    * Video câm vẫn phát bình thường.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av", reason="Cần PyAV để kiểm thử giải mã")

from subtitles_extractor.infrastructure.video.audio_sink_port import (  # noqa: E402
    AudioSinkPort,
)
from subtitles_extractor.infrastructure.video.pyav_audio_decoder import (  # noqa: E402
    PyAvAudioDecoder,
)
from subtitles_extractor.infrastructure.video.pyav_player_adapter import (  # noqa: E402
    PyAvPlayerAdapter,
)

_WIDTH, _HEIGHT, _FPS = 320, 180, 25
_FRAME_COUNT = 100
_SAMPLE_RATE, _CHANNELS = 48_000, 2
_TONE_HZ = 440.0
_BLOCK_WIDTH = _WIDTH // 8


def _draw_frame(index: int) -> np.ndarray:
    """Vẽ chỉ số frame bằng 8 khối đen/trắng (miễn nhiễm nén mất dữ liệu)."""
    image = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    for bit in range(8):
        if (index >> bit) & 1:
            image[:, bit * _BLOCK_WIDTH : (bit + 1) * _BLOCK_WIDTH, :] = 255
    return image


def _read_frame_index(image: np.ndarray) -> int:
    """Đọc lại chỉ số frame từ ảnh."""
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
def video_with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Video 4 giây kèm sóng sin 440Hz stereo 48kHz."""
    path = tmp_path_factory.mktemp("av") / "with_audio.mp4"
    container = av.open(str(path), "w")

    video_stream = container.add_stream("mpeg4", rate=_FPS)
    video_stream.width, video_stream.height = _WIDTH, _HEIGHT
    video_stream.pix_fmt = "yuv420p"
    audio_stream = container.add_stream("aac", rate=_SAMPLE_RATE)
    audio_stream.layout = "stereo"

    video_packets = []
    for index in range(_FRAME_COUNT):
        frame = av.VideoFrame.from_ndarray(_draw_frame(index), format="rgb24")
        video_packets.extend(video_stream.encode(frame))
    video_packets.extend(video_stream.encode())

    audio_packets = []
    total_samples = int(_SAMPLE_RATE * _FRAME_COUNT / _FPS)
    chunk = 1024
    offset = 0
    while offset < total_samples:
        count = min(chunk, total_samples - offset)
        times = (np.arange(offset, offset + count) / _SAMPLE_RATE).astype(np.float32)
        wave = (0.3 * np.sin(2 * np.pi * _TONE_HZ * times)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(
            np.stack([wave, wave]), format="fltp", layout="stereo"
        )
        frame.sample_rate = _SAMPLE_RATE
        frame.pts = offset
        frame.time_base = Fraction(1, _SAMPLE_RATE)
        audio_packets.extend(audio_stream.encode(frame))
        offset += count
    audio_packets.extend(audio_stream.encode())

    for packet in sorted(
        video_packets + audio_packets,
        key=lambda p: (p.pts or 0) * float(p.time_base or 0),
    ):
        container.mux(packet)
    container.close()
    return path


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Video 4 giây KHÔNG có tiếng."""
    path = tmp_path_factory.mktemp("av") / "silent.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=_FPS)
    stream.width, stream.height, stream.pix_fmt = _WIDTH, _HEIGHT, "yuv420p"
    for index in range(_FRAME_COUNT):
        for packet in stream.encode(
            av.VideoFrame.from_ndarray(_draw_frame(index), format="rgb24")
        ):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


class _FakeAudioSink:
    """Thiết bị âm thanh giả — kiểm soát chính xác 'đã phát bao nhiêu giây'."""

    CAPACITY = _SAMPLE_RATE * _CHANNELS * 2  # đệm 1 giây

    def __init__(self) -> None:
        self.buffer = b""
        self.played = 0.0
        self.volume = 1.0
        self.closed = False
        self.reset_count = 0

    @property
    def free_bytes(self) -> int:
        return max(0, self.CAPACITY - len(self.buffer))

    def write(self, pcm: bytes) -> int:
        accepted = min(len(pcm), self.free_bytes)
        self.buffer += pcm[:accepted]
        return accepted

    def played_seconds(self) -> float:
        return self.played

    def reset(self) -> None:
        self.buffer = b""
        self.played = 0.0
        self.reset_count += 1

    def set_volume(self, volume: float) -> None:
        self.volume = volume

    def close(self) -> None:
        self.closed = True

    def advance(self, seconds: float) -> None:
        """Giả lập thiết bị phát tiếp ``seconds`` giây."""
        self.played += seconds


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


# ── Bộ giải mã âm thanh ──────────────────────────────────────────────────────
def test_fake_sink_satisfies_port() -> None:
    assert isinstance(_FakeAudioSink(), AudioSinkPort)


def test_audio_decoder_opens_and_reports_format(video_with_audio: Path) -> None:
    decoder = PyAvAudioDecoder()
    try:
        assert decoder.open(video_with_audio) is True
        assert decoder.has_audio
        assert decoder.sample_rate == _SAMPLE_RATE
        assert decoder.channels == _CHANNELS
        assert decoder.bytes_per_frame == 4  # 2 kênh × 16-bit
    finally:
        decoder.release()


def test_audio_sample_count_matches_duration(video_with_audio: Path) -> None:
    """Số mẫu phải khớp thời lượng — đây là cơ sở của đồng bộ A/V."""
    decoder = PyAvAudioDecoder()
    try:
        decoder.open(video_with_audio)
        data = b""
        while True:
            chunk = decoder.read_chunk()
            if chunk is None:
                break
            data += chunk
        seconds = (len(data) // decoder.bytes_per_frame) / decoder.sample_rate
        # Bộ mã hoá AAC thêm đệm đầu/cuối nên cho phép lệch ~60ms.
        assert seconds == pytest.approx(_FRAME_COUNT / _FPS, abs=0.06)
    finally:
        decoder.release()


def test_resampling_preserves_tone_frequency(video_with_audio: Path) -> None:
    """Tái lấy mẫu đúng thì tần số phải giữ nguyên (kiểm bằng FFT)."""
    decoder = PyAvAudioDecoder()
    try:
        decoder.open(video_with_audio)
        data = decoder.read_duration(2.0)
        samples = np.frombuffer(data, dtype=np.int16).reshape(-1, _CHANNELS)
        mono = samples.mean(axis=1).astype(np.float64)
        segment = mono[_SAMPLE_RATE // 2 : _SAMPLE_RATE // 2 + 16384]
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        peak_hz = np.fft.rfftfreq(len(segment), 1 / _SAMPLE_RATE)[np.argmax(spectrum)]
        assert peak_hz == pytest.approx(_TONE_HZ, abs=5.0)
    finally:
        decoder.release()


def test_audio_seek_resets_counters(video_with_audio: Path) -> None:
    decoder = PyAvAudioDecoder()
    try:
        decoder.open(video_with_audio)
        decoder.read_duration(1.0)
        decoder.seek(2.0)
        assert decoder.samples_read == 0
        assert not decoder.is_exhausted
        assert decoder.read_chunk() is not None
    finally:
        decoder.release()


def test_silent_video_reports_no_audio(silent_video: Path) -> None:
    """Video câm KHÔNG phải lỗi — trả False và đọc ra None."""
    decoder = PyAvAudioDecoder()
    try:
        assert decoder.open(silent_video) is False
        assert not decoder.has_audio
        assert decoder.read_chunk() is None
    finally:
        decoder.release()


# ── Đồng bộ A/V ──────────────────────────────────────────────────────────────
def test_audio_is_master_clock(video_with_audio: Path) -> None:
    """Hình phải bám theo TIẾNG, không theo đồng hồ hệ thống."""
    sink, clock = _FakeAudioSink(), _FakeClock()
    player = PyAvPlayerAdapter(clock=clock, audio_sink=sink)
    try:
        player.load(video_with_audio)
        assert player.has_audio
        player.play()

        # Đồng hồ hệ thống chạy nhưng thiết bị CHƯA phát -> hình phải đứng yên.
        clock.advance(2.0)
        player.tick()
        assert player.position_sec < 0.1

        # Thiết bị phát 1 giây -> hình phải theo kịp.
        sink.advance(1.0)
        player.tick()
        assert player.position_sec == pytest.approx(1.0, abs=0.06)
    finally:
        player.release()


def test_sync_drift_stays_within_one_frame(video_with_audio: Path) -> None:
    """Phát thực tế (nhịp 4ms): lệch hình–tiếng không vượt quá 1 frame."""
    sink, clock = _FakeAudioSink(), _FakeClock()
    player = PyAvPlayerAdapter(clock=clock, audio_sink=sink)
    try:
        player.load(video_with_audio)
        player.play()
        worst_drift = 0.0
        for _ in range(250):  # 1 giây
            clock.advance(0.004)
            sink.advance(0.004)
            player.tick()
            worst_drift = max(
                worst_drift, abs(sink.played_seconds() - player.position_sec)
            )
        assert worst_drift <= 1.0 / _FPS + 0.005
    finally:
        player.release()


def test_seek_resyncs_audio(video_with_audio: Path) -> None:
    sink, clock = _FakeAudioSink(), _FakeClock()
    player = PyAvPlayerAdapter(clock=clock, audio_sink=sink)
    try:
        player.load(video_with_audio)
        player.play()
        before = sink.reset_count
        player.seek(2.5)
        assert sink.reset_count > before  # thiết bị được xoá đệm
        assert sink.played_seconds() == 0.0

        anchor = player.position_sec
        for _ in range(125):  # thiết bị phát 0.5s
            clock.advance(0.004)
            sink.advance(0.004)
            player.tick()
        assert player.position_sec == pytest.approx(anchor + 0.5, abs=0.05)
    finally:
        player.release()


def test_large_lag_triggers_seek_instead_of_falling_behind(
    video_with_audio: Path,
) -> None:
    """Thiết bị nhảy xa (máy khựng/ngủ dậy) -> hình nhảy thẳng, không tụt lại."""
    sink, clock = _FakeAudioSink(), _FakeClock()
    player = PyAvPlayerAdapter(clock=clock, audio_sink=sink)
    try:
        player.load(video_with_audio)
        player.play()
        sink.advance(2.0)  # nhảy 2 giây trong MỘT nhịp
        player.tick()
        assert player.position_sec == pytest.approx(2.0, abs=0.06)
    finally:
        player.release()


def test_speed_change_falls_back_to_wall_clock(video_with_audio: Path) -> None:
    """``speed != 1.0`` -> tắt tiếng, dùng đồng hồ hệ thống (giới hạn đã biết)."""
    sink, clock = _FakeAudioSink(), _FakeClock()
    player = PyAvPlayerAdapter(clock=clock, audio_sink=sink)
    try:
        player.load(video_with_audio)
        player.set_speed(2.0)
        player.play()
        for _ in range(10):
            clock.advance(0.04)  # thiết bị KHÔNG phát
            player.tick()
        assert player.position_sec == pytest.approx(0.8, abs=0.06)
    finally:
        player.release()


def test_volume_and_mute_reach_the_device(video_with_audio: Path) -> None:
    sink = _FakeAudioSink()
    player = PyAvPlayerAdapter(audio_sink=sink)
    try:
        player.load(video_with_audio)
        player.set_volume(50)
        assert sink.volume == pytest.approx(0.5)
        player.set_mute(True)
        assert sink.volume == pytest.approx(0.0)
        player.set_mute(False)
        assert sink.volume == pytest.approx(0.5)
    finally:
        player.release()


def test_silent_video_uses_wall_clock(silent_video: Path) -> None:
    """Video câm vẫn phát bình thường bằng đồng hồ hệ thống."""
    sink, clock = _FakeAudioSink(), _FakeClock()
    received: list[int] = []
    player = PyAvPlayerAdapter(
        on_frame=lambda img: received.append(_read_frame_index(img)),
        clock=clock,
        audio_sink=sink,
    )
    try:
        player.load(silent_video)
        assert not player.has_audio
        received.clear()
        player.play()
        for _ in range(10):
            clock.advance(1 / _FPS)
            player.tick()
        assert received == list(range(1, 11))
    finally:
        player.release()


def test_release_closes_audio_device(video_with_audio: Path) -> None:
    sink = _FakeAudioSink()
    player = PyAvPlayerAdapter(audio_sink=sink)
    player.load(video_with_audio)
    player.release()
    assert sink.closed
    player.release()  # idempotent


def test_audio_buffer_is_pumped_on_tick(video_with_audio: Path) -> None:
    """Nhịp đầu tiên phải bơm dữ liệu xuống thiết bị (nếu không sẽ im tiếng)."""
    sink, clock = _FakeAudioSink(), _FakeClock()
    player = PyAvPlayerAdapter(clock=clock, audio_sink=sink)
    try:
        player.load(video_with_audio)
        assert len(sink.buffer) == 0
        player.play()
        player.tick()
        assert len(sink.buffer) > 0
    finally:
        player.release()
