"""[v3.23.610] Âm thanh của đoạn video ngữ cảnh gửi Gemini: đúng giờ và rõ lời thoại.

Học từ ``extract_audio_from_video`` của gemini-srt-translator:

* ``aresample=async=1`` — "đồng bộ âm thanh theo mốc thời gian, chống lệch dần". Đo bằng
  ffmpeg thật: nguồn có khoảng hở 2 giây trong luồng âm thanh (thường gặp ở bản ghi TS/
  IPTV) thì đoạn cắt ra có mọi âm thanh SAU khoảng hở sớm 2 giây so với hình — Gemini
  nghe câu thoại trước khi thấy người nói. Thêm ``first_pts=0`` để âm thanh bắt đầu đúng
  0 s kể cả khi đoạn cắt rơi vào giữa khoảng hở (bộ đọc bỏ qua edit list MP4 vẫn lệch).
* Âm thanh 5.1: lời thoại nằm ở kênh giữa (FC); trộn mono mặc định cộng dồn mọi kênh nhạc/
  hiệu ứng. Trộn ưu tiên kênh giữa (giữ ít nhạc trước để còn ngữ cảnh) tăng tỉ lệ lời/nền
  khoảng 9 dB trên nguồn thử và hết bị chạm trần 0 dBFS.
* Chọn luồng âm thanh MẶC ĐỊNH thay vì luôn luồng đầu tiên (phim nhiều luồng tiếng).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.domain.ports.video_context_port import VideoChunk
from subtitles_extractor.infrastructure.translation.context_audio import (
    AudioStreamInfo,
    context_audio_args,
    select_context_audio,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="Cần ffmpeg trong PATH.")
SR = 16000


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG or "ffmpeg", "-nostdin", "-y", "-hide_banner",
                    "-loglevel", "error", *args], check=True)


def _decode_mono(path: Path) -> np.ndarray:
    raw = subprocess.run(
        [FFMPEG or "ffmpeg", "-v", "error", "-ignore_editlist", "1", "-i", str(path),
         "-vn", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.int16).astype(np.float64) / 32768.0


def _chunk_from(source: Path, tmp_path: Path, start: float, duration: float) -> Path:
    """Chạy đúng lệnh nén CPU của ứng dụng cho một đoạn."""
    provider = GeminiVideoContextProvider(work_dir=tmp_path)
    provider._nvenc_available = False
    out = tmp_path / f"chunk_{start:g}.mp4"
    chunk = VideoChunk(0, out, start, start + duration, False)
    commands = provider._build_encode_commands(FFMPEG or "ffmpeg", source, chunk, out)
    _gpu, command = commands[-1]
    subprocess.run(command, check=True, capture_output=True)
    return out


class TestArgs:
    def test_stereo_is_kept_but_resynced(self) -> None:
        args = context_audio_args(AudioStreamInfo(position=0, channels=2,
                                                  channel_names=("FL", "FR")))
        assert args[:2] == ["-map", "0:a:0?"]
        assert args[args.index("-af") + 1] == "aresample=async=1:first_pts=0"

    def test_surround_favours_the_centre_channel(self) -> None:
        info = AudioStreamInfo(
            position=1, channels=6, channel_names=("FL", "FR", "FC", "LFE", "SL", "SR")
        )
        args = context_audio_args(info)
        assert args[:2] == ["-map", "0:a:1?"]
        assert args[args.index("-af") + 1].startswith("pan=mono|c0<FC+")

    def test_unknown_stream_falls_back_to_the_first(self) -> None:
        assert context_audio_args(None)[:2] == ["-map", "0:a:0?"]

    def test_default_stream_wins(self) -> None:
        streams = [
            AudioStreamInfo(position=0, channels=2, channel_names=("FL", "FR")),
            AudioStreamInfo(position=1, channels=2, channel_names=("FL", "FR"),
                            is_default=True),
        ]
        selected = select_context_audio(streams)
        assert selected is not None and selected.position == 1
        assert select_context_audio(streams[:1]) == streams[0]
        assert select_context_audio([]) is None


@needs_ffmpeg
def test_audio_after_a_timestamp_gap_stays_in_sync(tmp_path: Path) -> None:
    """Tiếng bíp ở giây 10 của phim phải nằm ở giây 10 của đoạn cắt, không phải giây 8."""
    beep = tmp_path / "beep.wav"
    # Bíp ở 8 s nội dung; sau giây 5 mọi mốc dời +2 s (khoảng hở) → phát ở giây 10.
    _ffmpeg("-f", "lavfi", "-i",
            "aevalsrc='0.8*sin(2*PI*1000*t)*between(t,8,8.2)':s=48000:d=20", str(beep))
    source = tmp_path / "gap.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=22",
            "-i", str(beep), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
            "-preset", "ultrafast", "-af", "asetpts='if(gte(T,5),PTS+2/TB,PTS)'",
            "-c:a", "pcm_s16le", str(source))
    audio = _decode_mono(_chunk_from(source, tmp_path, 0.0, 18.0))
    onset = int(np.argmax(np.abs(audio) > 0.2)) / SR
    assert onset == pytest.approx(10.0, abs=0.15)


@needs_ffmpeg
def test_surround_dialogue_is_clearer_than_the_music(tmp_path: Path) -> None:
    """Thoại (1 kHz, kênh giữa) phải nổi hơn nhạc (220 Hz, mọi kênh còn lại)."""
    source = tmp_path / "surround.mkv"
    music = "sine=f=220:sample_rate=48000:duration=6"
    graph = (
        f"{music}[fl];{music}[fr];sine=f=1000:sample_rate=48000:duration=6[fc];"
        f"anullsrc=r=48000:cl=mono:d=6[lfe];{music}[sl];{music}[sr];"
        # Kênh mono tên là FC nên phải chỉ rõ từng đầu vào vào kênh nào.
        "[fl][fr][fc][lfe][sl][sr]join=inputs=6:channel_layout=5.1(side):"
        "map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR[a]"
    )
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=6",
            "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "libx264",
            "-preset", "ultrafast", "-c:a", "ac3", "-b:a", "384k", "-shortest",
            str(source))
    audio = _decode_mono(_chunk_from(source, tmp_path, 0.0, 5.0))
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1 / SR)

    def _band(freq: float) -> float:
        return float(spectrum[(freqs > freq - 15) & (freqs < freq + 15)].max())

    dialogue_over_music_db = 20 * np.log10(_band(1000) / _band(220))
    assert dialogue_over_music_db > 0
    assert float(np.abs(audio).max()) < 0.99  # không chạm trần
