"""Test xuất video hoàn chỉnh (ghép mềm / cháy vào hình / thay tiếng) — v3.23.314.

Phần **dựng lệnh** được kiểm thuần (không cần ffmpeg). Phần **chạy thật** chỉ chạy khi
máy có ffmpeg với ``libx264`` + bộ lọc ``subtitles`` (libass), ngược lại tự bỏ qua.

Trọng tâm: thoát ký tự đường dẫn Windows cho bộ lọc ``subtitles=``. Đây là nguồn lỗi
kinh điển vì ffmpeg dùng ``:`` và ``,`` làm dấu phân tách tham số bộ lọc, mà đường dẫn
Windows luôn có ``C:`` và ``\\``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video.video_render_command import (
    DuckLevel,
    RenderMode,
    RenderRequest,
    VideoRenderError,
    build_render_command,
    build_voice_over_filter,
    escape_filter_path,
)

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")


def _has_subtitles_filter() -> bool:
    """``True`` nếu ffmpeg có bộ lọc ``subtitles`` (cần libass)."""
    if _FFMPEG is None:
        return False
    result = subprocess.run(
        [_FFMPEG, "-hide_banner", "-h", "filter=subtitles"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "libass" in result.stdout.lower()


_CAN_RUN = _FFMPEG is not None and _FFPROBE is not None and _has_subtitles_filter()
_needs_ffmpeg = pytest.mark.skipif(
    not _CAN_RUN, reason="Cần ffmpeg + ffprobe + bộ lọc subtitles (libass)"
)


# ── Thoát ký tự đường dẫn ────────────────────────────────────────────────────
def test_escape_windows_drive_letter() -> None:
    """Ký tự ổ đĩa ``C:`` PHẢI được thoát, nếu không ffmpeg hiểu là dấu phân tách."""
    escaped = escape_filter_path(Path(r"C:\Videos\phim.ass"))
    assert r"C\:" in escaped
    assert "\\Videos" not in escaped  # dấu \ đã đổi thành /
    assert "/Videos/phim.ass" in escaped


def test_escape_converts_backslashes_to_slashes() -> None:
    escaped = escape_filter_path(Path(r"D:\a\b\c.srt"))
    assert "\\a" not in escaped
    assert escaped.count("/") >= 3


@pytest.mark.parametrize("char", ["'", "[", "]", ",", ";"])
def test_escape_special_filter_characters(char: str) -> None:
    """Các ký tự có nghĩa trong biểu thức bộ lọc phải được thêm ``\\``."""
    escaped = escape_filter_path(Path(f"/tmp/a{char}b.srt"))
    assert f"\\{char}" in escaped


def test_escape_keeps_spaces_intact() -> None:
    """Dấu cách KHÔNG cần thoát (tham số truyền dạng danh sách, không qua shell)."""
    escaped = escape_filter_path(Path("/tmp/co dau cach.srt"))
    assert "co dau cach.srt" in escaped


# ── Dựng lệnh: kiểm cấu trúc ─────────────────────────────────────────────────
@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    path = tmp_path / "src.mp4"
    path.write_bytes(b"\x00" * 128)  # chỉ cần tồn tại để qua bước kiểm tra
    return path


@pytest.fixture
def subtitle_file(tmp_path: Path) -> Path:
    path = tmp_path / "sub.srt"
    path.write_text(
        "1\n00:00:00,500 --> 00:00:02,500\nXin chào thế giới\n\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "vi.m4a"
    path.write_bytes(b"\x00" * 128)
    return path


def test_soft_sub_copies_streams_without_reencoding(
    tmp_path: Path, source_video: Path, subtitle_file: Path
) -> None:
    """Ghép mềm phải dùng ``-c copy`` — không mã hoá lại nên rất nhanh."""
    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            source_video, tmp_path / "out.mkv", RenderMode.SOFT_SUB,
            subtitle_path=subtitle_file,
        ),
    )
    # [v3.23.326] Đổi từ '-c copy' gộp sang '-c:v copy' + '-c:a copy' tách — hành vi
    # tương đương, nhưng tường minh hơn khi kết hợp nhiều chiều.
    # [v3.23.349] HÌNH vẫn copy (điểm quan trọng — không mã hoá lại video). TIẾNG nay
    # được trộn xuống stereo nên phải mã hoá lại: nguồn 5.1 giữ nguyên có thể KHÔNG NGHE
    # THẤY GÌ trên máy người xem. Muốn copy nguyên thì đặt `downmix_multichannel=False`.
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"
    assert "libx264" not in command
    assert command[command.index("-c:s") + 1] == "srt"


def test_soft_sub_uses_mov_text_for_mp4(
    tmp_path: Path, source_video: Path, subtitle_file: Path
) -> None:
    """MP4 không chứa được SubRip — phải chuyển sang ``mov_text``."""
    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            source_video, tmp_path / "out.mp4", RenderMode.SOFT_SUB,
            subtitle_path=subtitle_file,
        ),
    )
    assert command[command.index("-c:s") + 1] == "mov_text"


def test_soft_sub_rejects_container_without_subtitle_support(
    tmp_path: Path, source_video: Path, subtitle_file: Path
) -> None:
    with pytest.raises(VideoRenderError, match="không chứa được phụ đề"):
        build_render_command(
            "ffmpeg",
            RenderRequest(
                source_video, tmp_path / "out.webm", RenderMode.SOFT_SUB,
                subtitle_path=subtitle_file,
            ),
        )


def test_hard_sub_uses_subtitles_filter_and_encoder(
    tmp_path: Path, source_video: Path, subtitle_file: Path
) -> None:
    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            source_video, tmp_path / "out.mp4", RenderMode.HARD_SUB,
            subtitle_path=subtitle_file,
        ),
    )
    filter_arg = command[command.index("-vf") + 1]
    assert filter_arg.startswith("subtitles=")
    assert command[command.index("-c:v") + 1] == "libx264"
    assert "-crf" in command
    # [v3.23.349] Tiếng nay được trộn xuống stereo (nguồn 5.1 giữ nguyên có thể không
    # phát được), nên phải mã hoá lại. Tắt bằng `downmix_multichannel=False` nếu muốn
    # giữ nguyên luồng gốc.
    assert command[command.index("-c:a") + 1] == "aac"


def test_hard_sub_nvenc_uses_cq_not_crf(
    tmp_path: Path, source_video: Path, subtitle_file: Path
) -> None:
    """NVENC dùng thang ``-cq``; truyền ``-crf`` cho nó là sai."""
    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            source_video, tmp_path / "out.mp4", RenderMode.HARD_SUB,
            subtitle_path=subtitle_file, video_encoder="h264_nvenc",
        ),
    )
    assert "-cq" in command
    assert "-crf" not in command


def test_dub_audio_puts_vietnamese_track_first(
    tmp_path: Path, source_video: Path, audio_file: Path
) -> None:
    """Track tiếng Việt phải đứng TRƯỚC để trình phát chọn mặc định."""
    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            source_video, tmp_path / "out.mkv", RenderMode.DUB_AUDIO,
            audio_path=audio_file, keep_original_audio=True,
        ),
    )
    maps = [command[i + 1] for i, arg in enumerate(command) if arg == "-map"]
    assert maps == ["0:v", "1:a", "0:a?"]
    # '?' để video CÂM không làm cả lệnh thất bại.
    assert "0:a?" in maps
    assert command[command.index("-c:v") + 1] == "copy"


def test_dub_audio_can_drop_original(
    tmp_path: Path, source_video: Path, audio_file: Path
) -> None:
    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            source_video, tmp_path / "out.mkv", RenderMode.DUB_AUDIO,
            audio_path=audio_file, keep_original_audio=False,
        ),
    )
    maps = [command[i + 1] for i, arg in enumerate(command) if arg == "-map"]
    assert maps == ["0:v", "1:a"]


# ── Kiểm tra hợp lệ ──────────────────────────────────────────────────────────
def test_missing_source_video_raises(tmp_path: Path, subtitle_file: Path) -> None:
    with pytest.raises(VideoRenderError, match="video nguồn"):
        build_render_command(
            "ffmpeg",
            RenderRequest(
                tmp_path / "khong-co.mp4", tmp_path / "out.mkv",
                RenderMode.SOFT_SUB, subtitle_path=subtitle_file,
            ),
        )


def test_missing_subtitle_raises(tmp_path: Path, source_video: Path) -> None:
    with pytest.raises(VideoRenderError, match="cần tệp phụ đề"):
        build_render_command(
            "ffmpeg",
            RenderRequest(source_video, tmp_path / "out.mkv", RenderMode.SOFT_SUB),
        )


def test_missing_audio_raises(tmp_path: Path, source_video: Path) -> None:
    with pytest.raises(VideoRenderError, match="cần tệp âm thanh"):
        build_render_command(
            "ffmpeg",
            RenderRequest(source_video, tmp_path / "out.mkv", RenderMode.DUB_AUDIO),
        )


def test_output_same_as_source_raises(
    source_video: Path, subtitle_file: Path
) -> None:
    """Ghi đè lên chính tệp nguồn sẽ làm hỏng dữ liệu — phải chặn."""
    with pytest.raises(VideoRenderError, match="trùng tệp nguồn"):
        build_render_command(
            "ffmpeg",
            RenderRequest(
                source_video, source_video, RenderMode.SOFT_SUB,
                subtitle_path=subtitle_file,
            ),
        )


# ── Chạy thật với ffmpeg ─────────────────────────────────────────────────────
def _make_test_video(path: Path, *, size: str = "320x180", with_audio: bool = True) -> None:
    command = [
        _FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=black:size={size}:rate=25:duration=3",
    ]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3"]
    command += ["-c:v", "libx264", "-preset", "ultrafast"]
    if with_audio:
        command += ["-c:a", "aac", "-shortest"]
    command.append(str(path))
    subprocess.run(command, check=True, capture_output=True)


def _stream_summary(path: Path) -> list[tuple[str, str | None]]:
    result = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return [
        (stream["codec_type"], stream.get("tags", {}).get("language"))
        for stream in json.loads(result.stdout)["streams"]
    ]


@_needs_ffmpeg
def test_run_soft_sub_produces_subtitle_track(
    tmp_path: Path, subtitle_file: Path
) -> None:
    source = tmp_path / "src.mp4"
    _make_test_video(source)
    output = tmp_path / "out.mkv"
    command = build_render_command(
        _FFMPEG,
        RenderRequest(source, output, RenderMode.SOFT_SUB, subtitle_path=subtitle_file),
    )
    subprocess.run(command, check=True, capture_output=True)

    streams = _stream_summary(output)
    assert ("subtitle", "vie") in streams
    assert any(kind == "video" for kind, _ in streams)


@_needs_ffmpeg
def test_run_hard_sub_actually_draws_text(
    tmp_path: Path, subtitle_file: Path
) -> None:
    """Xác minh phụ đề THỰC SỰ được vẽ lên khung hình, đúng thời điểm và vị trí."""
    av = pytest.importorskip("av")
    import numpy as np

    source = tmp_path / "src.mp4"
    _make_test_video(source, size="720x1280", with_audio=False)  # dọc như CJK drama
    output = tmp_path / "hard.mp4"
    command = build_render_command(
        _FFMPEG,
        RenderRequest(source, output, RenderMode.HARD_SUB, subtitle_path=subtitle_file),
    )
    subprocess.run(command, check=True, capture_output=True)

    def frame_at(seconds: float):
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if float(frame.pts * stream.time_base) >= seconds:
                    return frame.to_ndarray(format="gray")
        return None

    with_text = frame_at(1.5)   # trong khoảng 0.5–2.5s
    without_text = frame_at(2.9)  # sau khi phụ đề kết thúc
    assert with_text is not None and without_text is not None

    drawn = int((with_text > 30).sum())
    background = int((without_text > 30).sum())
    assert background == 0, "nền phải đen tuyền khi không có phụ đề"
    assert drawn > 2000, f"chữ vẽ quá ít ({drawn} điểm) — libass có thể thiếu font"

    rows, cols = np.where(with_text > 30)
    height, width = with_text.shape
    assert rows.min() > height * 0.75, "phụ đề phải ở đáy khung hình"
    assert abs((cols.min() + cols.max()) / 2 - width / 2) < width * 0.1, "phải căn giữa"


@_needs_ffmpeg
def test_run_dub_audio_orders_tracks_correctly(tmp_path: Path) -> None:
    source = tmp_path / "src.mp4"
    _make_test_video(source)
    vietnamese = tmp_path / "vi.m4a"
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=880:duration=3", "-c:a", "aac", str(vietnamese)],
        check=True, capture_output=True,
    )
    output = tmp_path / "dub.mkv"
    command = build_render_command(
        _FFMPEG,
        RenderRequest(source, output, RenderMode.DUB_AUDIO, audio_path=vietnamese),
    )
    subprocess.run(command, check=True, capture_output=True)

    streams = _stream_summary(output)
    audio_tracks = [lang for kind, lang in streams if kind == "audio"]
    assert len(audio_tracks) == 2
    assert audio_tracks[0] == "vie", "track tiếng Việt phải đứng đầu"


# ── [v3.23.315] Thuyết minh trộn có ducking ──────────────────────────────────
@pytest.mark.parametrize("level", list(DuckLevel))
def test_duck_levels_map_to_measured_thresholds(level: DuckLevel) -> None:
    """Ba mức phải trỏ đúng ngưỡng ĐÃ ĐO thực nghiệm, thứ tự hạ dần."""
    assert 0.0 < level.threshold < 0.1
    assert level.approx_reduction_db < 0


def test_duck_levels_are_ordered() -> None:
    """Nhẹ hạ ít nhất, mạnh hạ nhiều nhất — ngưỡng thấp hơn = hạ mạnh hơn."""
    assert DuckLevel.GENTLE.threshold > DuckLevel.MEDIUM.threshold
    assert DuckLevel.MEDIUM.threshold > DuckLevel.STRONG.threshold
    assert (
        DuckLevel.GENTLE.approx_reduction_db
        > DuckLevel.MEDIUM.approx_reduction_db
        > DuckLevel.STRONG.approx_reduction_db
    )


def test_voice_over_filter_uses_sidechain_on_original(
    tmp_path: Path, source_video: Path, audio_file: Path
) -> None:
    """Tiếng GỐC phải là đầu vào bị nén, giọng thuyết minh là tín hiệu điều khiển."""
    request = RenderRequest(
        source_video, tmp_path / "out.mkv", RenderMode.VOICE_OVER, audio_path=audio_file
    )
    expression = build_voice_over_filter(request)
    # [0:a] = tiếng gốc -> nhãn [bg]; [1:a] = thuyết minh -> tách thành [narr] và [key].
    assert "[0:a]" in expression and "[bg]" in expression
    assert "asplit=2[narr][key]" in expression
    assert "[bg][key]sidechaincompress=" in expression
    # Trộn KHÔNG được tự chuẩn hoá, nếu không âm lượng bị chia đôi.
    assert "normalize=0" in expression


def test_voice_over_keeps_video_stream_untouched(
    tmp_path: Path, source_video: Path, audio_file: Path
) -> None:
    """Phần hình phải ``copy`` — thuyết minh không phải lý do để mã hoá lại video."""
    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            source_video, tmp_path / "out.mkv", RenderMode.VOICE_OVER,
            audio_path=audio_file,
        ),
    )
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-map") + 1] == "0:v"
    assert "[aout]" in command


def test_voice_over_requires_audio(tmp_path: Path, source_video: Path) -> None:
    with pytest.raises(VideoRenderError, match="cần tệp âm thanh"):
        build_render_command(
            "ffmpeg",
            RenderRequest(source_video, tmp_path / "out.mkv", RenderMode.VOICE_OVER),
        )


@_needs_ffmpeg
@pytest.mark.parametrize("level", list(DuckLevel))
def test_run_voice_over_ducks_background_but_keeps_it(
    tmp_path: Path, level: DuckLevel
) -> None:
    """ĐO THỰC: tiếng gốc phải NHỎ LẠI khi thuyết minh nói, nhưng KHÔNG tắt hẳn.

    Dùng hai tần số khác nhau (nền 200Hz, thuyết minh 1kHz) để tách từng nguồn bằng FFT.
    """
    av = pytest.importorskip("av")
    import numpy as np
    from av.audio.resampler import AudioResampler

    sample_rate = 48_000
    # Nền: 200Hz liên tục 6s — đại diện nhạc/tiếng động, mức KHÔNG đổi.
    background = tmp_path / "bg.wav"
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=200:duration=6", "-af", "volume=0.5",
         "-c:a", "pcm_s16le", str(background)],
        check=True, capture_output=True,
    )
    source = tmp_path / "src.mp4"
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=black:size=320x180:rate=25:duration=6", "-i", str(background),
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
         str(source)],
        check=True, capture_output=True,
    )
    # Thuyết minh: 1kHz CHỈ nói trong khoảng 2s–4s.
    narration = tmp_path / "tts.wav"
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=1000:duration=6", "-af",
         "volume=enable='between(t,2,4)':volume=0.6,"
         "volume=enable='not(between(t,2,4))':volume=0",
         "-c:a", "pcm_s16le", str(narration)],
        check=True, capture_output=True,
    )

    output = tmp_path / f"vo_{level.value}.mkv"
    subprocess.run(
        build_render_command(
            _FFMPEG,
            RenderRequest(
                source, output, RenderMode.VOICE_OVER,
                audio_path=narration, duck_level=level,
            ),
        ),
        check=True, capture_output=True,
    )

    with av.open(str(output)) as container:
        stream = container.streams.audio[0]
        resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks = [
            resampled.to_ndarray().reshape(-1)
            for frame in container.decode(stream)
            for resampled in resampler.resample(frame)
        ]
    samples = np.concatenate(chunks).astype(np.float64) / 32768.0

    def amplitude_at(seconds: float, frequency: float) -> float:
        start = int(seconds * sample_rate)
        segment = samples[start : start + int(0.4 * sample_rate)]
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        bins = np.fft.rfftfreq(len(segment), 1 / sample_rate)
        index = int(np.argmin(np.abs(bins - frequency)))
        return float(spectrum[max(0, index - 2) : index + 3].max()) / len(segment)

    quiet = np.mean([amplitude_at(t, 200) for t in (0.5, 1.4, 5.4)])
    ducked = np.mean([amplitude_at(t, 200) for t in (2.6, 3.4)])
    reduction_db = 20 * np.log10(ducked / quiet)

    # Mức hạ phải khớp giá trị đã đo (sai số 2 dB cho khác biệt phiên bản ffmpeg).
    assert reduction_db == pytest.approx(level.approx_reduction_db, abs=2.0)
    # QUAN TRỌNG: nền KHÔNG được tắt hẳn — nhạc/tiếng động phải còn.
    assert ducked > 0.0005, "tiếng gốc bị tắt hẳn thay vì chỉ nhỏ lại"
    # Giọng thuyết minh phải xuất hiện đúng lúc.
    assert np.mean([amplitude_at(t, 1000) for t in (2.6, 3.4)]) > 0.002
