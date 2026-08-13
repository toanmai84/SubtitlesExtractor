"""Test kết hợp phụ đề + âm thanh khi xuất bản — v3.23.326.

KHOẢNG TRỐNG ĐƯỢC LẤP: bốn chế độ cũ LOẠI TRỪ NHAU nên **không thể có phụ đề và
thuyết minh trong cùng một tệp** — đúng thứ người dùng yêu cầu từ đầu.

Nay phụ đề và âm thanh là hai chiều độc lập, kết hợp được cả 9 cặp.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video.video_render_command import (
    AudioMode,
    RenderMode,
    RenderRequest,
    SubtitleMode,
    build_render_command,
    resolve_modes,
)

_FFMPEG = shutil.which("ffmpeg")
_needs_ffmpeg = pytest.mark.skipif(_FFMPEG is None, reason="Cần ffmpeg")


@pytest.fixture
def parts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Trả về ``(video, phụ đề, âm thanh)`` — chỉ cần tồn tại để qua bước kiểm."""
    video = tmp_path / "phim.mp4"
    video.write_bytes(b"\x00")
    subtitle = tmp_path / "phim.tts.vi.srt"
    subtitle.write_text("1\n", encoding="utf-8")
    audio = tmp_path / "phim.flac"
    audio.write_bytes(b"\x00")
    return video, subtitle, audio


def _request(
    parts: tuple[Path, Path, Path],
    tmp_path: Path,
    subtitle_mode: SubtitleMode,
    audio_mode: AudioMode,
) -> RenderRequest:
    video, subtitle, audio = parts
    return RenderRequest(
        video_path=video,
        output_path=tmp_path / "ra.mkv",
        mode=RenderMode.SOFT_SUB,  # chỉ để tương thích; hai trường dưới quyết định
        subtitle_path=subtitle,
        audio_path=audio,
        subtitle_mode=subtitle_mode,
        audio_mode=audio_mode,
    )


# ── Tương thích ngược ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        (RenderMode.SOFT_SUB, (SubtitleMode.SOFT, AudioMode.ORIGINAL)),
        (RenderMode.HARD_SUB, (SubtitleMode.BURNED, AudioMode.ORIGINAL)),
        (RenderMode.VOICE_OVER, (SubtitleMode.NONE, AudioMode.VOICE_OVER)),
        (RenderMode.DUB_AUDIO, (SubtitleMode.NONE, AudioMode.REPLACE_TRACK)),
    ],
)
def test_legacy_mode_maps_to_expected_pair(
    parts: tuple[Path, Path, Path],
    tmp_path: Path,
    legacy: RenderMode,
    expected: tuple[SubtitleMode, AudioMode],
) -> None:
    """Mã cũ chỉ đặt ``mode`` phải cho ra đúng cặp tương ứng."""
    video, subtitle, audio = parts
    request = RenderRequest(
        video_path=video, output_path=tmp_path / "ra.mkv", mode=legacy,
        subtitle_path=subtitle, audio_path=audio,
    )
    assert resolve_modes(request) == expected


# ── Tổ hợp mới ───────────────────────────────────────────────────────────────
def test_subtitle_and_voice_over_can_combine(
    parts: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """ĐÂY LÀ KHOẢNG TRỐNG ĐƯỢC LẤP: phụ đề rời + thuyết minh trong một tệp."""
    command = build_render_command(
        "ffmpeg", _request(parts, tmp_path, SubtitleMode.SOFT, AudioMode.VOICE_OVER)
    )
    assert "-c:s" in command          # có ghép track phụ đề
    assert "-filter_complex" in command  # có trộn thuyết minh
    assert "[aout]" in command


def test_burned_subtitle_and_voice_over_use_single_filter_complex(
    parts: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Không thể dùng đồng thời ``-vf`` và ``-filter_complex`` — phải gộp làm một."""
    command = build_render_command(
        "ffmpeg", _request(parts, tmp_path, SubtitleMode.BURNED, AudioMode.VOICE_OVER)
    )
    assert "-vf" not in command
    assert command.count("-filter_complex") == 1
    expression = command[command.index("-filter_complex") + 1]
    assert "subtitles=" in expression  # chuỗi lọc hình
    assert "[vout]" in expression and "[aout]" in expression


def test_video_is_copied_unless_burning(
    parts: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Chỉ kiểu CHÁY mới cần mã hoá lại video — các kiểu khác phải ``copy``."""
    for subtitle_mode in (SubtitleMode.NONE, SubtitleMode.SOFT):
        for audio_mode in AudioMode:
            command = build_render_command(
                "ffmpeg", _request(parts, tmp_path, subtitle_mode, audio_mode)
            )
            assert command[command.index("-c:v") + 1] == "copy"


def test_burning_re_encodes_video(
    parts: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    command = build_render_command(
        "ffmpeg", _request(parts, tmp_path, SubtitleMode.BURNED, AudioMode.ORIGINAL)
    )
    assert command[command.index("-c:v") + 1] != "copy"


def test_voice_over_filter_uses_correct_input_index(
    parts: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Chỉ số input đổi theo tổ hợp — để cứng ``[1:a]`` sẽ trộn nhầm luồng."""
    command = build_render_command(
        "ffmpeg", _request(parts, tmp_path, SubtitleMode.SOFT, AudioMode.VOICE_OVER)
    )
    expression = command[command.index("-filter_complex") + 1]
    # Thứ tự input: 0=video, 1=âm thanh, 2=phụ đề.
    assert "[1:a]" in expression


def test_replace_track_puts_vietnamese_first(
    parts: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    command = build_render_command(
        "ffmpeg", _request(parts, tmp_path, SubtitleMode.SOFT, AudioMode.REPLACE_TRACK)
    )
    maps = [command[i + 1] for i, arg in enumerate(command) if arg == "-map"]
    assert maps.index("1:a") < maps.index("0:a?")


# ── Kiểm tra đầu vào ─────────────────────────────────────────────────────────
def test_missing_subtitle_blocks_any_subtitle_mode(
    parts: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    from subtitles_extractor.infrastructure.video.video_render_command import (
        VideoRenderError,
    )

    video, _subtitle, audio = parts
    for subtitle_mode in (SubtitleMode.SOFT, SubtitleMode.BURNED):
        request = RenderRequest(
            video_path=video, output_path=tmp_path / "ra.mkv",
            mode=RenderMode.SOFT_SUB, audio_path=audio,
            subtitle_mode=subtitle_mode, audio_mode=AudioMode.ORIGINAL,
        )
        with pytest.raises(VideoRenderError, match="phụ đề"):
            build_render_command("ffmpeg", request)


# ── Chạy thật ────────────────────────────────────────────────────────────────
@_needs_ffmpeg
@pytest.mark.parametrize("subtitle_mode", list(SubtitleMode))
@pytest.mark.parametrize("audio_mode", list(AudioMode))
def test_all_combinations_run(
    tmp_path: Path, subtitle_mode: SubtitleMode, audio_mode: AudioMode
) -> None:
    """Cả 9 tổ hợp phải chạy được thật với ffmpeg."""
    video = tmp_path / "src.mp4"
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=black:size=320x180:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
         str(video)],
        check=True, capture_output=True,
    )
    subtitle = tmp_path / "s.srt"
    subtitle.write_text(
        "1\n00:00:00,500 --> 00:00:02,000\nXin chào\n\n", encoding="utf-8"
    )
    audio = tmp_path / "vi.m4a"
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=1000:duration=3", "-c:a", "aac", str(audio)],
        check=True, capture_output=True,
    )

    output = tmp_path / "ra.mkv"
    command = build_render_command(
        _FFMPEG,
        RenderRequest(
            video_path=video, output_path=output, mode=RenderMode.SOFT_SUB,
            subtitle_path=subtitle, audio_path=audio,
            subtitle_mode=subtitle_mode, audio_mode=audio_mode,
            video_encoder="libx264",
        ),
    )
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-300:]
    assert output.is_file()
