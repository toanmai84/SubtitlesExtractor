"""Test ghép phụ đề MỀM (không phải cháy) kèm thuyết minh — v3.23.327.

Kiểm chứng theo yêu cầu: phụ đề mềm + thuyết minh có thực sự hoạt động không.

LỖI ĐƯỢC SỬA: codec phụ đề trước đây chọn CHỈ theo định dạng đích, nên ``.ass`` luôn
bị chuyển sang ``srt`` — mất sạch màu, kiểu chữ và vị trí. Đo thực tế xác nhận: mux
``.ass`` bằng codec ``srt`` làm mất ``&H0000FFFF``, tên style và thẻ ``{\\an8}``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video.video_render_command import (
    AudioMode,
    RenderMode,
    RenderRequest,
    SubtitleMode,
    _subtitle_codec_for,
    build_render_command,
    subtitle_styling_warning,
)

_FFMPEG = shutil.which("ffmpeg")
_needs_ffmpeg = pytest.mark.skipif(_FFMPEG is None, reason="Cần ffmpeg")

_STYLED_ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 360

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Alignment, MarginV
Style: Vang,Arial,28,&H0000FFFF,&H00000000,-1,2,30

[Events]
Format: Layer, Start, End, Style, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Vang,,Đại Càn Hãn Đao Hành
Dialogue: 0,0:00:04.00,0:00:06.25,Vang,,{\\an8}Bát đệ, ngươi đã về rồi à?
"""

_SRT = """1
00:00:01,000 --> 00:00:03,500
Đại Càn Hãn Đao Hành

2
00:00:04,000 --> 00:00:06,250
Bát đệ, ngươi đã về rồi à?
"""


# ── Chọn codec ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (".srt", ".mkv", "srt"),
        (".ass", ".mkv", "ass"),   # PHẢI giữ định dạng
        (".ssa", ".mkv", "ass"),
        (".srt", ".mp4", "mov_text"),
        (".ass", ".mp4", "mov_text"),  # MP4 không chứa được ASS
    ],
)
def test_codec_chosen_from_source_and_target(
    source: str, target: str, expected: str
) -> None:
    assert _subtitle_codec_for(Path(f"ra{target}"), Path(f"phu{source}")) == expected


def test_styled_subtitle_into_mkv_needs_no_warning() -> None:
    assert subtitle_styling_warning(Path("ra.mkv"), Path("phu.ass")) is None


def test_styled_subtitle_into_mp4_warns_about_lost_styling() -> None:
    """MP4 không giữ được định dạng — phải nói rõ để người dùng chọn .mkv."""
    warning = subtitle_styling_warning(Path("ra.mp4"), Path("phu.ass"))
    assert warning is not None
    assert ".mkv" in warning


def test_plain_subtitle_never_warns() -> None:
    assert subtitle_styling_warning(Path("ra.mp4"), Path("phu.srt")) is None


# ── Chạy thật ────────────────────────────────────────────────────────────────
@pytest.fixture
def media(tmp_path: Path) -> tuple[Path, Path]:
    """Trả về ``(video có tiếng, tệp thuyết minh)``."""
    video = tmp_path / "src.mp4"
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=navy:size=640x360:rate=25:duration=8",
         "-f", "lavfi", "-i", "sine=frequency=200:duration=8",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
         str(video)],
        check=True, capture_output=True,
    )
    narration = tmp_path / "vi.m4a"
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=1000:duration=8", "-c:a", "aac", str(narration)],
        check=True, capture_output=True,
    )
    return video, narration


def _extract_subtitle(source: Path, target: Path, codec: str = "srt") -> str:
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-map", "0:s:0", "-c:s", codec, str(target)],
        check=True, capture_output=True,
    )
    return target.read_text(encoding="utf-8")


@_needs_ffmpeg
def test_soft_subtitle_text_survives_muxing(
    tmp_path: Path, media: tuple[Path, Path]
) -> None:
    """Chữ phải nguyên vẹn 100%, kể cả dấu tiếng Việt và chữ Hán."""
    video, narration = media
    subtitle = tmp_path / "phu.srt"
    subtitle.write_text(_SRT, encoding="utf-8")
    output = tmp_path / "ra.mkv"

    subprocess.run(
        build_render_command(
            _FFMPEG,
            RenderRequest(
                video_path=video, output_path=output, mode=RenderMode.SOFT_SUB,
                subtitle_path=subtitle, audio_path=narration,
                subtitle_mode=SubtitleMode.SOFT, audio_mode=AudioMode.VOICE_OVER,
            ),
        ),
        check=True, capture_output=True,
    )

    recovered = _extract_subtitle(output, tmp_path / "back.srt")
    assert "Đại Càn Hãn Đao Hành" in recovered
    assert "Bát đệ, ngươi đã về rồi à?" in recovered


@_needs_ffmpeg
def test_soft_subtitle_timing_stays_within_one_frame(
    tmp_path: Path, media: tuple[Path, Path]
) -> None:
    """Mốc thời gian được phép lệch dưới MỘT khung hình (40ms ở 25fps).

    Mã hoá lại AAC có độ trễ mồi 1024 mẫu = 21.3ms ở 48kHz, làm cả tiếng lẫn phụ đề
    dịch cùng nhau. Dưới nửa khung hình nên không thể nhận ra — nhưng phải KHÔNG được
    tích luỹ (câu sau không lệch nhiều hơn câu trước).
    """
    video, narration = media
    subtitle = tmp_path / "phu.srt"
    subtitle.write_text(_SRT, encoding="utf-8")
    output = tmp_path / "ra.mkv"

    subprocess.run(
        build_render_command(
            _FFMPEG,
            RenderRequest(
                video_path=video, output_path=output, mode=RenderMode.SOFT_SUB,
                subtitle_path=subtitle, audio_path=narration,
                subtitle_mode=SubtitleMode.SOFT, audio_mode=AudioMode.VOICE_OVER,
            ),
        ),
        check=True, capture_output=True,
    )

    recovered = _extract_subtitle(output, tmp_path / "back.srt")
    starts = re.findall(r"(\d+):(\d+):(\d+),(\d+)\s*-->", recovered)
    assert len(starts) == 2

    expected = [1.000, 4.000]
    drifts = []
    for (hours, minutes, seconds, millis), want in zip(starts, expected):
        actual = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000
        drifts.append(abs(actual - want))

    one_frame = 1 / 25
    assert all(drift < one_frame for drift in drifts), f"lệch quá 1 khung: {drifts}"
    # KHÔNG được tích luỹ: câu sau lệch không nhiều hơn câu đầu quá 5ms.
    assert drifts[1] - drifts[0] < 0.005


@_needs_ffmpeg
def test_ass_styling_survives_into_mkv(
    tmp_path: Path, media: tuple[Path, Path]
) -> None:
    """LỖI ĐÃ SỬA: trước đây .ass bị hạ thành srt, mất màu/style/vị trí."""
    video, narration = media
    subtitle = tmp_path / "phu.ass"
    subtitle.write_text(_STYLED_ASS, encoding="utf-8")
    output = tmp_path / "ra.mkv"

    command = build_render_command(
        _FFMPEG,
        RenderRequest(
            video_path=video, output_path=output, mode=RenderMode.SOFT_SUB,
            subtitle_path=subtitle, audio_path=narration,
            subtitle_mode=SubtitleMode.SOFT, audio_mode=AudioMode.VOICE_OVER,
        ),
    )
    assert command[command.index("-c:s") + 1] == "ass"
    subprocess.run(command, check=True, capture_output=True)

    recovered = _extract_subtitle(output, tmp_path / "back.ass", codec="ass")
    assert "0000FFFF" in recovered.upper()   # màu vàng
    assert "Vang" in recovered               # tên style
    assert "an8" in recovered                # thẻ vị trí
    assert "Đại Càn Hãn Đao Hành" in recovered


@_needs_ffmpeg
def test_soft_subtitle_track_is_marked_default(
    tmp_path: Path, media: tuple[Path, Path]
) -> None:
    """Track phụ đề phải được đánh dấu mặc định để trình phát tự bật."""
    import json

    video, narration = media
    subtitle = tmp_path / "phu.srt"
    subtitle.write_text(_SRT, encoding="utf-8")
    output = tmp_path / "ra.mkv"

    subprocess.run(
        build_render_command(
            _FFMPEG,
            RenderRequest(
                video_path=video, output_path=output, mode=RenderMode.SOFT_SUB,
                subtitle_path=subtitle, audio_path=narration,
                subtitle_mode=SubtitleMode.SOFT, audio_mode=AudioMode.VOICE_OVER,
            ),
        ),
        check=True, capture_output=True,
    )

    info = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    subtitles = [s for s in info["streams"] if s["codec_type"] == "subtitle"]
    assert len(subtitles) == 1
    assert subtitles[0]["disposition"]["default"] == 1
    assert subtitles[0].get("tags", {}).get("language") == "vie"


@_needs_ffmpeg
def test_video_not_re_encoded_for_soft_subtitle(
    tmp_path: Path, media: tuple[Path, Path]
) -> None:
    """Phụ đề mềm KHÔNG được làm video phải mã hoá lại — đó là ưu điểm chính của nó."""
    video, narration = media
    subtitle = tmp_path / "phu.srt"
    subtitle.write_text(_SRT, encoding="utf-8")

    command = build_render_command(
        _FFMPEG,
        RenderRequest(
            video_path=video, output_path=tmp_path / "ra.mkv", mode=RenderMode.SOFT_SUB,
            subtitle_path=subtitle, audio_path=narration,
            subtitle_mode=SubtitleMode.SOFT, audio_mode=AudioMode.VOICE_OVER,
        ),
    )
    assert command[command.index("-c:v") + 1] == "copy"
