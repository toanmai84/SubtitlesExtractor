"""[v3.23.659] Rà BUG khâu Xuất bản: xuất không phụ đề + thuyết minh, nháy đơn trong đường dẫn phụ đề, nối cả bộ."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.application.services.concat_plan import find_concat_videos
from subtitles_extractor.infrastructure.video.video_render_command import (
    AudioMode,
    RenderMode,
    RenderRequest,
    SubtitleMode,
    build_render_command,
    escape_filter_path,
)


def test_no_subtitle_voice_over_builds_command(tmp_path: Path) -> None:
    """Trang Xuất bản luôn gửi ``mode=SOFT_SUB`` (trường cũ) — "Phụ đề: Không" + thuyết minh từng báo thiếu phụ đề."""
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"v")
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"a")
    request = RenderRequest(
        video_path=video, output_path=tmp_path / "ep_thuyetminh.mkv", mode=RenderMode.SOFT_SUB,
        subtitle_mode=SubtitleMode.NONE, audio_mode=AudioMode.VOICE_OVER, subtitle_path=None, audio_path=audio,
    )
    command = build_render_command("ffmpeg", request)
    assert str(audio) in command and "subtitles=" not in " ".join(command)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")
def test_apostrophe_in_subtitle_path_burns(tmp_path: Path) -> None:
    folder = tmp_path / "dir it's [a],b;c"
    folder.mkdir()
    subtitle = folder / "sub's.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
         "-vf", f"subtitles='{escape_filter_path(subtitle)}'", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-300:]


def test_concat_picks_exact_suffix_only(tmp_path: Path) -> None:
    """Lọc theo hậu tố từng là "chứa chuỗi": bản xem thử, tệp .partial và chế độ khác bị ghép vào trọn bộ."""
    for name in ("第1集_phude.mkv", "第1集_phude_xemthu.mkv", "第2集_phude.mkv", "第2集_phude.partial.mkv",
                 "第2集_phude_thuyetminh.mkv", "第10集_phude.mkv"):
        (tmp_path / name).write_bytes(b"v")
    found = [p.name for p in find_concat_videos(tmp_path, name_filter="_phude")]
    assert found == ["第1集_phude.mkv", "第2集_phude.mkv", "第10集_phude.mkv"]


def _episode(path: Path, *, fps: int, seconds: float, text: str) -> Path:
    """Tập "đã xuất bản" giống app tạo: hình + tiếng Việt (mono) + tiếng gốc (stereo) + phụ đề rời."""
    srt = path.with_suffix(".srt")
    srt.write_text(f"1\n00:00:00,500 --> 00:00:01,200\n{text}\n", encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc=size=64x64:rate={fps}:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=500:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds},aformat=channel_layouts=stereo",
         "-i", str(srt),
         "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-c:s", "srt",
         "-metadata:s:a:0", "language=vie", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return path


def _streams(path: Path) -> list[str]:
    import json

    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return [s["codec_type"] for s in json.loads(out)["streams"]]


def _subtitle_times(path: Path) -> list[float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s:0", "-show_entries",
                          "packet=pts_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return [float(x) for x in out.split() if x.strip()]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="cần ffmpeg")
@pytest.mark.parametrize("fps_b", [25, 30])
def test_concat_keeps_all_audio_and_subtitles(tmp_path: Path, fps_b: int) -> None:
    """Nối cả bộ từng bỏ phụ đề (tập lệch fps → chuẩn hoá) hoặc bỏ tiếng Việt (sao chép luồng giữ 1 track)."""
    from subtitles_extractor.infrastructure.video.ffmpeg_concat_adapter import concatenate_videos

    a = _episode(tmp_path / "t1_phude_tiengviet.mkv", fps=25, seconds=2.0, text="Một")
    b = _episode(tmp_path / "t2_phude_tiengviet.mkv", fps=fps_b, seconds=2.0, text="Hai")
    out = concatenate_videos([a, b], tmp_path / "bo_tron_bo.mkv")
    kinds = _streams(out)
    assert kinds.count("subtitle") == 1, kinds
    times = _subtitle_times(out)
    assert len(times) == 2 and times[0] == pytest.approx(0.5, abs=0.1) and times[1] == pytest.approx(2.5, abs=0.15)
    if fps_b == 25:
        assert kinds.count("audio") == 2, kinds      # sao chép luồng: giữ cả tiếng Việt lẫn tiếng gốc
    assert not list(tmp_path.glob("*.partial*"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")
def test_failed_concat_keeps_previous_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.video import ffmpeg_concat_adapter as module

    a = _episode(tmp_path / "t1.mkv", fps=25, seconds=1.0, text="A")
    b = _episode(tmp_path / "t2.mkv", fps=25, seconds=1.0, text="B")
    out = tmp_path / "bo_tron_bo.mkv"
    out.write_bytes(b"BAN CU")
    real = module._run_ffmpeg_concat

    def broken(command, output_path, *args, **kwargs):
        Path(output_path).write_bytes(b"hong")
        raise module.VideoConcatError("ffmpeg lỗi giả")

    monkeypatch.setattr(module, "_run_ffmpeg_concat", broken)
    with pytest.raises(module.VideoConcatError):
        module.concatenate_videos([a, b], out)
    assert out.read_bytes() == b"BAN CU"
    assert not list(tmp_path.glob("*.partial*"))
    monkeypatch.setattr(module, "_run_ffmpeg_concat", real)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="cần ffmpeg")
def test_cancel_during_subtitle_attach_stops_and_keeps_previous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rà độc lập: bước gắn phụ đề (chép lại cả tệp trọn bộ) bỏ qua nút Huỷ."""
    from subtitles_extractor.infrastructure.video import ffmpeg_concat_adapter as module

    a = _episode(tmp_path / "t1_phude.mkv", fps=25, seconds=1.0, text="A")
    b = _episode(tmp_path / "t2_phude.mkv", fps=30, seconds=1.0, text="B")  # lệch fps → nhánh chuẩn hoá + gắn phụ đề
    out = tmp_path / "bo_tron_bo.mkv"
    out.write_bytes(b"BAN CU")
    state = {"attach": False}
    real_attach = module._attach_episode_subtitles

    def attach(*args, **kwargs):
        state["attach"] = True
        return real_attach(*args, **kwargs)

    monkeypatch.setattr(module, "_attach_episode_subtitles", attach)
    with pytest.raises(module.VideoConcatError):
        module.concatenate_videos([a, b], out, cancel_check=lambda: state["attach"])
    assert out.read_bytes() == b"BAN CU"
    assert not list(tmp_path.glob("*.partial*"))
