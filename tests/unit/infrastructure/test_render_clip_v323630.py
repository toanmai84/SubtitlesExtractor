"""[v3.23.630] Xuất THỬ một đoạn: lệnh ffmpeg, cắt phụ đề, chạy thật (hình + tiếng + phụ đề khớp).

Log thật: 37 lần xuất cả phim 86 tập, mỗi lần ~3 phút — chỉ để nghe thử vài câu.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from subtitles_extractor.infrastructure.video.ffmpeg_video_renderer import render_video
from subtitles_extractor.infrastructure.video.subtitle_clip import clip_srt, clip_subtitle_file
from subtitles_extractor.infrastructure.video.video_render_command import (
    AudioMode,
    RenderMode,
    RenderRequest,
    SubtitleMode,
    VideoRenderError,
    build_render_command,
)

_FFMPEG, _FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
_needs_ffmpeg = pytest.mark.skipif(not (_FFMPEG and _FFPROBE), reason="Cần ffmpeg + ffprobe")

SRT = """1
00:00:01,000 --> 00:00:03,000
Trước đoạn

2
00:00:09,500 --> 00:00:12,000
Lấn vào đầu đoạn

3
00:00:13,000 --> 00:00:14,500
Trong đoạn

4
00:00:17,500 --> 00:00:19,000
Lấn ra cuối đoạn

5
00:00:25,000 --> 00:00:26,000
Sau đoạn
"""


class TestClipSrt:
    def test_keeps_overlapping_cues_shifted_and_clamped(self) -> None:
        out = clip_srt(SRT, 10.0, 8.0)         # đoạn 10 s → 18 s
        assert "Trước đoạn" not in out and "Sau đoạn" not in out
        assert out.splitlines()[:3] == ["1", "00:00:00,000 --> 00:00:02,000", "Lấn vào đầu đoạn"]
        assert "00:00:03,000 --> 00:00:04,500\nTrong đoạn" in out
        assert "00:00:07,500 --> 00:00:08,000\nLấn ra cuối đoạn" in out   # cắt ở 8 s
        assert out.count("-->") == 3 and "\n3\n" in out                  # đánh số lại

    def test_empty_clip_still_valid(self) -> None:
        out = clip_srt(SRT, 100.0, 10.0)
        assert out.count("-->") == 1 and "00:00:00,000" in out

    def test_tolerates_bom_crlf_dot_separator_and_multiline(self) -> None:
        text = "﻿1\r\n00:00:11.5 --> 00:00:12.25\r\nDòng một\r\nDòng hai\r\n\r\n"
        out = clip_srt(text, 10.0, 5.0)
        assert "00:00:01,500 --> 00:00:02,250\nDòng một\nDòng hai" in out

    def test_garbage_blocks_are_skipped(self) -> None:
        assert clip_srt("không phải srt", 0.0, 5.0).count("-->") == 1

    def test_only_srt_files(self, tmp_path: Path) -> None:
        ass = tmp_path / "a.ass"
        ass.write_text("x", encoding="utf-8")
        with pytest.raises(VideoRenderError, match="srt"):
            clip_subtitle_file(ass, tmp_path, 0.0, 5.0)
        srt = tmp_path / "a.srt"
        srt.write_text(SRT, encoding="utf-8")
        out = clip_subtitle_file(srt, tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, 10.0, 8.0)
        assert out.read_text(encoding="utf-8").count("-->") == 3


@pytest.fixture
def parts(tmp_path: Path) -> tuple[Path, Path, Path]:
    video = tmp_path / "phim.mp4"
    video.write_bytes(b"\x00")
    subtitle = tmp_path / "phim.srt"
    subtitle.write_text(SRT, encoding="utf-8")
    audio = tmp_path / "phim.flac"
    audio.write_bytes(b"\x00")
    return video, subtitle, audio


def _request(parts, tmp_path: Path, **kw) -> RenderRequest:
    video, subtitle, audio = parts
    defaults = dict(
        video_path=video, output_path=tmp_path / "ra.mkv", mode=RenderMode.SOFT_SUB,
        subtitle_path=subtitle, audio_path=audio, subtitle_mode=SubtitleMode.SOFT,
        audio_mode=AudioMode.REPLACE_TRACK, video_encoder="libx264",
    )
    defaults.update(kw)
    return RenderRequest(**defaults)


class TestCommand:
    def test_clip_seeks_every_input_limits_output_and_reencodes(self, parts, tmp_path) -> None:
        command = build_render_command(
            "ffmpeg", _request(parts, tmp_path, clip_start_sec=600.0, clip_duration_sec=90.0)
        )
        joined = " ".join(command)
        assert command.count("-ss") == 2                      # video + tiếng Việt
        assert joined.index("-ss 600.000 -i " + str(parts[0])) >= 0
        assert f"-ss 600.000 -i {parts[2]}" in joined
        assert command[command.index("-t") + 1] == "90.000"
        assert command.index("-t") > command.index("-i")      # -t là tuỳ chọn ĐẦU RA
        assert "-c:v libx264" in joined and "-preset veryfast" in joined
        assert "-c:v copy" not in joined                      # cắt copy làm tiếng lệch hình

    def test_without_clip_nothing_changes(self, parts, tmp_path) -> None:
        command = build_render_command("ffmpeg", _request(parts, tmp_path))
        assert "-ss" not in command and "-t" not in command
        assert "-c:v copy" in " ".join(command)

    def test_voice_over_clip_reencodes_too(self, parts, tmp_path) -> None:
        command = build_render_command("ffmpeg", _request(
            parts, tmp_path, audio_mode=AudioMode.VOICE_OVER, subtitle_mode=SubtitleMode.NONE,
            clip_start_sec=5.0, clip_duration_sec=20.0,
        ))
        joined = " ".join(command)
        assert "-c:v copy" not in joined and "-c:v libx264" in joined
        assert command.count("-ss") == 2

    def test_burned_clip_uses_fast_preset(self, parts, tmp_path) -> None:
        command = build_render_command("ffmpeg", _request(
            parts, tmp_path, subtitle_mode=SubtitleMode.BURNED, audio_mode=AudioMode.ORIGINAL,
            clip_start_sec=5.0, clip_duration_sec=20.0,
        ))
        assert "-preset veryfast" in " ".join(command)

    def test_negative_clip_is_rejected(self, parts, tmp_path) -> None:
        with pytest.raises(VideoRenderError):
            build_render_command(
                "ffmpeg", _request(parts, tmp_path, clip_start_sec=-1.0, clip_duration_sec=5.0)
            )


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Video 20 s (có tiếng 300 Hz), giọng Việt giả 20 s (440 Hz rồi 880 Hz từ giây 10), SRT."""
    video = tmp_path / "phim.mp4"
    subprocess.run(
        [_FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x284:rate=25:duration=20",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=20", "-c:v", "libx264", "-g", "100",
         "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )
    rate = 16000
    t = np.arange(20 * rate) / rate
    tone = np.where(t < 10, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)) * 0.4
    audio = tmp_path / "dub.wav"
    sf.write(audio, tone.astype(np.float32), rate)
    subtitle = tmp_path / "phim.srt"
    subtitle.write_text(SRT, encoding="utf-8")
    return video, subtitle, audio


def _dominant_hz(wav: Path) -> float:
    data, rate = sf.read(wav, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    spectrum = np.abs(np.fft.rfft(data))
    return float(np.argmax(spectrum) * rate / len(data))


@_needs_ffmpeg
class TestRealRender:
    def test_preview_clip_has_matching_picture_voice_and_subtitles(
        self, sample, tmp_path: Path
    ) -> None:
        video, subtitle, audio = sample
        output = tmp_path / "xemthu.mkv"
        render_video(RenderRequest(
            video_path=video, output_path=output, mode=RenderMode.SOFT_SUB,
            subtitle_path=subtitle, audio_path=audio, subtitle_mode=SubtitleMode.SOFT,
            audio_mode=AudioMode.REPLACE_TRACK, video_encoder="libx264",
            clip_start_sec=12.0, clip_duration_sec=6.0,
        ))
        info = _probe(output)
        kinds = [s["codec_type"] for s in info["streams"]]
        assert kinds.count("video") == 1 and kinds.count("audio") == 2 and "subtitle" in kinds
        assert float(info["format"]["duration"]) == pytest.approx(6.0, abs=0.35)

        # Track tiếng Việt (đứng đầu): đoạn 12→18 s phải toàn 880 Hz (nửa sau của giọng giả).
        extracted = tmp_path / "vi.wav"
        subprocess.run(
            [_FFMPEG, "-v", "error", "-y", "-i", str(output), "-map", "0:a:0", str(extracted)],
            check=True,
        )
        assert _dominant_hz(extracted) == pytest.approx(880, abs=15)

        # Phụ đề đã dời: "Trong đoạn" (13,0→14,5) nằm ở 1,0→2,5 s của đoạn thử.
        srt = tmp_path / "out.srt"
        subprocess.run(
            [_FFMPEG, "-v", "error", "-y", "-i", str(output), "-map", "0:s:0", str(srt)],
            check=True,
        )
        text = srt.read_text(encoding="utf-8")
        assert "Trong đoạn" in text and "Trước đoạn" not in text and "Sau đoạn" not in text
        match = re.search(r"00:00:0(\d),(\d{3}) --> 00:00:0(\d),(\d{3})\nTrong đoạn", text)
        assert match is not None
        begin = int(match[1]) + int(match[2]) / 1000
        finish = int(match[3]) + int(match[4]) / 1000
        # Lệch tối đa ~0,1 s do độ trễ mã hoá AAC đầu luồng — không phải 12 s như khi không dời.
        assert begin == pytest.approx(1.0, abs=0.1) and finish == pytest.approx(2.5, abs=0.1)

    def test_temp_clip_subtitle_is_removed(self, sample, tmp_path: Path, monkeypatch) -> None:
        import tempfile

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))
        video, subtitle, audio = sample
        render_video(RenderRequest(
            video_path=video, output_path=tmp_path / "t.mkv", mode=RenderMode.SOFT_SUB,
            subtitle_path=subtitle, subtitle_mode=SubtitleMode.SOFT,
            audio_mode=AudioMode.ORIGINAL, video_encoder="libx264",
            clip_start_sec=2.0, clip_duration_sec=4.0,
        ))
        assert not list(scratch.iterdir())

    def test_progress_is_measured_against_the_clip(self, sample, tmp_path: Path) -> None:
        video, subtitle, audio = sample
        seen: list[float] = []
        render_video(
            RenderRequest(
                video_path=video, output_path=tmp_path / "p.mkv", mode=RenderMode.SOFT_SUB,
                subtitle_path=subtitle, subtitle_mode=SubtitleMode.SOFT,
                audio_mode=AudioMode.ORIGINAL, video_encoder="libx264",
                clip_start_sec=0.0, clip_duration_sec=5.0,
            ),
            total_duration_sec=999.0,       # bị bỏ qua: đoạn thử lấy độ dài đoạn
            on_progress=seen.append,
        )
        assert seen and seen[-1] == 1.0 and max(seen[:-1] or [0]) <= 1.0


# ── Trang Xuất bản ───────────────────────────────────────────────────────────────

pytest.importorskip("PySide6")


@pytest.fixture
def publish_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from PySide6.QtWidgets import QApplication

    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import publish_page as module

    QApplication.instance() or QApplication([])
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(module.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    page = module.PublishPage(container)
    video = tmp_path / "phim.mp4"
    video.write_bytes(b"\x00")
    subtitle = tmp_path / "phim.srt"
    subtitle.write_text(SRT, encoding="utf-8")
    page._video_edit.setText(str(video))
    page._subtitle_edit.setText(str(subtitle))
    page._output_edit.setText(str(tmp_path / "phim_phude.mkv"))
    yield page, module
    page.close()


class TestPublishPageClip:
    def test_default_is_the_whole_film(self, publish_page) -> None:
        page, _module = publish_page
        request = page._build_request()
        assert request is not None and not request.is_clip
        assert request.output_path.name == "phim_phude.mkv"
        assert not page._preview_length.isEnabled()

    def test_preview_builds_a_clip_with_a_separate_file_name(self, publish_page) -> None:
        page, _module = publish_page
        page._preview_check.setChecked(True)
        page._preview_start.setValue(12)
        page._preview_length.setValue(75)
        request = page._build_request()
        assert request is not None and request.is_clip
        assert (request.clip_start_sec, request.clip_duration_sec) == (720.0, 75.0)
        assert request.output_path.name == "phim_phude_xemthu.mkv"   # không ghi đè bản đầy đủ
        assert page._preview_length.isEnabled()

    def test_start_beyond_the_video_is_refused_before_running(
        self, publish_page, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page, _module = publish_page
        page._preview_check.setChecked(True)
        page._preview_start.setValue(30)
        monkeypatch.setattr(page, "_probe_duration_sec", lambda _path: 600.0)
        started: list[object] = []
        monkeypatch.setattr(page, "_warn", lambda message: started.append(message))
        page._start_render()
        assert page._thread is None and len(started) == 1 and "10.0" in started[0]
        assert page._start_button.isEnabled()
