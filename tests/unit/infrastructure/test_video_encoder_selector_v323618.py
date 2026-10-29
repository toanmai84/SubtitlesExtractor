"""[v3.23.618] Chọn bộ mã hoá GPU nếu máy hỗ trợ (nối cả bộ + xuất bản), CPU nếu không.

ffmpeg liệt kê ``hevc_nvenc``/``hevc_qsv`` cả trên máy không có GPU (đã thấy thật trên máy
kiểm thử), nên chỉ THỬ MÃ HOÁ mới biết dùng được hay không.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video import video_encoder_selector as sel


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    sel.reset_encoder_cache()
    yield
    sel.reset_encoder_cache()


def _fake_machine(monkeypatch: pytest.MonkeyPatch, working: set[str]) -> list[str]:
    """Giả lập máy chỉ mã hoá được bằng các bộ trong ``working``; trả danh sách đã thử."""
    tried: list[str] = []

    def fake_probe(_ffmpeg: str, encoder: str) -> bool:
        tried.append(encoder)
        return encoder in working

    monkeypatch.setattr(sel, "probe_encoder", fake_probe)
    monkeypatch.setattr(sel.sys, "platform", "win32")
    return tried


def test_prefers_nvidia_gpu_when_it_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_machine(monkeypatch, {"hevc_nvenc", "hevc_qsv", "libx265"})
    assert sel.pick_video_encoder("ffmpeg", "hevc") == "hevc_nvenc"


def test_intel_gpu_used_when_no_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_machine(monkeypatch, {"hevc_qsv", "libx265"})
    assert sel.pick_video_encoder("ffmpeg", "hevc") == "hevc_qsv"


def test_cpu_only_machine_gets_software_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    tried = _fake_machine(monkeypatch, {"libx265", "libx264"})
    assert sel.pick_video_encoder("ffmpeg", "hevc") == "libx265"
    assert tried == ["hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265"]  # không thử VideoToolbox


def test_hevc_falls_back_to_h264_without_libx265(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_machine(monkeypatch, {"libx264"})
    assert sel.pick_video_encoder("ffmpeg", "hevc") == "libx264"


def test_result_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    tried = _fake_machine(monkeypatch, {"hevc_nvenc"})
    sel.pick_video_encoder("ffmpeg", "hevc")
    sel.pick_video_encoder("ffmpeg", "hevc")
    assert tried == ["hevc_nvenc"]


def test_broken_gpu_is_skipped_next_time(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_machine(monkeypatch, {"hevc_nvenc", "libx265"})
    assert sel.pick_video_encoder("ffmpeg", "hevc") == "hevc_nvenc"
    sel.mark_encoder_broken("ffmpeg", "hevc_nvenc")
    assert sel.pick_video_encoder("ffmpeg", "hevc") == "libx265"


def test_macos_only_tries_videotoolbox(monkeypatch: pytest.MonkeyPatch) -> None:
    tried = _fake_machine(monkeypatch, {"hevc_videotoolbox"})
    monkeypatch.setattr(sel.sys, "platform", "darwin")
    assert sel.pick_video_encoder("ffmpeg", "hevc") == "hevc_videotoolbox"
    assert tried == ["hevc_videotoolbox"]


@pytest.mark.parametrize(
    ("encoder", "must_have", "must_not_have"),
    [
        ("libx264", ["-crf", "20"], ["-cq"]),
        ("libx265", ["-crf", "22", "-preset", "fast"], ["-cq"]),
        ("hevc_nvenc", ["-cq", "22", "-b:v", "0"], ["-crf"]),
        ("hevc_qsv", ["-global_quality", "22"], ["-crf"]),
        ("hevc_amf", ["-qp_i", "22"], ["-crf"]),
        ("hevc_videotoolbox", ["-q:v", "61"], ["-crf"]),
    ],
)
def test_quality_args_use_each_encoder_scale(
    encoder: str, must_have: list[str], must_not_have: list[str]
) -> None:
    args = sel.encoder_quality_args(encoder, 20)
    joined = " ".join(args)
    assert all(a in args for a in must_have), joined
    assert not any(a in args for a in must_not_have)


def test_hevc_in_mp4_is_tagged_hvc1() -> None:
    assert "hvc1" in sel.encoder_quality_args("libx265", output_path=Path("a.mp4"))
    assert "hvc1" not in sel.encoder_quality_args("libx265", output_path=Path("a.mkv"))
    assert "hvc1" not in sel.encoder_quality_args("libx264", output_path=Path("a.mp4"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="Cần ffmpeg")
def test_real_probe_accepts_cpu_and_rejects_unknown_encoder() -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert sel.probe_encoder(ffmpeg, "libx264")
    assert not sel.probe_encoder(ffmpeg, "khong_ton_tai_encoder")


# ── Trang Xuất bản ───────────────────────────────────────────────────────────


def test_publish_auto_choice_is_no_longer_hardcoded_nvenc() -> None:
    from subtitles_extractor.presentation.pages.publish_page import _ENCODER_CHOICES

    assert _ENCODER_CHOICES[0] == ("publish.enc_auto", "auto")


def test_render_resolves_auto_and_falls_back_to_cpu_when_gpu_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.infrastructure.video import ffmpeg_video_renderer as renderer
    from subtitles_extractor.infrastructure.video.video_render_command import (
        RenderMode,
        RenderRequest,
        VideoRenderError,
    )

    subtitle = tmp_path / "a.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nxin chào\n", encoding="utf-8")
    (tmp_path / "in.mp4").write_bytes(b"video")
    monkeypatch.setattr(renderer, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(renderer, "pick_video_encoder", lambda _f, _c: "h264_nvenc")
    used: list[str] = []

    def fake_run(command: list[str], request: RenderRequest, *_args: object) -> Path:
        encoder = command[command.index("-c:v") + 1]
        used.append(encoder)
        if encoder == "h264_nvenc":
            raise VideoRenderError("ffmpeg lỗi (mã 1): OpenEncodeSessionEx failed")
        request.output_path.write_bytes(b"ok")  # [v3.23.652] ffmpeg thật tạo tệp (tạm) rồi mới được thay vào đích
        return request.output_path

    monkeypatch.setattr(renderer, "_run_render", fake_run)
    request = RenderRequest(
        tmp_path / "in.mp4", tmp_path / "out.mp4", RenderMode.HARD_SUB,
        subtitle_path=subtitle, video_encoder="auto",
    )
    renderer.render_video(request)
    assert used == ["h264_nvenc", "libx264"]


def test_render_does_not_retry_failures_unrelated_to_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kiểu chỉ sao chép hình (không mã hoá) lỗi thì báo lỗi luôn, không chạy lại."""
    from subtitles_extractor.infrastructure.video import ffmpeg_video_renderer as renderer
    from subtitles_extractor.infrastructure.video.video_render_command import (
        RenderMode,
        RenderRequest,
        VideoRenderError,
    )

    subtitle = tmp_path / "a.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nxin chào\n", encoding="utf-8")
    (tmp_path / "in.mp4").write_bytes(b"video")
    monkeypatch.setattr(renderer, "find_ffmpeg", lambda: "ffmpeg")
    calls: list[int] = []

    def fake_run(*_args: object) -> Path:
        calls.append(1)
        raise VideoRenderError("ffmpeg lỗi (mã 1): No such file")

    monkeypatch.setattr(renderer, "_run_render", fake_run)
    request = RenderRequest(
        tmp_path / "in.mp4", tmp_path / "out.mkv", RenderMode.SOFT_SUB,
        subtitle_path=subtitle, video_encoder="h264_nvenc",
    )
    with pytest.raises(VideoRenderError):
        renderer.render_video(request)
    assert calls == [1]


# ── Nối cả bộ ────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Cần ffmpeg + ffprobe",
)
def test_concat_retries_with_cpu_when_gpu_encoder_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.infrastructure.video import ffmpeg_concat_adapter as adapter

    clips = []
    for name, fps in (("01", 25), ("02", 30)):
        path = tmp_path / f"{name}.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc2=s=160x284:r={fps}:d=1",
             "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
            check=True, timeout=60,
        )
        clips.append(path)

    # Máy "có" NVENC theo phép thử nhưng mã hoá thật thì lỗi (vd hết phiên).
    monkeypatch.setattr(adapter, "pick_video_encoder", lambda _f, _c: "hevc_nvenc")
    output = adapter.concatenate_videos(clips, tmp_path / "tron_bo.mkv")
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert codec == "hevc"  # đã chạy lại bằng libx265
