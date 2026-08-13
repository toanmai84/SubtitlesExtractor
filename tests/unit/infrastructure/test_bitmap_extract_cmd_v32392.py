"""[v3.23.93] Tách phụ đề bitmap: sub2video (filter_complex) + lọc frame trống."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from subtitles_extractor.domain.ports.embedded_subtitle_port import (
    EmbeddedSubtitleTrack,
)
from subtitles_extractor.infrastructure.video.ffmpeg_embedded_subtitle_adapter import (
    FfmpegEmbeddedSubtitleAdapter,
)


def test_bitmap_command_uses_sub2video_filter(tmp_path: Path, monkeypatch) -> None:
    adapter = FfmpegEmbeddedSubtitleAdapter()
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(adapter, "_run_ffmpeg",
                        lambda command, label, timeout=None: captured.__setitem__("command", command))
    monkeypatch.setattr(adapter, "_probe_bitmap_timings", lambda *a, **k: [])

    video = tmp_path / "phim.mkv"
    video.write_bytes(b"x")
    track = EmbeddedSubtitleTrack(0, "hdmv_pgs_subtitle", is_bitmap=True)
    adapter._extract_bitmap_via_sub2video(video, track)

    cmd = captured["command"]
    # PHẢI chuyển subtitle -> video qua filter_complex (sub2video), KHÔNG map thẳng.
    assert "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert fc == "[0:s:0]copy[v]"
    assert cmd[cmd.index("-map") + 1] == "[v]"
    assert cmd[cmd.index("-c:v") + 1] == "png"
    assert cmd[cmd.index("-f") + 1] == "image2"


def test_is_blank_image_detects_empty_frames() -> None:
    tmp = Path(tempfile.mkdtemp())
    is_blank = FfmpegEmbeddedSubtitleAdapter._is_blank_image

    transparent = tmp / "t.png"
    Image.new("RGBA", (80, 30), (0, 0, 0, 0)).save(transparent)
    assert is_blank(transparent) is True

    solid = tmp / "s.png"
    Image.new("RGB", (80, 30), (0, 0, 0)).save(solid)
    assert is_blank(solid) is True

    content = tmp / "c.png"
    img = Image.new("RGBA", (80, 30), (0, 0, 0, 0))
    img.putpixel((5, 5), (255, 255, 255, 255))
    img.save(content)
    assert is_blank(content) is False


def test_is_blank_image_unreadable_kept() -> None:
    tmp = Path(tempfile.mkdtemp())
    bad = tmp / "bad.png"
    bad.write_bytes(b"not a real png")
    # Không đọc được -> coi là KHÔNG trống (giữ lại, an toàn).
    assert FfmpegEmbeddedSubtitleAdapter._is_blank_image(bad) is False


def test_extract_bitmap_routes_dvdsub_to_vobsub(tmp_path, monkeypatch) -> None:
    # [v3.23.95] Router phải đọc track.codec (không phải codec_name) và route đúng.
    from subtitles_extractor.domain.ports.embedded_subtitle_port import (
        EmbeddedExtractionResult,
    )

    adapter = FfmpegEmbeddedSubtitleAdapter()
    calls: list[str] = []
    monkeypatch.setattr(adapter, "_extract_vobsub",
                        lambda v, t: calls.append("vobsub") or ["frame"])
    monkeypatch.setattr(adapter, "_extract_bitmap_via_sub2video",
                        lambda v, t: calls.append("sub2video")
                        or EmbeddedExtractionResult(bitmap_frames=[], is_bitmap=True))

    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")

    dvd = EmbeddedSubtitleTrack(0, "dvd_subtitle", is_bitmap=True)
    adapter._extract_bitmap(video, dvd)
    assert calls == ["vobsub"]

    calls.clear()
    pgs = EmbeddedSubtitleTrack(0, "hdmv_pgs_subtitle", is_bitmap=True)
    adapter._extract_bitmap(video, pgs)
    assert calls == ["sub2video"]


def test_extract_vobsub_skips_non_mkv(tmp_path) -> None:
    # [v3.23.98] Đường VobSub thuần Python chỉ áp dụng cho MKV; phi-MKV trả [] (để fallback).
    adapter = FfmpegEmbeddedSubtitleAdapter()
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # không phải magic EBML
    track = EmbeddedSubtitleTrack(0, "dvd_subtitle", is_bitmap=True)
    assert adapter._extract_vobsub(video, track) == []
