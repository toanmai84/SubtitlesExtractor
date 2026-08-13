"""[v3.23.361] Kiểm thử decoder PGS/.sup và đường trích PGS trực tiếp (SE-style)."""
from __future__ import annotations

import struct
from pathlib import Path

from subtitles_extractor.domain.ports.embedded_subtitle_port import EmbeddedSubtitleTrack
from subtitles_extractor.infrastructure.video import pgs_decoder
from subtitles_extractor.infrastructure.video.ffmpeg_embedded_subtitle_adapter import (
    FfmpegEmbeddedSubtitleAdapter,
)
from subtitles_extractor.infrastructure.video.pgs_decoder import _decode_rle


def _seg(seg_type: int, pts_ms: int, payload: bytes) -> bytes:
    return struct.pack(">HIIBH", 0x5047, pts_ms * 90, 0, seg_type, len(payload)) + payload


def _sample_sup() -> bytes:
    """Một Display Set hiển thị ảnh 4×2 tại (10,20), xoá sau 3s."""
    rle = bytes([0x01, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00])
    ods = (
        struct.pack(">HBB", 0, 0, 0xC0)
        + struct.pack(">I", len(rle) + 4)[1:]
        + struct.pack(">HH", 4, 2)
        + rle
    )
    pds = bytes([0, 0, 1, 0xEB, 0x80, 0x80, 0xFF])
    pcs = struct.pack(">HHBHBBBB", 1920, 1080, 0x10, 0, 0x80, 0, 0, 1) + struct.pack(
        ">HBBHH", 0, 0, 0, 10, 20
    )
    pcs_clear = struct.pack(">HHBHBBBB", 1920, 1080, 0x10, 1, 0, 0, 0, 0)
    return (
        _seg(0x16, 1000, pcs)
        + _seg(0x14, 1000, pds)
        + _seg(0x15, 1000, ods)
        + _seg(0x80, 1000, b"")
        + _seg(0x16, 4000, pcs_clear)
        + _seg(0x80, 4000, b"")
    )


def test_rle_single_pixels() -> None:
    assert _decode_rle(bytes([3, 4, 5, 0x00, 0x00]), 3, 1)[0] == [3, 4, 5]


def test_rle_short_color_run() -> None:
    # 00 10LLLLLL CCCCCCCC → L pixel màu C
    assert _decode_rle(bytes([0x00, 0x80 | 5, 7, 0x00, 0x00]), 5, 1)[0] == [7] * 5


def test_rle_long_zero_run() -> None:
    length = 300
    data = bytes([0x00, 0x40 | (length >> 8), length & 0xFF, 0x00, 0x00])
    assert _decode_rle(data, length, 1)[0] == [0] * length


def test_rle_long_color_run() -> None:
    length = 200
    data = bytes([0x00, 0xC0 | (length >> 8), length & 0xFF, 9, 0x00, 0x00])
    assert _decode_rle(data, length, 1)[0] == [9] * length


def test_parse_sup_timing_and_crop() -> None:
    subs = pgs_decoder.parse_sup(_sample_sup())
    assert len(subs) == 1
    sub = subs[0]
    assert sub.start_ms == 1000
    assert sub.end_ms == 4000
    # 2 pixel trắng liền nhau → ảnh crop 2×1.
    assert sub.image.size == (2, 1)
    r, g, b, a = sub.image.convert("RGBA").load()[0, 0]
    assert a == 255 and r > 200 and g > 200 and b > 200


def test_parse_sup_empty_returns_nothing() -> None:
    assert pgs_decoder.parse_sup(b"") == []
    assert pgs_decoder.parse_sup(b"garbage-not-pg") == []


def test_extract_pgs_via_sup_demux_and_decode(monkeypatch) -> None:
    adapter = FfmpegEmbeddedSubtitleAdapter()
    sup_bytes = _sample_sup()

    def fake_run(command: list[str], label: str, timeout: float | None = None) -> None:
        # ffmpeg giả: ghi .sup vào đường dẫn output (phần tử cuối lệnh).
        Path(command[-1]).write_bytes(sup_bytes)

    monkeypatch.setattr(adapter, "_run_ffmpeg", fake_run)
    track = EmbeddedSubtitleTrack(track_index=0, codec="hdmv_pgs_subtitle", is_bitmap=True)
    frames = adapter._extract_pgs_via_sup(Path("/fake.mkv"), track)
    assert len(frames) == 1
    assert frames[0].start_sec == 1.0
    assert frames[0].end_sec == 4.0
    assert frames[0].image_path.exists()
