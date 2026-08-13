"""[v3.23.95] Test giải mã VobSub (.idx + SPU MPEG-PS tổng hợp) -> bitmap."""

from __future__ import annotations

from subtitles_extractor.infrastructure.video import vobsub_decoder


def _build_synthetic_sub() -> bytes:
    """Dựng .sub MPEG-PS chứa 1 SPU 4x2: dòng 0 màu 1, dòng 1 màu 2."""
    top_rle = bytes([0x11])  # length 4, color 1
    bot_rle = bytes([0x12])  # length 4, color 2
    rle = top_rle + bot_rle
    ctrl = bytes([0x00, 0x00])               # delay
    ctrl += bytes([0x00, 0x06])              # next_dcsq = self -> dừng
    ctrl += bytes([0x05, 0x00, 0x00, 0x03, 0x00, 0x00, 0x01])  # area 4x2
    ctrl += bytes([0x06, 0x00, 0x04, 0x00, 0x05])  # top_addr=4, bottom_addr=5
    ctrl += bytes([0x03, 0x01, 0x23])        # color idx [0,1,2,3]
    ctrl += bytes([0x04, 0x0F, 0xFF])        # alpha [0,F,F,F]
    ctrl += bytes([0x01, 0xFF])              # start + end
    body = rle + ctrl
    dcsqt_offset = 4 + len(rle)
    total = 4 + len(body)
    spu = bytes([(total >> 8) & 0xFF, total & 0xFF,
                 (dcsqt_offset >> 8) & 0xFF, dcsqt_offset & 0xFF]) + body

    pack = b"\x00\x00\x01\xBA" + b"\x00" * 10
    pes_len = 4 + len(spu)
    pes = b"\x00\x00\x01\xBD" + bytes([(pes_len >> 8) & 0xFF, pes_len & 0xFF])
    pes += bytes([0x80, 0x00, 0x00, 0x20])   # flags, pts_flags, hdr_len=0, sub-stream id
    pes += spu
    return pack + pes


def test_decode_event_renders_correct_pixels() -> None:
    palette = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)] + [(0, 0, 0)] * 12
    img = vobsub_decoder.decode_event(_build_synthetic_sub(), 0, palette)
    assert img is not None
    assert img.size == (4, 2)
    assert img.getpixel((0, 0)) == (255, 0, 0, 255)  # dòng top = palette[1]
    assert img.getpixel((3, 1)) == (0, 255, 0, 255)  # dòng bottom = palette[2]


def test_parse_idx_palette_and_events() -> None:
    idx = (
        "palette: 000000, ff0000, 00ff00, 0000ff\n"
        "timestamp: 00:00:01:234, filepos: 000000000\n"
        "timestamp: 00:00:05:000, filepos: 0000000ab\n"
    )
    parsed = vobsub_decoder.parse_idx(idx)
    assert parsed.palette[1] == (255, 0, 0)
    assert [(e.start_ms, e.filepos) for e in parsed.events] == [(1234, 0), (5000, 0xAB)]


def test_decode_all_pairs_timing() -> None:
    idx = (
        "palette: 000000, ff0000, 00ff00, 0000ff\n"
        "timestamp: 00:00:02:000, filepos: 000000000\n"
    )
    results = vobsub_decoder.decode_all(idx, _build_synthetic_sub())
    assert len(results) == 1
    start_ms, end_ms, img = results[0]
    assert start_ms == 2000
    assert end_ms == 5000  # sự kiện cuối +3000ms
    assert img.size == (4, 2)


def test_parse_spu_timing_extracts_display_duration() -> None:
    # [v3.23.101] Thời lượng lấy từ lệnh 0x02 trong DCSQ thứ hai (khớp Subtitle Edit).
    rle = bytes([0x11, 0x12])
    dcsq1 = bytes([0, 0]) + bytes([0, 12]) + bytes([0x01, 0xFF])      # bật tại delay 0
    dcsq2 = (502).to_bytes(2, "big") + bytes([0, 12]) + bytes([0x02, 0xFF])  # tắt tại 502
    body = rle + dcsq1 + dcsq2
    dcsqt = 4 + len(rle)
    total = 4 + len(body)
    spu = bytes([total >> 8, total & 0xFF, dcsqt >> 8, dcsqt & 0xFF]) + body

    start_off, end_off = vobsub_decoder.parse_spu_timing(spu)
    assert start_off == 0
    assert end_off == round(502 * 1024 / 90)  # = 5712 ms
