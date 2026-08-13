"""[v3.23.98] Test parser MKV/EBML trích track VobSub (palette + SPU blocks)."""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.infrastructure.video import mkv_vobsub, vobsub_decoder


def _enc_size(value: int) -> bytes:
    length = 1
    while value >= (1 << (7 * length)) - 1:
        length += 1
    return (value | (1 << (7 * length))).to_bytes(length, "big")


def _elem(id_hex: str, data: bytes) -> bytes:
    return bytes.fromhex(id_hex) + _enc_size(len(data)) + data


def _synthetic_spu() -> bytes:
    rle = bytes([0x11, 0x12])
    ctrl = bytes([0, 0, 0, 6, 0x05, 0, 0, 3, 0, 0, 1, 0x06, 0, 4, 0, 5,
                  0x03, 1, 0x23, 0x04, 0x0F, 0xFF, 0x01, 0xFF])
    body = rle + ctrl
    dcsqt = 4 + len(rle)
    total = 4 + len(body)
    return bytes([total >> 8, total & 0xFF, dcsqt >> 8, dcsqt & 0xFF]) + body


def _build_mkv(spu: bytes, idx_text: bytes) -> bytes:
    track_entry = (_elem("d7", bytes([1])) + _elem("83", bytes([0x11]))
                   + _elem("86", b"S_VOBSUB") + _elem("63a2", idx_text))
    tracks = _elem("1654ae6b", _elem("ae", track_entry))
    info = _elem("1549a966", _elem("2ad7b1", (1_000_000).to_bytes(3, "big")))
    block_data = (bytes([0x81]) + (2000).to_bytes(2, "big", signed=True)
                  + bytes([0x00]) + spu)
    cluster = _elem("1f43b675", _elem("e7", bytes([0])) + _elem("a3", block_data))
    segment = _elem("18538067", info + tracks + cluster)
    ebml_header = _elem("1a45dfa3", b"\x42\x86\x81\x01")
    return ebml_header + segment


def test_extract_and_decode_vobsub_from_mkv(tmp_path: Path) -> None:
    idx_text = b"palette: 000000, ff0000, 00ff00, 0000ff\nsize: 4x2\n"
    mkv = _build_mkv(_synthetic_spu(), idx_text)
    path = tmp_path / "sample.mkv"
    path.write_bytes(mkv)

    assert mkv_vobsub.is_mkv(path)
    idx, blocks = mkv_vobsub.extract_vobsub_track(path)
    assert "palette:" in idx
    assert len(blocks) == 1

    start_ms, spu_bytes = blocks[0]
    assert start_ms == 2000  # cluster_ts 0 + rel_ts 2000, timescale 1ms

    palette = vobsub_decoder.parse_idx(idx).palette
    image = vobsub_decoder.decode_spu(spu_bytes, palette)
    assert image is not None
    assert image.size == (4, 2)
    assert image.getpixel((0, 0)) == (255, 0, 0, 255)
    assert image.getpixel((3, 1)) == (0, 255, 0, 255)


def test_is_mkv_rejects_non_matroska(tmp_path: Path) -> None:
    path = tmp_path / "v.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    assert not mkv_vobsub.is_mkv(path)


def test_extract_returns_empty_when_no_vobsub(tmp_path: Path) -> None:
    # MKV hợp lệ nhưng track CodecID khác (không S_VOBSUB) -> không block.
    track_entry = _elem("d7", bytes([1])) + _elem("86", b"S_TEXT/UTF8")
    tracks = _elem("1654ae6b", _elem("ae", track_entry))
    segment = _elem("18538067", tracks)
    path = tmp_path / "notvob.mkv"
    path.write_bytes(_elem("1a45dfa3", b"\x42\x86\x81\x01") + segment)

    _, blocks = mkv_vobsub.extract_vobsub_track(path)
    assert blocks == []


def test_extract_decodes_zlib_compressed_block(tmp_path: Path) -> None:
    # [v3.23.100] Track VobSub thường nén zlib mỗi block (ContentEncoding) -> giải nén.
    import zlib

    spu = _synthetic_spu()
    comp = _elem("5034", _elem("4254", bytes([0])))           # ContentCompAlgo = 0 (zlib)
    enc = _elem("6d80", _elem("6240", comp))
    track_entry = (_elem("d7", bytes([3])) + _elem("83", bytes([0x11]))
                   + _elem("86", b"S_VOBSUB")
                   + _elem("63a2", b"palette: 000000, ff0000, 00ff00, 0000ff\n") + enc)
    tracks = _elem("1654ae6b", _elem("ae", track_entry))
    block_data = (bytes([0x83]) + (2000).to_bytes(2, "big", signed=True)
                  + bytes([0x00]) + zlib.compress(spu))
    cluster = _elem("1f43b675", _elem("e7", bytes([0])) + _elem("a3", block_data))
    segment = _elem("18538067", tracks + cluster)
    path = tmp_path / "comp.mkv"
    path.write_bytes(_elem("1a45dfa3", b"\x42\x86\x81\x01") + segment)

    idx, blocks = mkv_vobsub.extract_vobsub_track(path)
    assert len(blocks) == 1
    _, spu_out = blocks[0]
    assert spu_out == spu  # đã giải nén về SPU gốc

    image = vobsub_decoder.decode_spu(spu_out, vobsub_decoder.parse_idx(idx).palette)
    assert image is not None
    assert image.getpixel((0, 0)) == (255, 0, 0, 255)
