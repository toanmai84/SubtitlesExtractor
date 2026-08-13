"""[v3.23.98] Đọc track VobSub (S_VOBSUB) trực tiếp từ file Matroska (.mkv).

Thuần Python, KHÔNG cần ffmpeg/mkvextract. ffmpeg KHÔNG có muxer 'vobsub' (chỉ có
demuxer) nên không xuất được .idx/.sub. Subtitle Edit đọc thẳng container Matroska:
``CodecPrivate`` của track S_VOBSUB CHÍNH là nội dung .idx (palette + size), và mỗi
block phụ đề chứa một SPU (subpicture) thô kèm mốc thời gian. Module này phân tích
EBML đủ để lấy hai thứ đó, rồi ``vobsub_decoder.decode_spu`` dựng bitmap.

Chỉ hỗ trợ block KHÔNG lacing (mỗi block 1 SPU - mặc định VobSub). Block lacing bỏ qua.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# ── Element ID Matroska (giữ nguyên marker bit) ────────────────────────────────
_ID_EBML = 0x1A45DFA3
_ID_SEGMENT = 0x18538067
_ID_INFO = 0x1549A966
_ID_TIMESTAMP_SCALE = 0x2AD7B1
_ID_TRACKS = 0x1654AE6B
_ID_TRACK_ENTRY = 0xAE
_ID_TRACK_NUMBER = 0xD7
_ID_TRACK_TYPE = 0x83
_ID_CODEC_ID = 0x86
_ID_CODEC_PRIVATE = 0x63A2
_ID_CONTENT_ENCODINGS = 0x6D80
_ID_CONTENT_ENCODING = 0x6240
_ID_CONTENT_COMPRESSION = 0x5034
_ID_CONTENT_COMP_ALGO = 0x4254
_ID_CONTENT_COMP_SETTINGS = 0x4255
_ID_CLUSTER = 0x1F43B675
_ID_TIMESTAMP = 0xE7
_ID_SIMPLE_BLOCK = 0xA3
_ID_BLOCK_GROUP = 0xA0
_ID_BLOCK = 0xA1

# Container ta cần đi sâu vào; phần còn lại thì bỏ qua (seek).
_CONTAINERS = {_ID_SEGMENT, _ID_INFO, _ID_TRACKS, _ID_TRACK_ENTRY, _ID_CLUSTER,
               _ID_BLOCK_GROUP}
_ENCODING_CONTAINERS = {_ID_CONTENT_ENCODINGS, _ID_CONTENT_ENCODING,
                        _ID_CONTENT_COMPRESSION}


@dataclass
class _VobSubTrack:
    track_number: int = -1
    codec_private: bytes = b""
    comp_algo: int = -1  # -1: không nén; 0: zlib; 3: header stripping
    comp_settings: bytes = field(default=b"")


def _read_vint(buf: bytes, pos: int, keep_marker: bool) -> tuple[int, int] | None:
    """Đọc số biến độ dài EBML tại ``buf[pos]``. Trả (giá trị, độ_dài_byte) hoặc None."""
    if pos >= len(buf):
        return None
    b0 = buf[pos]
    if b0 == 0:
        return None
    mask = 0x80
    length = 1
    while not (b0 & mask):
        mask >>= 1
        length += 1
        if length > 8:
            return None
    if pos + length > len(buf):
        return None
    value = b0 if keep_marker else (b0 & (mask - 1))
    for i in range(1, length):
        value = (value << 8) | buf[pos + i]
    return value, length


def _read_element(buf: bytes, pos: int) -> tuple[int, int, int, bool] | None:
    """Đọc header element: trả (id, data_start, data_size, unknown_size)."""
    id_res = _read_vint(buf, pos, keep_marker=True)
    if id_res is None:
        return None
    elem_id, id_len = id_res
    size_res = _read_vint(buf, pos + id_len, keep_marker=False)
    if size_res is None:
        return None
    size, size_len = size_res
    unknown = size == (1 << (7 * size_len)) - 1
    return elem_id, pos + id_len + size_len, size, unknown


def _parse_uint(buf: bytes, start: int, size: int) -> int:
    value = 0
    for i in range(size):
        value = (value << 8) | buf[start + i]
    return value


def _parse_content_encodings(buf: bytes, start: int, end: int) -> tuple[int, bytes]:
    """Đọc cây ContentEncodings -> (thuật_toán_nén, settings). algo=-1 nếu không có."""
    algo = -1
    settings = b""
    pos = start
    while pos < end:
        elem = _read_element(buf, pos)
        if elem is None:
            break
        elem_id, data_start, size, _ = elem
        data_end = min(data_start + size, end)
        if elem_id in _ENCODING_CONTAINERS:
            sub_algo, sub_settings = _parse_content_encodings(buf, data_start, data_end)
            if sub_algo >= 0:
                algo = sub_algo
            if sub_settings:
                settings = sub_settings
        elif elem_id == _ID_CONTENT_COMP_ALGO:
            algo = _parse_uint(buf, data_start, size)
        elif elem_id == _ID_CONTENT_COMP_SETTINGS:
            settings = buf[data_start:data_end]
        pos = data_end
    return algo, settings


def _decompress_frame(data: bytes, algo: int, settings: bytes) -> bytes:
    """Giải nén dữ liệu frame theo ContentCompAlgo của Matroska.

    Hỗ trợ zlib (algo 0, hoặc tự nhận diện qua magic 0x78) và header-stripping (algo 3).
    """
    is_zlib = len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x9C, 0xDA)
    if algo == 0 or is_zlib:
        try:
            return zlib.decompress(data)
        except zlib.error:
            return data
    if algo == 3:  # header stripping: nối lại phần byte đã lược bỏ
        return settings + data
    return data


def _walk(buf: bytes, start: int, end: int, target: _VobSubTrack,
          state: dict, results: list) -> None:
    """Duyệt các element trong [start, end); thu palette + SPU blocks cho track VobSub."""
    pos = start
    while pos < end:
        elem = _read_element(buf, pos)
        if elem is None:
            break
        elem_id, data_start, size, unknown = elem
        data_end = end if unknown else min(data_start + size, end)

        if elem_id == _ID_TIMESTAMP_SCALE:
            state["timescale_ns"] = _parse_uint(buf, data_start, size)
        elif elem_id == _ID_TRACK_ENTRY:
            _parse_track_entry(buf, data_start, data_end, target)
        elif elem_id == _ID_TIMESTAMP:  # mốc thời gian của Cluster
            state["cluster_ts"] = _parse_uint(buf, data_start, size)
        elif elem_id in (_ID_SIMPLE_BLOCK, _ID_BLOCK):
            _parse_block(buf, data_start, data_end, target, state, results)
        elif elem_id in _CONTAINERS:
            _walk(buf, data_start, data_end, target, state, results)

        pos = data_end


def _parse_track_entry(buf: bytes, start: int, end: int, target: _VobSubTrack) -> None:
    """Đọc một TrackEntry; nếu là S_VOBSUB thì ghi track_number + codec_private."""
    number = -1
    codec_id = b""
    codec_private = b""
    comp_algo = -1
    comp_settings = b""
    pos = start
    while pos < end:
        elem = _read_element(buf, pos)
        if elem is None:
            break
        elem_id, data_start, size, _ = elem
        data_end = min(data_start + size, end)
        if elem_id == _ID_TRACK_NUMBER:
            number = _parse_uint(buf, data_start, size)
        elif elem_id == _ID_CODEC_ID:
            codec_id = buf[data_start:data_end]
        elif elem_id == _ID_CODEC_PRIVATE:
            codec_private = buf[data_start:data_end]
        elif elem_id == _ID_CONTENT_ENCODINGS:
            comp_algo, comp_settings = _parse_content_encodings(buf, data_start, data_end)
        pos = data_end
    if codec_id.startswith(b"S_VOBSUB") and target.track_number < 0:
        target.track_number = number
        target.codec_private = codec_private
        target.comp_algo = comp_algo
        target.comp_settings = comp_settings


def _parse_block(buf: bytes, start: int, end: int, target: _VobSubTrack,
                 state: dict, results: list) -> None:
    """Lấy SPU + thời gian từ (Simple)Block nếu thuộc track VobSub và không lacing.

    (Lacing bị bỏ qua vì VobSub mặc định một SPU mỗi block.)
    """
    if target.track_number < 0:
        return
    track_res = _read_vint(buf, start, keep_marker=False)
    if track_res is None:
        return
    track_num, tlen = track_res
    if track_num != target.track_number:
        return
    rel_pos = start + tlen
    if rel_pos + 3 > end:
        return
    rel_ts = int.from_bytes(buf[rel_pos:rel_pos + 2], "big", signed=True)
    flags = buf[rel_pos + 2]
    if flags & 0x06:  # có lacing -> bỏ qua (VobSub thường không lacing)
        return
    spu = buf[rel_pos + 3:end]
    spu = _decompress_frame(bytes(spu), target.comp_algo, target.comp_settings)
    timescale_ns = state.get("timescale_ns", 1_000_000)
    abs_ticks = state.get("cluster_ts", 0) + rel_ts
    start_ms = int(abs_ticks * timescale_ns / 1_000_000)
    results.append((start_ms, spu))


def extract_vobsub_track(mkv_path: Path) -> tuple[str, list[tuple[int, bytes]]]:
    """Trích track VobSub đầu tiên từ file MKV.

    Returns:
        ``(idx_text, [(start_ms, spu_bytes), …])`` - ``idx_text`` là CodecPrivate (chứa
        palette) để ``vobsub_decoder.parse_idx`` lấy bảng màu; mỗi phần tử là một SPU thô.
        Danh sách rỗng nếu không tìm thấy track S_VOBSUB.
    """
    buf = mkv_path.read_bytes()
    pos = 0
    target = _VobSubTrack()
    state: dict = {"timescale_ns": 1_000_000, "cluster_ts": 0}
    results: list[tuple[int, bytes]] = []
    n = len(buf)
    while pos < n:
        elem = _read_element(buf, pos)
        if elem is None:
            break
        elem_id, data_start, size, unknown = elem
        data_end = n if unknown else min(data_start + size, n)
        if elem_id == _ID_SEGMENT:
            _walk(buf, data_start, data_end, target, state, results)
        pos = data_end
    idx_text = ""
    if target.codec_private:
        idx_text = target.codec_private.decode("utf-8", errors="ignore")
    results.sort(key=lambda item: item[0])
    logger.info("MKV VobSub: track #{}, {} block.", target.track_number, len(results))
    return idx_text, results


def is_mkv(path: Path) -> bool:
    """Nhận diện file Matroska qua magic bytes EBML (1A 45 DF A3)."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x1a\x45\xdf\xa3"
    except OSError:
        return False


__all__ = ["extract_vobsub_track", "is_mkv"]
