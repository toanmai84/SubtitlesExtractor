"""[v3.23.361] Giải mã phụ đề PGS (Presentation Graphic Stream / ``.sup``) — SE-style.

Tham khảo cách Subtitle Edit / BDSup2Sub xử lý phụ đề Blu-ray: thay vì để ffmpeg
"sub2video" render từng frame video (quét cả file, chậm), ta DEMUX thẳng stream PGS ra
``.sup`` rồi tự giải mã các *segment* trong bộ nhớ. Mỗi phụ đề là một bitmap + mốc thời
gian lấy TRỰC TIẾP từ PTS (không cần lượt ffprobe thứ hai).

Định dạng theo tài liệu chuẩn (US Patent US 20090185789 A1; mô tả byte-level của
TheScorpius). Mỗi segment có header 13 byte::

    magic "PG" (2) | PTS (4) | DTS (4) | type (1) | size (2)

Type: 0x14=PDS (palette), 0x15=ODS (ảnh RLE), 0x16=PCS (điều phối/timing), 0x17=WDS
(cửa sổ), 0x80=END (kết thúc Display Set). PTS ở 90 kHz → ms = PTS / 90.

Module thuần (không I/O đĩa, không phụ thuộc ffmpeg); chỉ dùng Pillow để dựng ảnh.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field

from PIL import Image

logger = logging.getLogger(__name__)

# Mã segment PGS.
_SEG_PDS = 0x14
_SEG_ODS = 0x15
_SEG_PCS = 0x16
_SEG_WDS = 0x17
_SEG_END = 0x80

_PG_MAGIC = 0x5047  # "PG"
_PTS_CLOCK_HZ = 90  # PTS 90 kHz → chia 90 ra mili-giây.
_SEGMENT_HEADER_LEN = 13


@dataclass(frozen=True, slots=True)
class PgsSubtitle:
    """Một phụ đề PGS đã giải mã: ảnh RGBA + khoảng thời gian hiển thị."""

    start_ms: int
    end_ms: int
    image: Image.Image


@dataclass(slots=True)
class _Segment:
    seg_type: int
    pts_ms: int
    payload: bytes


@dataclass(slots=True)
class _PaletteState:
    """Palette hiện hành (index → RGBA), cập nhật bởi các PDS trong epoch."""

    entries: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)


def _iter_segments(data: bytes) -> list[_Segment]:
    """Tách chuỗi byte ``.sup`` thành danh sách segment (bỏ qua rác cuối file)."""
    segments: list[_Segment] = []
    pos = 0
    total = len(data)
    while pos + _SEGMENT_HEADER_LEN <= total:
        magic, pts, _dts, seg_type, seg_size = struct.unpack_from(">HIIBH", data, pos)
        if magic != _PG_MAGIC:
            # Mất đồng bộ — không thể tin phần còn lại; dừng an toàn.
            logger.warning("PGS: sai magic tại offset %d, dừng phân tích.", pos)
            break
        body_start = pos + _SEGMENT_HEADER_LEN
        body_end = body_start + seg_size
        if body_end > total:
            logger.warning("PGS: segment vượt quá dữ liệu tại offset %d, dừng.", pos)
            break
        segments.append(
            _Segment(
                seg_type=seg_type,
                pts_ms=pts // _PTS_CLOCK_HZ,
                payload=data[body_start:body_end],
            )
        )
        pos = body_end
    return segments


def _parse_palette(payload: bytes, state: _PaletteState) -> None:
    """Cập nhật palette hiện hành từ một PDS (YCbCr+alpha → RGBA)."""
    # payload: palette_id(1) version(1) rồi N×(entry_id(1) Y(1) Cr(1) Cb(1) A(1)).
    pos = 2
    while pos + 5 <= len(payload):
        entry_id, y, cr, cb, alpha = payload[pos : pos + 5]
        state.entries[entry_id] = _ycbcr_to_rgba(y, cb, cr, alpha)
        pos += 5


def _ycbcr_to_rgba(y: int, cb: int, cr: int, alpha: int) -> tuple[int, int, int, int]:
    """Chuyển YCbCr (BT.709, dải TV) + alpha sang RGBA 8-bit đã kẹp [0,255].

    Với OCR, điều quan trọng là tương phản chữ/nền và alpha; BT.709 dải TV là chuẩn cho
    nội dung HD Blu-ray. Sai lệch nhỏ về màu không ảnh hưởng OCR.
    """
    y_lin = (y - 16) * 255.0 / 219.0
    cb_d = cb - 128.0
    cr_d = cr - 128.0
    r = y_lin + 1.5748 * cr_d
    g = y_lin - 0.1873 * cb_d - 0.4681 * cr_d
    b = y_lin + 1.8556 * cb_d
    return (_clamp8(r), _clamp8(g), _clamp8(b), alpha)


def _clamp8(value: float) -> int:
    if value <= 0.0:
        return 0
    if value >= 255.0:
        return 255
    return int(value + 0.5)


def _decode_rle(rle: bytes, width: int, height: int) -> list[list[int]]:
    """Giải nén RLE của ODS thành ma trận CHỈ SỐ palette (height×width).

    Bảng mã (theo US 7912305 B1):
      * ``C`` (C≠0)                         → 1 pixel màu C
      * ``00 00LLLLLL``                     → L pixel màu 0 (1..63)
      * ``00 01LLLLLL LLLLLLLL``            → L pixel màu 0 (64..16383)
      * ``00 10LLLLLL CCCCCCCC``            → L pixel màu C (3..63)
      * ``00 11LLLLLL LLLLLLLL CCCCCCCC``   → L pixel màu C (64..16383)
      * ``00 00``                           → hết dòng
    """
    rows: list[list[int]] = []
    row: list[int] = []
    pos = 0
    length = len(rle)
    while pos < length:
        first = rle[pos]
        pos += 1
        if first != 0:
            row.append(first)
            continue
        if pos >= length:
            break
        second = rle[pos]
        pos += 1
        if second == 0:  # hết dòng
            rows.append(row)
            row = []
            continue
        run_flag = second & 0xC0
        if run_flag == 0x00:  # 00LLLLLL → L pixel màu 0
            run_len = second & 0x3F
            color = 0
        elif run_flag == 0x40:  # 01LLLLLL LLLLLLLL → 14-bit, màu 0
            run_len = ((second & 0x3F) << 8) | rle[pos]
            pos += 1
            color = 0
        elif run_flag == 0x80:  # 10LLLLLL CCCCCCCC → L pixel màu C
            run_len = second & 0x3F
            color = rle[pos]
            pos += 1
        else:  # 11LLLLLL LLLLLLLL CCCCCCCC → 14-bit, màu C
            run_len = ((second & 0x3F) << 8) | rle[pos]
            color = rle[pos + 1]
            pos += 2
        row.extend([color] * run_len)
    if row:
        rows.append(row)

    # Chuẩn hoá về đúng kích thước width×height (đệm/cắt cho chắc chắn).
    normalized: list[list[int]] = []
    for y in range(height):
        if y < len(rows):
            line = rows[y]
        else:
            line = []
        if len(line) < width:
            line = line + [0] * (width - len(line))
        elif len(line) > width:
            line = line[:width]
        normalized.append(line)
    return normalized


@dataclass(slots=True)
class _ObjectAccumulator:
    """Gom các mảnh ODS (một ảnh có thể bị chia nhiều fragment)."""

    width: int = 0
    height: int = 0
    data: bytearray = field(default_factory=bytearray)


def _parse_ods(payload: bytes, store: dict[int, _ObjectAccumulator]) -> None:
    """Đọc một ODS; gom mảnh RLE theo object_id tới khi 'last in sequence'."""
    if len(payload) < 4:
        return
    object_id = struct.unpack_from(">H", payload, 0)[0]
    sequence_flag = payload[3]
    is_first = bool(sequence_flag & 0x80)
    if is_first:
        # ObjectDataLength(3) + Width(2) + Height(2) rồi RLE. Lưu ý: ObjectDataLength
        # ĐÃ TÍNH cả 4 byte Width+Height (theo đính chính của tài liệu).
        width, height = struct.unpack_from(">HH", payload, 7)
        acc = _ObjectAccumulator(width=width, height=height)
        acc.data.extend(payload[11:])
        store[object_id] = acc
    else:
        acc = store.get(object_id)
        if acc is not None:
            acc.data.extend(payload[4:])


def _compose_image(
    canvas_w: int,
    canvas_h: int,
    placements: list[tuple[int, int, _ObjectAccumulator]],
    palette: _PaletteState,
) -> Image.Image | None:
    """Dựng ảnh RGBA từ các object đã đặt vị trí, rồi cắt về vùng có nội dung.

    Ánh xạ chỉ-số-palette → RGBA bằng LUT numpy (nhanh, tránh vòng lặp per-pixel Python
    vốn rất chậm với phụ đề lớn × hàng trăm câu). Trả None nếu không có pixel nhìn thấy.
    """
    if canvas_w <= 0 or canvas_h <= 0:
        return None

    import numpy as np

    lut = np.zeros((256, 4), dtype=np.uint8)
    for index, rgba in palette.entries.items():
        lut[index] = rgba

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    placed_any = False
    for x_off, y_off, acc in placements:
        if acc.width <= 0 or acc.height <= 0 or not acc.data:
            continue
        indices = _decode_rle(bytes(acc.data), acc.width, acc.height)
        index_array = np.array(indices, dtype=np.uint8)  # (h, w)
        rgba_array = lut[index_array]  # (h, w, 4) qua LUT
        obj = Image.fromarray(rgba_array, "RGBA")
        canvas.alpha_composite(obj, dest=(max(0, x_off), max(0, y_off)))
        placed_any = True

    if not placed_any:
        return None
    bbox = canvas.getbbox()  # vùng bao nội dung không trong suốt
    if bbox is None:
        return None
    return canvas.crop(bbox)


def parse_sup(data: bytes) -> list[PgsSubtitle]:
    """Giải mã toàn bộ chuỗi ``.sup`` PGS thành danh sách phụ đề (ảnh + thời gian).

    Args:
        data: Nội dung nhị phân của file/stream ``.sup`` (đã có magic "PG").

    Returns:
        Danh sách :class:`PgsSubtitle` theo thứ tự thời gian. Rỗng nếu không giải mã
        được segment nào.
    """
    segments = _iter_segments(data)
    palette = _PaletteState()
    objects: dict[int, _ObjectAccumulator] = {}
    # Trạng thái của Display Set đang gom (giữa PCS và END).
    pending_pcs: bytes | None = None
    pending_pts_ms = 0

    results: list[PgsSubtitle] = []
    open_start_ms: int | None = None  # phụ đề đang hiển thị chờ mốc kết thúc

    for seg in segments:
        if seg.seg_type == _SEG_PCS:
            pending_pcs = seg.payload
            pending_pts_ms = seg.pts_ms
            # PCS mở đầu epoch mới (Epoch Start) làm mới object store; palette giữ lại
            # để palette-only update vẫn hoạt động, nhưng object cũ không còn hợp lệ.
            objects = {}
        elif seg.seg_type == _SEG_PDS:
            _parse_palette(seg.payload, palette)
        elif seg.seg_type == _SEG_ODS:
            _parse_ods(seg.payload, objects)
        elif seg.seg_type == _SEG_END:
            if pending_pcs is None:
                continue
            placements, canvas_w, canvas_h = _read_pcs_objects(pending_pcs, objects)
            if placements:
                # Bắt đầu một phụ đề: nếu có cái đang mở mà chưa đóng, đóng nó tại đây.
                if open_start_ms is not None and results:
                    results[-1] = _with_end(results[-1], pending_pts_ms)
                image = _compose_image(canvas_w, canvas_h, placements, palette)
                if image is not None:
                    results.append(
                        PgsSubtitle(
                            start_ms=pending_pts_ms,
                            end_ms=pending_pts_ms,  # tạm; đóng khi có DS xoá kế tiếp
                            image=image,
                        )
                    )
                    open_start_ms = pending_pts_ms
            else:
                # PCS rỗng (0 object) = XOÁ phụ đề → chốt mốc kết thúc cái đang mở.
                if open_start_ms is not None and results:
                    results[-1] = _with_end(results[-1], pending_pts_ms)
                    open_start_ms = None
            pending_pcs = None

    # Phụ đề cuối chưa có DS xoá → cho hiển thị mặc định 3s.
    if open_start_ms is not None and results and results[-1].end_ms <= results[-1].start_ms:
        results[-1] = _with_end(results[-1], results[-1].start_ms + 3000)
    return results


def _with_end(sub: PgsSubtitle, end_ms: int) -> PgsSubtitle:
    if end_ms <= sub.start_ms:
        end_ms = sub.start_ms + 500
    return PgsSubtitle(start_ms=sub.start_ms, end_ms=end_ms, image=sub.image)


def _read_pcs_objects(
    pcs: bytes, objects: dict[int, _ObjectAccumulator]
) -> tuple[list[tuple[int, int, _ObjectAccumulator]], int, int]:
    """Đọc PCS → danh sách (x, y, object) đã có ODS + kích thước khung (canvas)."""
    if len(pcs) < 11:
        return [], 0, 0
    canvas_w, canvas_h = struct.unpack_from(">HH", pcs, 0)
    num_objects = pcs[10]
    placements: list[tuple[int, int, _ObjectAccumulator]] = []
    pos = 11
    for _ in range(num_objects):
        if pos + 8 > len(pcs):
            break
        object_id, _window_id, cropped_flag = struct.unpack_from(">HBB", pcs, pos)
        x_off, y_off = struct.unpack_from(">HH", pcs, pos + 4)
        pos += 8
        if cropped_flag == 0x40:
            pos += 8  # bỏ qua 4 trường cropping (không dùng cho OCR)
        acc = objects.get(object_id)
        if acc is not None:
            placements.append((x_off, y_off, acc))
    return placements, canvas_w, canvas_h


__all__ = ["PgsSubtitle", "parse_sup"]
