"""[v3.23.95] Giải mã phụ đề VobSub (.idx/.sub) thành bitmap.

Tham chiếu thuật toán của Subtitle Edit / VobSub2SRT.

VÌ SAO: ffmpeg KHÔNG rasterize được phụ đề ảnh (dvd_subtitle) thành PNG (lỗi "image2 codec
none"). Subtitle Edit nhanh vì đọc THẲNG các gói VobSub rời rạc (mỗi sự kiện 1 bitmap) rồi
giải mã RLE tại chỗ - KHÔNG xử lý theo từng frame video. Module này làm đúng điều đó:

* ``parse_idx`` đọc tệp ``.idx``: bảng màu (palette) + danh sách (mốc thời gian, filepos).
* ``decode_event`` đọc gói MPEG-PS tại ``filepos`` trong ``.sub``, gom payload subpicture,
  phân tích SPU (Sub-Picture Unit), giải mã RLE 2 bit/điểm (xen kẽ trường) -> ảnh RGBA.

Định dạng SPU theo chuẩn DVD subpicture (SP_DCSQT + lệnh 0x01..0x06, 0xFF).
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

# ── Bảng màu mặc định (khi .idx không có palette) ──────────────────────────────
_DEFAULT_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0), (255, 255, 255), (0, 0, 0), (127, 127, 127),
] + [(0, 0, 0)] * 12


@dataclass(frozen=True)
class VobSubEvent:
    """Một sự kiện phụ đề VobSub: mốc thời gian (ms) + vị trí byte trong .sub."""

    start_ms: int
    filepos: int


@dataclass(frozen=True)
class VobSubIndex:
    """Kết quả phân tích .idx: bảng màu 16 màu + danh sách sự kiện theo thứ tự."""

    palette: list[tuple[int, int, int]]
    events: list[VobSubEvent]


def _parse_timestamp_ms(value: str) -> int:
    """'HH:MM:SS:mmm' -> mili-giây."""
    parts = value.strip().split(":")
    if len(parts) != 4:
        return 0
    hours, minutes, seconds, millis = (int(p) for p in parts)
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _parse_palette(value: str) -> list[tuple[int, int, int]]:
    """Dòng 'palette: rrggbb, rrggbb, ...' (YUV hoặc RGB hex) -> list RGB.

    VobSub lưu palette dạng hex RGB (đã chuyển từ YUV khi tạo .idx). Ta đọc trực tiếp.
    """
    colors: list[tuple[int, int, int]] = []
    for token in value.split(","):
        token = token.strip()
        if len(token) == 6:
            try:
                rgb = int(token, 16)
            except ValueError:
                continue
            colors.append(((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF))
    while len(colors) < 16:
        colors.append((0, 0, 0))
    return colors[:16]


def parse_idx(idx_text: str) -> VobSubIndex:
    """Phân tích nội dung tệp ``.idx`` của VobSub.

    Args:
        idx_text: Toàn bộ nội dung văn bản tệp .idx.

    Returns:
        :class:`VobSubIndex` gồm bảng màu và danh sách sự kiện (mốc thời gian + filepos).
    """
    palette = list(_DEFAULT_PALETTE)
    events: list[VobSubEvent] = []
    for raw in idx_text.splitlines():
        line = raw.strip()
        if line.startswith("palette:"):
            palette = _parse_palette(line[len("palette:"):])
        elif line.startswith("timestamp:"):
            # 'timestamp: 00:00:01:234, filepos: 000000000'
            try:
                ts_part, pos_part = line.split(",", 1)
                start_ms = _parse_timestamp_ms(ts_part.split(":", 1)[1])
                filepos = int(pos_part.split(":", 1)[1].strip(), 16)
            except (ValueError, IndexError):
                continue
            events.append(VobSubEvent(start_ms=start_ms, filepos=filepos))
    return VobSubIndex(palette=palette, events=events)


def _collect_spu_payload(sub_bytes: bytes, filepos: int) -> bytes:
    """Gom payload SPU từ các gói MPEG-PS bắt đầu tại ``filepos``.

    Phân tích pack header (0x000001BA) + PES private_stream_1 (0x000001BD), nối phần
    payload subpicture cho tới khi đủ kích thước SPU khai báo ở 2 byte đầu.
    """
    payload = bytearray()
    offset = filepos
    total_size = -1
    n = len(sub_bytes)
    while offset + 6 <= n:
        if sub_bytes[offset:offset + 3] != b"\x00\x00\x01":
            break
        stream_id = sub_bytes[offset + 3]
        if stream_id == 0xBA:  # pack header MPEG-2: 14 byte
            offset += 14
            continue
        if stream_id != 0xBD:  # chỉ quan tâm private_stream_1
            break
        pes_len = (sub_bytes[offset + 4] << 8) | sub_bytes[offset + 5]
        header_data_len = sub_bytes[offset + 8]
        data_start = offset + 9 + header_data_len + 1  # +1: byte stream-id phụ đề
        data_end = offset + 6 + pes_len
        payload += sub_bytes[data_start:data_end]
        if total_size < 0 and len(payload) >= 2:
            total_size = (payload[0] << 8) | payload[1]
        offset = data_end
        if total_size >= 0 and len(payload) >= total_size:
            break
    return bytes(payload)


def _decode_rle(data: bytes, start_nibble: int, width: int, lines: range,
                bitmap: list[list[int]]) -> None:
    """Giải mã RLE 2 bit/điểm cho một trường (chẵn hoặc lẻ) vào ``bitmap``.

    Args:
        data: Toàn bộ payload SPU.
        start_nibble: Vị trí nibble bắt đầu của trường này.
        width: Chiều rộng vùng phụ đề (điểm ảnh).
        lines: Các chỉ số dòng (y) thuộc trường này.
        bitmap: Ma trận màu [y][x] (giá trị 0..3) để ghi kết quả.
    """
    nibble_pos = start_nibble

    def next_nibble() -> int:
        nonlocal nibble_pos
        byte = data[nibble_pos >> 1]
        value = (byte >> 4) if (nibble_pos & 1) == 0 else (byte & 0x0F)
        nibble_pos += 1
        return value

    for y in lines:
        x = 0
        while x < width:
            run = next_nibble()
            if run < 0x4:
                run = (run << 4) | next_nibble()
                if run < 0x10:
                    run = (run << 4) | next_nibble()
                    if run < 0x40:
                        run = (run << 4) | next_nibble()
            color = run & 0x3
            length = run >> 2
            if length == 0:  # tô hết phần còn lại của dòng
                length = width - x
            length = min(length, width - x)
            if 0 <= y < len(bitmap):
                for px in range(x, x + length):
                    bitmap[y][px] = color
            x += length
        if nibble_pos & 1:  # mỗi dòng bắt đầu ở ranh giới byte
            nibble_pos += 1


def inspect_spu(spu: bytes) -> dict:
    """[v3.23.99] Chẩn đoán nhanh một SPU thô: trả các trường header để soi lỗi giải mã.

    Dùng để log vài block đầu khi giải mã thất bại, giúp xác định đúng cấu trúc thật.
    """
    info: dict = {"len": len(spu), "head_hex": spu[:24].hex()}
    if len(spu) >= 4:
        total_size = (spu[0] << 8) | spu[1]
        dcsqt_offset = (spu[2] << 8) | spu[3]
        info["total_size"] = total_size
        info["dcsqt_offset"] = dcsqt_offset
        info["dcsqt_in_range"] = 4 <= dcsqt_offset < len(spu)
        info["size_matches_len"] = abs(total_size - len(spu)) <= 8
    return info


def parse_spu_timing(spu: bytes) -> tuple[int | None, int | None]:
    """[v3.23.101] Trích thời điểm bật/tắt hiển thị (ms, lệch so với PTS) từ SPU.

    DVD subpicture mã hoá thời lượng trong chuỗi lệnh: SP_DCSQ có trường delay; lệnh 0x01
    (bật hiển thị) và 0x02 (tắt hiển thị). Đổi delay -> ms theo ``delay * 1024 / 90``.

    Returns:
        ``(start_offset_ms, end_offset_ms)`` - lệch so với mốc block. Trả None cho
        trường không có. Nhờ ``end_offset_ms`` mà thời lượng khớp Subtitle Edit
        (mỗi phụ đề tắt đúng lúc thay vì kéo dài tới block kế tiếp).
    """
    if len(spu) < 4:
        return (None, None)
    dcsqt_offset = (spu[2] << 8) | spu[3]
    if not (4 <= dcsqt_offset < len(spu)):
        return (None, None)

    start_offset: int | None = None
    end_offset: int | None = None
    pos = dcsqt_offset
    guard = 0
    while pos + 4 <= len(spu) and guard < 64:
        guard += 1
        delay = (spu[pos] << 8) | spu[pos + 1]
        next_dcsq = (spu[pos + 2] << 8) | spu[pos + 3]
        delay_ms = round(delay * 1024 / 90)
        cmd_pos = pos + 4
        while cmd_pos < len(spu):
            cmd = spu[cmd_pos]
            cmd_pos += 1
            if cmd == 0xFF:
                break
            if cmd == 0x01:  # bật hiển thị
                if start_offset is None:
                    start_offset = delay_ms
            elif cmd == 0x02:  # tắt hiển thị
                end_offset = delay_ms
            elif cmd == 0x00:
                continue
            elif cmd == 0x03 or cmd == 0x04:
                cmd_pos += 2
            elif cmd == 0x05:
                cmd_pos += 6
            elif cmd == 0x06:
                cmd_pos += 4
            else:
                break
        if next_dcsq == pos or next_dcsq < dcsqt_offset:
            break
        pos = next_dcsq
    return (start_offset, end_offset)


def decode_event(sub_bytes: bytes, filepos: int,
                 palette: list[tuple[int, int, int]]):
    """Giải mã một sự kiện VobSub tại ``filepos`` trong tệp ``.sub`` (bọc MPEG-PS).

    Returns:
        ``PIL.Image`` RGBA của bitmap phụ đề, hoặc ``None`` nếu không giải mã được.
    """
    spu = _collect_spu_payload(sub_bytes, filepos)
    return decode_spu(spu, palette)


def decode_spu(spu: bytes, palette: list[tuple[int, int, int]]):
    """Giải mã một SPU (Sub-Picture Unit) THÔ thành ảnh RGBA (PIL).

    SPU thô = dữ liệu subpicture bắt đầu bằng 2 byte kích thước (dùng cho block MKV
    S_VOBSUB - không bọc MPEG-PS). Trả ``None`` nếu không giải mã được.
    """
    from PIL import Image

    if len(spu) < 4:
        return None
    dcsqt_offset = (spu[2] << 8) | spu[3]
    # [v3.23.99] Chỉ yêu cầu dcsqt_offset nằm trong tầm; KHÔNG ràng buộc theo total_size
    # (một số encoder đặt total_size không khớp độ dài block). Biên vẫn an toàn nhờ kiểm
    # trong vòng lặp lệnh + RLE.
    if not (4 <= dcsqt_offset < len(spu)):
        return None

    color_idx = [0, 1, 2, 3]
    alpha = [0, 0xF, 0xF, 0xF]
    x1 = y1 = 0
    x2 = y2 = 0
    top_addr = bottom_addr = 0

    # ── Phân tích chuỗi lệnh điều khiển (SP_DCSQ) ──────────────────────────────
    pos = dcsqt_offset
    guard = 0
    while pos + 4 <= len(spu) and guard < 64:
        guard += 1
        next_dcsq = (spu[pos + 2] << 8) | spu[pos + 3]
        cmd_pos = pos + 4
        while cmd_pos < len(spu):
            cmd = spu[cmd_pos]
            cmd_pos += 1
            if cmd == 0xFF:
                break
            if cmd in (0x00, 0x01, 0x02):
                continue
            if cmd == 0x03 and cmd_pos + 2 <= len(spu):
                packed = (spu[cmd_pos] << 8) | spu[cmd_pos + 1]
                color_idx = [(packed >> 12) & 0xF, (packed >> 8) & 0xF,
                             (packed >> 4) & 0xF, packed & 0xF]
                cmd_pos += 2
            elif cmd == 0x04 and cmd_pos + 2 <= len(spu):
                packed = (spu[cmd_pos] << 8) | spu[cmd_pos + 1]
                alpha = [(packed >> 12) & 0xF, (packed >> 8) & 0xF,
                         (packed >> 4) & 0xF, packed & 0xF]
                cmd_pos += 2
            elif cmd == 0x05 and cmd_pos + 6 <= len(spu):
                b = spu[cmd_pos:cmd_pos + 6]
                x1 = (b[0] << 4) | (b[1] >> 4)
                x2 = ((b[1] & 0xF) << 8) | b[2]
                y1 = (b[3] << 4) | (b[4] >> 4)
                y2 = ((b[4] & 0xF) << 8) | b[5]
                cmd_pos += 6
            elif cmd == 0x06 and cmd_pos + 4 <= len(spu):
                top_addr = (spu[cmd_pos] << 8) | spu[cmd_pos + 1]
                bottom_addr = (spu[cmd_pos + 2] << 8) | spu[cmd_pos + 3]
                cmd_pos += 4
            else:
                break
        if next_dcsq == pos or next_dcsq < dcsqt_offset:
            break
        pos = next_dcsq

    width = x2 - x1 + 1
    height = y2 - y1 + 1
    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        return None
    if top_addr == 0:
        return None

    bitmap = [[0] * width for _ in range(height)]
    _decode_rle(spu, top_addr * 2, width, range(0, height, 2), bitmap)
    if bottom_addr:
        _decode_rle(spu, bottom_addr * 2, width, range(1, height, 2), bitmap)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            color_value = bitmap[y][x]
            pal_index = color_idx[color_value]
            a = int(alpha[color_value] * 255 / 15)
            if a == 0:
                continue
            r, g, b = palette[pal_index] if pal_index < len(palette) else (255, 255, 255)
            pixels[x, y] = (r, g, b, a)
    return image


def decode_all(idx_text: str, sub_bytes: bytes):
    """Giải mã toàn bộ sự kiện VobSub -> list (start_ms, end_ms, PIL.Image).

    ``end_ms`` lấy từ mốc bắt đầu của sự kiện kế tiếp (xấp xỉ); sự kiện cuối +3000ms.
    Bỏ qua sự kiện không giải mã được.
    """
    index = parse_idx(idx_text)
    results = []
    events = index.events
    for i, event in enumerate(events):
        try:
            image = decode_event(sub_bytes, event.filepos, index.palette)
        except (IndexError, ValueError, OSError) as exc:
            logger.debug("Bỏ qua sự kiện VobSub tại {}: {}", event.filepos, exc)
            continue
        if image is None:
            continue
        end_ms = events[i + 1].start_ms if i + 1 < len(events) else event.start_ms + 3000
        results.append((event.start_ms, end_ms, image))
    return results


__all__ = [
    "VobSubEvent",
    "VobSubIndex",
    "decode_all",
    "decode_event",
    "decode_spu",
    "inspect_spu",
    "parse_idx",
    "parse_spu_timing",
]
