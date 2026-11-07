"""[v3.23.675] Dựng ``data/app.ico`` chuẩn Windows từ ``data/app.png`` (512 px).

Ảnh chụp bản exe trên Windows 11: thanh tiêu đề hiện đúng icon (Qt vẽ từ tệp .ico), nhưng NÚT TRÊN THANH TÁC VỤ hiện
icon "ứng dụng không có icon" mặc định của Windows. Gốc: mọi khung trong app.ico cũ (16…256) đều nén PNG. Microsoft chỉ
quy định PNG cho khung 256 px; các khung nhỏ mà shell/thanh tác vụ đọc qua tài nguyên RT_ICON của exe nên là bitmap
(DIB 32 bit + mặt nạ AND). Bộ cỡ cũ còn thiếu 20/24/40 px (thanh tác vụ ở 100/125/150 % DPI) nên Windows phải co giãn.

Chạy:  python tools/make_app_icon.py   (ghi đè src/subtitles_extractor/data/app.ico)
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

from PIL import Image

#: Cỡ khung: 16-48 cho thanh tiêu đề/thanh tác vụ/Explorer ở các mức DPI, 64-256 cho icon lớn.
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
#: Khung từ cỡ này trở lên lưu PNG (chuẩn Windows Vista+), nhỏ hơn lưu bitmap.
PNG_MIN_SIZE = 256


def _bmp_entry(img: Image.Image) -> bytes:
    """Khung ICO dạng DIB: BITMAPINFOHEADER + điểm ảnh BGRA (dưới lên) + mặt nạ AND 1 bit."""
    w, h = img.size
    rgba = img.convert("RGBA")
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, 0, 0, 0, 0, 0)
    pixels = bytearray()
    for y in range(h - 1, -1, -1):
        for x in range(w):
            r, g, b, a = rgba.getpixel((x, y))
            pixels += bytes((b, g, r, a))
    row_bytes = ((w + 31) // 32) * 4
    mask = bytearray()
    for y in range(h - 1, -1, -1):
        row = bytearray(row_bytes)
        for x in range(w):
            if rgba.getpixel((x, y))[3] == 0:
                row[x // 8] |= 0x80 >> (x % 8)
        mask += row
    return header + bytes(pixels) + bytes(mask)


def _png_entry(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_ico(source: Image.Image, sizes: tuple[int, ...] = ICON_SIZES) -> bytes:
    """Trả nội dung tệp .ico gồm ``sizes`` (khung < 256 là bitmap, 256 là PNG)."""
    src = source.convert("RGBA")
    entries = []
    for size in sizes:
        frame = src.resize((size, size), Image.Resampling.LANCZOS)
        data = _png_entry(frame) if size >= PNG_MIN_SIZE else _bmp_entry(frame)
        entries.append((size, data))
    out = io.BytesIO()
    out.write(struct.pack("<HHH", 0, 1, len(entries)))
    offset = 6 + 16 * len(entries)
    for size, data in entries:
        dim = 0 if size >= 256 else size
        out.write(struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    for _size, data in entries:
        out.write(data)
    return out.getvalue()


def main() -> int:
    data_dir = Path(__file__).resolve().parents[1] / "src" / "subtitles_extractor" / "data"
    source = Image.open(data_dir / "app.png")
    target = data_dir / "app.ico"
    target.write_bytes(build_ico(source))
    print(f"Đã ghi {target} ({target.stat().st_size // 1024} KB, cỡ {', '.join(map(str, ICON_SIZES))}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
