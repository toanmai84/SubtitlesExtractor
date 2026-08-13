"""[v3.23.155] Dò MÃ HÓA file phụ đề — đọc đúng mọi bảng mã phổ biến.

Trước đây importer đọc cứng ``utf-8-sig`` + ``errors="replace"`` nên file phụ đề mã
hóa GB18030/Big5/UTF-16/Shift-JIS/CP1252 bị thay ký tự \ufffd hàng loạt (HỎNG ÂM THẦM,
không báo lỗi). Module này dò theo ba lớp, KHÔNG cần thư viện ngoài:

1. **BOM**: UTF-8-SIG / UTF-16 LE-BE / UTF-32 LE-BE — chắc chắn tuyệt đối.
2. **UTF-8 strict**: đa số file hiện đại; decode strict thành công là dùng ngay.
3. **Chấm điểm ứng viên**: thử strict lần lượt các bảng mã phổ biến cho phụ đề
   (UTF-16 không BOM, GB18030, Big5, Shift-JIS, CP1252, Latin-1) rồi chọn bản decode
   "hợp lý" nhất theo tỉ lệ ký tự chữ/CJK và mức phạt ký tự điều khiển — cần thiết vì
   GB18030/CP1252 hầu như decode được mọi chuỗi byte nên "decode thành công" chưa đủ.
"""

# ruff: noqa: RUF001, RUF002 — dấu câu fullwidth CJK là CHỦ ĐÍCH (dò mã hóa)
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# (BOM, tên bảng mã) — thứ tự dài trước ngắn để UTF-32 không bị nhận nhầm UTF-16.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

# Ứng viên thử khi không BOM và không phải UTF-8 (thứ tự chỉ để ưu tiên khi ĐỒNG điểm).
_CANDIDATES: tuple[str, ...] = (
    "utf-16", "gb18030", "big5", "shift_jis", "cp1252", "latin-1",
)


_NATIVE_PUNCT = set("。，、！？；：「」『』（）…—·.,!?\"'()-")


def _plausibility_score(decoded_text: str) -> float:
    """Điểm "hợp lý" của một bản decode.

    Nhiều CHỮ là tốt; DẤU CÂU bản địa còn nguyên là dấu hiệu decode ĐÚNG BỘ (bộ sai
    thường biến 、。， thành Hán tự lạ — vd Shift-JIS decode nhầm bằng GB18030); ký
    tự điều khiển/\ufffd bị phạt nặng.
    """
    if not decoded_text:
        return 0.0
    sample = decoded_text[:8000]
    letters = 0
    native_puncts = 0
    penalties = 0
    for ch in sample:
        if ch in ("\n", "\r", "\t"):
            continue
        if ch in _NATIVE_PUNCT:
            native_puncts += 1
            continue
        category = unicodedata.category(ch)
        if category.startswith("L"):  # chữ cái mọi hệ chữ (Latin/CJK/Kana/Hangul...)
            letters += 1
        elif category in ("Cc", "Co", "Cn") or ch == "\ufffd":
            penalties += 1
    total = max(1, len(sample))
    return (letters + 3.0 * native_puncts) / total - 3.0 * (penalties / total)


def decode_subtitle_bytes(raw: bytes) -> tuple[str, str]:
    """Decode nội dung phụ đề với mã hóa được dò tự động.

    Args:
        raw: Toàn bộ byte của file phụ đề.

    Returns:
        ``(văn_bản, tên_bảng_mã_đã_dùng)``. Không bao giờ raise: trường hợp xấu nhất
        rơi về ``latin-1`` (decode được mọi byte) để không mất dữ liệu dòng thời gian.
    """
    for bom, encoding_name in _BOMS:
        if raw.startswith(bom):
            # Cắt BOM trước khi decode bảng LE/BE tường minh (nếu không, ký tự
            # \ufeff lọt vào đầu văn bản làm hỏng dòng phụ đề đầu tiên).
            payload = raw[len(bom):]
            codec = "utf-8" if encoding_name == "utf-8-sig" else encoding_name
            return payload.decode(codec, errors="replace"), encoding_name

    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    best_text = ""
    best_encoding = "latin-1"
    best_score = float("-inf")
    for encoding_name in _CANDIDATES:
        try:
            candidate = raw.decode(encoding_name)
        except (UnicodeDecodeError, LookupError):
            continue
        score = _plausibility_score(candidate)
        if score > best_score:
            best_text, best_encoding, best_score = candidate, encoding_name, score
    if not best_text:
        best_text = raw.decode("latin-1", errors="replace")
    return best_text, best_encoding


def read_subtitle_text(source_path: Path) -> str:
    """Đọc file phụ đề với mã hóa dò tự động (API dùng chung cho mọi importer).

    Args:
        source_path: Đường dẫn file phụ đề (.srt/.ass/...).

    Returns:
        Nội dung văn bản đã decode đúng bảng mã.

    Raises:
        OSError: không đọc được file (không tồn tại/không có quyền).
    """
    raw = source_path.read_bytes()
    text, encoding_name = decode_subtitle_bytes(raw)
    if encoding_name not in ("utf-8", "utf-8-sig"):
        logger.info(
            "Phụ đề '%s' dùng mã hóa %s (dò tự động).", source_path.name, encoding_name
        )
    return text
