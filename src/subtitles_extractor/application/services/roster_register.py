"""[v3.23.637] Sửa TẤT ĐỊNH bảng xưng hô sai "thế giới" của phim (hàm thuần, không gọi AI).

Dữ liệu thật — tệp chuẩn đoán dịch cả bộ 86 tập (``series_context.characters``, 19 nhân vật):

* ``Diệp Nhiên … với Nam Cung Mộng tự xưng 'cháu' gọi 'sư thúc'`` (phim tu tiên, phải là "đệ tử");
* ``với Thiên Thanh xưng 'tôi' gọi 'ông / sư bá'``, ``Tần Chấp … xưng 'tôi' gọi 'cậu'``,
  ``Ma Viêm … xưng 'tôi' gọi 'ngươi'``, ``Ma Hải … xưng 'tôi' gọi 'ngươi'``.

Bảng được lưu theo phiên và dùng lại ở các lần dịch sau, nên bản cũ (sinh trước khi prompt phân
tích dặn dùng đại từ cổ phong) cứ được đưa vào khâu dịch: phép gán đại từ sẵn cho từng dòng
(``pron``) ép dòng thoại ra "tôi"/"cậu" giữa phim cổ phong — lỗi nặng theo tiêu chí chấm.

Chỉ sửa khi bảng + tóm tắt NHẬN RA là phim cổ phong/tiên hiệp (nhiều từ khoá tông môn/tu luyện/
sư thúc…) và chỉ các cặp thay thế không nhập nhằng: ``tôi → ta``, ``cậu → ngươi``, ``cháu → đệ tử``
(khi gọi bậc trên) và ``ta → đệ tử`` (người dưới tự xưng với bậc trên: xem
:mod:`junior_self_term`). Phim hiện đại không bị đụng tới.
"""

from __future__ import annotations

import re

from subtitles_extractor.application.services.junior_self_term import (
    ELDER_TITLES,
    JUNIOR_SELF_TERM,
    fix_junior_self_terms,
)

__all__ = ["looks_ancient", "normalize_roster_register"]

_ANCIENT_MARKERS = (
    "tông môn", "tu tiên", "tu luyện", "tu vi", "sư thúc", "sư phụ", "sư tôn", "chưởng giáo",
    "đệ tử", "kim đan", "nguyên anh", "phi thăng", "linh khí", "ma môn", "phong chủ", "đạo nhân",
    "trưởng lão", "sư bá", "tông chủ", "thiên kiêu", "đại thừa", "hóa thần", "luyện khí",
)
#: Số từ khoá khác nhau tối thiểu để coi là phim cổ phong/tiên hiệp.
_MIN_MARKERS = 5

_SELF_TOI = re.compile(r"(xưng\s+(?:là\s+)?')tôi(')", re.IGNORECASE)
_CALL_CAU = re.compile(r"(gọi\s+(?:là\s+)?')cậu(')", re.IGNORECASE)
_SELF_CHAU = re.compile(
    r"(?P<head>xưng\s+(?:là\s+)?')cháu(?P<mid>'\s+gọi\s+(?:là\s+)?')(?P<pre>(?:[^'/]*/\s*)?)(?P<title>"
    + "|".join(re.escape(t) for t in ELDER_TITLES) + r")(?P<tail>(?:\s+(?:sư\s+)?(?:bá|thúc|tổ|phụ|tôn))?')",
    re.IGNORECASE,
)


def looks_ancient(*texts: str) -> bool:
    """Bảng/tóm tắt thuộc phim cổ phong, tiên hiệp, huyền huyễn (đủ nhiều từ khoá đặc trưng)."""
    blob = " ".join(texts).lower()
    return sum(1 for marker in _ANCIENT_MARKERS if marker in blob) >= _MIN_MARKERS


def normalize_roster_register(characters: str, overview: str = "") -> tuple[str, int]:
    """Đưa đại từ hiện đại trong mệnh đề xưng hô về đại từ cổ phong; người dưới tự xưng 'ta' → 'đệ tử'.

    Args:
        characters: Bảng nhân vật (mỗi nhân vật một dòng).
        overview: Tóm tắt phim (giúp nhận ra thể loại).

    Returns:
        ``(bảng đã sửa, số mệnh đề đã đổi)``; nguyên bảng nếu không phải phim cổ phong.
    """
    if not characters or not looks_ancient(characters, overview):
        return characters, 0
    changes = 0
    text = characters
    text, n = _SELF_CHAU.subn(
        lambda m: f"{m.group('head')}{JUNIOR_SELF_TERM}{m.group('mid')}{m.group('pre')}{m.group('title')}{m.group('tail')}",
        text,
    )
    changes += n
    text, n = _SELF_TOI.subn(r"\1ta\2", text)
    changes += n
    text, n = _CALL_CAU.subn(r"\1ngươi\2", text)
    changes += n
    text, n = fix_junior_self_terms(text)
    return text, changes + n
