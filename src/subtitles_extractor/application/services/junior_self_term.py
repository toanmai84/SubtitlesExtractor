"""[v3.23.637] Sửa TẤT ĐỊNH tự xưng 'ta' của người DƯỚI nói với bậc TRÊN trong bảng nhân vật.

Đo thật (phân tích phim ghép 12 tập, bản 3.23.636): cùng prompt, có lượt bảng ghi
``Diệp Nhiên … với Nam Cung Mộng tự xưng 'đệ tử' gọi 'sư thúc'`` (đúng), có lượt ghi
``tự xưng 'ta' gọi 'sư thúc'`` — 3/6 lượt gemini-3.1-flash-lite, 3/5 lượt gemini-3.8-flash; thêm câu
dặn "người dưới tự xưng đệ tử" vào prompt không giảm được (flash-lite 2/6 → 3/6). Hậu quả đo bằng
chấm mù tập 9 và 12 dịch với bảng sai đó: 17 lỗi nặng (đệ tử tự xưng "ta" với sư thúc ở 6 dòng,
gọi sư thúc là "ngươi"…) so với 4 lỗi ở lượt bảng đúng. Prompt không đủ → sửa tại nguồn bằng luật.

Chỉ sửa ca không nhập nhằng: mệnh đề ``tự xưng 'ta' gọi '<chức danh bậc trên>'`` (gọi người nghe là
sư phụ/sư thúc/chưởng giáo… nghĩa là người nghe ở vai trên). 'tôi' (phim hiện đại; "sư phụ" ở đó là
tài xế/thợ) không bị đụng tới.
"""

from __future__ import annotations

import re

__all__ = ["ELDER_TITLES", "JUNIOR_SELF_TERM", "fix_junior_self_terms"]

JUNIOR_SELF_TERM = "đệ tử"
ELDER_TITLES = (
    "sư phụ", "sư tôn", "sư thúc", "sư bá", "sư tổ", "sư công", "lão tổ", "chưởng giáo", "chưởng môn",
    "tông chủ", "trưởng lão", "thái thượng trưởng lão",
)
_CLAUSE_RE = re.compile(
    r"(?P<head>xưng\s+(?:là\s+)?')ta(?P<mid>'\s+gọi\s+(?:là\s+)?')(?P<pre>(?:[^'/]*/\s*)?)(?P<title>"
    + "|".join(re.escape(t) for t in ELDER_TITLES) + r")(?P<tail>(?:\s+(?:sư\s+)?(?:bá|thúc|tổ|phụ|tôn))?')",
    re.IGNORECASE,
)


def fix_junior_self_terms(characters: str) -> tuple[str, int]:
    """Đổi ``xưng 'ta' gọi 'sư thúc'`` thành ``xưng 'đệ tử' gọi 'sư thúc'``.

    Args:
        characters: Bảng nhân vật (mỗi nhân vật một dòng).

    Returns:
        ``(bảng đã sửa, số mệnh đề đã đổi)``.
    """
    if not characters:
        return characters, 0
    return _CLAUSE_RE.subn(
        lambda m: f"{m.group('head')}{JUNIOR_SELF_TERM}{m.group('mid')}{m.group('pre')}{m.group('title')}{m.group('tail')}",
        characters,
    )
