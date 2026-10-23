"""[v3.23.605] Lùi dòng mà giai đoạn tinh chỉnh chỉ CHÈN TỪ ĐỆM làm vượt ngân sách đọc.

Chuẩn đoán thật 11/10 21:10 (灭门惨案II): giai đoạn dịch thô chạy gemini-3.7-flash chỉ
để 122 dòng vượt ngân sách ký tự (``max_chars``); hai giai đoạn tinh chỉnh chạy
gemini-3.5-flash-lite đẩy lên 202 dòng. Số dòng TTS phải nén trên 1.6 lần tăng từ 44
lên 59 so với lần chạy trước. Trong đó có loại thuần đệm chữ: bản sau GIỮ NGUYÊN mọi từ
của bản trước, chỉ chèn thêm hư từ/trợ từ —

* "Mày quá đáng lắm, giờ tính sao?" → "Mày quá đáng lắm rồi đấy, giờ tính sao đây?"
* "Tôi không đền cho anh nữa đâu." → "Tôi sẽ không đền cho anh nữa đâu."
* "Chắc A Quân tới." → "Chắc A Quân tới rồi."

Những từ chèn đó không thêm nghĩa nhưng bắt giọng đọc nén nhanh hơn. Chỉ lùi khi MỌI từ
chèn thêm thuộc :data:`FILLER_WORDS`; chèn đại từ ("Mày đúng là đồ khốn nạn!") hay từ
mang nghĩa ("Từ từ ăn đi.") thì giữ.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = [
    "FILLER_WORDS",
    "find_padding_regressions",
    "inserted_words",
]

#: Hư từ / trợ từ tiếng Việt chèn thêm mà không đổi nghĩa mệnh đề. Cố ý KHÔNG có:
#: đại từ (xưng hô do lưới khác quyết); phủ định; từ hay mang nghĩa khi đứng chèn —
#: "đi"/"lại"/"nữa"/"vẫn" (đi, lặp lại, thêm, còn), "cũng" ("tôi cũng đi" = đi theo),
#: "cả"/"hết" ("cả nhà", "ăn hết"), "nào" ("thế nào", "lúc nào" — đo thật:
#: "phế nhân à?" → "phế nhân thế nào à?").
FILLER_WORDS = frozenset({
    "à", "ạ", "ấy", "chứ", "cơ", "đấy", "đây", "đó", "hả", "hử", "kìa", "nhé", "nhỉ",
    "nha", "nhá", "thôi", "vậy", "thế", "mà", "rồi", "luôn", "là", "của", "với", "thì",
    "các", "những", "một", "cái", "chiếc", "này", "đã", "đang", "sẽ", "chính",
})

_WORD_RE = re.compile(r"[^\W_]+")
#: Dấu đóng câu/ngoặc kép có thể đứng sau "?" ("… chứ?\"", "… à?…").
_CLOSING_MARKS = ".…!\"'\u201d\u2019)"


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text or "")]


def _is_question(text: str) -> bool:
    return (text or "").rstrip().rstrip(_CLOSING_MARKS).endswith("?")


def inserted_words(before: str, after: str) -> list[str] | None:
    """Các từ ``after`` chèn thêm vào ``before`` khi ``after`` chỉ là bản chèn thêm.

    Args:
        before: Bản trước.
        after: Bản sau.

    Returns:
        Danh sách từ chèn thêm (chữ thường, theo thứ tự xuất hiện); ``None`` nếu ``after``
        bỏ hoặc đổi bất kỳ từ nào của ``before``.
    """
    before_words = _words(before)
    matched = 0
    inserted: list[str] = []
    for word in _words(after):
        if matched < len(before_words) and before_words[matched] == word:
            matched += 1
        else:
            inserted.append(word)
    return inserted if matched == len(before_words) else None


def find_padding_regressions(
    previous: Mapping[int, str],
    current: Mapping[int, str],
    budgets: Mapping[int, int],
) -> list[int]:
    """Dòng vượt ngân sách chỉ vì chèn thêm từ đệm so với bản trước.

    Args:
        previous: Bản dịch đầu vào của giai đoạn theo số dòng.
        current: Bản dịch giai đoạn vừa tạo theo số dòng.
        budgets: Ngân sách ký tự (``max_chars``) theo số dòng; thiếu hoặc 0 = bỏ qua dòng.

    Returns:
        Số dòng (tăng dần) cần lùi về bản trước.
    """
    found: list[int] = []
    for index, after in current.items():
        budget = budgets.get(index, 0)
        if budget <= 0 or len(after or "") <= budget:
            continue
        before = previous.get(index, "")
        if not _words(before) or _is_question(before) != _is_question(after):
            # Đổi câu kể thành câu hỏi ("… không có tiền" → "… không có tiền chứ?") là đổi
            # nghĩa, dù từ chèn thêm chỉ là trợ từ.
            continue
        added = inserted_words(before, after)
        if added and all(word in FILLER_WORDS for word in added):
            found.append(index)
    return sorted(found)
