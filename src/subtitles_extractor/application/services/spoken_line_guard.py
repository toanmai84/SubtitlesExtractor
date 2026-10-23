"""[v3.23.605] Dòng thoại không được bị giai đoạn tinh chỉnh biến thành chỉ còn tag.

Ca thật 11/10 21:10 (灭门惨案II, dòng 263 "當當" — tiếng "tèn ten" khi y tá đưa đồ): bản
tinh chỉnh "Đương Đương...", giai đoạn bản địa hoá (bật sinh tag cảm xúc) trả về đúng
"[cười]". Tag cảm xúc chỉ được CHÈN KÈM lời; khi dòng chỉ còn tag, khâu chuẩn bị TTS bỏ
trống dòng (VieNeu không đọc được tag đứng một mình) và phụ đề xuất ra cũng lọc mất tag
→ dòng thoại biến mất khỏi cả phụ đề lẫn lồng tiếng ("Bị bỏ (không đọc): 1 — Cửa sổ quá
ngắn / text rỗng"). Cùng dòng đó ở v3.23.586: "Trời ạ." → "[cười]".

Ngoại lệ: câu gốc chỉ là tiếng cười/thở dài/ho ("哈哈哈", "唉") — để tag đứng một mình là
đúng ý, không có lời nào để giữ.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = [
    "NON_VERBAL_SOURCE_CHARS",
    "find_unspoken_regressions",
    "has_spoken_words",
    "is_non_verbal_source",
]

#: Chữ gốc chỉ là âm thanh không lời: cười (哈呵嘿嘻, ㅋㅎ tiếng Hàn), thở dài (唉),
#: ho (咳).
NON_VERBAL_SOURCE_CHARS = frozenset("哈呵嘿嘻唉咳ㅋㅎ")

#: Đoạn trong ngoặc: tag cảm xúc, chú thích âm thanh, ghi chú của người dịch (gồm ngoặc
#: toàn khổ \uff08\uff09 và \u3010\u3011 của phụ đề CJK).
_BRACKETED_RE = re.compile(
    r"\[[^\]]*\]|\([^)]*\)|\uff08[^\uff09]*\uff09|\u3010[^\u3011]*\u3011|\{[^}]*\}"
)
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af\u3131-\u318e]"
)
_LATIN_WORD_RE = re.compile(r"[^\W\d_]+")
_LATIN_NON_VERBAL_RE = re.compile(
    r"(?:[hk][aeiou]h?)+|sigh|ahem|cough|lol", re.IGNORECASE
)


def has_spoken_words(text: str) -> bool:
    """True nếu bỏ hết phần trong ngoặc mà vẫn còn chữ hoặc số để đọc.

    Args:
        text: Một dòng phụ đề.

    Returns:
        False với dòng rỗng, chỉ có tag ("[cười]") hoặc chỉ có dấu câu.
    """
    return any(char.isalnum() for char in _BRACKETED_RE.sub(" ", text or ""))


def is_non_verbal_source(source: str) -> bool:
    """True nếu câu gốc không có lời: chỉ tiếng cười/thở dài/ho, hoặc chỉ ngoặc/dấu câu.

    Args:
        source: Câu gốc.

    Returns:
        True khi để bản dịch chỉ còn tag cảm xúc là chấp nhận được.
    """
    unbracketed = _BRACKETED_RE.sub(" ", source or "")
    letters = [char for char in unbracketed if char.isalnum()]
    if not letters:
        return True
    if all(_CJK_RE.match(char) for char in letters):
        return all(char in NON_VERBAL_SOURCE_CHARS for char in letters)
    words = _LATIN_WORD_RE.findall(unbracketed)
    return bool(words) and all(_LATIN_NON_VERBAL_RE.fullmatch(word) for word in words)


def find_unspoken_regressions(
    sources: Mapping[int, str],
    previous: Mapping[int, str],
    current: Mapping[int, str],
) -> list[int]:
    """Dòng mà giai đoạn vừa chạy làm mất hết lời dù bản trước còn lời và câu gốc có lời.

    Args:
        sources: Câu gốc theo số dòng.
        previous: Bản dịch đầu vào của giai đoạn theo số dòng.
        current: Bản dịch giai đoạn vừa tạo theo số dòng.

    Returns:
        Số dòng (tăng dần) cần lùi về bản trước.
    """
    return sorted(
        index
        for index, text in current.items()
        if not has_spoken_words(text)
        and has_spoken_words(previous.get(index, ""))
        and not is_non_verbal_source(sources.get(index, ""))
    )
