"""[v3.23.616] Viết hoa chữ đầu của dòng MỞ CÂU MỚI trong bản dịch cuối.

Model dịch theo lô và hay nối mạch câu xuyên dòng; khi dòng trước đã kết câu bằng
``.``/``!``/``?`` mà model vẫn coi dòng sau là phần tiếp, dòng sau bị viết thường chữ
đầu. Ca của chuẩn đoán 14/10 (phim "天骄女友退婚" tập 1): dòng 59 "Từ hôn?!" rồi dòng 60
"sắp chết bỗng giật mình ngồi dậy,". Đo trên toàn bộ file chuẩn đoán dịch thật đã có (hơn
114 nghìn cặp dòng): sau dấu kết câu có 113 207 dòng viết hoa và 258 dòng viết thường;
soát các dòng viết thường đều là lỗi chính tả kiểu trên ("Ấy da." → "sống còn chẳng bằng
con chó nhà tao.").

Chỉ sửa khi chắc chắn: dòng trước (dòng có chữ gần nhất) kết bằng dấu kết câu — KHÔNG
tính dấu lửng ``...``/``…`` (câu bỏ lửng thường nói tiếp) và chữ viết tắt (``v.v.``,
``TP.``) — và ký tự đầu dòng sau là CHỮ viết thường. Dòng mở bằng số, ngoặc nhãn
``[thở dài]``, dấu gạch hội thoại hay dấu lửng được giữ nguyên.

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: Dấu đóng có thể đứng sau dấu kết câu: nháy, ngoặc kép kiểu Pháp, ngoặc đơn/vuông.
_CLOSERS = "\"'\u201d\u2019\u00bb)\\]"
_TERMINAL_END_RE = re.compile(f"[.!?]+[{_CLOSERS}]*\\s*$")
_ELLIPSIS_END_RE = re.compile(f"(?:\\.\\.|\u2026)[{_CLOSERS}]*\\s*$")
_ABBREVIATION_END_RE = re.compile(
    r"(?:(?<![^\W\d_])[^\W\d_]\.){2,}\s*$|(?<![^\W\d_])(?:tp|mr|mrs|ms|dr|st|vs|etc)\.\s*$",
    re.IGNORECASE,
)

_FIRST_WORD_RE = re.compile(r"[^\W\d_]+")

#: Ngôn ngữ có quy tắc hoa/thường riêng mà ``str.upper`` không làm đúng (chữ ``i``).
_LOCALE_CASED_LANGUAGES = ("turk", "azerbaij")


def supports_sentence_case(target_language: str) -> bool:
    """Ngôn ngữ đích viết hoa đầu câu được bằng ``str.upper`` không (hàm thuần).

    Args:
        target_language: Tên ngôn ngữ đích (vd ``"Vietnamese"``).

    Returns:
        ``False`` với tiếng Thổ Nhĩ Kỳ/Azerbaijan (``i`` hoa là ``İ``), ``True`` còn lại;
        chữ không phân hoa/thường (Hán, Thái…) tự giữ nguyên.
    """
    lowered = (target_language or "").lower()
    return not any(name in lowered for name in _LOCALE_CASED_LANGUAGES)


def _ends_sentence(text: str) -> bool:
    """Dòng kết thúc bằng dấu kết câu thật (không phải dấu lửng/chữ viết tắt)."""
    return (
        bool(_TERMINAL_END_RE.search(text))
        and not _ELLIPSIS_END_RE.search(text)
        and not _ABBREVIATION_END_RE.search(text)
    )


def _capitalized_first_letter(text: str) -> str | None:
    """Bản viết hoa chữ đầu nếu dòng mở bằng chữ viết thường; ``None`` nếu không sửa.

    Từ đầu dòng có chữ hoa ở giữa (tên riêng kiểu ``iPhone``, ``eBay``) giữ nguyên.
    """
    stripped = text.lstrip()
    first = stripped[:1]
    if not first.isalpha() or not first.islower():
        return None
    first_word = _FIRST_WORD_RE.match(stripped)
    if first_word and any(ch.isupper() for ch in first_word.group()[1:]):
        return None
    upper = first.upper()
    if len(upper) != 1:
        return None
    return text[: len(text) - len(stripped)] + upper + stripped[1:]


def capitalize_sentence_starts(texts: Sequence[str]) -> dict[int, str]:
    """Viết hoa chữ đầu các dòng mở câu mới sau một dòng đã kết câu (hàm thuần).

    Args:
        texts: Bản dịch theo thứ tự thời gian (dòng rỗng được bỏ qua khi tìm dòng trước).

    Returns:
        ``vị trí → bản đã sửa`` cho các dòng cần sửa; rỗng nếu không có gì để sửa.
    """
    fixed: dict[int, str] = {}
    previous = ""
    for position, text in enumerate(texts):
        current = text or ""
        if not current.strip():
            continue
        if previous and _ends_sentence(previous):
            capitalized = _capitalized_first_letter(current)
            if capitalized is not None:
                fixed[position] = capitalized
                current = capitalized
        previous = current.rstrip()
    return fixed


__all__ = ["capitalize_sentence_starts", "supports_sentence_case"]
