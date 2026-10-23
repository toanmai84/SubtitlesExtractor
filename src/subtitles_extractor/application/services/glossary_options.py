"""[v3.23.587] Các PHƯƠNG ÁN DỊCH của một mục thuật ngữ, theo từng cách viết câu gốc.

Bảng thuật ngữ ghi ``gốc = dịch``; cả hai vế đều có thể ghi nhiều cách ngăn bằng ``/``:

* vế dịch ``衝動 = bốc đồng / manh động`` — hai phương án, người dịch CHỌN MỘT;
* vế gốc ``老山 / 魯山 = Lão Sơn`` — hai cách viết của cùng một từ (thường do OCR);
* cả hai vế ``炸药 / 炸弹 = chất nổ / bom`` — khi số cách viết bằng số phương án thì chúng
  ứng với nhau theo THỨ TỰ.

Đo trên 51 chuẩn đoán dịch thật (10 079 mục thuật ngữ):

* 3 084 mục có vế dịch dạng ``A / B``. Khối nhắc thuật ngữ theo lô đưa nguyên chuỗi đó vào
  ``must_translate_as`` nên model chép cả hai phương án vào lời thoại: 84 dòng dịch thô ở
  28 lượt chạy ("Mày nói gì thế, đừng bốc đồng / manh động chứ.").
* 98 mục có vế gốc dạng ``A / B``. Khối nhắc tìm NGUYÊN chuỗi ``"老山 / 魯山"`` trong câu
  gốc nên không bao giờ khớp: 228 dòng chứa các thuật ngữ ấy không hề được nhắc.

Module này tách hai vế thành từng cặp (cách viết gốc → các phương án dịch) để khối nhắc
theo lô đưa đúng dữ liệu, và để lưới ``glossary_note_guard`` nhận ra hai phương án bị chép
liền nhau. Toàn bộ là HÀM THUẦN: không I/O, không trạng thái.
"""

# ruff: noqa: RUF001 — ngoặc fullwidth CJK là CHỦ ĐÍCH (có trong bảng thuật ngữ)

from __future__ import annotations

import re
from dataclasses import dataclass

from subtitles_extractor.application.services.roster_format import parse_glossary

#: Một chú thích trong ngoặc (ngoặc thường hoặc fullwidth), nhóm 1 là nội dung.
GLOSSARY_NOTE_RE = re.compile(r"[(（]([^()（）]+)[)）]")
_SPACE_RE = re.compile(r"\s+")
_TERM_ALIAS_SPLIT_RE = re.compile(r"\s*[/|]\s*")
#: Dấu ``/`` có khoảng trắng ở ít nhất một phía: ngăn hai PHƯƠNG ÁN ("bãi mìn / mìn").
_SPACED_SLASH_RE = re.compile(r"\s+/\s*|\s*/\s+")
#: Dấu ``/`` viết liền giữa hai TỪ thuần chữ ("xã/phường"); không khớp "24/24".
_TIGHT_CHOICE_RE = re.compile(r"(?<![\w/-])([^\W\d_]{2,})/([^\W\d_]{2,})(?![\w/-])")
#: Từ đứng SAU dấu ``/`` cho biết đó là "trên mỗi" (một triệu tệ/năm), không phải "hoặc".
_RATE_DENOMINATORS = frozenset({
    "giây", "phút", "giờ", "tiếng", "ngày", "đêm", "tuần", "tháng", "quý", "năm",
    "lần", "lượt", "người", "suất", "cân", "ký", "lít", "mét", "tấn",
})


@dataclass(frozen=True, slots=True)
class TermRendering:
    """Một cách viết của thuật ngữ ở câu gốc cùng các phương án dịch của nó.

    Attributes:
        term: Cách viết ở câu gốc (một vế của ``A / B`` nếu mục có nhiều cách viết).
        options: Các phương án dịch dùng được trong lời thoại; luôn có ít nhất một.
    """

    term: str
    options: tuple[str, ...]


def split_glossary_note(translation: str) -> tuple[str, str]:
    """Tách vế dịch của một mục thuật ngữ thành (phần dùng trong lời thoại, chú thích).

    Args:
        translation: Vế dịch, vd ``"thanh thiên bạch nhật (giữa ban ngày ban mặt)"``.

    Returns:
        ``("thanh thiên bạch nhật", "giữa ban ngày ban mặt")``. Vế dịch không có ngoặc,
        hoặc chỉ gồm phần trong ngoặc, thì trả nguyên vế dịch và chú thích rỗng.
    """
    text = (translation or "").strip()
    notes = [match.group(1).strip() for match in GLOSSARY_NOTE_RE.finditer(text)]
    main = _SPACE_RE.sub(" ", GLOSSARY_NOTE_RE.sub(" ", text)).strip()
    if not notes or not main:
        return text, ""
    return main, "; ".join(note for note in notes if note)


def _tight_choices(usable: str) -> tuple[str, ...]:
    """Hai phương án của một vế dịch viết liền kiểu ``"đồn công an xã/phường"``.

    Chỉ nhận khi chắc đó là "hoặc": đúng một dấu ``/``, hai phía là hai từ thường thuần
    chữ, cả vế không có chữ số, và từ đứng sau không phải mẫu số của một tỷ lệ
    ("tệ/năm", "km/giờ" giữ nguyên).

    Args:
        usable: Vế dịch đã bỏ chú thích.

    Returns:
        Hai phương án đã khai triển; rỗng nếu dấu ``/`` không phải là "hoặc".
    """
    if usable.count("/") != 1 or any(char.isdigit() for char in usable):
        return ()
    match = _TIGHT_CHOICE_RE.search(usable)
    if match is None:
        return ()
    left, right = match.group(1), match.group(2)
    if not (left.islower() and right.islower()) or right in _RATE_DENOMINATORS:
        return ()
    head, tail = usable[: match.start()], usable[match.end() :]
    return (f"{head}{left}{tail}", f"{head}{right}{tail}")


def translation_options(translation: str) -> tuple[str, ...]:
    """Các phương án dịch DÙNG ĐƯỢC TRONG LỜI THOẠI của một mục thuật ngữ.

    Args:
        translation: Vế dịch, có thể kèm chú thích trong ngoặc và nhiều phương án.

    Returns:
        ``"bãi mìn / trận địa mìn"`` → ``("bãi mìn", "trận địa mìn")``;
        ``"đồn công an xã/phường"`` → ``("đồn công an xã", "đồn công an phường")``;
        ``"túc trực 24/24"`` → ``("túc trực 24/24",)``. Vế dịch rỗng trả bộ rỗng.
    """
    usable, _note = split_glossary_note(translation)
    usable = _SPACE_RE.sub(" ", usable).strip()
    if not usable:
        return ()
    if _SPACED_SLASH_RE.search(usable):
        parts = [part.strip() for part in _SPACED_SLASH_RE.split(usable)]
    else:
        parts = list(_tight_choices(usable)) or [usable]
    options: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.casefold()
        if part and key not in seen:
            seen.add(key)
            options.append(part)
    return tuple(options)


def term_aliases(term: str) -> tuple[str, ...]:
    """Các cách viết ở câu gốc của một mục thuật ngữ (``"老山 / 魯山"`` → hai cách)."""
    return tuple(
        alias.strip() for alias in _TERM_ALIAS_SPLIT_RE.split(term or "") if alias.strip()
    )


def term_renderings(glossary: str) -> list[TermRendering]:
    """Tách bảng thuật ngữ thành từng cặp (cách viết gốc → các phương án dịch).

    Mục có số cách viết BẰNG số phương án (từ hai trở lên) được ghép theo thứ tự:
    ``炸药 / 炸弹 = chất nổ / bom`` → ``炸药 → chất nổ`` và ``炸弹 → bom``. Còn lại, mỗi
    cách viết nhận đủ mọi phương án.

    Args:
        glossary: Bảng thuật ngữ, mỗi dòng ``gốc = dịch``.

    Returns:
        Các cặp theo thứ tự của bảng; mục thiếu vế gốc hoặc vế dịch bị bỏ qua.
    """
    renderings: list[TermRendering] = []
    for entry in parse_glossary(glossary):
        aliases = term_aliases(entry.term)
        options = translation_options(entry.translation)
        if not aliases or not options:
            continue
        if len(aliases) > 1 and len(aliases) == len(options):
            renderings.extend(
                TermRendering(alias, (option,))
                for alias, option in zip(aliases, options, strict=True)
            )
        else:
            renderings.extend(TermRendering(alias, options) for alias in aliases)
    return renderings


__all__ = [
    "GLOSSARY_NOTE_RE",
    "TermRendering",
    "split_glossary_note",
    "term_aliases",
    "term_renderings",
    "translation_options",
]
