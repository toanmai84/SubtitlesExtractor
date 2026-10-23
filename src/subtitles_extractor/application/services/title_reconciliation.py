"""[v3.23.589] Chức danh trong bảng nhân vật phải theo BẢNG THUẬT NGỮ.

Chuẩn đoán thật v3.23.588, phim "灭门惨案II": bảng thuật ngữ ghi ``所長 = Trưởng đồn công
an`` còn bảng nhân vật ghi bốn quy tắc xưng hô "gọi 'Sở trưởng'" (âm Hán-Việt của chính
chữ 所長). Hai bảng do cùng một lượt phân tích sinh ra nhưng không đối chiếu nhau. Kết
quả: dịch thô theo bảng thuật ngữ 15/15 dòng; giai đoạn phong cách đổi lại "Sở trưởng"
ở những dòng người nói có quy tắc ấy. Bản cuối của ba lượt chạy cùng phân tích này: 3,
0 và 6 trên 15 dòng gọi "Sở trưởng", các dòng còn lại gọi "Trưởng đồn" — một người hai
chức danh, giọng đọc nghe như hai người. Những phân tích mà hai bảng ghi GIỐNG nhau
(``所長 = sở trưởng``) thì bản cuối nhất quán 15/15.

Phim khác (5540 dòng, 6 lượt chạy): ``厂长 = giám đốc xưởng`` nhưng bảng nhân vật ghi "gọi
'xưởng trưởng'", "gọi 'Phùng xưởng trưởng'" — bản cuối 99 dòng "xưởng trưởng" / 50 dòng
"giám đốc xưởng".

Cách sửa: bảng thuật ngữ là chuẩn (dịch thô đã theo nó, và là bảng người dùng sửa khi muốn
đổi cách dịch). Cách gọi trong bảng nhân vật mà đúng là ÂM HÁN-VIỆT của một thuật ngữ chức
danh — trong khi bảng thuật ngữ dịch thuật ngữ đó khác đi — được đổi sang cách dịch của
bảng thuật ngữ.

Giới hạn có chủ ý (chỉ làm đúng lớp lỗi đã đo):

* Chỉ thuật ngữ chức danh kết thúc bằng ``長``/``长`` đọc là "trưởng". Các mục khác của
  bảng thuật ngữ thường là thành ngữ, tên riêng: âm Hán-Việt của chúng trong bảng nhân
  vật có thể chính là TÊN nhân vật (``老表 (Lão Biểu)``).
* Chỉ sửa ở VỊ TRÍ CÁCH GỌI: chuỗi trong nháy đơn và "(danh xưng: …)", khi cả phương án
  đúng bằng âm Hán-Việt. Văn xuôi không đụng — "gia trưởng", "sở trường" còn là từ thường.
* Phiên âm trong ngoặc sau tên nhân vật (``冯厂长 (Phùng xưởng trưởng)``) giữ nguyên: đó
  còn là khoá để tra người nghe của các quy tắc xưng hô.

Toàn bộ là HÀM THUẦN: không I/O, không trạng thái.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

from subtitles_extractor.application.services.glossary_options import term_renderings
from subtitles_extractor.application.services.name_reading_check import (
    NameReadings,
    syllable_key,
)
from subtitles_extractor.application.services.roster_format import parse_roster

#: Chữ cuối của một thuật ngữ chức danh và âm của nó khi là chức danh ("trưởng", không
#: phải "trường" — "sở trường" là từ thường).
TITLE_SUFFIX_CHARS = frozenset("長长")
TITLE_SUFFIX_READING = "trưởng"
#: Thuật ngữ chức danh dài 2-4 chữ (``所長``, ``董事长``, ``副院长``).
MIN_TITLE_CHARS = 2
MAX_TITLE_CHARS = 4
#: Họ đứng trước chức danh trong tên nhân vật (``冯厂长``, ``欧阳局长``): tối đa hai chữ.
MAX_SURNAME_CHARS = 2

_CJK_RE = re.compile(r"^[㐀-䶿一-鿿]+$")
_NAME_ALIAS_SPLIT_RE = re.compile(r"\s*[/|]\s*")
_QUOTED_FORM_RE = re.compile(r"(?P<head>')(?P<form>[^'\n]*)(?P<tail>')")
_TITLE_NOTE_RE = re.compile(
    r"(?P<head>\(danh xưng:\s*)(?P<form>[^()\n]*)(?P<tail>\))", re.IGNORECASE
)
_ALTERNATIVE_SPLIT_RE = re.compile(r"(\s*/\s*)")
#: Ký tự làm vỡ cú pháp bảng nhân vật nếu nằm trong cách gọi thay vào.
_UNSAFE_REPLACEMENT_RE = re.compile(r"['()/\n]")

_Keys = tuple[frozenset[str], ...]


@dataclass(frozen=True)
class TitleFix:
    """Một cách gọi trong bảng nhân vật đã đổi theo bảng thuật ngữ.

    Attributes:
        term: Thuật ngữ chức danh trong bảng thuật ngữ (vd ``所長``).
        written: Cách gọi bảng nhân vật đã ghi (vd ``Sở trưởng``).
        replacement: Cách gọi thay vào (vd ``Trưởng đồn công an``).
        count: Số chỗ đã đổi.
    """

    term: str
    written: str
    replacement: str
    count: int


@dataclass(frozen=True)
class _TitleRule:
    """Âm Hán-Việt theo từng chữ của một chức danh (kèm họ nếu có) và cách dịch đúng."""

    term: str
    surname_words: int
    keys: _Keys
    replacement: str


class _ContextTexts(Protocol):
    characters: str
    glossary: str


ContextT = TypeVar("ContextT", bound=_ContextTexts)


def _is_title_term(term: str) -> bool:
    return (
        MIN_TITLE_CHARS <= len(term) <= MAX_TITLE_CHARS
        and term[-1] in TITLE_SUFFIX_CHARS
        and _CJK_RE.match(term) is not None
    )


def _char_keys(chars: str, readings: NameReadings) -> _Keys | None:
    """Khoá âm tiết hợp lệ của từng chữ; None nếu có chữ không rõ âm."""
    keys: list[frozenset[str]] = []
    for char in chars:
        accepted = readings.accepted(char)
        if not accepted:
            return None
        keys.append(frozenset(syllable_key(reading) for reading in accepted))
    return tuple(keys)


def _title_keys(term: str, readings: NameReadings) -> _Keys | None:
    stem = _char_keys(term[:-1], readings)
    if stem is None:
        return None
    return (*stem, frozenset({syllable_key(TITLE_SUFFIX_READING)}))


def _reads_as(words: list[str], keys: _Keys) -> bool:
    """``words`` đúng là âm Hán-Việt theo từng chữ (không phân biệt hoa/thường)."""
    return len(words) == len(keys) and all(
        syllable_key(word) in key for word, key in zip(words, keys, strict=True)
    )


def _roster_cjk_names(characters: str) -> list[str]:
    return [
        alias.strip()
        for entry in parse_roster(characters or "")
        for alias in _NAME_ALIAS_SPLIT_RE.split(entry.name or "")
        if _CJK_RE.match(alias.strip())
    ]


def _title_rules(
    characters: str, glossary: str, readings: NameReadings
) -> list[_TitleRule]:
    """Các chức danh mà bảng thuật ngữ KHÔNG dịch bằng âm Hán-Việt, dài trước ngắn sau."""
    renderings = [r for r in term_renderings(glossary or "") if _is_title_term(r.term)]
    glossary_terms = {rendering.term for rendering in renderings}
    names = _roster_cjk_names(characters)
    rules: list[_TitleRule] = []
    for rendering in renderings:
        keys = _title_keys(rendering.term, readings)
        replacement = rendering.options[0]
        if keys is None or _UNSAFE_REPLACEMENT_RE.search(replacement):
            continue
        # Bảng thuật ngữ tự dùng âm Hán-Việt ("所長 = sở trưởng") thì hai bảng đã khớp.
        if any(_reads_as(option.split(), keys) for option in rendering.options):
            continue
        rules.append(_TitleRule(rendering.term, 0, keys, replacement))
        for name in names:
            surname = name[: -len(rendering.term)]
            if (
                not name.endswith(rendering.term)
                or not 1 <= len(surname) <= MAX_SURNAME_CHARS
                # Tên có mục riêng trong bảng thuật ngữ thì mục đó quyết định.
                or name in glossary_terms
            ):
                continue
            surname_keys = _char_keys(surname, readings)
            if surname_keys is not None:
                rules.append(
                    _TitleRule(
                        rendering.term, len(surname), (*surname_keys, *keys), replacement
                    )
                )
    return sorted(rules, key=lambda rule: len(rule.keys), reverse=True)


def _match_case(replacement: str, like: str) -> str:
    if like[:1].isupper() and replacement[:1].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _fix_alternative(text: str, rules: list[_TitleRule]) -> tuple[str, _TitleRule] | None:
    """Cách gọi thay cho ``text`` nếu cả ``text`` là âm Hán-Việt của một chức danh."""
    words = text.split()
    for rule in rules:
        if not _reads_as(words, rule.keys):
            continue
        if rule.surname_words:
            # "Phùng xưởng trưởng" → "giám đốc xưởng Phùng": tiếng Việt đặt họ SAU.
            surname = " ".join(words[: rule.surname_words])
            return f"{rule.replacement} {surname}", rule
        return _match_case(rule.replacement, text), rule
    return None


def reconcile_titles(
    characters: str, glossary: str, readings: NameReadings | None
) -> tuple[str, list[TitleFix]]:
    """Đổi cách gọi chức danh trong bảng nhân vật theo bảng thuật ngữ (hàm thuần).

    Args:
        characters: Bảng nhân vật (có các quy tắc "gọi '…'" và "(danh xưng: …)").
        glossary: Bảng thuật ngữ ``gốc = dịch``.
        readings: Bảng âm Hán-Việt theo chữ; None/rỗng thì không làm gì.

    Returns:
        (bảng nhân vật đã đổi, các lần đổi gộp theo cách gọi). Không có gì để đổi thì
        trả đúng chuỗi ``characters`` và danh sách rỗng.
    """
    if not readings or not (characters or "").strip() or not (glossary or "").strip():
        return characters, []
    rules = _title_rules(characters, glossary, readings)
    if not rules:
        return characters, []
    counts: dict[tuple[str, str, str], int] = {}

    def _fix_form(match: re.Match[str]) -> str:
        parts = _ALTERNATIVE_SPLIT_RE.split(match.group("form"))
        for position in range(0, len(parts), 2):
            written = parts[position].strip()
            fixed = _fix_alternative(written, rules)
            if fixed is None:
                continue
            replacement, rule = fixed
            parts[position] = parts[position].replace(written, replacement)
            key = (rule.term, written, replacement)
            counts[key] = counts.get(key, 0) + 1
        return f"{match.group('head')}{''.join(parts)}{match.group('tail')}"

    reconciled = _TITLE_NOTE_RE.sub(_fix_form, _QUOTED_FORM_RE.sub(_fix_form, characters))
    if not counts:
        return characters, []
    return reconciled, [TitleFix(*key, count) for key, count in counts.items()]


def reconcile_context_titles(
    context: ContextT, readings: NameReadings | None
) -> tuple[ContextT, list[TitleFix]]:
    """Áp :func:`reconcile_titles` lên bảng nhân vật của một ngữ cảnh (hàm thuần).

    Args:
        context: Dataclass có ``characters`` và ``glossary`` (vd
            ``SubtitleContextAnalysis``, ``TranslationContext``).
        readings: Bảng âm Hán-Việt; None/rỗng thì trả nguyên ngữ cảnh.

    Returns:
        (ngữ cảnh đã đổi, các lần đổi). Không đổi gì thì trả đúng đối tượng cũ.
    """
    characters, fixes = reconcile_titles(context.characters, context.glossary, readings)
    if not fixes:
        return context, []
    return replace(context, characters=characters), fixes  # type: ignore[type-var]


def describe_title_fix(fix: TitleFix) -> str:
    """Mô tả ngắn cho log/thông báo, vd ``所長: 'Sở trưởng' → 'Trưởng đồn' (4 chỗ)``."""
    return f"{fix.term}: '{fix.written}' → '{fix.replacement}' ({fix.count} chỗ)"


__all__ = [
    "TitleFix",
    "describe_title_fix",
    "reconcile_context_titles",
    "reconcile_titles",
]
