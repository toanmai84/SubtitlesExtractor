"""[v3.23.586] Gỡ CHÚ THÍCH của bảng thuật ngữ bị chép vào lời thoại.

Bảng thuật ngữ ghi ``gốc = bản dịch (chú thích)`` — phần trong ngoặc chỉ để khâu dịch hiểu
nghĩa. Đo trên 50 chuẩn đoán dịch thật: chú thích bị chép nguyên văn vào bản dịch thô ở
459 dòng (mọi lượt chạy đều có), còn sót tới bản cuối ở 25 dòng của 11 lượt:

* ``我给你打幺二零`` → "tôi gọi 120 (cấp cứu y tế) cho cô."
* ``这辆大G是你的？`` → "chiếc xe Mercedes G-Class (Đại G) này là của ai cơ?"
* ``死相`` → "Đồ chết tiệt (câu mắng yêu)"

Phụ đề không có chỗ cho lời giải thích trong ngoặc, và khâu thuyết minh sẽ ĐỌC cả phần đó
lên. Nguyên nhân: khối thuật ngữ nhắc theo từng lô ghi ``must_translate_as`` bằng cả vế
dịch lẫn chú thích (đã sửa ở adapter); module này là lưới tất định phía sau.

Chỉ gỡ ngoặc khi câu GỐC không có ngoặc và nội dung trong ngoặc là:

* chú thích hoặc một phương án dịch của một mục trong bảng thuật ngữ, hoặc
* chính cụm từ đứng ngay trước ngoặc ("Lạng Sơn (Lạng Sơn)").

Ngoặc khác (vd chỉ dẫn "(Thở dốc)") được giữ nguyên.

[v3.23.587] Cùng nguyên nhân, cùng lưới: vế dịch ``A / B`` (hai PHƯƠNG ÁN, chọn một) bị
chép cả hai vào lời thoại — 84 dòng dịch thô ở 28/51 lượt chạy ("Mày nói gì thế, đừng bốc
đồng / manh động chứ."), còn tới bản cuối ở lượt mới nhất ("đến đồn công an xã/phường.").
Khi câu GỐC không có dấu ``/`` mà bản dịch có hai phương án của CÙNG một mục thuật ngữ nối
bằng ``/`` thì giữ phương án đứng trước. Dấu ``/`` khác ("24/24", "tệ/năm") không bị đụng.

Toàn bộ là HÀM THUẦN: không I/O.
"""

# ruff: noqa: RUF001, RUF002 — dấu câu fullwidth CJK là CHỦ ĐÍCH (ngoặc trong câu gốc)

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from subtitles_extractor.application.services.glossary_options import (
    GLOSSARY_NOTE_RE as _NOTE_RE,
)
from subtitles_extractor.application.services.glossary_options import (
    split_glossary_note,
    translation_options,
)
from subtitles_extractor.application.services.roster_format import parse_glossary
from subtitles_extractor.application.services.translation_diagnostics import (
    glossary_alternatives,
)

_PARENTHESIS_CHARS_RE = re.compile(r"[(（]")
_SLASH_CHARS_RE = re.compile(r"[/／]")
_SPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?…])")


@dataclass(frozen=True, slots=True)
class NoteLeak:
    """Một dòng dịch bị chèn chú thích thuật ngữ.

    Attributes:
        index: Chỉ số dòng phụ đề.
        notes: Các chú thích đã gỡ (nguyên văn trong ngoặc).
        text_before: Bản dịch trước khi gỡ.
        text_after: Bản dịch sau khi gỡ.
    """

    index: int
    notes: tuple[str, ...]
    text_before: str
    text_after: str


def _normalise(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip().lower()


def glossary_note_phrases(glossary: str) -> frozenset[str]:
    """Mọi cụm (chữ thường) mà nếu nằm trong ngoặc ở bản dịch thì là chú thích bị chép.

    Gồm nguyên văn từng chú thích và từng phương án dịch của MỤC CÓ CHÚ THÍCH (model có
    lúc viết ngược: ``婚礼 = hôn lễ (lễ cưới)`` → "lễ cưới (hôn lễ)").

    Args:
        glossary: Bảng thuật ngữ, mỗi dòng ``gốc = dịch``.

    Returns:
        Tập cụm đã chuẩn hoá khoảng trắng.
    """
    phrases: set[str] = set()
    for entry in parse_glossary(glossary):
        notes = [match.group(1) for match in _NOTE_RE.finditer(entry.translation)]
        if not notes:
            continue
        phrases.update(_normalise(note) for note in notes)
        phrases.update(glossary_alternatives(entry.translation))
    phrases.discard("")
    return frozenset(phrases)


def _repeats_preceding_words(text: str, match: re.Match[str]) -> bool:
    """Nội dung trong ngoặc lặp lại đúng cụm từ đứng ngay trước nó."""
    inner = _normalise(match.group(1))
    before = _normalise(text[: match.start()])
    if not inner or not before.endswith(inner):
        return False
    boundary = len(before) - len(inner)
    return boundary == 0 or not before[boundary - 1].isalnum()


def strip_leaked_notes(
    text: str, source: str, phrases: frozenset[str]
) -> tuple[str, tuple[str, ...]]:
    """Gỡ các chú thích bị chép khỏi MỘT dòng dịch.

    Args:
        text: Bản dịch của dòng.
        source: Câu gốc của dòng.
        phrases: Kết quả :func:`glossary_note_phrases`.

    Returns:
        ``(bản dịch đã gỡ, các chú thích đã gỡ)``. Không gỡ gì thì trả nguyên ``text``.
    """
    if not text or _PARENTHESIS_CHARS_RE.search(source or ""):
        return text, ()
    removed: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        if _normalise(match.group(1)) in phrases or _repeats_preceding_words(text, match):
            removed.append(match.group(1).strip())
            return " "
        return match.group(0)

    stripped = _NOTE_RE.sub(_replace, text)
    if not removed:
        return text, ()
    cleaned = _SPACE_RE.sub(" ", stripped).strip()
    cleaned = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", cleaned)
    # Dòng chỉ có mỗi chú thích: giữ nguyên, để các lưới khác (dịch thiếu) xử lý.
    if not cleaned or not any(char.isalnum() for char in cleaned):
        return text, ()
    return cleaned, tuple(removed)


def glossary_choice_patterns(glossary: str) -> tuple[re.Pattern[str], ...]:
    """Mẫu nhận ra hai PHƯƠNG ÁN của một mục thuật ngữ bị chép liền nhau ("A / B").

    Mỗi mẫu khớp một phương án (nhóm ``keep``) theo sau bởi một hay nhiều phương án khác
    của CHÍNH mục đó, ngăn bằng ``/``. Với vế dịch viết liền (``đồn công an xã/phường``),
    phần sau dấu ``/`` là mẩu thô ("phường") nên các mẩu thô cũng được nhận ở vế sau.

    Args:
        glossary: Bảng thuật ngữ, mỗi dòng ``gốc = dịch``.

    Returns:
        Các mẫu đã biên dịch, không trùng; rỗng nếu không mục nào có từ hai phương án.
    """
    patterns: dict[str, re.Pattern[str]] = {}
    for entry in parse_glossary(glossary):
        options = translation_options(entry.translation)
        if len(options) < 2:
            continue
        usable, _note = split_glossary_note(entry.translation)
        followers = {*options, *(piece.strip() for piece in usable.split("/"))}
        followers.discard("")
        head = "|".join(_phrase_pattern(option) for option in _longest_first(options))
        tail = "|".join(_phrase_pattern(piece) for piece in _longest_first(followers))
        source = rf"(?<!\w)(?P<keep>{head})(?:\s*/\s*(?:{tail}))+(?!\w)"
        if source not in patterns:
            patterns[source] = re.compile(source, re.IGNORECASE)
    return tuple(patterns.values())


def _longest_first(phrases: tuple[str, ...] | set[str]) -> list[str]:
    """Cụm dài trước: "bệnh bạch cầu cấp tính" phải được thử trước "bệnh bạch cầu"."""
    return sorted(phrases, key=lambda phrase: (-len(phrase), phrase))


def _phrase_pattern(phrase: str) -> str:
    """Biểu thức khớp một cụm, chấp nhận mọi kiểu khoảng trắng giữa các từ."""
    return r"\s+".join(re.escape(word) for word in phrase.split())


def strip_joined_options(
    text: str, source: str, patterns: tuple[re.Pattern[str], ...]
) -> tuple[str, tuple[str, ...]]:
    """Giữ MỘT phương án khi bản dịch chép liền nhiều phương án của một thuật ngữ.

    Args:
        text: Bản dịch của dòng.
        source: Câu gốc của dòng.
        patterns: Kết quả :func:`glossary_choice_patterns`.

    Returns:
        ``(bản dịch đã sửa, các phương án đã bỏ)``. Không sửa gì thì trả nguyên ``text``.
    """
    if "/" not in text or _SLASH_CHARS_RE.search(source or ""):
        return text, ()
    dropped: list[str] = []

    def _keep_first(match: re.Match[str]) -> str:
        kept = match.group("keep")
        dropped.append(match.group(0)[len(kept) :].strip())
        return kept

    for pattern in patterns:
        text = pattern.sub(_keep_first, text)
        if "/" not in text:
            break
    return text, tuple(dropped)


def find_note_leaks(
    sources: Mapping[int, str], translations: Mapping[int, str], glossary: str
) -> list[NoteLeak]:
    """Tìm mọi dòng dịch bị chèn chú thích hoặc chép liền nhiều phương án (hàm thuần).

    Args:
        sources: Câu gốc theo chỉ số dòng.
        translations: Bản dịch theo chỉ số dòng.
        glossary: Bảng thuật ngữ của lượt dịch.

    Returns:
        Các dòng cần sửa, theo thứ tự chỉ số. ``notes`` gồm cả chú thích đã gỡ lẫn các
        phương án đã bỏ (dạng ``"/ manh động"``).
    """
    phrases = glossary_note_phrases(glossary)
    choice_patterns = glossary_choice_patterns(glossary)
    leaks: list[NoteLeak] = []
    for index in sorted(translations):
        text = translations[index]
        source = sources.get(index, "")
        cleaned, removed = strip_leaked_notes(text, source, phrases)
        cleaned, dropped = strip_joined_options(cleaned, source, choice_patterns)
        if removed or dropped:
            leaks.append(NoteLeak(index, (*removed, *dropped), text, cleaned))
    return leaks


__all__ = [
    "NoteLeak",
    "find_note_leaks",
    "glossary_choice_patterns",
    "glossary_note_phrases",
    "split_glossary_note",
    "strip_joined_options",
    "strip_leaked_notes",
]
