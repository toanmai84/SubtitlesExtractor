"""[v3.23.567] Soát phiên âm Hán-Việt của tên nhân vật theo TỪNG CHỮ, sửa lỗi chắc chắn.

Chuẩn đoán thật v3.23.566 (phân tích bằng gemini-3.6-flash): bảng nhân vật ghi
``柳慕云 (Liễu Mộc Vân)`` — nhưng 慕 đọc **mộ**, không phải "mộc" (âm của 木/沐).
Bản dịch chép đúng bảng nên nữ chính thứ hai mang tên sai ở **44/45** dòng nhắc tới cô,
tóm tắt phim sai 28 chỗ. Soát trên 40 bảng nhân vật thật khác còn thấy
``赵可可 (Triệu Kha Kha)`` (可 là **khả**). Model đọc sai âm Hán-Việt là giới hạn của
chính model — nhắc trong prompt không chữa được, nhưng đối chiếu với bảng âm theo từng chữ
thì kiểm được tất định.

Nguyên tắc an toàn (đo trên dữ liệu thật trước khi chọn):

* Chỉ TỰ SỬA mục trong **bảng nhân vật** (tên riêng). Bảng thuật ngữ đầy thành ngữ dịch
  theo nghĩa và âm quy ước ngành (``心包经`` → "Tâm Bào Kinh", ``兑`` → "Đoài") nên chỉ
  báo trong file chuẩn đoán.
* Chỉ sửa khi tên RÕ là phiên âm (ít nhất một chữ khác khớp âm, số chữ khớp ≥ số chữ lệch)
  và âm viết sai chỉ LỆCH MỘT CHÚT so với đúng một âm hợp lệ (bỏ dấu, khác ≤ 1 ký tự):
  "Mộc"→"mộ", "Kha"→"khả". Tên Việt hoá có chủ ý (``吴胖`` → "Ngô Béo") lệch xa nên chỉ
  bị báo, không bị sửa.
* Âm đề xuất chỉ lấy từ nguồn Hán-Việt chính; âm phụ (lẫn âm Nôm) chỉ để CHẤP NHẬN.

[v3.23.591] Đọc nhầm sang âm của CHỮ KHÁC. Chuẩn đoán thật v3.23.590 (gemini-3.8-flash):
``甄劍聲 (Chân Kiếm Sanh)`` — 聲 đọc **thanh**; "sanh" là âm của 甥/笙 (đồng âm *shēng*
trong tiếng phổ thông). Âm sai lệch xa âm đúng nên quy tắc "lệch một chút" không sửa:
tên nam chính sai 16 chỗ trong bảng nhân vật, 53 chỗ trong tóm tắt và ở mọi dòng dịch gọi
tên anh.
Soát 20 bảng nhân vật khác nhau của kho chuẩn đoán: 5 tên bị báo mà không được tự sửa —

* ``甄劍聲 (Chân Kiếm Sanh)``, ``洪大嫂 (Hồng Đại Sảo)``, ``莫嫂 (Mạc Sảo)`` (嫂 đọc
  **tẩu**; "sảo" là âm của 稍): model phiên âm nhưng lấy âm của chữ khác;
* ``宾客 (Quan khách)``, ``王梅 (Chị Mai)``: dịch NGHĨA, đúng.

Dấu hiệu tách hai nhóm: âm viết sai tự nó là một âm Hán-Việt chính của chữ khác ("Sanh",
"Sảo" — còn "Béo", "Chị" thì không), VÀ tên có từ ba chữ trở lên với đúng một chữ lệch
(họ tên đầy đủ; "Quan khách" hai chữ cũng có "quan" là âm Hán-Việt nên chưa đủ chắc). Chỉ
sửa khi chữ đó có ĐÚNG MỘT âm Hán-Việt chính — chữ nhiều âm (曾 tằng/tăng, 單 đơn/thiền/
thiện) không bị đụng tới. Tên ngắn hơn chỉ được sửa THEO: cùng chữ, cùng âm sai đã được
xác nhận ở một tên dài trong chính bảng đó ("Hồng Đại Sảo" → "Mạc Sảo").

[v3.23.611] Tên gọi thân mật hai chữ cũng đủ chắc: ``阿彪 (A Tiêu / Cục trưởng Bưu)``
(chuẩn đoán v3.23.609) — tiền tố 阿/小/老 đã phiên âm đúng cho biết chữ sau là tên riêng;
"tiêu" là âm của 標/鏢. Trừ chữ chỉ quan hệ/vai vế ("阿叔 (A Chú)", "老闆 (Lão Chủ)"):
những cụm đó là danh xưng dịch nghĩa. Đo trên 26 bảng nhân vật thật: chỉ "A Tiêu" được sửa
thêm.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Protocol, TypeVar

from subtitles_extractor.application.services.roster_format import (
    parse_glossary,
    parse_roster,
)

#: Khoảng cách tối đa (sau khi bỏ mọi dấu) giữa âm viết sai và âm đúng để được TỰ SỬA.
MAX_AUTO_FIX_DISTANCE = 1

_TONE_MARKS = frozenset({"̀", "́", "̃", "̉", "̣"})
_CJK_NAME_RE = re.compile(r"^[㐀-䶿一-鿿]+$")
_SYLLABLE_RE = re.compile(r"^[^\W\d_]+$")
_ALIAS_SPLIT_RE = re.compile(r"\s*[/|]\s*")
#: "y" cuối sau phụ âm ⇔ "i" ("Lý"/"Lí", "Kỳ"/"Kì") — hai lối chính tả cùng một âm.
_FINAL_Y_RE = re.compile(r"^(qu|[^aeiouyăâêôơư]*)y$")

SOURCE_ROSTER = "roster"
SOURCE_GLOSSARY = "glossary"


@dataclass(frozen=True)
class NameReadings:
    """Âm Hán-Việt theo từng chữ Hán.

    Attributes:
        primary: Âm Hán-Việt chuẩn — được phép dùng làm âm đề xuất khi sửa.
        extra: Âm bổ sung (có thể lẫn âm Nôm) — chỉ để chấp nhận, không đề xuất.
    """

    primary: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    extra: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_encoded(cls, encoded: Mapping[str, str]) -> NameReadings:
        """Dựng từ dạng mã hoá ``{chữ: "âm chính…|âm phụ…"}`` của file dữ liệu.

        Args:
            encoded: Ánh xạ chữ → chuỗi âm (ngăn bằng dấu cách, ``|`` tách âm phụ).

        Returns:
            Bảng âm đã tách âm chính/âm phụ.
        """
        primary: dict[str, tuple[str, ...]] = {}
        extra: dict[str, tuple[str, ...]] = {}
        for char, value in encoded.items():
            main, _, rest = str(value).partition("|")
            if main.split():
                primary[char] = tuple(main.split())
            if rest.split():
                extra[char] = tuple(rest.split())
        return cls(primary=primary, extra=extra)

    def accepted(self, char: str) -> tuple[str, ...]:
        """Mọi âm được coi là hợp lệ của ``char``."""
        return (*self.primary.get(char, ()), *self.extra.get(char, ()))

    def suggestable(self, char: str) -> tuple[str, ...]:
        """Âm được phép đề xuất khi sửa ``char``."""
        return tuple(self.primary.get(char, ()))

    @cached_property
    def _primary_syllables(self) -> frozenset[str]:
        """Khoá của mọi âm Hán-Việt chính trong bảng (dựng một lần, ~13 nghìn chữ)."""
        return frozenset(
            syllable_key(reading)
            for readings in self.primary.values()
            for reading in readings
        )

    def is_hanviet_syllable(self, syllable: str) -> bool:
        """[v3.23.591] ``syllable`` có phải âm Hán-Việt chính của MỘT chữ nào đó không.

        Args:
            syllable: Một âm tiết tiếng Việt (hoa/thường, vị trí dấu tuỳ ý).

        Returns:
            True nếu có ít nhất một chữ Hán mang âm này trong nguồn Hán-Việt chính.
        """
        return syllable_key(syllable) in self._primary_syllables

    def __bool__(self) -> bool:
        return bool(self.primary or self.extra)


@dataclass(frozen=True)
class NameReadingIssue:
    """Một chữ trong tên được phiên âm khác mọi âm Hán-Việt của nó.

    Attributes:
        source: ``"roster"`` hoặc ``"glossary"``.
        name: Tên gốc (một bí danh chữ Hán) được đối chiếu.
        written: Phiên âm như model đã viết (một phương án trong ngoặc).
        char: Chữ Hán bị phiên âm lệch.
        wrote: Âm model đã viết cho chữ đó.
        expected: Các âm hợp lệ của chữ.
        positions: Vị trí (0-based) của âm lệch trong ``written``.
        suggestion: Âm sửa (giữ kiểu viết hoa) nếu đủ chắc để TỰ SỬA, ngược lại rỗng.
    """

    source: str
    name: str
    written: str
    char: str
    wrote: str
    expected: tuple[str, ...]
    positions: tuple[int, ...]
    suggestion: str = ""

    def as_dict(self) -> dict[str, object]:
        """Dạng JSON cho file chuẩn đoán."""
        return {
            "source": self.source,
            "name": self.name,
            "written": self.written,
            "char": self.char,
            "wrote": self.wrote,
            "expected": list(self.expected),
            "auto_fix": self.suggestion,
        }


class _ContextTexts(Protocol):
    characters: str
    overview: str
    glossary: str
    visual_cues: str


ContextT = TypeVar("ContextT", bound=_ContextTexts)


def syllable_key(syllable: str) -> str:
    """Khoá so sánh một âm tiết: không phân biệt hoa/thường, vị trí dấu thanh, y/i.

    ``"Hoà"`` và ``"hòa"``, ``"Lý"`` và ``"lí"`` cho cùng khoá (hàm thuần).

    Args:
        syllable: Một âm tiết tiếng Việt.

    Returns:
        Khoá dạng ``"<phần chữ>|<dấu thanh>"``.
    """
    decomposed = unicodedata.normalize("NFD", syllable.strip().lower())
    tone = "".join(ch for ch in decomposed if ch in _TONE_MARKS)
    base = unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if ch not in _TONE_MARKS)
    )
    if _FINAL_Y_RE.match(base):
        base = base[:-1] + "i"
    return f"{base}|{tone}"


def _bare(syllable: str) -> str:
    """Bỏ MỌI dấu (thanh lẫn mũ/móc) và đ→d: "Mộc" → "moc"."""
    decomposed = unicodedata.normalize("NFD", syllable.lower().replace("đ", "d"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _nearest_reading(wrote: str, candidates: tuple[str, ...]) -> str:
    """Âm gần nhất với ``wrote`` nếu DUY NHẤT và đủ gần, ngược lại rỗng."""
    lowered = unicodedata.normalize("NFC", wrote.lower())
    # Xếp theo (khoảng cách bỏ dấu, khoảng cách đủ dấu): hoà ở tiêu chí đầu thì tiêu chí
    # sau phân xử; hoà cả hai thì không đủ chắc để chọn.
    scored = sorted(
        (
            (
                _levenshtein(_bare(lowered), _bare(reading)),
                _levenshtein(lowered, unicodedata.normalize("NFC", reading)),
            ),
            reading,
        )
        for reading in candidates
    )
    if not scored or scored[0][0][0] > MAX_AUTO_FIX_DISTANCE:
        return ""
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return ""
    return scored[0][1]


def _match_case(reading: str, like: str) -> str:
    return reading[:1].upper() + reading[1:] if like[:1].isupper() else reading


def _split_aliases(value: str) -> list[str]:
    return [part.strip() for part in _ALIAS_SPLIT_RE.split(value or "") if part.strip()]


@dataclass(frozen=True)
class _Alignment:
    name: str
    matched: frozenset[str]
    wrong: dict[str, tuple[str, tuple[int, ...]]]
    distance: int


#: [v3.23.591] Tên từ ngần này chữ trở lên mới đủ chắc là HỌ TÊN phiên âm (xem đầu
#: module).
OTHER_CHARACTER_MIN_NAME_CHARS = 3

#: [v3.23.611] Tiền tố gọi thân mật ("阿彪", "小蓮", "老莫"): khi chính tiền tố được phiên
#: âm ("A", "Tiểu", "Lão") thì chữ sau chắc chắn là tên riêng đang được phiên âm.
NICKNAME_PREFIXES = frozenset("阿小老")

#: [v3.23.611] Chữ chỉ quan hệ/vai vế hay đứng sau tiền tố: "阿叔 (A Chú)", "老闆 (Lão
#: Chủ)" là danh xưng dịch nghĩa có chủ ý, dù "chú", "chủ" cũng là âm Hán-Việt của chữ
#: khác.
_KINSHIP_CHARS = frozenset(
    "哥姐姊弟妹兄叔伯嬸婶姨婆公爺爷爹媽妈娘爸舅姑嫂妻夫頭头板闆仔"
)


def _sure_transliteration(alignment: _Alignment) -> bool:
    """Tên chắc chắn đang được PHIÊN ÂM, đủ để sửa âm đọc nhầm sang chữ khác.

    Họ tên từ :data:`OTHER_CHARACTER_MIN_NAME_CHARS` chữ, hoặc [v3.23.611] tên gọi thân
    mật hai chữ có tiền tố đã phiên âm đúng và chữ sau không phải danh xưng quan hệ —
    cả hai chỉ khi có ĐÚNG MỘT chữ lệch.

    Args:
        alignment: Kết quả đối chiếu tên với phiên âm.

    Returns:
        True nếu đủ chắc.
    """
    if len(alignment.wrong) != 1:
        return False
    name = alignment.name
    if len(name) >= OTHER_CHARACTER_MIN_NAME_CHARS:
        return True
    return (
        len(name) == 2
        and name[0] in NICKNAME_PREFIXES
        and name[0] in alignment.matched
        and name[1] not in _KINSHIP_CHARS
    )


def _other_character_reading(
    alignment: _Alignment, char: str, readings: NameReadings
) -> str:
    """[v3.23.591] Âm đúng khi model đọc ``char`` bằng âm Hán-Việt của một CHỮ KHÁC.

    ``聲`` viết "Sanh" (âm của 甥/笙) → "thanh". Chỉ trả âm khi: tên chắc chắn đang
    được phiên âm (:func:`_sure_transliteration`); âm đã viết là một âm Hán-Việt chính có
    thật (đang phiên âm, không phải dịch nghĩa như "Béo"); và chữ chỉ có ĐÚNG MỘT âm
    Hán-Việt chính (không phải đoán giữa nhiều âm).

    Args:
        alignment: Kết quả đối chiếu tên với phiên âm.
        char: Chữ Hán bị lệch đang xét.
        readings: Bảng âm.

    Returns:
        Âm đúng, hoặc rỗng nếu không đủ chắc.
    """
    if not _sure_transliteration(alignment):
        return ""
    wrote, _ = alignment.wrong[char]
    candidates = readings.suggestable(char)
    if len(candidates) != 1 or not readings.is_hanviet_syllable(wrote):
        return ""
    return candidates[0]


#: [v3.23.615] Số đếm đọc bằng số từ THUẦN VIỆT là dịch nghĩa, không phải đọc nhầm âm:
#: ``两位长老 (Hai vị trưởng lão)``, ``三胖子 (Ba Béo)``. Chẩn đoán thật 14/10
#: (gemini-3.8-flash) báo "两 đọc lưỡng/lượng/lạng" cho nhãn "Hai vị trưởng lão" — cảnh
#: báo oan duy nhất của lượt phân tích đó. Đo trên 28 bảng nhân vật thật: chỉ mục này hết
#: bị báo.
_NATIVE_NUMERALS: dict[str, frozenset[str]] = {
    char: frozenset(syllable_key(word) for word in words)
    for chars, words in (
        ("一壹", ("một",)),
        ("二两兩贰貳", ("hai",)),
        ("三叁參", ("ba",)),
        ("四肆", ("bốn",)),
        ("五伍", ("năm",)),
        ("六陆陸", ("sáu",)),
        ("七柒", ("bảy", "bẩy")),
        ("八捌", ("tám",)),
        ("九玖", ("chín",)),
        ("十拾", ("mười", "mươi")),
        ("百佰", ("trăm",)),
        ("千仟", ("nghìn", "ngàn")),
    )
    for char in chars
}


def _align(name: str, syllables: list[str], readings: NameReadings) -> _Alignment:
    matched: set[str] = set()
    wrong: dict[str, tuple[str, tuple[int, ...]]] = {}
    distance = 0
    for position, (char, syllable) in enumerate(zip(name, syllables, strict=True)):
        accepted = readings.accepted(char)
        if not accepted:
            continue
        if syllable_key(syllable) in _NATIVE_NUMERALS.get(char, frozenset()):
            matched.add(char)
            continue
        if syllable_key(syllable) in {syllable_key(r) for r in accepted}:
            matched.add(char)
            continue
        distance += min(_levenshtein(_bare(syllable), _bare(r)) for r in accepted)
        wrote, positions = wrong.get(char, (syllable, ()))
        wrong[char] = (wrote, (*positions, position))
    return _Alignment(name, frozenset(matched - set(wrong)), wrong, distance)


def _best_alignment(
    names: list[str], syllables: list[str], readings: NameReadings
) -> _Alignment | None:
    candidates = [
        _align(name, syllables, readings)
        for name in names
        if _CJK_NAME_RE.match(name) and len(name) == len(syllables) >= 2
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda a: (-len(a.matched), a.distance))


def _issues_for_pair(
    source: str, name_field: str, aka_field: str, readings: NameReadings
) -> list[NameReadingIssue]:
    names = _split_aliases(name_field)
    issues: list[NameReadingIssue] = []
    for written in _split_aliases(aka_field):
        syllables = written.split()
        if not syllables or not all(_SYLLABLE_RE.match(s) for s in syllables):
            continue
        alignment = _best_alignment(names, syllables, readings)
        if alignment is None or not alignment.wrong or not alignment.matched:
            continue
        # Chỉ coi là PHIÊN ÂM (không phải dịch nghĩa) khi phần khớp không ít hơn phần lệch
        # và mọi âm lệch đều viết hoa như tên riêng.
        if len(alignment.matched) < len(alignment.wrong):
            continue
        if not all(wrote[:1].isupper() for wrote, _ in alignment.wrong.values()):
            continue
        suggestions = {
            char: _nearest_reading(wrote, readings.suggestable(char))
            or _other_character_reading(alignment, char, readings)
            for char, (wrote, _) in alignment.wrong.items()
        }
        fixable = source == SOURCE_ROSTER and all(suggestions.values())
        for char, (wrote, positions) in alignment.wrong.items():
            issues.append(
                NameReadingIssue(
                    source=source,
                    name=alignment.name,
                    written=written,
                    char=char,
                    wrote=wrote,
                    expected=readings.accepted(char),
                    positions=positions,
                    suggestion=_match_case(suggestions[char], wrote) if fixable else "",
                )
            )
    return issues


def _follow_confirmed_misreadings(
    issues: list[NameReadingIssue],
) -> list[NameReadingIssue]:
    """[v3.23.591] Sửa THEO: cùng chữ, cùng âm sai đã được xác nhận ở tên khác trong bảng.

    Model đọc nhầm một chữ thì nhầm ở mọi tên có chữ đó: ``洪大嫂 (Hồng Đại Sảo)`` đủ chắc
    để sửa thì ``莫嫂 (Mạc Sảo)`` — tên hai chữ, tự nó chưa đủ chắc — cũng sửa theo, để
    một bảng không vừa "Tẩu" vừa "Sảo". Chỉ áp cho tên có đúng một chữ lệch.

    Args:
        issues: Lệch âm của bảng nhân vật.

    Returns:
        Danh sách cùng thứ tự; mục sửa theo được điền ``suggestion``.
    """
    confirmed = {
        (issue.char, syllable_key(issue.wrote)): issue.suggestion
        for issue in issues
        if issue.suggestion
    }
    if not confirmed:
        return issues
    wrong_per_name: dict[tuple[str, str], int] = {}
    for issue in issues:
        key = (issue.name, issue.written)
        wrong_per_name[key] = wrong_per_name.get(key, 0) + 1
    followed: list[NameReadingIssue] = []
    for issue in issues:
        suggestion = confirmed.get((issue.char, syllable_key(issue.wrote)), "")
        if (
            not issue.suggestion
            and suggestion
            and wrong_per_name[(issue.name, issue.written)] == 1
        ):
            issue = replace(issue, suggestion=_match_case(suggestion, issue.wrote))
        followed.append(issue)
    return followed


def find_name_reading_issues(
    characters: str, glossary: str, readings: NameReadings
) -> list[NameReadingIssue]:
    """Tìm các chữ trong tên được phiên âm khác âm Hán-Việt (hàm thuần).

    Args:
        characters: Bảng nhân vật (quy cách ``Tên (Phiên âm) …``).
        glossary: Bảng thuật ngữ (``gốc = dịch``).
        readings: Bảng âm Hán-Việt theo chữ.

    Returns:
        Danh sách lệch âm; mục có ``suggestion`` là mục đủ chắc để tự sửa.
    """
    if not readings:
        return []
    issues: list[NameReadingIssue] = []
    for entry in parse_roster(characters or ""):
        issues.extend(_issues_for_pair(SOURCE_ROSTER, entry.name, entry.aka, readings))
    issues = _follow_confirmed_misreadings(issues)
    for term in parse_glossary(glossary or ""):
        issues.extend(
            _issues_for_pair(SOURCE_GLOSSARY, term.term, term.translation, readings)
        )
    return issues


def _other_characters_akas(characters: str, fixed_names: set[str]) -> set[str]:
    """Phiên âm của các nhân vật KHÔNG bị sửa (bí danh của chính nhân vật đang sửa, vd
    ``Kha Kha`` của ``赵可可 / 可可``, phải được sửa theo chứ không được che chắn)."""
    return {
        unicodedata.normalize("NFC", alt)
        for entry in parse_roster(characters or "")
        if not set(_split_aliases(entry.name)) & fixed_names
        for alt in _split_aliases(entry.aka)
    }


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def name_corrections(
    issues: list[NameReadingIssue], characters: str = ""
) -> dict[str, str]:
    """Các cụm cần thay: mọi đoạn ≥ 2 âm tiết của tên có chứa âm sai (hàm thuần).

    ``Liễu Mộc Vân`` (sai "Mộc") cho ``Liễu Mộc Vân``, ``Liễu Mộc``, ``Mộc Vân`` → bản
    sửa; không thay một âm tiết đứng riêng ("mộc" còn là từ thường). Cụm trùng phiên âm
    ĐÚNG của một nhân vật khác trong bảng bị loại để không sửa nhầm người khác.

    Args:
        issues: Lệch âm đã có ``suggestion`` (các mục không có bị bỏ qua).
        characters: Bảng nhân vật, để loại cụm trùng tên người khác.

    Returns:
        Ánh xạ cụm sai → cụm đúng.
    """
    fixed_syllables: dict[str, list[str]] = {}
    for issue in issues:
        if not issue.suggestion:
            continue
        syllables = fixed_syllables.setdefault(
            issue.written, unicodedata.normalize("NFC", issue.written).split()
        )
        for position in issue.positions:
            syllables[position] = unicodedata.normalize("NFC", issue.suggestion)
    others = _other_characters_akas(
        characters, {issue.name for issue in issues if issue.suggestion}
    )
    corrections: dict[str, str] = {}
    for written, fixed in fixed_syllables.items():
        original = unicodedata.normalize("NFC", written).split()
        for start in range(len(original)):
            for end in range(start + 2, len(original) + 1):
                if original[start:end] == fixed[start:end]:
                    continue
                phrase = " ".join(original[start:end])
                if any(_contains_phrase(alt, phrase) for alt in others):
                    continue
                corrections[phrase] = " ".join(fixed[start:end])
    return corrections


def apply_name_corrections(text: str, corrections: Mapping[str, str]) -> str:
    """Thay các cụm tên sai trong ``text`` (nguyên từ, cụm dài trước) — hàm thuần.

    Args:
        text: Văn bản bất kỳ (bảng nhân vật, tóm tắt, cues…).
        corrections: Ánh xạ từ :func:`name_corrections`.

    Returns:
        Văn bản đã sửa (giữ nguyên nếu không có gì để thay).
    """
    if not text or not corrections:
        return text
    normalized = unicodedata.normalize("NFC", text)
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(key) for key in sorted(corrections, key=len, reverse=True))
        + r")(?!\w)"
    )
    if pattern.search(normalized) is None:
        return text
    return pattern.sub(lambda match: corrections[match.group(0)], normalized)


def correct_context_names(
    context: ContextT, readings: NameReadings | None
) -> tuple[ContextT, list[NameReadingIssue]]:
    """Sửa phiên âm tên sai CHẮC CHẮN trong mọi trường văn bản của ngữ cảnh (hàm thuần).

    Args:
        context: Dataclass có ``characters``/``overview``/``glossary``/``visual_cues``
            (vd ``SubtitleContextAnalysis``, ``TranslationContext``).
        readings: Bảng âm; None/rỗng thì trả nguyên ngữ cảnh.

    Returns:
        (ngữ cảnh đã sửa, các lệch âm đã sửa). Không sửa gì thì trả đúng đối tượng cũ.
    """
    if not readings:
        return context, []
    issues = find_name_reading_issues(context.characters, context.glossary, readings)
    fixes = [issue for issue in issues if issue.suggestion]
    corrections = name_corrections(fixes, context.characters)
    if not corrections:
        return context, []
    corrected = replace(
        context,  # type: ignore[type-var]
        characters=apply_name_corrections(context.characters, corrections),
        overview=apply_name_corrections(context.overview, corrections),
        glossary=apply_name_corrections(context.glossary, corrections),
        visual_cues=apply_name_corrections(context.visual_cues, corrections),
    )
    return corrected, fixes


def describe_fix(issue: NameReadingIssue) -> str:
    """Mô tả ngắn một lần sửa cho log/thông báo: ``柳慕云: "Mộc" → "Mộ" (慕 đọc mộ)``."""
    return (
        f'{issue.name}: "{issue.wrote}" → "{issue.suggestion}" '
        f"({issue.char} đọc {'/'.join(issue.expected)})"
    )


__all__ = [
    "MAX_AUTO_FIX_DISTANCE",
    "NICKNAME_PREFIXES",
    "OTHER_CHARACTER_MIN_NAME_CHARS",
    "SOURCE_GLOSSARY",
    "SOURCE_ROSTER",
    "NameReadingIssue",
    "NameReadings",
    "apply_name_corrections",
    "correct_context_names",
    "describe_fix",
    "find_name_reading_issues",
    "name_corrections",
    "syllable_key",
]
