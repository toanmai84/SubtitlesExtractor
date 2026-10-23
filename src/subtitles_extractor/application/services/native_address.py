"""[v3.23.612] Xưng hô THUẦN VIỆT: "anh Kiếm" thay "Kiếm ca", "chị Hồng" thay "Hồng
đại tẩu".

Model dịch phim Trung hay chép tiếng gọi tên kèm vai vế theo âm Hán-Việt: ``劍哥`` →
"Kiếm ca", ``洪大嫂`` → "Hồng đại tẩu", ``甄先生`` → "Chân tiên sinh". Cách gọi đó hợp
phim cổ trang, kiếm hiệp. Với phim hiện đại, đời thường thì người Việt nói "anh Kiếm",
"chị Hồng", "ông Chân". Bản cuối phim "灭门惨案II" (v3.23.609) có 28 lần "Kiếm ca" và 8
lần "đại tẩu". Phim "Hệ thống sủng thê" có 28 lần "Lâm tiên sinh", 23 lần "Cường ca"
và "Dật ca". Một phim đô thị khác có 20 lần "Cẩu ca", 10 lần "Tống tiên sinh".

Từ vai vế đứng trước tên KHÔNG cố định. Cảnh sát gọi ``甄大嫂`` là "cô Chân" (bảng
nhân vật ghi cảnh sát gọi cô ấy 'cô'), em chồng thì gọi "chị Chân". Vì vậy từ vai vế
lấy theo đúng đại từ người nói dùng để gọi người đó trong bảng nhân vật. Không có quy
tắc, hoặc quy tắc dùng "mày"/"tôi"…, thì lấy từ mặc định theo hậu tố (哥 → anh, 嫂 →
chị, 先生 → ông, hoặc "anh" khi người đó dưới 40 tuổi).

An toàn (hàm thuần, không I/O):

* Dòng dịch chỉ được đổi khi câu gốc có chữ Hán của hậu tố (``哥`` cho "ca") và chữ đứng
  trước hậu tố là âm tiết TÊN của một nhân vật trong bảng. Nhờ vậy "ca hát", "hai tỷ
  đồng" hay tiếng lóng "đại ca" không bị đụng tới.
* Bảng nhân vật, tóm tắt, thuật ngữ không có câu gốc để đối chiếu, nên chỉ đổi những
  tiếng gọi mà chính bảng nhân vật/thuật ngữ ghi là tên gọi của một nhân vật.
* Giữ nguyên phiên âm tên ("Kiếm", "Chân") và kiểu "A + tên" ("A Tú").
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

from subtitles_extractor.application.services.roster_format import (
    RosterEntry,
    address_rule_map,
    lookup_address_rule,
    name_aliases,
    parse_glossary,
    parse_roster,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
    gender_from_roster,
)

GENDER_MALE = "nam"
GENDER_FEMALE = "nữ"

LOCATION_ROSTER = "roster"
LOCATION_OVERVIEW = "overview"
LOCATION_GLOSSARY = "glossary"
LOCATION_LINE = "line"

#: Người được gọi "先生" dưới tuổi này thì gọi "anh", từ tuổi này trở lên gọi "ông".
SENIOR_AGE = 40


@dataclass(frozen=True, slots=True)
class CalqueSuffix:
    """Một hậu tố vai vế bị dịch theo âm Hán-Việt sau tên ("Kiếm ca").

    Attributes:
        words: Các cách viết hậu tố trong bản dịch (chữ thường).
        cjk: Chữ Hán của hậu tố — câu gốc phải có một trong số này.
        prefix: Từ vai vế thuần Việt mặc định, đặt TRƯỚC tên.
        gender: Giới tính người được gọi mà hậu tố cho biết (rỗng nếu không rõ).
        younger: Hậu tố gọi người NHỎ hơn người nói (弟, 妹) — chỉ hợp "em".
        not_before: Từ đứng ngay sau làm hậu tố thành từ ghép khác nghĩa ("đệ tử",
            "ca hát", "tỷ lệ") — gặp thì không phải tiếng gọi.
    """

    words: tuple[str, ...]
    cjk: tuple[str, ...]
    prefix: str
    gender: str
    younger: bool = False
    not_before: frozenset[str] = frozenset()


#: Thứ tự quan trọng: hậu tố dài đứng trước ("đại tẩu" trước "tẩu", "ca ca" trước "ca").
NAME_CALQUES: tuple[CalqueSuffix, ...] = (
    CalqueSuffix(("đại tẩu",), ("大嫂",), "chị", GENDER_FEMALE),
    CalqueSuffix(("tẩu tử",), ("嫂子",), "chị", GENDER_FEMALE),
    CalqueSuffix(("tiên sinh",), ("先生",), "ông", GENDER_MALE),
    CalqueSuffix(("ca ca",), ("哥哥",), "anh", GENDER_MALE),
    CalqueSuffix(("tỷ tỷ", "tỉ tỉ"), ("姐姐",), "chị", GENDER_FEMALE),
    CalqueSuffix(("đệ đệ",), ("弟弟",), "em", GENDER_MALE, younger=True),
    CalqueSuffix(("muội muội",), ("妹妹",), "em", GENDER_FEMALE, younger=True),
    CalqueSuffix(
        ("ca",), ("哥",), "anh", GENDER_MALE,
        not_before=frozenset({"hát", "sĩ", "khúc", "từ", "dao", "kịch", "múa", "ngợi",
                              "tụng", "trưởng"}),
    ),
    CalqueSuffix(
        ("tỷ", "tỉ"), ("姐",), "chị", GENDER_FEMALE,
        not_before=frozenset({"lệ", "phú", "số", "suất", "đồng", "giá", "mỉ"}),
    ),
    CalqueSuffix(
        ("tẩu",), ("嫂",), "chị", GENDER_FEMALE,
        not_before=frozenset({"thoát", "tán", "hỏa", "hoả"}),
    ),
    CalqueSuffix(("muội",), ("妹",), "em", GENDER_FEMALE, younger=True),
    CalqueSuffix(
        ("đệ",), ("弟",), "em", GENDER_MALE, younger=True,
        not_before=frozenset({"tử", "nhất", "nhị", "tam", "trình", "đơn"}),
    ),
    CalqueSuffix(
        ("thúc",), ("叔",), "chú", GENDER_MALE,
        not_before=frozenset({"giục", "đẩy", "ép"}),
    ),
    CalqueSuffix(
        ("di",), ("姨",), "dì", GENDER_FEMALE,
        not_before=frozenset({"chúc", "sản", "chuyển", "động", "tích", "cư", "dân",
                              "truyền", "ngôn", "hài", "căn", "tản", "hận", "chứng"}),
    ),
    CalqueSuffix(
        ("thẩm",), ("嬸", "婶"), "thím", GENDER_FEMALE,
        not_before=frozenset({"vấn", "định", "phán", "tra", "mỹ", "quyền", "thấu",
                              "duyệt", "cứu"}),
    ),
)
_SIR = NAME_CALQUES[2]

#: Danh xưng đứng một mình (không kèm tên): đổi nguyên cụm.
TITLE_NOUNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("đại tẩu", ("大嫂",), "chị dâu"),
    ("tẩu tử", ("嫂子",), "chị dâu"),
    ("biểu ca", ("表哥",), "anh họ"),
    ("biểu tỷ", ("表姐",), "chị họ"),
    ("biểu đệ", ("表弟",), "em họ"),
    ("biểu muội", ("表妹",), "em họ"),
)

#: Từ vai vế đặt trước tên, theo giới tính người được gọi.
_MALE_KIN = frozenset({"anh", "chú", "ông", "cậu", "dượng"})
_FEMALE_KIN = frozenset({"chị", "cô", "bà", "dì", "thím", "mợ"})
_NEUTRAL_KIN = frozenset({"bác"})
_YOUNGER_KIN = frozenset({"em"})

#: Tiền tố gọi thân mật đứng trước tên ("A Tú tỷ" → "chị Tú"): bỏ khi đã có từ vai vế.
_NICKNAME_WORDS = frozenset({"a", "tiểu", "lão"})
#: "这位先生" → "Vị Lâm tiên sinh": "vị" thừa khi đã có từ vai vế ("ông Lâm").
_DROPPED_DETERMINERS = frozenset({"vị"})
#: Chữ viết hoa trong phiên âm nhưng không phải âm tiết tên ("Vợ Hồng Chấn").
_NOT_NAME_WORDS = frozenset({
    "vợ", "chồng", "anh", "chị", "em", "ông", "bà", "cô", "chú", "bác", "cậu", "dì",
    "thím", "mợ", "con", "cháu", "bố", "mẹ", "cha", "má", "ba", "đại", "tiểu", "lão", "a",
    "vị", "biểu", "thiếu", "tổng", "cục", "đồn", "sở", "trưởng", "bạn", "gái", "trai",
    "người", "nhà", "họ", "bé", "thằng", "cả", "hai", "út",
})
#: Tên gọi kèm hậu tố dài tối đa ngần này âm tiết ("Tống Tiền đệ đệ").
_MAX_CORE_WORDS = 3

_WORD_RE = re.compile(r"[^\W\d_]+")
_AGE_RE = re.compile(r"\d+")
_CALL_RE = re.compile(r"gọi\s+'([^']*)'", re.IGNORECASE)
_TITLE_CLAUSE_RE = re.compile(
    r"gọi\s+'(?P<call>[^']*)'(?P<gap>\s*)\(danh xưng:\s*(?P<title>[^)]*)\)",
    re.IGNORECASE,
)
#: Dấu đứng ngay trước chỗ bắt đầu câu mới.
_SENTENCE_OPENERS = frozenset(".!?…:\"“«-—–")  # noqa: RUF001 — dấu câu chữ Việt cố ý
#: Dấu ngăn tiếng gọi với phần còn lại của câu.
_VOCATIVE_BEFORE = frozenset(",;:—–-")  # noqa: RUF001 — gạch ngang cố ý
_VOCATIVE_AFTER = frozenset(",.!?…;:")


class _ContextTexts(Protocol):
    characters: str
    overview: str
    glossary: str


ContextT = TypeVar("ContextT", bound=_ContextTexts)


@dataclass(frozen=True, slots=True)
class NativeAddressChange:
    """Một tiếng gọi đã được đổi sang thuần Việt.

    Attributes:
        location: :data:`LOCATION_ROSTER`, :data:`LOCATION_OVERVIEW`,
            :data:`LOCATION_GLOSSARY` hoặc :data:`LOCATION_LINE`.
        before: Cụm cũ ("Kiếm ca").
        after: Cụm mới ("anh Kiếm").
        index: Chỉ số dòng phụ đề (0 với bảng nhân vật/tóm tắt/thuật ngữ).
    """

    location: str
    before: str
    after: str
    index: int = 0


@dataclass(frozen=True, slots=True)
class _Calque:
    """Một tiếng gọi Hán-Việt tìm thấy trong văn bản."""

    start: int
    end: int
    core: str
    suffix: CalqueSuffix
    text: str
    suffix_text: str


def _roster_gender(written: str) -> str:
    """Giới tính ghi trong bảng → :data:`GENDER_MALE`/:data:`GENDER_FEMALE`/rỗng."""
    voice = gender_from_roster(written)
    if voice == VOICE_FEMALE:
        return GENDER_FEMALE
    return GENDER_MALE if voice == VOICE_MALE else ""


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _key(text: str) -> str:
    """Khoá so sánh một cụm: NFC, chữ thường, một khoảng trắng giữa các từ."""
    return " ".join(_nfc(text).lower().split())


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Khớp nguyên cụm (khoảng trắng giữa các từ tuỳ ý), không dính vào từ khác."""
    body = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _prefix_fits(word: str, suffix: CalqueSuffix, gender: str) -> bool:
    """Từ vai vế hợp hậu tố (bậc trên/dưới) và giới tính người được gọi."""
    if suffix.younger:
        return word in _YOUNGER_KIN
    if word in _NEUTRAL_KIN:
        return True
    if word in _MALE_KIN:
        return gender in ("", GENDER_MALE)
    if word in _FEMALE_KIN:
        return gender in ("", GENDER_FEMALE)
    return False


def _capitalized(word: str) -> bool:
    return word[:1].isupper()


def _is_kin_word(word: str) -> bool:
    lowered = word.lower()
    return lowered in _MALE_KIN or lowered in _FEMALE_KIN or lowered in (
        _NEUTRAL_KIN | _YOUNGER_KIN
    )


def _split_calque_phrase(words: list[str]) -> tuple[list[str], CalqueSuffix] | None:
    """``["Hồng", "đại", "tẩu"]`` → (``["Hồng"]``, hậu tố "đại tẩu"); không phải → None.

    Tiền tố gọi thân mật đầu tên bị bỏ như khi đổi trong dòng dịch: "Tiểu Tiền đệ đệ" →
    phần tên "Tiền" ("em Tiền").
    """
    for suffix in NAME_CALQUES:
        for spelling in suffix.words:
            tail = spelling.split()
            if not len(tail) < len(words) <= len(tail) + _MAX_CORE_WORDS:
                continue
            if [word.lower() for word in words[-len(tail):]] != tail:
                continue
            core = words[:-len(tail)]
            if not all(_capitalized(word) for word in core):
                continue
            if len(core) > 1 and core[0].lower() in _NICKNAME_WORDS:
                core = core[1:]
            return core, suffix
    return None


def _cjk_confirms(cjk_names: Iterable[str], suffix: CalqueSuffix) -> bool:
    """Một tên chữ Hán kết thúc bằng chữ Hán của hậu tố (王姨 → hậu tố 姨)."""
    return any(name.endswith(suffix.cjk) for name in cjk_names)


def _confirmed_split(
    words: list[str], cjk_names: Iterable[str]
) -> tuple[list[str], CalqueSuffix] | None:
    """Như :func:`_split_calque_phrase` nhưng hậu tố VIẾT HOA cần tên chữ Hán xác nhận.

    "Lâm Di" (林怡) trùng âm "di" (姨) nhưng là tên riêng; "Vương Di" (王姨) mới là tiếng
    gọi. Hậu tố viết thường ("Kiếm ca") luôn là tiếng gọi.
    """
    split = _split_calque_phrase(words)
    if split is None:
        return None
    suffix_words = words[-len(split[1].words[0].split()):]
    if any(_capitalized(word) for word in suffix_words) and not _cjk_confirms(
        cjk_names, split[1]
    ):
        return None
    return split


def _alias_name_words(alias: str, cjk_names: tuple[str, ...]) -> list[str]:
    """Phần TÊN của một bí danh trong bảng nhân vật, có đối chiếu tên chữ Hán."""
    words = _nfc(alias).split()
    split = _confirmed_split(words, cjk_names)
    if split is not None:
        return split[0]
    if _split_calque_phrase(words) is not None:
        # Hậu tố viết hoa mà tên chữ Hán không xác nhận → cả cụm là tên ("Lâm Di").
        return words if all(_capitalized(word) for word in words) else []
    return _name_words(words)


def _name_words(words: list[str]) -> list[str]:
    """Phần TÊN của một cách gọi: "Kiếm ca" → Kiếm; "anh Kiếm" → Kiếm; "A Tú" → A Tú.

    Cách gọi đã thuần Việt ("anh Kiếm" — bảng nhân vật đã đổi) vẫn cho ra tên, để dòng
    dịch còn sót "Kiếm ca" vẫn được nhận ra.
    """
    calque = _split_calque_phrase(words)
    if calque is not None:
        return calque[0]
    if len(words) > 1 and _is_kin_word(words[0]) and all(
        _capitalized(word) for word in words[1:]
    ):
        return words[1:]
    if words and all(_capitalized(word) for word in words):
        return words
    return []


def name_syllables(characters: str) -> frozenset[str]:
    """Âm tiết TÊN riêng (chữ thường) lấy từ phiên âm trong bảng nhân vật (hàm thuần).

    Lấy từ các bí danh viết hoa toàn bộ ("Chân Kiếm Thanh", "A Tú") và phần tên của
    tiếng gọi kiểu Hán-Việt ("Kiếm ca", "Hồng đại tẩu"); bỏ từ vai vế, chức danh.

    Args:
        characters: Bảng nhân vật.

    Returns:
        Tập âm tiết tên.
    """
    syllables: set[str] = set()
    for entry in parse_roster(characters or ""):
        cjk_names = tuple(name_aliases(entry.name))
        for alias in name_aliases(entry.aka):
            words = _alias_name_words(alias, cjk_names)
            if not words:
                continue
            syllables.update(
                word.lower() for word in words
                if _WORD_RE.fullmatch(word) and word.lower() not in _NOT_NAME_WORDS
            )
    return frozenset(syllables)


def _suffix_pattern() -> re.Pattern[str]:
    spellings = sorted(
        {spelling for suffix in NAME_CALQUES for spelling in suffix.words},
        key=len, reverse=True,
    )
    body = "|".join(r"\s+".join(re.escape(w) for w in s.split()) for s in spellings)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


_SUFFIX_RE = _suffix_pattern()


def _suffix_for(spelling: str) -> CalqueSuffix:
    normalized = _key(spelling)
    return next(suffix for suffix in NAME_CALQUES if normalized in suffix.words)


def _find_calques(text: str, syllables: frozenset[str]) -> list[_Calque]:
    """Mọi cụm "<tên> <hậu tố Hán-Việt>" mà tên là âm tiết tên trong bảng nhân vật."""
    found: list[_Calque] = []
    for match in _SUFFIX_RE.finditer(text):
        suffix = _suffix_for(match.group())
        following = _WORD_RE.match(text[match.end():].lstrip())
        if (
            following is not None
            and text[match.end():match.end() + 1].isspace()
            and following.group().lower() in suffix.not_before
        ):
            continue
        words = list(_WORD_RE.finditer(text[:match.start()]))
        core: list[re.Match[str]] = []
        for word in reversed(words):
            if len(core) == _MAX_CORE_WORDS:
                break
            gap = text[word.end():(core[0].start() if core else match.start())]
            if gap.strip() or not gap or not _capitalized(word.group()):
                break
            if word.group().lower() not in syllables:
                break
            core.insert(0, word)
        if not core:
            continue
        start = core[0].start()
        before = [word for word in words if word.end() <= start]
        if before:
            previous = before[-1]
            gap = text[previous.end():start]
            if gap and not gap.strip() and previous.group().lower() in (
                _NICKNAME_WORDS | _DROPPED_DETERMINERS
            ):
                start = previous.start()
        if found and start < found[-1].end:
            continue
        found.append(
            _Calque(
                start=start,
                end=match.end(),
                core=" ".join(word.group() for word in core),
                suffix=suffix,
                text=text[start:match.end()],
                suffix_text=match.group(),
            )
        )
    return found


def _sentence_start(text: str, position: int) -> bool:
    head = text[:position].rstrip()
    return not head or head[-1] in _SENTENCE_OPENERS


def _render(prefix: str, core: str, *, capitalize: bool) -> str:
    word = prefix[:1].upper() + prefix[1:] if capitalize else prefix
    return f"{word} {core}"


def _call_prefix(value: str) -> str:
    """Từ vai vế của một cách gọi: "anh" ← "anh", "anh Tống Tiền"; rỗng ← "chú em".

    Từ đứng sau mà viết thường thì cả cụm là một danh xưng khác nghĩa từ đầu ("chú em"
    là gọi người nhỏ tuổi hơn, "chị dâu" là vợ của anh) — không dùng làm từ vai vế.
    """
    words = _nfc(value).split()
    if not words or not all(_capitalized(word) for word in words[1:]):
        return ""
    return words[0].lower()


def _single_call(rule: str) -> str:
    """Từ vai vế của cách GỌI trong quy tắc; nhiều cách gọi khác nhau → rỗng."""
    values = _CALL_RE.findall(rule or "")
    calls = {_call_prefix(value) for value in values if value.strip()}
    return next(iter(calls)) if len(calls) == 1 else ""


def _entry_age(entry: RosterEntry | None) -> int | None:
    match = _AGE_RE.search(entry.age if entry is not None else "")
    return int(match.group()) if match else None


@dataclass(frozen=True, slots=True)
class NativeRoster:
    """Bảng nhân vật đã đọc sẵn để chọn từ vai vế (dựng bằng :func:`build_roster`)."""

    entries: tuple[RosterEntry, ...]
    rules: Mapping[tuple[str, str], str]
    syllables: frozenset[str]
    titles: Mapping[str, str]
    title_cores: Mapping[str, frozenset[str]]

    @classmethod
    def build(cls, characters: str) -> NativeRoster:
        """Đọc bảng nhân vật: quy tắc xưng hô, âm tiết tên, tiếng gọi → nhân vật."""
        entries = tuple(parse_roster(characters or ""))
        rules = address_rule_map(characters or "")
        titles: dict[str, str] = {}
        title_cores: dict[str, set[str]] = {}
        for entry in entries:
            for alias in name_aliases(entry.aka):
                titles.setdefault(_key(alias), entry.name.strip())
        for (_speaker, listener), rule in rules.items():
            for match in _TITLE_CLAUSE_RE.finditer(rule):
                titles.setdefault(_key(match.group("title")), listener)
                core = _name_words(_nfc(match.group("title")).split())
                if core:
                    title_cores.setdefault(_key(" ".join(core)), set()).add(listener)
        return cls(
            entries, rules, name_syllables(characters), titles,
            {key: frozenset(names) for key, names in title_cores.items()},
        )

    def entry_of(self, name: str) -> RosterEntry | None:
        """Mục bảng nhân vật có chung một bí danh với ``name``."""
        aliases = set(name_aliases(name))
        return next(
            (entry for entry in self.entries if aliases & set(name_aliases(entry.name))),
            None,
        )

    def gender_of(self, name: str) -> str:
        """Giới tính trong bảng (:data:`GENDER_MALE`/:data:`GENDER_FEMALE`), rỗng nếu
        không rõ. [v3.23.614] Đọc theo nghĩa ("Nam giới", "không rõ")."""
        entry = self.entry_of(name)
        return _roster_gender(entry.gender) if entry is not None else ""

    def referent(self, core: str, phrase: str, suffix: CalqueSuffix) -> str:
        """Nhân vật được gọi bằng tiếng gọi; không xác định được duy nhất → rỗng."""
        titled = self.titles.get(_key(phrase))
        if titled:
            return titled
        words = [word.lower() for word in _nfc(core).split()]
        owners: list[str] = []
        for entry in self.entries:
            if not self._gender_fits(entry.gender, suffix):
                continue
            for alias in name_aliases(entry.aka):
                alias_words = [word.lower() for word in _nfc(alias).split()]
                if any(
                    alias_words[i:i + len(words)] == words
                    for i in range(len(alias_words) - len(words) + 1)
                ):
                    owners.append(entry.name.strip())
                    break
        if len(owners) == 1:
            return owners[0]
        if owners:
            return ""
        # Tên chỉ có trong danh xưng của một quy tắc ("danh xưng: Chân đại tẩu" — vợ
        # Kiếm Thanh mang họ chồng, phiên âm của cô không có chữ "Chân").
        titled_owners = {
            listener for listener in self.title_cores.get(_key(core), frozenset())
            if self._gender_fits(self.gender_of(listener), suffix)
        }
        return next(iter(titled_owners)) if len(titled_owners) == 1 else ""

    def confirms_suffix(self, phrase: str, suffix: CalqueSuffix) -> bool:
        """Bảng ghi ``phrase`` là tên gọi của người có tên chữ Hán tận cùng là hậu tố."""
        owner = self.titles.get(_key(phrase))
        entry = self.entry_of(owner) if owner else None
        return entry is not None and _cjk_confirms(name_aliases(entry.name), suffix)

    @staticmethod
    def _gender_fits(gender: str, suffix: CalqueSuffix) -> bool:
        gender = _roster_gender(gender)
        return not suffix.gender or not gender or gender == suffix.gender

    def default_prefix(self, suffix: CalqueSuffix, referent: str) -> str:
        """Từ vai vế mặc định; "先生" gọi người dưới :data:`SENIOR_AGE` tuổi là "anh"."""
        if suffix is _SIR:
            age = _entry_age(self.entry_of(referent)) if referent else None
            if age is not None and age < SENIOR_AGE:
                return "anh"
        return suffix.prefix

    def prefix_for(
        self, suffix: CalqueSuffix, referent: str, call: str = ""
    ) -> str:
        """Từ vai vế: ``call`` (cách gọi theo bảng) nếu hợp hậu tố, không thì mặc định."""
        gender = suffix.gender or (self.gender_of(referent) if referent else "")
        if call and _prefix_fits(call, suffix, gender):
            return call
        return self.default_prefix(suffix, referent)

    def call_of(self, speaker: str, referent: str, phrase: str = "") -> str:
        """Cách ``speaker`` gọi ``referent`` theo bảng, hoặc rỗng.

        Quy tắc ghi danh xưng đúng bằng ``phrase`` ("gọi 'ông' (danh xưng: Chân tiên
        sinh)") thì lấy cách gọi đi kèm danh xưng đó — quy tắc có điều kiện giai đoạn
        ("trước khi thua … gọi 'mày', sau khi thua … gọi 'ông'") vẫn chọn được. Ngoài ra
        chỉ nhận quy tắc có MỘT cách gọi.
        """
        if not speaker or not referent:
            return ""
        rule = lookup_address_rule(self.rules, speaker, referent)
        wanted = _key(phrase)
        # Danh xưng có thể đã được đổi sang thuần Việt ("danh xưng: ông Chân") trong khi
        # dòng dịch còn "Chân tiên sinh" — so cả phần tên.
        wanted_core = _key(" ".join(_name_words(_nfc(phrase).split())))
        for match in _TITLE_CLAUSE_RE.finditer(rule):
            title = match.group("title")
            title_core = _key(" ".join(_name_words(_nfc(title).split())))
            if wanted and (
                _key(title) == wanted or (wanted_core and title_core == wanted_core)
            ):
                return _call_prefix(match.group("call"))
        return _single_call(rule)


def build_roster(characters: str) -> NativeRoster:
    """Đọc bảng nhân vật một lần cho nhiều dòng (dùng với :func:`nativize_line`)."""
    return NativeRoster.build(characters)


def _vocative(text: str, calque: _Calque) -> bool:
    """Tiếng gọi đứng riêng: đầu/cuối dòng hoặc tách khỏi câu bằng dấu câu."""
    head = text[:calque.start].strip()
    tail = text[calque.end:].strip()
    opens = not head or head[-1] in _VOCATIVE_BEFORE
    closes = not tail or tail[0] in _VOCATIVE_AFTER
    return opens and closes


def _replace_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(spans, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _after_name(text: str, position: int) -> bool:
    """Ngay trước ``position`` (chỉ cách khoảng trắng) là một từ viết hoa.

    Có thể là tên riêng ("Mạc đại tẩu") nên danh xưng không được đổi thành "Mạc chị
    dâu". Từ viết hoa vì đứng đầu câu ("Chào đại tẩu") cũng bị bỏ qua — thà sót một cụm
    còn hơn ghép sai vào tên.
    """
    head = text[:position]
    words = list(_WORD_RE.finditer(head))
    if not words or head[words[-1].end():].strip():
        return False
    return _capitalized(words[-1].group())


def _title_nouns(text: str, source: str) -> tuple[str, list[tuple[str, str]]]:
    """Đổi danh xưng đứng một mình ("đại tẩu" → "chị dâu") khi câu gốc có chữ Hán đó."""
    changes: list[tuple[str, str]] = []
    for words, cjk, native in TITLE_NOUNS:
        if not any(char in source for char in cjk):
            continue
        spans: list[tuple[int, int, str]] = []
        for match in _phrase_pattern(words).finditer(text):
            if _after_name(text, match.start()):
                continue
            replacement = (
                native[:1].upper() + native[1:] if _capitalized(match.group()) else native
            )
            spans.append((match.start(), match.end(), replacement))
            changes.append((match.group(), replacement))
        text = _replace_spans(text, spans)
    return text, changes


def nativize_line(
    text: str,
    source: str,
    *,
    roster: NativeRoster,
    speaker: str = "",
    listener: str = "",
) -> tuple[str, list[tuple[str, str]]]:
    """Đổi tiếng gọi Hán-Việt trong MỘT dòng dịch (hàm thuần).

    Args:
        text: Bản dịch của dòng.
        source: Câu gốc (phải có chữ Hán của hậu tố thì mới đổi).
        roster: Bảng nhân vật đã đọc (:func:`build_roster`).
        speaker: Người nói theo Visual Cues.
        listener: Người nghe theo Visual Cues.

    Returns:
        (bản dịch mới, danh sách (cụm cũ, cụm mới)).
    """
    text = _nfc(text)
    source = source or ""
    spans: list[tuple[int, int, str]] = []
    changes: list[tuple[str, str]] = []
    for calque in _find_calques(text, roster.syllables):
        suffix = calque.suffix
        if not any(char in source for char in suffix.cjk):
            continue
        # [v3.23.613] Hậu tố viết hoa có thể là âm tiết của chính tên ("Lâm Di" = 林怡,
        # câu gốc có 姨 ở chỗ khác): chỉ đổi khi bảng nhân vật xác nhận.
        if _capitalized(calque.suffix_text) and not roster.confirms_suffix(
            calque.text, suffix
        ):
            continue
        referent = roster.referent(calque.core, calque.text, suffix)
        # Không tra được người được gọi mà tiếng gọi đứng riêng → gọi chính người nghe
        # (nếu giới tính không trái hậu tố). Tra được thì tin tên hơn nhãn người nghe:
        # nhãn Visual Cues có thể sai người.
        if not referent and _vocative(text, calque) and listener and (
            not suffix.gender or roster.gender_of(listener) in ("", suffix.gender)
        ):
            referent = listener
        prefix = roster.prefix_for(
            suffix, referent, roster.call_of(speaker, referent, calque.text)
        )
        replacement = _render(
            prefix, calque.core, capitalize=_sentence_start(text, calque.start)
        )
        spans.append((calque.start, calque.end, replacement))
        changes.append((calque.text, replacement))
    text = _replace_spans(text, spans)
    text, title_changes = _title_nouns(text, source)
    return text, changes + title_changes


def nativize_lines(
    translations: Mapping[int, str],
    sources: Mapping[int, str],
    characters: str,
    *,
    speakers: Mapping[int, str] | None = None,
    listeners: Mapping[int, str] | None = None,
) -> tuple[dict[int, str], list[NativeAddressChange]]:
    """Đổi tiếng gọi Hán-Việt trong các dòng dịch (hàm thuần).

    Args:
        translations: ``chỉ số → bản dịch``.
        sources: ``chỉ số → câu gốc``.
        characters: Bảng nhân vật (quy tắc xưng hô, âm tiết tên).
        speakers: ``chỉ số → người nói`` (Visual Cues).
        listeners: ``chỉ số → người nghe`` (Visual Cues).

    Returns:
        (chỉ các dòng đã đổi, danh sách thay đổi).
    """
    roster = build_roster(characters)
    speakers = speakers or {}
    listeners = listeners or {}
    changed: dict[int, str] = {}
    changes: list[NativeAddressChange] = []
    for index, text in translations.items():
        updated, line_changes = nativize_line(
            text or "", sources.get(index, ""), roster=roster,
            speaker=speakers.get(index, ""), listener=listeners.get(index, ""),
        )
        if line_changes and updated != _nfc(text or ""):
            changed[index] = updated
            changes.extend(
                NativeAddressChange(LOCATION_LINE, before, after, index)
                for before, after in line_changes
            )
    return changed, changes


def _known_calques(characters: str, glossary: str) -> dict[str, tuple[str, CalqueSuffix]]:
    """Tiếng gọi Hán-Việt mà bảng nhân vật/thuật ngữ ghi là tên gọi của một người.

    Returns:
        ``cụm → (phần tên, hậu tố)``.
    """
    # (cụm, chữ Hán của người được gọi — rỗng nếu không có để đối chiếu)
    phrases: list[tuple[str, tuple[str, ...]]] = []
    for entry in parse_roster(characters or ""):
        names = tuple(name_aliases(entry.name))
        phrases.extend((alias, names) for alias in name_aliases(entry.aka))
        phrases.extend(
            (m.group("title"), ()) for m in _TITLE_CLAUSE_RE.finditer(entry.role)
        )
        phrases.extend((value, ()) for value in _CALL_RE.findall(entry.role))
    for term in parse_glossary(glossary or ""):
        phrases.append((term.translation.split("(")[0].strip(), (term.term.strip(),)))
    syllables = name_syllables(characters)
    known: dict[str, tuple[str, CalqueSuffix]] = {}
    for phrase, cjk_names in phrases:
        words = _nfc(phrase).split()
        # Hậu tố viết hoa ("Vương Nhị Thẩm", "Lý Thẩm") có thể là chính tên riêng: chỉ
        # nhận khi tên chữ Hán kết thúc bằng đúng hậu tố đó (王二婶).
        split = _confirmed_split(words, cjk_names)
        if split is None:
            continue
        core, suffix = split
        # Phần tên phải là tên một nhân vật: "北妹 = Bắc muội" (gái miền Bắc) không phải.
        if not all(word.lower() in syllables for word in core):
            continue
        known[" ".join(words)] = (" ".join(core), suffix)
    return known


def _replace_known(
    text: str,
    known: Mapping[str, tuple[str, CalqueSuffix]],
    roster: NativeRoster,
    location: str,
    *,
    prose: bool,
    call: str = "",
) -> tuple[str, list[NativeAddressChange]]:
    """Đổi các tiếng gọi đã biết trong một văn bản ngữ cảnh.

    Args:
        text: Văn bản.
        known: Kết quả :func:`_known_calques`.
        roster: Bảng nhân vật đã đọc.
        location: Nơi đổi (ghi vào thay đổi).
        prose: Văn xuôi (tóm tắt) — viết hoa từ vai vế ở đầu câu.
        call: Cách gọi của mệnh đề xưng hô chứa văn bản này (rỗng = dùng mặc định).
    """
    changes: list[NativeAddressChange] = []
    text = _nfc(text)
    for phrase in sorted(known, key=len, reverse=True):
        core, suffix = known[phrase]
        referent = roster.referent(core, phrase, suffix)
        prefix = roster.prefix_for(suffix, referent, call)
        spans: list[tuple[int, int, str]] = []
        for match in _phrase_pattern(phrase).finditer(text):
            replacement = _render(
                prefix, core, capitalize=prose and _sentence_start(text, match.start())
            )
            spans.append((match.start(), match.end(), replacement))
            changes.append(NativeAddressChange(location, match.group(), replacement))
        text = _replace_spans(text, spans)
    return text, changes


def _nativize_roster(
    characters: str, known: Mapping[str, tuple[str, CalqueSuffix]], roster: NativeRoster
) -> tuple[str, list[NativeAddressChange]]:
    """Bảng nhân vật: danh xưng của mỗi mệnh đề theo đúng cách gọi của mệnh đề đó."""
    changes: list[NativeAddressChange] = []
    lines: list[str] = []
    for line in _nfc(characters).split("\n"):

        def _title(match: re.Match[str]) -> str:
            title, title_changes = _replace_known(
                match.group("title"), known, roster, LOCATION_ROSTER,
                prose=False, call=_call_prefix(match.group("call")),
            )
            changes.extend(title_changes)
            return f"gọi '{match.group('call')}'{match.group('gap')}(danh xưng: {title})"

        line = _TITLE_CLAUSE_RE.sub(_title, line)
        line, line_changes = _replace_known(
            line, known, roster, LOCATION_ROSTER, prose=False
        )
        changes.extend(line_changes)
        lines.append(line)
    return "\n".join(lines), changes


def nativize_context(context: ContextT) -> tuple[ContextT, list[NativeAddressChange]]:
    """Đổi tiếng gọi Hán-Việt trong bảng nhân vật, tóm tắt, thuật ngữ (hàm thuần).

    Args:
        context: ``SubtitleContextAnalysis`` hoặc ``TranslationContext``.

    Returns:
        (ngữ cảnh mới, các thay đổi); không có gì để đổi thì trả ĐÚNG đối tượng cũ.
    """
    characters = context.characters or ""
    glossary = context.glossary or ""
    known = _known_calques(characters, glossary)
    if not known:
        return context, []
    roster = build_roster(characters)
    new_characters, roster_changes = _nativize_roster(characters, known, roster)
    new_overview, overview_changes = _replace_known(
        context.overview or "", known, roster, LOCATION_OVERVIEW, prose=True
    )
    new_glossary, glossary_changes = _replace_known(
        glossary, known, roster, LOCATION_GLOSSARY, prose=False
    )
    changes = roster_changes + overview_changes + glossary_changes
    if not changes:
        return context, []
    return (
        replace(  # type: ignore[type-var]
            context,
            characters=new_characters,
            overview=new_overview,
            glossary=new_glossary,
        ),
        changes,
    )


def describe_changes(changes: Iterable[NativeAddressChange], limit: int = 6) -> str:
    """Tóm tắt thay đổi cho nhật ký/thông báo: ``"Kiếm ca → anh Kiếm (26), …"``."""
    counts = Counter((change.before, change.after) for change in changes)
    parts = [
        f"{before} → {after}" + (f" ({count})" if count > 1 else "")
        for (before, after), count in counts.most_common(limit)
    ]
    more = len(counts) - limit
    return ", ".join(parts) + (f" và {more} cách gọi khác" if more > 0 else "")


__all__ = [
    "GENDER_FEMALE",
    "GENDER_MALE",
    "LOCATION_GLOSSARY",
    "LOCATION_LINE",
    "LOCATION_OVERVIEW",
    "LOCATION_ROSTER",
    "NAME_CALQUES",
    "SENIOR_AGE",
    "TITLE_NOUNS",
    "CalqueSuffix",
    "NativeAddressChange",
    "NativeRoster",
    "build_roster",
    "describe_changes",
    "name_syllables",
    "nativize_context",
    "nativize_line",
    "nativize_lines",
]
