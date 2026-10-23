"""[v3.23.570] Dòng GỌI TÊN: câu gốc chỉ là một cái tên thì bản dịch phải là CHÍNH tên đó.

Chuẩn đoán thật v3.23.569: dòng 2532 gốc chỉ có ``柳慕云`` (Thẩm Minh Nguyệt gọi Liễu
Mộ Vân) lại dịch thành "Thẩm Minh Nguyệt,"; dòng 3075 gốc ``徐俊豪`` (Tống Tiền gọi Từ
Tuấn Hào) dịch thành "Tống Tiền,". Lỗi lặp ở gần như MỌI lượt dịch phim này (2532: 6
lượt, 3075: 5 lượt) và ở phim khác (``林逸`` → "Tần Nguyệt", ``秦月`` → "Lâm Dật"): trên
100 bản dịch thật có 28 dòng gọi tên bị tráo sang tên người khác.

Nguyên nhân: Visual Cues gán NGƯỢC người nói/người nghe (``spk=柳慕云, to=沈明月``). Model
tin nhãn hơn câu gốc nên "sửa" tiếng gọi thành tên người nghe trong nhãn. Nhưng câu gốc
là sự thật: cả dòng chỉ là tên X thì người đang được gọi LÀ X. Hai việc tất định:

1. Sửa nhãn: dòng gọi tên X mà nhãn ghi X là NGƯỜI NÓI (và người nghe là người khác) thì
   đảo lại — trước khi khâu dịch và lưới xưng hô dùng tới nhãn đó.
2. Sửa bản dịch: dòng gọi tên X mà bản dịch không có tên X nhưng có tên một nhân vật
   KHÁC, và ngoài tên đó chỉ còn TIỂU TỪ gọi ("à", "ơi"…) → thay bằng tên X.

[v3.23.571] Tiếng gọi kèm VAI VẾ (``宋钱弟弟``, ``宋钱哥``, ``慕云姐``): hậu tố cho biết
người nói coi người được gọi là em hay anh/chị — câu gốc quyết định, không phải nhãn cues.
Ca thật (5 lượt dịch, 15 dòng): ``宋钱弟弟`` gắn nhãn người nói ``张依贝`` — người mà bảng
nhân vật ghi "gọi 'anh Tống Tiền'" — nên ra "Anh Tống Tiền, anh đi trước đi!" (12/15 dòng
sai vai), và ra "Chị Đào Hoa ơi," khi nhãn ghi ngược ``spk=宋钱``. Chỉ sửa TỪ "Anh" thành
"Em" sẽ ra câu lai "Em Tống Tiền, anh đi trước đi!" vì các dòng nói tiếp đã dịch theo nhãn
sai. Nên sửa ở NHÃN, trước khi dịch: người nói trái vai với hậu tố → đổi sang người nói ở
dòng kề mà quy tắc xưng hô với người được gọi KHỚP hậu tố, áp cho cả lượt lời cùng nhãn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from subtitles_extractor.application.services.roster_format import (
    RosterEntry,
    address_rule_map,
    lookup_address_rule,
    parse_roster,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue

#: Ngoài cái tên, bản dịch dòng gọi tên chỉ được có thêm các TIỂU TỪ gọi này. Có từ
#: khác (nhất là danh xưng: "Chị dâu Xuân Hoa") là câu của dòng khác bị lệch sang — ca
#: thật dòng 175 — thay tên vào sẽ ra "Chị dâu Tống Tiền", nên không đụng.
VOCATIVE_PARTICLES = frozenset({
    "à", "ơi", "này", "ạ", "hả", "hở", "hử", "nhé", "nè", "ê", "ấy", "kìa", "đấy",
    "a", "ô", "ồ", "ha", "hừ", "chứ", "sao", "thế", "ư",
})

_ALIAS_SPLIT_RE = re.compile(r"\s*[/|]\s*")
_CJK_RE = re.compile(r"^[㐀-䶿一-鿿]{2,}$")
#: Dấu câu và thán từ đứng ĐẦU/CUỐI tiếng gọi ("哎呀许桃花", "宋钱啊!").
_EDGE_FILLER = (
    "[\\s,.!?:;\\-\"'()"
    "\uff0c\u3001\u3002\uff01\uff1f\u2026\uff5e~\uff1a\uff1b\u2014"
    "\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f\uff08\uff09"
    "啊呀哎哟喂嘛呢吧哦噢诶欸]"
)
_LEADING_FILLER_RE = re.compile(rf"^{_EDGE_FILLER}+")

_TRAILING_FILLER_RE = re.compile(rf"{_EDGE_FILLER}+$")

#: [v3.23.571] Hậu tố vai vế → (từ vai vế thuần Việt, từ vai vế Hán-Việt) đúng.
KINSHIP_SUFFIXES: dict[str, tuple[str, str]] = {
    "哥哥": ("anh", "ca"), "哥": ("anh", "ca"),
    "姐姐": ("chị", "tỷ"), "姐": ("chị", "tỷ"),
    "弟弟": ("em", "đệ"), "弟": ("em", "đệ"),
    "妹妹": ("em", "muội"), "妹": ("em", "muội"),
}
#: Từ vai vế trong bản dịch → vai (anh/chị/em) nó biểu thị.
_KIN_WORD_ROLE = {
    "anh": "anh", "ca": "anh", "chị": "chị", "tỷ": "chị",
    "em": "em", "đệ": "em", "muội": "em",
}
_HAN_VIET_KIN_WORDS = frozenset({"ca", "tỷ", "đệ", "muội"})

#: [v3.23.571] Người nói coi người được gọi là bậc TRÊN (anh/chị) hay bậc DƯỚI (em).
OLDER, YOUNGER = "older", "younger"
_ROLE_RANK = {"anh": OLDER, "chị": OLDER, "em": YOUNGER}
_ADDRESS_TERM_RE = re.compile(r"gọi\s+'([^']*)'", re.IGNORECASE)
_SELF_TERM_RE = re.compile(r"xưng\s+'([^']*)'", re.IGNORECASE)
#: Tự xưng anh/chị là coi người nghe bậc dưới; tự xưng em là coi người nghe bậc trên.
_SELF_WORD_RANK = {"anh": YOUNGER, "chị": YOUNGER, "em": OLDER}
#: Lượt lời được đổi nhãn tối đa bao nhiêu dòng (tính cả dòng gọi tên) — ca thật: 2-5.
KIN_RELABEL_MAX_LINES = 5
#: Tìm người nói thay thế trong bao nhiêu dòng trước dòng gọi tên / sau lượt lời.
KIN_SPEAKER_SEARCH_LINES = 3

#: [v3.23.603] Câu HỎI TÊN ("你叫甚麼名字", "如何称呼", "貴姓") ở dòng ngay trước: dòng
#: chỉ-là-tên theo sau là câu TRẢ LỜI của người được hỏi, không phải tiếng gọi. "你叫我
#: 什么" (gọi tôi là gì) KHÔNG phải câu hỏi tên — chữ 我 chen giữa nên không khớp.
NAME_QUESTION_RE = re.compile(
    "叫(?:甚麼|什麼|什么|啥|咩|乜)"
    "|(?:甚麼|什麼|什么|啥|咩|乜)名"
    "|名字(?:是|叫)?(?:甚麼|什麼|什么|啥)"
    "|[貴贵尊]姓"
    "|[稱称]呼"
    "|[你您妳](?:是|係)(?:誰|谁|邊個)"
)
#: [v3.23.603] Dòng ngay trước là nửa đầu lời TỰ GIỚI THIỆU bị tách dòng ("我是" /
#: "柳慕云!") — dòng tên theo sau vẫn là lời của chính người đó.
SELF_INTRO_TAIL_RE = re.compile(
    "(?:我|俺|本人|在下|老子)(?:就)?(?:是|叫(?:做)?|係)"
    "[\\s,.!~\uff0c\u3002\uff01\u2026\uff5e]*$"
)


def _aliases(value: str) -> list[str]:
    return [part.strip() for part in _ALIAS_SPLIT_RE.split(value or "") if part.strip()]


@dataclass(frozen=True)
class VocativeNameFix:
    """Một dòng gọi tên được sửa lại đúng tên.

    Attributes:
        index: Chỉ số dòng.
        wrong_name: Tên nhân vật khác đã bị dịch vào.
        right_name: Tên đúng (người được gọi theo câu gốc).
        text_before: Bản dịch trước khi sửa.
        text_after: Bản dịch sau khi sửa.
    """

    index: int
    wrong_name: str
    right_name: str
    text_before: str
    text_after: str


@dataclass(frozen=True)
class _Callee:
    entry: RosterEntry
    alias: str
    kinship: tuple[str, str] | None = None


def _alias_index(entries: list[RosterEntry]) -> dict[str, RosterEntry]:
    """Bí danh chữ Hán (≥ 2 chữ) → nhân vật; bí danh trùng giữa hai người bị loại."""
    index: dict[str, RosterEntry] = {}
    ambiguous: set[str] = set()
    for entry in entries:
        for alias in _aliases(entry.name):
            if not _CJK_RE.match(alias):
                continue
            if alias in index and index[alias] is not entry:
                ambiguous.add(alias)
            index[alias] = entry
    for alias in ambiguous:
        index.pop(alias, None)
    return index


def vocative_alias_index(characters: str) -> dict[str, RosterEntry]:
    """[v3.23.577] Bảng tra tiếng gọi (bí danh chữ Hán → nhân vật) của bảng nhân vật."""
    return _alias_index(parse_roster(characters or ""))


def vocative_callee(source_text: str, index: dict[str, RosterEntry]) -> _Callee | None:
    """Nhân vật được gọi nếu câu gốc CHỈ là một tên, có thể kèm hậu tố vai vế.

    Bỏ dấu câu/thán từ ở hai đầu; tên trùng nguyên văn một bí danh được ưu tiên (bí danh
    như ``狗哥`` tự nó đã kết thúc bằng 哥).
    """
    core = _TRAILING_FILLER_RE.sub("", _LEADING_FILLER_RE.sub("", source_text or ""))
    entry = index.get(core)
    if entry is not None:
        return _Callee(entry, core)
    for suffix in sorted(KINSHIP_SUFFIXES, key=len, reverse=True):
        stem = core[: -len(suffix)]
        if core.endswith(suffix) and stem in index:
            return _Callee(index[stem], stem, KINSHIP_SUFFIXES[suffix])
    return None


def _label_is(label: str, entry: RosterEntry) -> bool:
    names = set(_aliases(entry.name))
    return bool(names & set(_aliases(label)))


def is_name_answer(sources: dict[int, str], line_no: int) -> bool:
    """[v3.23.603] Dòng chỉ-là-tên đang TRẢ LỜI tên, không phải tiếng gọi (hàm thuần).

    Đúng khi câu gốc dòng ngay trước là câu hỏi tên (:data:`NAME_QUESTION_RE`) hoặc nửa
    đầu lời tự giới thiệu (:data:`SELF_INTRO_TAIL_RE`).

    Args:
        sources: ``số dòng → câu gốc``.
        line_no: Dòng chỉ-là-tên.

    Returns:
        True nếu dòng là câu trả lời/tự giới thiệu.
    """
    previous = sources.get(line_no - 1, "")
    return bool(NAME_QUESTION_RE.search(previous) or SELF_INTRO_TAIL_RE.search(previous))


def fix_vocative_cues(
    cues: list[VisualCue], sources: dict[int, str], characters: str
) -> tuple[list[VisualCue], list[int]]:
    """Đảo nhãn người nói/nghe ở dòng gọi tên bị gán ngược (hàm thuần).

    [v3.23.603] Bỏ qua dòng TRẢ LỜI tên (:func:`is_name_answer`): ở đó người nói CHÍNH
    LÀ người mang tên, đảo nhãn là làm sai một nhãn đang đúng.

    Args:
        cues: Visual Cues.
        sources: ``số dòng → câu gốc``.
        characters: Bảng nhân vật.

    Returns:
        (cues đã sửa, các dòng đã đảo nhãn). Không sửa gì thì trả đúng danh sách cũ.
    """
    index = _alias_index(parse_roster(characters or ""))
    if not index:
        return cues, []
    fixed: list[VisualCue] = []
    swapped: list[int] = []
    for cue in cues:
        callee = vocative_callee(sources.get(cue.line_no, ""), index)
        if (
            callee is not None
            and cue.addressee
            and _label_is(cue.speaker, callee.entry)
            and not _label_is(cue.addressee, callee.entry)
            and not is_name_answer(sources, cue.line_no)
        ):
            fixed.append(replace(cue, speaker=cue.addressee, addressee=cue.speaker))
            swapped.append(cue.line_no)
            continue
        fixed.append(cue)
    return (fixed, swapped) if swapped else (cues, [])


def _answer_labels(
    cue: VisualCue, previous: VisualCue, previous_source: str, callee: RosterEntry
) -> tuple[str, str] | None:
    """``(người nói, người nghe)`` đúng của dòng trả lời tên; ``None`` nếu không suy ra
    được hoặc nhãn đã đúng."""
    if _label_is(cue.speaker, callee):
        return None
    if NAME_QUESTION_RE.search(previous_source):
        # Người hỏi hỏi CHÍNH người mang tên → người đáp là người đó, nói với người hỏi.
        asker, asked = previous.speaker, previous.addressee
        if not asker or _label_is(asker, callee) or not _label_is(asked, callee):
            return None
        return asked, asker
    if SELF_INTRO_TAIL_RE.search(previous_source) and _label_is(previous.speaker, callee):
        return previous.speaker, previous.addressee
    return None


def fix_name_answer_cues(
    cues: list[VisualCue], sources: dict[int, str], characters: str
) -> tuple[list[VisualCue], list[int]]:
    """[v3.23.603] Dòng TRẢ LỜI tên mà nhãn ghi người nói là người khác → sửa (hàm thuần).

    Ca thật 灭门惨案II dòng 126-127: Long Truyền Nhân hỏi "你叫甚麼名字", Tần Khả Tú đáp
    "秦可秀" — cả 22 lượt phân tích đã xuất của phim đều ghi dòng đáp là lời của NGƯỜI
    HỎI (lưới đảo nhãn gọi tên còn biến nhãn đúng thành sai), nên lồng tiếng đọc tên cô
    bằng giọng nam. Phim khác dòng 2443-2444 "我是" / "柳慕云!": 16 lượt ghi dòng tên là
    lời 徐俊豪 dù dòng "我是" ngay trước ghi đúng 柳慕云.

    Chỉ sửa khi suy ra chắc: câu hỏi tên do người KHÁC hỏi CHÍNH người mang tên (theo
    nhãn dòng hỏi), hoặc nửa đầu lời tự giới thiệu do chính người mang tên nói.

    Args:
        cues: Visual Cues.
        sources: ``số dòng → câu gốc``.
        characters: Bảng nhân vật.

    Returns:
        (cues đã sửa, các dòng đã sửa nhãn). Không sửa gì thì trả đúng danh sách cũ.
    """
    index = _alias_index(parse_roster(characters or ""))
    if not index:
        return cues, []
    by_line = {cue.line_no: cue for cue in cues}
    fixed: list[VisualCue] = []
    changed: list[int] = []
    for cue in cues:
        callee = vocative_callee(sources.get(cue.line_no, ""), index)
        previous = by_line.get(cue.line_no - 1)
        labels = (
            _answer_labels(
                cue, previous, sources.get(cue.line_no - 1, ""), callee.entry
            )
            if callee is not None and previous is not None
            else None
        )
        if labels is None:
            fixed.append(cue)
            continue
        fixed.append(replace(cue, speaker=labels[0], addressee=labels[1]))
        changed.append(cue.line_no)
    return (fixed, changed) if changed else (cues, [])


def _ranks_in(terms: list[str], ranks: dict[str, str]) -> set[str]:
    # Phân biệt hoa/thường: từ vai vế viết thường ("gọi 'em Tiền'"), còn chữ viết hoa là
    # một phần của TÊN ("gọi 'Tú Anh'" — 英 đọc Anh) và không nói gì về vai vế.
    return {
        ranks[word]
        for term in terms
        for word in re.findall(r"[^\W\d_]+", term)
        if word in ranks
    }


def address_rank(rule: str) -> str | None:
    """Quy tắc xưng hô coi người nghe là bậc trên (:data:`OLDER`) hay bậc dưới.

    Đọc từ vai vế trong cách GỌI ("gọi 'em Tiền'", "gọi 'anh Tống Tiền'"); cách gọi
    không có từ vai vế ("gọi 'chú Tiền'") thì suy từ cách TỰ XƯNG. Quy tắc nhiều giai
    đoạn trái vai nhau ("ban đầu gọi 'anh', sau gọi 'em Tiền'") là không rõ.

    Args:
        rule: Quy tắc xưng hô, vd ``"tự xưng 'em' gọi 'anh Tống Tiền'"``.

    Returns:
        :data:`OLDER`, :data:`YOUNGER`, hoặc ``None`` nếu không rõ.
    """
    ranks = _ranks_in(_ADDRESS_TERM_RE.findall(rule or ""), _KIN_WORD_ROLE)
    ranks = {_ROLE_RANK[role] for role in ranks}
    if not ranks:
        ranks = _ranks_in(_SELF_TERM_RE.findall(rule or ""), _SELF_WORD_RANK)
    return ranks.pop() if len(ranks) == 1 else None


def _speaker_rank(
    rules: dict[tuple[str, str], str], speaker: str, callee: RosterEntry
) -> str | None:
    return address_rank(lookup_address_rule(rules, speaker, callee.name))


def _utterance_run(cue: VisualCue, by_line: dict[int, VisualCue]) -> list[VisualCue]:
    """Dòng gọi tên và các dòng ngay sau có CÙNG cặp nhãn — cùng một lượt lời."""
    run = [cue]
    while len(run) < KIN_RELABEL_MAX_LINES:
        following = by_line.get(run[-1].line_no + 1)
        if following is None or (following.speaker, following.addressee) != (
            cue.speaker,
            cue.addressee,
        ):
            break
        run.append(following)
    return run


def _matching_neighbour_speaker(
    run: list[VisualCue],
    by_line: dict[int, VisualCue],
    rules: dict[tuple[str, str], str],
    callee: RosterEntry,
    wanted_rank: str,
) -> str | None:
    """Người nói ở dòng kề khớp vai với người được gọi — xét trước lượt lời, rồi sau."""
    first, last = run[0].line_no, run[-1].line_no
    nearby = [first - step for step in range(1, KIN_SPEAKER_SEARCH_LINES + 1)] + [
        last + step for step in range(1, KIN_SPEAKER_SEARCH_LINES + 1)
    ]
    for line_no in nearby:
        neighbour = by_line.get(line_no)
        if (
            neighbour is None
            or not neighbour.speaker
            or neighbour.speaker == run[0].speaker
            or _label_is(neighbour.speaker, callee)
        ):
            continue
        if _speaker_rank(rules, neighbour.speaker, callee) == wanted_rank:
            return neighbour.speaker
    return None


def fix_kin_vocative_speakers(
    cues: list[VisualCue], sources: dict[int, str], characters: str
) -> tuple[list[VisualCue], list[int]]:
    """[v3.23.571] Đổi người nói ở tiếng gọi kèm vai vế có nhãn trái vai (hàm thuần).

    Dòng ``宋钱弟弟`` (gọi "em") mà nhãn người nói là người gọi Tống Tiền là "anh" (hoặc
    chính Tống Tiền) thì nhãn sai: đổi sang người nói ở dòng kề mà quy tắc xưng hô với
    người được gọi khớp hậu tố. Nhãn người nói trái vai thì cả lượt lời (các dòng ngay sau
    cùng cặp nhãn) cũng đổi theo, để câu nói tiếp dịch cùng một vai; nhãn ghi CHÍNH người
    được gọi thì chỉ đổi dòng gọi tên — dòng sau có thể là lời đáp của người đó.

    Args:
        cues: Visual Cues.
        sources: ``số dòng → câu gốc``.
        characters: Bảng nhân vật.

    Returns:
        (cues đã sửa, các dòng GỌI TÊN đã đổi nhãn). Không sửa gì thì trả đúng danh
        sách cũ.
    """
    index = _alias_index(parse_roster(characters or ""))
    rules = address_rule_map(characters or "")
    if not index or not rules:
        return cues, []
    by_line = {cue.line_no: cue for cue in cues}
    relabeled: dict[int, VisualCue] = {}
    vocative_lines: list[int] = []
    for cue in cues:
        callee = vocative_callee(sources.get(cue.line_no, ""), index)
        if callee is None or callee.kinship is None:
            continue
        wanted = _ROLE_RANK[callee.kinship[0]]
        speaker_is_callee = _label_is(cue.speaker, callee.entry)
        if not speaker_is_callee:
            current = _speaker_rank(rules, cue.speaker, callee.entry)
            if current is None or current == wanted:
                continue
        run = [cue] if speaker_is_callee else _utterance_run(cue, by_line)
        speaker = _matching_neighbour_speaker(run, by_line, rules, callee.entry, wanted)
        if speaker is None:
            continue
        addressee = (
            cue.addressee if _label_is(cue.addressee, callee.entry) else callee.entry.name
        )
        for line_cue in run:
            relabeled[line_cue.line_no] = replace(
                line_cue, speaker=speaker, addressee=addressee
            )
        vocative_lines.append(cue.line_no)
    if not vocative_lines:
        return cues, []
    return [relabeled.get(cue.line_no, cue) for cue in cues], vocative_lines


@dataclass(frozen=True)
class VocativeCueFixes:
    """Kết quả sửa nhãn Visual Cues ở các dòng gọi tên.

    Attributes:
        cues: Cues sau khi sửa.
        kin_relabeled: Dòng gọi kèm vai vế đã đổi người nói
            (:func:`fix_kin_vocative_speakers`).
        swapped: Dòng gọi tên đã đảo người nói/nghe (:func:`fix_vocative_cues`).
        name_answers: [v3.23.603] Dòng trả lời tên đã sửa người nói
            (:func:`fix_name_answer_cues`).
    """

    cues: list[VisualCue]
    kin_relabeled: list[int]
    swapped: list[int]
    name_answers: list[int] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Có dòng nào được sửa nhãn."""
        return bool(self.kin_relabeled or self.swapped or self.name_answers)


def fix_vocative_cue_labels(
    cues: list[VisualCue], sources: dict[int, str], characters: str
) -> VocativeCueFixes:
    """Sửa mọi nhãn cues sai ở dòng gọi tên, đúng thứ tự (hàm thuần).

    Vai vế trước: dòng ``宋钱弟弟`` có nhãn ``spk=宋钱`` được đổi sang người nói KHỚP
    vai ở dòng kề; đảo nhãn chỉ là phương án cuối — người nghe trong nhãn chưa chắc là
    người gọi. [v3.23.603] Dòng TRẢ LỜI tên được sửa riêng (không đảo) — chạy lại trên
    nhãn đã sửa không đổi gì thêm, nên dùng được cho cả nhãn đã lưu.

    Args:
        cues: Visual Cues.
        sources: ``số dòng → câu gốc``.
        characters: Bảng nhân vật.

    Returns:
        :class:`VocativeCueFixes`.
    """
    after_kin, kin_relabeled = fix_kin_vocative_speakers(cues, sources, characters)
    after_answers, name_answers = fix_name_answer_cues(after_kin, sources, characters)
    fixed, swapped = fix_vocative_cues(after_answers, sources, characters)
    return VocativeCueFixes(
        cues=fixed,
        kin_relabeled=kin_relabeled,
        swapped=swapped,
        name_answers=name_answers,
    )


def _contains(text: str, phrase: str) -> bool:
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _own_forms(entry: RosterEntry) -> set[str]:
    """Mọi cách bản dịch có thể gọi đúng nhân vật: phiên âm đủ, tên riêng, âm cuối."""
    forms: set[str] = set()
    for aka in _aliases(entry.aka):
        forms.add(aka)
        syllables = aka.split()
        if len(syllables) >= 2:
            forms.add(" ".join(syllables[1:]))
            forms.add(syllables[-1])
    return forms


def _full_names(entry: RosterEntry) -> list[str]:
    """Phiên âm ĐỦ (≥ 2 âm tiết, viết hoa) — đủ chắc là tên người."""
    return [
        aka for aka in _aliases(entry.aka) if len(aka.split()) >= 2 and aka[:1].isupper()
    ]


def _name_for_alias(entry: RosterEntry, alias: str) -> str:
    """Phiên âm ứng với bí danh được gọi (theo vị trí khi hai danh sách cùng độ dài)."""
    names = _aliases(entry.name)
    akas = _aliases(entry.aka)
    if len(names) == len(akas) and alias in names:
        return akas[names.index(alias)]
    return akas[0] if akas else ""


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text)


def _other_names(entry: RosterEntry) -> list[str]:
    """Tên đủ và tên riêng ≥ 2 âm tiết ("Đào Hoa" của "Hứa Đào Hoa") của một nhân vật."""
    names = list(_full_names(entry))
    for full in _full_names(entry):
        syllables = full.split()
        if len(syllables) >= 3:
            names.append(" ".join(syllables[1:]))
    return names


def _match_case(word: str, like: str) -> str:
    return word[:1].upper() + word[1:] if like[:1].isupper() else word


def _fix_kinship_word(text: str, kinship: tuple[str, str]) -> str:
    """Sửa TỪ VAI VẾ đi cùng cái tên vừa thay ("Chị Tống Tiền" cho 宋钱弟弟 → "Em …").

    Chỉ dùng khi đã thay tên: cả dòng là của người khác nên từ vai vế cũng của người đó.
    Dòng đã đúng tên mà sai vai thì KHÔNG sửa ở đây — các dòng nói tiếp đã dịch theo cùng
    nhãn sai, sửa riêng một từ sẽ ra câu lai; chỗ sửa là nhãn (:func:`fix_kin_vocative_
    speakers`). Chỉ sửa khi có ĐÚNG MỘT từ vai vế; giữ lối Hán-Việt (ca/tỷ/đệ/muội) hay
    thuần Việt như bản dịch đang dùng.

    Args:
        text: Bản dịch dòng gọi tên.
        kinship: (từ thuần Việt, từ Hán-Việt) đúng theo hậu tố câu gốc.

    Returns:
        Bản dịch đã sửa, hoặc nguyên văn nếu không cần/không chắc.
    """
    found = [
        match
        for match in re.finditer(r"[^\W\d_]+", text)
        if match.group(0).lower() in _KIN_WORD_ROLE
    ]
    if len(found) != 1:
        return text
    word = found[0].group(0)
    if _KIN_WORD_ROLE[word.lower()] == kinship[0]:
        return text
    target = kinship[1] if word.lower() in _HAN_VIET_KIN_WORDS else kinship[0]
    start, end = found[0].span()
    return text[:start] + _match_case(target, word) + text[end:]


def _neighbour_mentions(
    sources: dict[int, str], line_index: int, other: RosterEntry
) -> bool:
    """Câu gốc dòng kề có tên nhân vật kia → bản dịch nhiều khả năng bị LỆCH dòng sang."""
    aliases = [a for a in _aliases(other.name) if len(a) >= 2]
    return any(
        alias in sources.get(neighbour, "")
        for neighbour in (line_index - 1, line_index + 1)
        for alias in aliases
    )


def find_vocative_name_fixes(
    sources: dict[int, str], translations: dict[int, str], characters: str
) -> list[VocativeNameFix]:
    """Dòng gọi tên X bị dịch thành tên nhân vật khác → sửa về tên X (hàm thuần).

    Args:
        sources: ``index → câu gốc``.
        translations: ``index → bản dịch``.
        characters: Bảng nhân vật.

    Returns:
        Danh sách sửa, sắp theo chỉ số dòng.
    """
    entries = parse_roster(characters or "")
    index = _alias_index(entries)
    if not index:
        return []
    fixes: list[VocativeNameFix] = []
    for line_index, source in sorted(sources.items()):
        callee = vocative_callee(source, index)
        text = translations.get(line_index, "")
        if callee is None or not text.strip():
            continue
        if any(_contains(text, form) for form in _own_forms(callee.entry)):
            continue
        fix = _name_fix(line_index, text, callee, entries, sources)
        if fix is not None:
            fixes.append(fix)
    return fixes


def _name_fix(
    line_index: int,
    text: str,
    callee: _Callee,
    entries: list[RosterEntry],
    sources: dict[int, str],
) -> VocativeNameFix | None:
    """Sửa tên nhân vật khác bị dịch vào dòng gọi tên (kèm từ vai vế nếu có)."""
    intruders = {
        name: other
        for other in entries
        if other is not callee.entry
        for name in (_other_names(other) if callee.kinship else _full_names(other))
        if _contains(text, name)
    }
    if len({id(other) for other in intruders.values()}) != 1:
        return None
    wrong = max(intruders, key=len)
    # Tiếng gọi kèm vai vế được phép có thêm từ vai vế nên dễ nhầm với câu của dòng kề
    # bị lệch sang hơn — thêm chốt: tên lạ không được có trong câu gốc dòng kề.
    if callee.kinship and _neighbour_mentions(sources, line_index, intruders[wrong]):
        return None
    right = _name_for_alias(callee.entry, callee.alias)
    pattern = rf"(?<!\w){re.escape(wrong)}(?!\w)"
    rest = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    allowed = VOCATIVE_PARTICLES | (set(_KIN_WORD_ROLE) if callee.kinship else set())
    if not right or not {word.lower() for word in _words(rest)} <= allowed:
        return None
    fixed = re.sub(pattern, right, text, count=1, flags=re.IGNORECASE)
    if callee.kinship:
        fixed = _fix_kinship_word(fixed, callee.kinship)
    return VocativeNameFix(line_index, wrong, right, text, fixed)


__all__ = [
    "KINSHIP_SUFFIXES",
    "KIN_RELABEL_MAX_LINES",
    "KIN_SPEAKER_SEARCH_LINES",
    "NAME_QUESTION_RE",
    "OLDER",
    "SELF_INTRO_TAIL_RE",
    "VOCATIVE_PARTICLES",
    "YOUNGER",
    "VocativeCueFixes",
    "VocativeNameFix",
    "address_rank",
    "find_vocative_name_fixes",
    "fix_kin_vocative_speakers",
    "fix_name_answer_cues",
    "fix_vocative_cue_labels",
    "fix_vocative_cues",
    "is_name_answer",
    "vocative_alias_index",
    "vocative_callee",
]
