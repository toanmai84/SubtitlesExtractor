"""[v3.23.599] Soát người nói từng dòng LỒNG TIẾNG: dòng chưa có người nói, dòng chắc sai.

Người nói lấy từ nhãn Visual Cues của khâu phân tích — model xem video rồi đoán, không gì
bảo đảm đúng: 18 lượt phân tích thật phim "灭门惨案II" khác nhau ở 23,7% số dòng.
Không thể biết chắc người nói của MỌI dòng nếu không xem phim, nhưng có những dòng mà
chính câu gốc cho thấy nhãn SAI. Mô-đun (hàm thuần) chỉ ra các dòng đó để người dùng soát
trước khi tổng hợp, thay vì nghe hết cả phim:

* :data:`REVIEW_UNKNOWN` — dòng không có người nói (đọc bằng giọng chung). Lượt dịch mới
  nhất của hai phim thật còn 103/3902 và 125/5540 dòng như vậy.
* :data:`REVIEW_SELF_CALL` — câu gốc mở đầu bằng tiếng GỌI tên (hoặc tên thân mật) của
  chính người được gán là người nói rồi nói tiếp: "阿秀, 你幫我" mà nhãn là 秦可秀 (A Tú).
  Trên 58 lượt dịch thật của 3 phim có 85 dòng như vậy, cả 85 đều khác nhãn đa số của
  các lượt.
* :data:`REVIEW_VOICE_ONLY` — [v3.23.600] model chỉ nghe được giọng ("Giọng nam chưa
  rõ"…), chưa biết là nhân vật nào.
* :data:`REVIEW_GENDER` — câu gốc tự xưng theo giới ("你们让我嫁给他" — người nói là nữ)
  trái giới tính của vai được gán (1 dòng trên 58 lượt, khác nhãn đa số).

Mô-đun KHÔNG tự đổi người nói: tiếng gọi cho biết ai KHÔNG nói chứ không cho biết ai nói —
đổi sang người nghe của nhãn chỉ đúng 17/32 ca đo được.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subtitles_extractor.application.services.roster_format import (
    RosterEntry,
    parse_roster,
)
from subtitles_extractor.domain.value_objects.speaker_label import VOICE_ONLY_SPEAKERS
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
)

REVIEW_UNKNOWN = "unknown"
REVIEW_SELF_CALL = "self_call"
REVIEW_GENDER = "gender"
#: [v3.23.600] Chỉ biết giọng (nhãn "Giọng nam chưa rõ"…), chưa biết là nhân vật nào.
REVIEW_VOICE_ONLY = "voice_only"

_ALIAS_SPLIT_RE = re.compile(r"\s*[/|]\s*")
_CJK_NAME_RE = re.compile(r"^[㐀-䶿一-鿿]{2,4}$")
#: Ngoặc kép / ngoặc mở đứng trước tiếng gọi.
_LEADING_RE = re.compile("^[\\s\"'“”‘’「『（(]+")  # noqa: RUF001 — ngoặc chữ Hán cố ý
#: Dấu câu sau tiếng gọi (chữ Hán và Latinh).
_AFTER_PUNCTUATION = frozenset(",.!?:;~ 、。…，！？．～：；")  # noqa: RUF001 — cố ý
#: Sau tiếng gọi còn có thể là đại từ ngôi 1-2: "阿秀你幫我", "阿震我嫁了你幾十年".
_AFTER_CALL = _AFTER_PUNCTUATION | frozenset("你妳您我")
#: Đuôi đi liền tiếng gọi: vai vế và tiểu từ gọi ("劍哥", "明月姐", "阿秀啊").
_CALL_SUFFIXES = ("哥哥", "姐姐", "哥", "姐", "啊", "呀", "哇")
#: Tên thân mật từ chữ cuối tên ("阿秀", "小蓮"); tên hai chữ bắt đầu bằng các chữ này đã
#: là tên thân mật sẵn.
_NICK_PREFIXES = ("阿", "小")
#: Tên thân mật tự dựng trùng từ thường gặp đầu câu ("小心" = coi chừng) — không dùng.
_COMMON_WORDS = frozenset({"小心", "小姐", "小时", "小時", "阿姨"})
#: Câu gốc tự xưng theo giới (đã đo: không khớp nhầm câu nào trên 58 lượt dịch thật).
_FEMALE_SELF_WORDS = ("我老公", "我丈夫", "我嫁")
_MALE_SELF_WORDS = ("我老婆", "我太太", "我妻子", "我媳婦", "我媳妇", "我娶")


@dataclass(frozen=True, slots=True)
class SpeakerIssue:
    """Một dòng cần soát người nói.

    Attributes:
        reason: ``REVIEW_*``.
        detail: Tên chuẩn nhân vật được gọi (:data:`REVIEW_SELF_CALL`) hoặc giới tính câu
            gốc gợi ra (:data:`REVIEW_GENDER`); rỗng với :data:`REVIEW_UNKNOWN`.
        word: Chữ trong câu gốc làm bằng chứng (tiếng gọi, từ tự xưng).
    """

    reason: str
    detail: str = ""
    word: str = ""


def nickname_index(characters: str) -> dict[str, str]:
    """Tiếng gọi chữ Hán → tên chuẩn nhân vật (bí danh + tên thân mật tự dựng).

    Tên ba chữ 秦可秀 cho: 秦可秀, 可秀, 阿秀, 小秀, 秀秀; tên hai chữ 洪震 cho: 洪震,
    阿震, 小震. Tiếng gọi trùng giữa hai nhân vật bị bỏ.

    Args:
        characters: Bảng nhân vật.

    Returns:
        ``{tiếng gọi: tên chuẩn (trường name của bảng)}``.
    """
    index: dict[str, str] = {}
    ambiguous: set[str] = set()
    for entry in parse_roster(characters or ""):
        canonical = entry.name.strip()
        for form in _call_forms(entry):
            if index.get(form, canonical) != canonical:
                ambiguous.add(form)
            index[form] = canonical
    for form in ambiguous:
        index.pop(form, None)
    return index


def leading_call(source_text: str, index: Mapping[str, str]) -> tuple[str, str] | None:
    """Nhân vật được GỌI ở đầu câu gốc rồi câu nói tiếp.

    Câu CHỈ có cái tên ("秦月", "慕云。") không tính: có thể là tự giới thiệu hay đọc
    tên — trên dữ liệu thật, đa số các lượt phân tích giữ nhãn "người nói = người được
    gọi" ở 31/57 dòng như vậy.

    Args:
        source_text: Câu gốc.
        index: Kết quả :func:`nickname_index`.

    Returns:
        ``(tên chuẩn, tiếng gọi)`` hoặc ``None``.
    """
    text = _LEADING_RE.sub("", source_text or "")
    for form in sorted(index, key=len, reverse=True):
        if not text.startswith(form):
            continue
        rest = text[len(form):]
        for suffix in _CALL_SUFFIXES:
            if rest.startswith(suffix):
                rest = rest[len(suffix):]
                break
        # Tên lặp chữ ("明明") còn là trạng từ "rõ ràng" ("明明你知道") — chỉ nhận khi
        # theo sau là dấu câu.
        doubled = len(form) == 2 and form[0] == form[1]
        allowed = _AFTER_PUNCTUATION if doubled else _AFTER_CALL
        said_more = rest.strip("".join(_AFTER_PUNCTUATION))
        if rest and rest[0] in allowed and said_more:
            return index[form], form
    return None


def self_gender(source_text: str) -> tuple[str, str] | None:
    """Giới tính người nói mà câu gốc tự xưng ra (``我嫁`` → nữ, ``我老婆`` → nam).

    Returns:
        ``(VOICE_FEMALE | VOICE_MALE, từ tự xưng)`` hoặc ``None``.
    """
    for words, gender in (
        (_FEMALE_SELF_WORDS, VOICE_FEMALE), (_MALE_SELF_WORDS, VOICE_MALE)
    ):
        for word in words:
            if word in (source_text or ""):
                return gender, word
    return None


def review_speakers(
    speakers: Sequence[str],
    source_texts: Sequence[str],
    characters: str,
    genders: Mapping[str, str],
    skip: frozenset[int] = frozenset(),
) -> dict[int, SpeakerIssue]:
    """Các dòng cần soát người nói.

    Args:
        speakers: Người nói từng dòng TTS (đã áp sửa tay; rỗng = chưa rõ).
        source_texts: Câu gốc của từng dòng TTS (rỗng nếu không ghép được câu gốc).
        characters: Bảng nhân vật.
        genders: ``{tên vai: giới}``.
        skip: Dòng người dùng đã chọn tay người nói (đã soát — không báo).

    Returns:
        ``{vị trí dòng: vấn đề}``; mỗi dòng một vấn đề, ưu tiên tiếng gọi > giới tính.
    """
    index = nickname_index(characters)
    issues: dict[int, SpeakerIssue] = {}
    for position, (speaker, source_text) in enumerate(
        zip(speakers, source_texts, strict=True)
    ):
        if position in skip:
            continue
        issue = _line_issue(speaker, source_text, index, genders)
        if issue is not None:
            issues[position] = issue
    return issues


def _line_issue(
    speaker: str,
    source_text: str,
    index: Mapping[str, str],
    genders: Mapping[str, str],
) -> SpeakerIssue | None:
    if not speaker:
        return SpeakerIssue(REVIEW_UNKNOWN)
    if speaker in VOICE_ONLY_SPEAKERS:
        return SpeakerIssue(REVIEW_VOICE_ONLY, speaker)
    called = leading_call(source_text, index)
    if called is not None and called[0] == speaker:
        return SpeakerIssue(REVIEW_SELF_CALL, called[0], called[1])
    implied = self_gender(source_text)
    role_gender = genders.get(speaker, "")
    if implied is not None and role_gender and role_gender != implied[0]:
        return SpeakerIssue(REVIEW_GENDER, implied[0], implied[1])
    return None


def _call_forms(entry: RosterEntry) -> set[str]:
    forms: set[str] = set()
    for alias in (part.strip() for part in _ALIAS_SPLIT_RE.split(entry.name or "")):
        if not _CJK_NAME_RE.match(alias):
            continue
        forms.add(alias)
        last = alias[-1]
        if len(alias) == 3:
            forms |= {alias[1:], last * 2, *(prefix + last for prefix in _NICK_PREFIXES)}
        elif len(alias) == 2 and alias[0] not in (*_NICK_PREFIXES, "老"):
            forms |= {prefix + last for prefix in _NICK_PREFIXES}
    return forms - _COMMON_WORDS


__all__ = [
    "REVIEW_GENDER",
    "REVIEW_SELF_CALL",
    "REVIEW_UNKNOWN",
    "REVIEW_VOICE_ONLY",
    "SpeakerIssue",
    "leading_call",
    "nickname_index",
    "review_speakers",
    "self_gender",
]
