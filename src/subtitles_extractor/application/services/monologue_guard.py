"""[v3.23.577] Dòng ĐỘC THOẠI NỘI TÂM: báo rõ cho khâu dịch là không có người nghe.

Chuẩn đoán thật (phim "傻子落水…", 3902 dòng, v3.23.576): khâu phân tích ghi độc thoại
nội tâm bằng nhãn cues "người nói = người nghe" (``宋钱 → 宋钱``, 90 dòng). Khâu dịch nhận
``speaker: 宋钱, addressee: 宋钱`` — không hiểu là tự nói với mình — nên mang theo cách
xưng hô của câu thoại ngay trước đó. Kết quả ở MỌI lần dịch phim này:

* 125-126 "Nhưng **em** có truyền thừa của Thần Tiên tỷ tỷ, trên đời này có bệnh gì làm
  khó được **em**?" — nhân vật chính đang một mình;
* 21 (Hứa Đào Hoa nghĩ thầm) "nhưng **chị** thủ tiết hai năm";
* 469 "**Em** cũng chẳng còn cách nào khác."

Đo trên 31 lần dịch thật: ở dòng độc thoại mà câu gốc có đúng một ``我`` và bản dịch có
đúng một đại từ tự xưng, 45/175 dòng (26%) tự xưng bằng từ vai vế (em/anh/chị/con…) —
riêng phim này 5-6 dòng mỗi lần, gần một nửa số dòng đo được.

Không phải mọi nhãn "người nói = người nghe" đều là độc thoại — có khi là nhãn sai của
một lượt thoại thật (đàn em Ngô Nhị Cẩu "狗哥我控制不住了", "你想造反啊"). Tách theo
LƯỢT (các dòng liền nhau cùng nhãn tự-nói): lượt có ``你/您``, ``我们`` hay tiếng gọi tên
là đang nói với người khác → bỏ nhãn người nghe (không biết là ai); còn lại là độc thoại.
Soát tay trên phim này: 42 lượt độc thoại (126 dòng) đều là nhân vật nghĩ thầm / nói một
mình, trừ một lượt ra lệnh cho gia nhân ("来人!…") không có đại từ nào; 8 lượt (47 dòng)
bị nhận là nói với người khác.

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.application.services.roster_format import name_aliases
from subtitles_extractor.application.services.translation_diagnostics import (
    ADDRESS_WORD_RE,
    PLURAL_SOURCE_RE,
    SELF_SOURCE_RE,
    YOU_SOURCE_RE,
)
from subtitles_extractor.application.services.vocative_guard import (
    vocative_alias_index,
    vocative_callee,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue

#: Nhãn người nghe gửi cho khâu dịch ở dòng độc thoại (thay cho tên chính người nói).
MONOLOGUE_ADDRESSEE = "(độc thoại — không nói với ai)"
#: Quy tắc xưng hô gắn vào dòng độc thoại (cùng cơ chế trường ``address``).
MONOLOGUE_RULE = (
    "độc thoại nội tâm, KHÔNG có người nghe: tự xưng 'mình' (hoặc 'tôi'), KHÔNG tự xưng "
    "anh/em/chị/con/cháu; người được nhắc tới gọi ở ngôi thứ ba"
)
#: Từ vai vế — tự xưng bằng các từ này là đang nói VỚI ai đó.
KIN_SELF_WORDS = frozenset({
    "anh", "em", "chị", "con", "cháu", "cô", "chú", "bác", "ông", "bà",
})

_SECOND_PERSON_RE = re.compile("[你妳您]")
_GROUP_SELF_RE = re.compile("[我咱俺][们們]")
#: Ngay sau tiếng gọi tên là dấu ngắt hoặc chính người nói xưng "tôi".
_AFTER_VOCATIVE_RE = re.compile("[\uff0c,\u3001\uff01!\uff1f?\uff1a: \u3000我俺]")
#: "mình" không nằm trong danh sách đại từ chung (còn là danh từ "cơ thể") nhưng là cách
#: tự xưng phổ biến nhất khi độc thoại.
_MINH_RE = re.compile(r"(?<!\w)mình(?!\w)", re.IGNORECASE)
#: Từ vai vế đứng trước TÊN ("chị Đào Hoa", "anh Thiết Trụ") là ngôi ba, không tự xưng.
_BEFORE_NAME_RE = re.compile(r"\s+[A-ZÀ-Ỹ]")


@dataclass(frozen=True, slots=True)
class SelfAddressedRun:
    """Một lượt các dòng liền nhau có nhãn "người nói = người nghe".

    Attributes:
        speaker: Nhãn người nói.
        lines: Các dòng của lượt (tăng dần).
        monologue: True nếu là độc thoại; False nếu lượt thực ra nói với người khác.
    """

    speaker: str
    lines: tuple[int, ...]
    monologue: bool


def is_self_addressed(speaker: str, addressee: str) -> bool:
    """Nhãn người nói và người nghe có chung bí danh (``"宋钱 / 宋"`` với ``"宋钱"``)."""
    return bool(set(name_aliases(speaker)) & set(name_aliases(addressee)))


def _speaks_to_someone(
    lines: list[int], sources: dict[int, str], vocatives: dict[str, Any]
) -> bool:
    """Lượt có dấu hiệu đang nói với người khác: ngôi hai, "chúng tôi", tiếng gọi tên."""
    for line_no in lines:
        source = sources.get(line_no, "")
        if _SECOND_PERSON_RE.search(source) or _GROUP_SELF_RE.search(source):
            return True
        if vocative_callee(source, vocatives) is not None:
            return True
        # Mở đầu bằng tiếng gọi tên rồi nói tiếp ("狗哥我控制不住了", "老表, 别…"). Tên
        # làm CHỦ NGỮ ("吴二狗居然自称土狗", "张依贝吗") thì vẫn là nghĩ thầm.
        if any(
            source.startswith(alias) and _AFTER_VOCATIVE_RE.match(source, len(alias))
            for alias in vocatives
        ):
            return True
    return False


def self_addressed_runs(
    cues: list[VisualCue], sources: dict[int, str], characters: str
) -> list[SelfAddressedRun]:
    """Tách các dòng nhãn "người nói = người nghe" thành lượt và phân loại (hàm thuần).

    Args:
        cues: Visual Cues.
        sources: ``số dòng → câu gốc`` của MỌI dòng (dòng không có cue ngắt lượt).
        characters: Bảng nhân vật (để nhận tiếng gọi tên).

    Returns:
        Các lượt theo thứ tự thời gian.
    """
    by_line = {cue.line_no: cue for cue in cues}
    vocatives = vocative_alias_index(characters)
    runs: list[SelfAddressedRun] = []
    current: list[int] = []
    speaker = ""

    def close() -> None:
        if current:
            monologue = not _speaks_to_someone(current, sources, vocatives)
            runs.append(SelfAddressedRun(speaker, tuple(current), monologue))

    for line_no in sorted(sources):
        cue = by_line.get(line_no)
        own = (
            cue is not None
            and bool(cue.speaker)
            and is_self_addressed(cue.speaker, cue.addressee)
        )
        if own and current and is_self_addressed(cue.speaker, speaker):
            current.append(line_no)
            continue
        close()
        current = [line_no] if own else []
        speaker = cue.speaker if own and cue is not None else ""
    close()
    return runs


def _self_words(text: str) -> set[str]:
    """Các đại từ có thể là tự xưng trong câu dịch (bỏ từ vai vế đứng trước tên riêng)."""
    words = {
        match.group(1).lower()
        for match in ADDRESS_WORD_RE.finditer(text)
        if not _BEFORE_NAME_RE.match(text, match.end())
    }
    if _MINH_RE.search(text):
        words.add("mình")
    return words


def monologue_self_forms(
    cues: list[VisualCue],
    sources: dict[int, str],
    translations: dict[int, str],
    characters: str,
) -> list[dict[str, Any]]:
    """[Chuẩn đoán] Dòng độc thoại tự xưng bằng từ vai vế, gom theo người nói.

    Chỉ xét dòng mà vai của đại từ chắc chắn: câu gốc có ``我`` số ít, không có ``你``, và
    bản dịch có đúng một đại từ.

    Returns:
        ``[{"speaker", "lines", "variants"}]`` — ``variants`` đếm MỌI cách tự xưng của
        người đó khi độc thoại, ``lines`` là các dòng tự xưng bằng từ vai vế.
    """
    findings: dict[str, dict[str, Any]] = {}
    for run in self_addressed_runs(cues, sources, characters):
        if not run.monologue:
            continue
        for line_no in run.lines:
            source = sources.get(line_no, "")
            if (
                PLURAL_SOURCE_RE.search(source)
                or not SELF_SOURCE_RE.search(source)
                or YOU_SOURCE_RE.search(source)
            ):
                continue
            words = _self_words(translations.get(line_no, ""))
            if len(words) != 1:
                continue
            word = next(iter(words))
            entry = findings.setdefault(
                run.speaker, {"speaker": run.speaker, "lines": [], "variants": {}}
            )
            entry["variants"][word] = entry["variants"].get(word, 0) + 1
            if word in KIN_SELF_WORDS:
                entry["lines"].append(line_no)
    return [entry for entry in findings.values() if entry["lines"]]


__all__ = [
    "KIN_SELF_WORDS",
    "MONOLOGUE_ADDRESSEE",
    "MONOLOGUE_RULE",
    "SelfAddressedRun",
    "is_self_addressed",
    "monologue_self_forms",
    "self_addressed_runs",
]
