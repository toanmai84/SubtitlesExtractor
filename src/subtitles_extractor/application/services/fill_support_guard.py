"""[v3.23.602] Nhãn người nói vừa HỎI LẠI phải có chỗ dựa trong chính cảnh đó.

Chuẩn đoán thật 灭门惨案II (v3.23.601, dịch bằng gemini-3.5-flash-lite): bước hỏi lại
điền đủ 39/39 dòng thiếu người nói, nhưng các dòng dân làng hỏi trưởng đồn (103-114,
131-132, 154-156) được gán LUÂN PHIÊN ba cái tên trong bảng nhân vật — 胖公安, 老伯,
朱正軍, 胖公安, 老伯, 朱正軍… — kể cả một câu nói liền ba dòng (154-156) thành ba người.
Các lượt phân tích khác của cùng phim gọi những dòng này là "dân làng"; chỉ 5/39 nhãn hỏi
lại trùng nhãn đa số của chúng.

Tín hiệu đo được: người nói THẬT hầu như luôn xuất hiện (nói hoặc được nói tới) trong các
nhãn quanh dòng của mình. Trên 153 746 nhãn tên của 57 lượt phân tích (3 phim), chỉ 0.83%
người nói vắng mặt khỏi nhãn ±10 dòng quanh đó; ở 34 nhãn tên vừa hỏi lại kể trên: 14 dòng
(41%) — và không dòng nào trong 14 dòng đó trùng nhãn đa số của các lượt khác. Mô phỏng
trên nhãn thật (cứ 7 nhãn tên coi 1 nhãn như vừa hỏi lại, 3 lượt mới nhất mỗi phim): lưới
hạ nhầm 189/31 340 nhãn (0.60%).

Đã thử và BỎ: coi "người nói = người nghe đã biết" là mâu thuẫn (dòng 138). Nhãn phân tích
thật ghi người nghe trùng người nói rất thường (độc thoại) — mô phỏng hạ nhầm 1057 nhãn,
và chính dòng 138 thì 14 lượt phân tích khác đều nói là 龍傳人.

Nhãn như vậy bị HẠ xuống nhãn giọng theo giới tính của tên model đưa ra ("Giọng nam chưa
rõ"): giữ phần model nghe được (giọng nam/nữ — lồng tiếng vẫn đúng giới), bỏ danh tính
không có chỗ dựa (không còn quy tắc xưng hô của một cặp nhân vật bịa ra, không còn vai
riêng cho từng câu của đám đông). Toàn bộ là hàm thuần.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from subtitles_extractor.application.services.roster_format import (
    name_aliases,
    parse_roster,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue
from subtitles_extractor.domain.value_objects.speaker_label import (
    UNKNOWN_FEMALE_SPEAKER,
    UNKNOWN_MALE_SPEAKER,
    is_fallback_speaker,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
    gender_from_roster,
)

#: Người nói thật phải có mặt trong nhãn của ngần này dòng mỗi bên (số đo: docstring).
FILL_SUPPORT_WINDOW_LINES = 10
#: Cảnh có ít nhãn tin cậy hơn mức này thì không đủ chứng cứ để nói ai "vắng mặt" (vd cả
#: một đoạn dài đều thiếu nhãn) — giữ nguyên nhãn hỏi lại.
MIN_SCENE_LABELS = 3

#: Lý do: người nói không có mặt trong cảnh.
REASON_ABSENT = "absent"

_GENDER_LABELS = {VOICE_MALE: UNKNOWN_MALE_SPEAKER, VOICE_FEMALE: UNKNOWN_FEMALE_SPEAKER}


@dataclass(frozen=True, slots=True)
class UnsupportedFill:
    """Một nhãn hỏi lại không có chỗ dựa.

    Attributes:
        line_no: Số dòng.
        speaker: Tên model đã gán.
        reason: :data:`REASON_ABSENT`.
        replacement: Nhãn thay thế (nhãn giọng theo giới; rỗng = bỏ người nói).
    """

    line_no: int
    speaker: str
    reason: str
    replacement: str


def _people(label: str) -> frozenset[str]:
    return frozenset(name_aliases(label))


def _roster_genders(characters: str) -> dict[str, str]:
    """``bí danh → giới tính`` của mọi nhân vật trong bảng."""
    genders: dict[str, str] = {}
    for entry in parse_roster(characters):
        gender = gender_from_roster(entry.gender)
        for alias in name_aliases(entry.name):
            genders[alias] = gender
    return genders


def _scene_people(
    trusted: dict[int, VisualCue], line_no: int, window: int
) -> tuple[frozenset[str], int]:
    """Người nói/người nghe trong nhãn tin cậy ±``window`` dòng (trừ chính dòng đó).

    Returns:
        ``(tập bí danh, số dòng nhãn tin cậy trong cửa sổ)``.
    """
    people: set[str] = set()
    labelled = 0
    for other in range(line_no - window, line_no + window + 1):
        cue = trusted.get(other)
        if other == line_no or cue is None:
            continue
        labelled += 1
        people |= _people(cue.speaker) | _people(cue.addressee)
    return frozenset(people), labelled


def unsupported_fills(
    original: Sequence[VisualCue],
    completed: Sequence[VisualCue],
    filled: Sequence[int],
    characters: str,
    window: int = FILL_SUPPORT_WINDOW_LINES,
) -> list[UnsupportedFill]:
    """Nhãn TÊN vừa hỏi lại mà người đó không có mặt trong cảnh.

    Chỉ xét tên CÓ trong bảng nhân vật: nhãn giọng/đám đông không khẳng định danh tính,
    nhãn vai ngoài bảng ("dân làng") không có gì để đối chiếu.

    Args:
        original: Cues trước khi hỏi lại (nhãn tin cậy).
        completed: Cues sau khi điền.
        filled: Các dòng vừa được điền.
        characters: Bảng nhân vật.
        window: Số dòng mỗi bên để tìm người nói trong cảnh.

    Returns:
        Danh sách nhãn không có chỗ dựa, theo thứ tự dòng.
    """
    if not filled:
        return []
    genders = _roster_genders(characters)
    if not genders:
        return []
    filled_set = set(filled)
    before = {cue.line_no: cue for cue in original}
    trusted = {
        line: cue for line, cue in before.items()
        if line not in filled_set and (cue.speaker or cue.addressee)
    }
    after = {cue.line_no: cue for cue in completed}
    found: list[UnsupportedFill] = []
    for line_no in sorted(filled_set):
        cue = after.get(line_no)
        if cue is None or is_fallback_speaker(cue.speaker):
            continue
        speaker = _people(cue.speaker)
        known = [alias for alias in speaker if alias in genders]
        if not known:
            continue
        scene, labelled = _scene_people(trusted, line_no, window)
        if labelled >= MIN_SCENE_LABELS and not speaker & scene:
            replacement = _GENDER_LABELS.get(genders[known[0]], "")
            found.append(
                UnsupportedFill(line_no, cue.speaker, REASON_ABSENT, replacement)
            )
    return found


def demote_unsupported_fills(
    original: Sequence[VisualCue],
    completed: Sequence[VisualCue],
    filled: Sequence[int],
    characters: str,
    window: int = FILL_SUPPORT_WINDOW_LINES,
) -> tuple[list[VisualCue], list[UnsupportedFill]]:
    """Hạ nhãn tên không có chỗ dựa xuống nhãn giọng theo giới tính.

    Không rõ giới tính thì trả dòng về nhãn cũ (trước khi hỏi lại).

    Args:
        original: Cues trước khi hỏi lại.
        completed: Cues sau khi điền.
        filled: Các dòng vừa được điền.
        characters: Bảng nhân vật.
        window: Số dòng mỗi bên để tìm người nói trong cảnh.

    Returns:
        ``(cues, các nhãn đã hạ)``.
    """
    findings = unsupported_fills(original, completed, filled, characters, window)
    if not findings:
        return list(completed), []
    by_finding = {finding.line_no: finding for finding in findings}
    previous = {cue.line_no: cue for cue in original}
    result: list[VisualCue] = []
    for cue in completed:
        finding = by_finding.get(cue.line_no)
        if finding is None:
            result.append(cue)
        elif finding.replacement:
            result.append(replace(cue, speaker=finding.replacement))
        elif cue.line_no in previous:
            result.append(previous[cue.line_no])
    return result, findings


def demote_unsupported_scan_labels(
    cues: Sequence[VisualCue], characters: str
) -> tuple[list[VisualCue], list[UnsupportedFill]]:
    """Áp lưới cho Visual Cues ĐÃ LƯU (phiên không ghi nguồn gốc nhãn).

    Nhãn của bộ quét/hỏi lại nhận ra vì luôn kèm mô tả bối cảnh, nhãn của khâu phân tích
    thì không (:func:`described_scan_lines`). Phiên của chuẩn đoán thật đã lưu 39 nhãn hỏi
    lại kèm mô tả — đúng 39 dòng — trong đó 14 tên luân phiên không có mặt trong cảnh;
    không áp ở đây thì lồng tiếng và lượt dịch sau vẫn dùng chúng như nhãn tin cậy.

    Args:
        cues: Visual Cues đã quy nhãn về bảng nhân vật.
        characters: Bảng nhân vật.

    Returns:
        ``(cues, các nhãn đã hạ)``; nguyên ``cues`` khi không phân biệt được nguồn gốc.
    """
    from subtitles_extractor.application.services.cue_contradiction_guard import (
        described_scan_lines,
    )

    scanned = described_scan_lines(cues)
    if not scanned:
        return list(cues), []
    return demote_unsupported_fills(cues, cues, sorted(scanned), characters)


__all__ = [
    "FILL_SUPPORT_WINDOW_LINES",
    "MIN_SCENE_LABELS",
    "REASON_ABSENT",
    "UnsupportedFill",
    "demote_unsupported_fills",
    "demote_unsupported_scan_labels",
    "unsupported_fills",
]
