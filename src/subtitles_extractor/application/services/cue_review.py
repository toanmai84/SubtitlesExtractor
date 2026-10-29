"""[v3.23.654] Chấm dòng Visual Cues đáng nghi — không tốn request (hàm thuần).

Gom về trang Dịch các bộ kiểm trước đây chỉ chạy ở trang TTS (tiếng gọi chính người nói, câu tự xưng trái giới —
:mod:`speaker_review`) và thêm luật mới:

* người nói = người nghe;
* thiếu người nói ở câu có lời;
* từ tự xưng gắn vai/giới (老夫, 本王… / 本姑娘, 妾身…) trái giới của người nói theo bảng nhân vật;
* người nói không có trong bảng nhân vật (nhãn lạ — dễ là AI bịa/nhầm tên).

Cue người dùng đã chọn (``origin == user``) được coi là đã soát, chỉ còn bị báo khi người nói = người nghe.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from subtitles_extractor.application.services.dubbing_cast import build_cast
from subtitles_extractor.application.services.roster_format import parse_roster
from subtitles_extractor.application.services.speaker_review import (
    REVIEW_GENDER,
    REVIEW_SELF_CALL,
    review_speakers,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import CUE_ORIGIN_USER, VisualCue
from subtitles_extractor.domain.value_objects.speaker_label import FALLBACK_SPEAKERS
from subtitles_extractor.domain.value_objects.voice_profile import VOICE_FEMALE, VOICE_MALE

CUE_SAME_PERSON = "same_person"
CUE_NO_SPEAKER = "no_speaker"
CUE_SELF_CALL = "self_call"
CUE_GENDER = "gender"
CUE_UNKNOWN_NAME = "unknown_name"

#: Từ tự xưng gắn giới (cổ trang/hiện đại). Chỉ dùng khi câu gốc BẮT ĐẦU hoặc chứa nguyên cụm.
_MALE_TITLES = ("老夫", "本王", "本少爷", "小爷", "老子", "本公子", "贫僧", "老衲", "在下")
_FEMALE_TITLES = ("本姑娘", "妾身", "老娘", "本小姐", "奴婢", "臣妾", "哀家", "本宫", "贫尼")


#: Nhãn ngoài bảng nhân vật dùng không quá bấy nhiêu câu mới bị nghi là nhãn lạ.
_RARE_LABEL_MAX = 2


@dataclass(frozen=True, slots=True)
class CueIssue:
    """Một lý do khiến cue đáng nghi.

    Attributes:
        reason: ``CUE_*``.
        word: Chữ trong câu gốc làm bằng chứng (nếu có).
        detail: Thông tin thêm (tên được gọi, giới tính câu gốc gợi ra).
    """

    reason: str
    word: str = ""
    detail: str = ""


def _title_gender(source: str) -> tuple[str, str] | None:
    for word in _FEMALE_TITLES:
        if word in source:
            return VOICE_FEMALE, word
    for word in _MALE_TITLES:
        if word in source:
            return VOICE_MALE, word
    return None


def roster_labels(characters: str) -> set[str]:
    """Mọi tên/bí danh/phiên âm trong bảng nhân vật (để nhận ra nhãn lạ)."""
    labels: set[str] = set()
    for entry in parse_roster(characters or ""):
        for part in re.split(r"\s*[/|]\s*", f"{entry.name}/{entry.aka}"):
            if part.strip():
                labels.add(part.strip())
    return labels


def review_cues(
    cues: Sequence[VisualCue], source_texts: Sequence[str], characters: str,
) -> dict[int, list[CueIssue]]:
    """Các cue đáng nghi, theo ``line_no`` (1-based, = vị trí câu gốc + 1).

    Args:
        cues: Visual Cues.
        source_texts: Câu gốc theo thứ tự (``source_texts[i]`` ↔ ``line_no = i + 1``).
        characters: Bảng nhân vật.

    Returns:
        ``{line_no: [vấn đề, …]}`` — dòng không có vấn đề không xuất hiện.
    """
    by_line = {cue.line_no: cue for cue in cues}
    count = len(source_texts)
    speakers = [by_line[i + 1].speaker.strip() if i + 1 in by_line else "" for i in range(count)]
    user_lines = {i for i in range(count) if i + 1 in by_line and by_line[i + 1].origin == CUE_ORIGIN_USER}
    genders = {member.name: member.gender for member in build_cast(speakers, characters)}
    known = roster_labels(characters)
    # Nhãn ngoài bảng nhưng dùng nhất quán nhiều câu ("Hệ thống", "Người dẫn chuyện") là một vai thật — chỉ nghi nhãn
    # hiếm (≤ 2 câu cả phim): dễ là AI gõ nhầm/bịa tên. Đo trên 12 tập mẫu: luật cũ báo 60 câu đều của vai "Hệ thống".
    label_counts: dict[str, int] = {}
    for speaker in speakers:
        if speaker:
            label_counts[speaker] = label_counts.get(speaker, 0) + 1
    issues: dict[int, list[CueIssue]] = {}

    def add(line_no: int, issue: CueIssue) -> None:
        bucket = issues.setdefault(line_no, [])
        if all(existing.reason != issue.reason for existing in bucket):
            bucket.append(issue)

    for position, issue in review_speakers(
        speakers, list(source_texts), characters, genders, frozenset(user_lines),
    ).items():
        if issue.reason == REVIEW_SELF_CALL:
            add(position + 1, CueIssue(CUE_SELF_CALL, issue.word, issue.detail))
        elif issue.reason == REVIEW_GENDER:
            add(position + 1, CueIssue(CUE_GENDER, issue.word, issue.detail))
    for position, source in enumerate(source_texts):
        line_no = position + 1
        cue = by_line.get(line_no)
        speaker = cue.speaker.strip() if cue is not None else ""
        addressee = cue.addressee.strip() if cue is not None else ""
        if speaker and addressee and speaker == addressee:
            add(line_no, CueIssue(CUE_SAME_PERSON, detail=speaker))
        if position in user_lines:
            continue
        if not speaker:
            if (source or "").strip():
                add(line_no, CueIssue(CUE_NO_SPEAKER))
            continue
        implied = _title_gender(source or "")
        if implied is not None and genders.get(speaker) and genders[speaker] != implied[0]:
            add(line_no, CueIssue(CUE_GENDER, implied[1], implied[0]))
        if (
            known and speaker not in known and speaker not in FALLBACK_SPEAKERS
            and label_counts.get(speaker, 0) <= _RARE_LABEL_MAX
        ):
            add(line_no, CueIssue(CUE_UNKNOWN_NAME, detail=speaker))
    return dict(sorted(issues.items()))


__all__ = [
    "CUE_GENDER",
    "CUE_NO_SPEAKER",
    "CUE_SAME_PERSON",
    "CUE_SELF_CALL",
    "CUE_UNKNOWN_NAME",
    "CueIssue",
    "review_cues",
    "roster_labels",
]
