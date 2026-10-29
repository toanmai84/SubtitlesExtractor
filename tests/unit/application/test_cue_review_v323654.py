"""[v3.23.654] Chấm dòng Visual Cues đáng nghi ở trang Dịch (không tốn request)."""

from __future__ import annotations

from subtitles_extractor.application.services.cue_review import (
    CUE_GENDER,
    CUE_NO_SPEAKER,
    CUE_SAME_PERSON,
    CUE_SELF_CALL,
    CUE_UNKNOWN_NAME,
    review_cues,
)
from subtitles_extractor.application.services.visual_cue_serializer import deserialize_visual_cues
from subtitles_extractor.domain.ports.subtitle_translator_port import CUE_ORIGIN_USER, VisualCue
from tests.unit.presentation.test_dubbing_line_edits_v323598 import LINES, P495, ROSTER, _source


def _reasons(issues, line_no):
    return {issue.reason for issue in issues.get(line_no, [])}


def test_real_wrong_label_calling_own_name_is_flagged() -> None:
    cues = deserialize_visual_cues(_source().visual_cues)
    texts = [row[6] for row in LINES]
    issues = review_cues(cues, texts, ROSTER)
    assert CUE_SELF_CALL in _reasons(issues, P495 + 1)   # "阿秀，你幫我…" mang nhãn chính A Tú


def test_rules() -> None:
    roster = "林逸 (Lâm Dật) [nam, ~20] — chính\n苏瑶 (Tô Dao) [nữ, ~18] — sư muội"
    texts = ["你来了", "本姑娘不怕", "老夫来也", "好", "", "走吧", "看", "看"]
    cues = [
        VisualCue(1, speaker="林逸", addressee="林逸"),          # người nói = người nghe
        VisualCue(2, speaker="林逸", addressee="苏瑶"),          # 本姑娘 mà người nói là nam
        VisualCue(3, speaker="苏瑶", addressee="林逸"),          # 老夫 mà người nói là nữ
        VisualCue(4, speaker=""),                                 # thiếu người nói (câu có lời)
        VisualCue(5, speaker=""),                                 # câu rỗng: không báo
        VisualCue(6, speaker="王麻子"),                           # tên lạ, chỉ 1 câu
        VisualCue(7, speaker="Hệ thống"), VisualCue(8, speaker="Hệ thống"), VisualCue(9, speaker="Hệ thống"),
    ]
    texts.append("提示")
    issues = review_cues(cues, texts, roster)
    assert _reasons(issues, 1) == {CUE_SAME_PERSON}
    assert _reasons(issues, 2) == {CUE_GENDER}
    assert _reasons(issues, 3) == {CUE_GENDER}
    assert _reasons(issues, 4) == {CUE_NO_SPEAKER}
    assert 5 not in issues
    assert _reasons(issues, 6) == {CUE_UNKNOWN_NAME}
    assert 7 not in issues and 9 not in issues           # vai ngoài bảng dùng nhất quán: không nghi


def test_user_cues_are_trusted() -> None:
    roster = "林逸 (Lâm Dật) [nam, ~20] — chính"
    cues = [VisualCue(1, speaker="林逸", origin=CUE_ORIGIN_USER), VisualCue(2, speaker="X", addressee="X", origin=CUE_ORIGIN_USER)]
    issues = review_cues(cues, ["本姑娘来了", "好"], roster)
    assert 1 not in issues
    assert _reasons(issues, 2) == {CUE_SAME_PERSON}
