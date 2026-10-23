"""[v3.23.614] Giới tính của nhãn vai chữ Hán theo DANH TỪ CHÍNH (đứng cuối cụm).

Ca thật 14/10: vai "女弟子" (nữ đệ tử) được giao giọng NAM vì chữ "弟" (em trai) trong
"弟子" (đệ tử — không chỉ giới) khớp trước chữ "女". Danh từ chính trong cụm chữ Hán đứng
CUỐI ("伯母" là bác gái, "舅妈" là mợ, "姐夫" là anh rể) nên lấy gợi ý ở vị trí cuối cùng,
và bỏ các từ chứa chữ chỉ giới nhưng không chỉ giới ("弟子", "徒弟", "子弟").
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.dubbing_cast import _gender_from_label
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
)


@pytest.mark.parametrize(
    ("label", "gender"),
    [
        ("女弟子", VOICE_FEMALE),     # ca thật 14/10
        ("男弟子", VOICE_MALE),
        ("大弟子", ""),
        ("小弟", VOICE_MALE),
        ("伯母", VOICE_FEMALE),
        ("舅妈", VOICE_FEMALE),
        ("姐夫", VOICE_MALE),
        ("姨夫", VOICE_MALE),
        ("女婿", VOICE_MALE),
        ("洪震情婦", VOICE_FEMALE),   # nhãn thật trong kho chẩn đoán
        ("老妇", VOICE_FEMALE),       # nhãn thật trong kho chẩn đoán
        ("老伯", VOICE_MALE),
        ("洪大嫂", VOICE_FEMALE),
        ("钁柱妻子", VOICE_FEMALE),
        ("医生", ""),
        ("大夫", ""),
    ],
)
def test_head_noun_decides(label: str, gender: str) -> None:
    assert _gender_from_label(label) == gender


def test_real_cast_reads_the_female_disciple_as_female() -> None:
    from types import SimpleNamespace

    from subtitles_extractor.application.services.dubbing_cast import (
        DubbingSource,
        build_cast,
        line_speakers,
    )
    from tests.unit.application.test_roster_sentence_label_v323614 import ROSTER

    cues = (
        '[{"id": 1, "spk": "楚瑶", "to": "叶然"},'
        ' {"id": 2, "spk": "女弟子", "to": "叶然"},'
        ' {"id": 3, "spk": "叶然", "to": "楚瑶"}]'
    )
    spans = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    source = DubbingSource(
        characters=ROSTER, visual_cues=cues, source_spans=spans,
        source_texts=("师兄", "叶然你还不明白吗", "一言为定"),
    )
    events = [SimpleNamespace(text="x", start_sec=a, end_sec=b) for a, b in spans]
    cast = {member.name: member for member in build_cast(
        line_speakers(events, source), ROSTER
    )}
    assert cast["楚瑶"].gender == VOICE_FEMALE
    assert cast["楚瑶"].display_name == "Sở Dao"
    assert cast["女弟子"].gender == VOICE_FEMALE
    assert cast["叶然"].gender == VOICE_MALE
