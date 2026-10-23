"""[v3.23.615] Từ trung tính chỉ bị bỏ khi đứng riêng, không khi nằm trong cụm chỉ giới.

Tự rà bản 3.23.614: để "女弟子" (nữ đệ tử) hết bị đọc giọng nam, bản đó xoá "大夫" (thầy
thuốc, không chỉ giới) ở MỌI nơi trong nhãn — kéo theo "大夫人" (phu nhân cả) mất cụm
"夫人" và thành chưa rõ giới, trong khi bản 3.23.613 đọc đúng là nữ.
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
        ("大夫人", VOICE_FEMALE),
        ("二夫人", VOICE_FEMALE),
        ("大夫", ""),
        ("女弟子", VOICE_FEMALE),
        ("男弟子", VOICE_MALE),
        ("师弟", VOICE_MALE),
        ("女大夫", VOICE_FEMALE),
    ],
)
def test_neutral_words_inside_gender_compounds(label: str, gender: str) -> None:
    assert _gender_from_label(label) == gender
