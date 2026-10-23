"""[v3.23.614] Giới tính trong bảng nhân vật đọc theo NGHĨA, không theo đúng chữ.

Bản 3.23.614 đọc ``[~70]`` (chưa rõ giới, có tuổi) đúng thành tuổi; giới tính ghi kiểu
"Nam giới"/"nữ giới"/"không rõ" trước đây không được hiểu: lồng tiếng coi "Nam giới" là
chưa rõ giới, còn xưng hô thuần Việt coi "không rõ" là một giới khác cả nam lẫn nữ nên
loại nhân vật khỏi mọi tiếng gọi.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.native_address import (
    GENDER_FEMALE,
    GENDER_MALE,
    build_roster,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
    gender_from_roster,
)


@pytest.mark.parametrize(
    ("written", "voice"),
    [
        ("nam", VOICE_MALE), ("Nam giới", VOICE_MALE), ("male", VOICE_MALE),
        ("nữ", VOICE_FEMALE), ("Nữ giới", VOICE_FEMALE), ("female", VOICE_FEMALE),
        ("không rõ", ""), ("", ""), ("~70", ""),
    ],
)
def test_gender_from_roster(written: str, voice: str) -> None:
    assert gender_from_roster(written) == voice


def test_native_roster_normalizes_gender() -> None:
    roster = build_roster("\n".join([
        "王老 (Vương lão) [Nam giới, ~70] — trưởng lão",
        "李姐 (Lý tỷ) [nữ giới, ~30] — chủ quán",
        "系统 (Hệ thống) [không rõ] — hệ thống",
    ]))
    assert roster.gender_of("王老") == GENDER_MALE
    assert roster.gender_of("李姐") == GENDER_FEMALE
    assert roster.gender_of("系统") == ""
