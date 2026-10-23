"""[v3.23.601] "con ranh" là lời chửi, không phải đại từ "con" — không báo mơ hồ.

Chuẩn đoán thật v3.23.600 (灭门惨案II): cờ ``roster_ambiguous_address`` cho
``洪志權 / 阿權 → Tần Khả Tú: xưng 'tao' gọi 'mày/con ranh'`` kèm lời khuyên phân tích
lại cả phim. Đo trên 24 bảng nhân vật của kho: đây là cờ DUY NHẤT thay đổi (26 → 25).
"""

from __future__ import annotations

from subtitles_extractor.application.services.roster_format import (
    ambiguous_address_rules,
)

_ROSTER = (
    "秦可秀 / 阿秀 (Tần Khả Tú / A Tú) [nữ, ~28] — vợ Chân Kiếm Thanh; xưng hô: với "
    "Hồng Chí Quyền xưng 'tôi' gọi 'cậu'\n"
    "洪震 (Hồng Chấn) [nam, ~45] — Ác bá đầu gấu trong làng; xưng hô: với Hồng Chí "
    "Quyền tự xưng 'bố' gọi 'con'\n"
    "洪志權 / 阿權 (Hồng Chí Quyền / A Quyền) [nam, ~20] — Con trai của Hồng Chấn, cùng "
    "cha tác oai tác quái tống tiền và làm nhục Khả Tú; xưng hô: với Hồng Chấn tự xưng "
    "'con' gọi 'bố'; với Tần Khả Tú xưng 'tao' gọi 'mày/con ranh'"
)


def test_con_ranh_khong_phai_dai_tu() -> None:
    assert ambiguous_address_rules(_ROSTER) == []


def test_hai_dai_tu_that_van_bao() -> None:
    roster = _ROSTER.replace("'mày/con ranh'", "'mày/cô'")
    assert ambiguous_address_rules(roster) == [
        "洪志權 / 阿權 → Tần Khả Tú: xưng 'tao' gọi 'mày/cô'"
    ]


def test_con_gai_van_la_dai_tu_con() -> None:
    # "con gái" (gọi con cái) vẫn là đại từ "con" → cùng 'cháu' là hai đại từ.
    roster = _ROSTER.replace("'mày/con ranh'", "'cháu/con gái'")
    assert len(ambiguous_address_rules(roster)) == 1
