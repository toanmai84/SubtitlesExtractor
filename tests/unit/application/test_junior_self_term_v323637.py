"""[v3.23.637] Người dưới tự xưng 'ta' với bậc trên → 'đệ tử' (sửa tất định ở bảng nhân vật).

Ca thật (phân tích phim ghép 12 tập, 3.1-flash-lite 3/6 lượt, 3.8-flash 3/5 lượt): Diệp Nhiên "tự xưng
'ta' gọi 'sư thúc'"; dịch với bảng đó: 17 lỗi nặng so với 4 ở lượt bảng đúng (chấm mù tập 9, 12).
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.junior_self_term import fix_junior_self_terms
from subtitles_extractor.application.services.roster_format import address_rule_map


@pytest.mark.parametrize(("before", "after"), [
    ("với Nam Cung Mộng tự xưng 'ta' gọi 'sư thúc'", "với Nam Cung Mộng tự xưng 'đệ tử' gọi 'sư thúc'"),
    ("với Nam Cung Mộng xưng 'ta' gọi 'sư thúc'", "với Nam Cung Mộng xưng 'đệ tử' gọi 'sư thúc'"),
    ("với Sư phụ tự xưng 'ta' gọi 'sư phụ'; với Dao Dao tự xưng 'ta' gọi 'muội'",
     "với Sư phụ tự xưng 'đệ tử' gọi 'sư phụ'; với Dao Dao tự xưng 'ta' gọi 'muội'"),
    ("với chưởng giáo tự xưng 'ta' gọi 'Chưởng giáo sư bá'",
     "với chưởng giáo tự xưng 'đệ tử' gọi 'Chưởng giáo sư bá'"),
])
def test_fixes_junior_to_elder(before: str, after: str) -> None:
    fixed, n = fix_junior_self_terms(before)
    assert fixed == after and n >= 1


@pytest.mark.parametrize("keep", [
    "với Nam Cung Mộng tự xưng 'ta' gọi 'ngươi'",                    # bậc trên nói với người dưới
    "với Diệp Nhiên tự xưng 'ta' gọi 'sư đệ'",                       # ngang vai
    "với tài xế tự xưng 'tôi' gọi 'sư phụ'",                         # hiện đại: 'tôi', "sư phụ" = tài xế
    "với Nam Cung Mộng tự xưng 'đệ tử' gọi 'sư thúc'",               # đã đúng
    "với Nam Cung Mộng tự xưng 'ta' gọi 'sư thúc muội'",             # không phải chức danh trong danh sách
    "",
])
def test_leaves_everything_else(keep: str) -> None:
    assert fix_junior_self_terms(keep) == (keep, 0)


def test_fixed_roster_gives_the_translator_a_readable_rule() -> None:
    roster = (
        "叶然 (Diệp Nhiên) [nam, ~20] — chính; xưng hô: với 南宫梦 tự xưng 'ta' gọi 'sư thúc'\n"
        "南宫梦 (Nam Cung Mộng) [nữ, ~30] — sư thúc; xưng hô: với 叶然 tự xưng 'ta' gọi 'ngươi'"
    )
    fixed, n = fix_junior_self_terms(roster)
    assert n == 1
    rules = address_rule_map(fixed)
    assert rules[("叶然", "南宫梦")] == "tự xưng 'đệ tử' gọi 'sư thúc'"
    assert rules[("南宫梦", "叶然")] == "tự xưng 'ta' gọi 'ngươi'"
