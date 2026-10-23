"""[v3.23.636] Quy tắc 'A' hoặc 'B' (lọt qua prompt phân tích) vẫn cho phép gán pron — lấy phương án đầu.

Ca thật (tập 12 phim mẫu): ``tự xưng 'đệ tử' hoặc 'con' gọi 'sư thúc'`` làm pron trống 11/18 dòng
và bản dịch tự xưng lúc "con", lúc "đệ tử", lúc "ta".
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.address_gloss import (
    address_pronoun_gloss,
    first_alternative,
)


_TITLE_NOTE = ' (tự xưng bằng danh xưng: viết nguyên "{t}", KHÔNG đổi thành "ta")'


@pytest.mark.parametrize(("rule", "source", "expected"), [
    ("tự xưng 'đệ tử' hoặc 'con' gọi 'sư thúc'", "我现在就出去看看",
     '我="đệ tử"' + _TITLE_NOTE.format(t="đệ tử")),
    ("tự xưng 'con' hoặc 'đệ tử' gọi 'sư thúc'", "我怕你", "我=con; 你=sư thúc"),
    ("tự xưng 'ta' gọi 'ngươi' hoặc 'tên tiểu tử'", "你过来", "你=ngươi"),
    ("tự xưng 'tôi/chị' gọi 'cậu'", "你来了", ""),            # '/' vẫn để model chọn (v3.23.550)
    ("trước khi thua tự xưng 'ta' hoặc 'tôi' gọi 'mày'", "你来了", ""),   # có điều kiện
])
def test_gloss_takes_first_of_hoac(rule: str, source: str, expected: str) -> None:
    assert address_pronoun_gloss(rule, source) == expected


def test_title_self_term_keeps_second_person_gloss() -> None:
    got = address_pronoun_gloss("tự xưng 'đồ nhi' gọi 'sư phụ'", "我怕你")
    assert got.startswith('我="đồ nhi"') and got.endswith("; 你=sư phụ")


def test_first_alternative_leaves_plain_rules_alone() -> None:
    rule = "tự xưng 'ta' gọi 'ngươi'"
    assert first_alternative(rule) == rule
    assert first_alternative("") == ""
