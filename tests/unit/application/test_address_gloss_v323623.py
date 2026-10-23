"""[v3.23.623] Phép gán 我/你 cho từng dòng + người nghe viết lệch tên vẫn tra được quy tắc."""
from __future__ import annotations

import pytest

from subtitles_extractor.application.services.address_gloss import address_pronoun_gloss
from subtitles_extractor.application.services.roster_format import (
    address_clauses,
    address_rule_map,
    lookup_address_rule,
)

RULE = "tự xưng 'muội' gọi 'sư huynh'"


@pytest.mark.parametrize(("rule", "source", "expected"), [
    (RULE, "师兄我就先走了", "我=muội"),          # lời gọi + 我: 我 vẫn là người nói
    (RULE, "你走吧", "你=sư huynh"),
    (RULE, "没有我在，你什么都做不成", "我=muội; 你=sư huynh"),
    (RULE, "我们一起走", ""),                      # số nhiều: không gán
    (RULE, "你们别吵了", ""),
    (RULE, "天气真好", ""),
    ("trước khi thua tự xưng 'ta' gọi 'mày'", "你来了", ""),   # có điều kiện: để model chọn
    ("tự xưng 'tôi/chị' gọi 'cậu'", "你来了", ""),              # nhiều phương án
    ("tự xưng 'ta' gọi 'con'; tự xưng 'ta' gọi 'ngươi'", "你来了", ""),
    ("", "你来了", ""),
])
def test_gloss(rule: str, source: str, expected: str) -> None:
    assert address_pronoun_gloss(rule, source) == expected


ROSTER = (
    "叶然 (Diệp Nhiên) [nam, ~20] — nhân vật chính; xưng hô: với Diêu Dao tự xưng 'sư huynh' "
    "gọi 'sư muội', với Nam Cung Mộng tự xưng 'con' gọi 'sư thúc'\n"
    "瑶瑶 (Dao Dao) [nữ, ~20] — sư muội; xưng hô: với Diệp Nhiên tự xưng 'muội' gọi 'sư huynh'\n"
    "南宫梦 (Nam Cung Mộng) [nữ, trung niên] — sư thúc; xưng hô: với Diệp Nhiên tự xưng 'ta' "
    "gọi 'con'"
)


def test_misspelled_listener_resolves_to_the_only_close_character() -> None:
    clauses = {(c.speakers[0], c.raw_listener): c.listeners for c in address_clauses(ROSTER)}
    assert clauses[("叶然", "Diêu Dao")] == ("瑶瑶",)
    rules = address_rule_map(ROSTER)
    assert lookup_address_rule(rules, "叶然", "瑶瑶") == "tự xưng 'sư huynh' gọi 'sư muội'"


def test_unrelated_listener_is_not_guessed() -> None:
    roster = ROSTER.replace("Diêu Dao", "Tống Tiền")
    clauses = {c.raw_listener: c.listeners for c in address_clauses(roster)}
    assert clauses["Tống Tiền"] == ()  # không có ai gần → không đoán
