"""[v3.23.562] Hồi quy: chốt "vị trí sở hữu" của lưới xưng hô chặn oan.

Chuẩn đoán thật v3.23.562 (Visual Cues đã phủ 100% bằng tên nhân vật): 24 dòng lệch quy
tắc xưng hô, 9 dòng bị bỏ qua với lý do "đại từ ở vị trí sở hữu". Hai nguyên nhân:

1. Nhận "tên riêng viết hoa" bằng lớp ký tự ``[A-ZĐÀ-Ỹ]`` — dải ``À-Ỹ`` chứa cả chữ THƯỜNG
   có dấu, nên "còn ép tôi", "đánh tôi", "ở tôi" đều bị coi là "sau tên riêng".
2. Mọi dạng sở hữu đều bị chặn, kể cả "nhà tôi", "mẹ tôi" ở dòng tự xưng — vốn chỉ có thể
   là của người nói, đổi theo cách xưng của họ vẫn giữ nguyên nghĩa.
"""

from __future__ import annotations

import json

import pytest

from subtitles_extractor.application.services.address_rule_guard import (
    find_address_rule_fixes,
    is_possessive_position,
    possessive_swap_is_safe,
)

CHARACTERS = (
    "宋钱 (Tống Tiền) [nam, ~22] — nhân vật chính\n"
    "张依贝 (Trương Y Bối) [nữ, ~20] — hàng xóm; xưng hô: với Tống Tiền tự xưng 'em' "
    "gọi 'anh'\n"
)


class TestProperNounDetection:
    @pytest.mark.parametrize(
        "text",
        ["còn ép tôi phải bỏ học cấp ba,", "hắn đánh tôi", "ở tôi mãi", "ông ấy ưa tôi"],
    )
    def test_lowercase_accented_word_is_not_a_proper_noun(self, text: str) -> None:
        assert is_possessive_position(text, "tôi") is False

    def test_real_proper_noun_still_counts(self) -> None:
        assert is_possessive_position("Tốt tốt, nhà họ Liễu ta đúng là", "ta") is True


class TestPossessiveSwapSafety:
    @pytest.mark.parametrize(
        ("text", "found", "expected"),
        [
            ("ở phòng khám nhà tôi", "tôi", "em"),
            ("phòng khám đó là do bố tôi mở,", "tôi", "em"),
            ("dàn cảnh ăn vạ tiểu thư nhà tao.", "tao", "tôi"),
            ("đánh gãy chân mẹ tôi,", "tôi", "tao"),
        ],
    )
    def test_real_cases_are_safe(self, text: str, found: str, expected: str) -> None:
        assert possessive_swap_is_safe(text, found, expected) is True

    @pytest.mark.parametrize(
        ("text", "found", "expected"),
        [
            ("anh tôi đi làm rồi", "tôi", "em"),  # → "anh em" = anh em ruột
            ("mẹ tôi gọi đấy", "tôi", "con"),  # → "mẹ con" = mẹ và con
            ("chị tôi bảo thế", "tôi", "chị"),  # → "chị chị"
            ("nhà ta ở đây", "ta", "tôi"),  # 'ta' có thể là số nhiều
            ("nhà mình nghèo", "mình", "em"),  # 'mình' có thể là số nhiều
            ("hai tao đi", "tao", "tôi"),  # số từ đứng trước
            ("Tống Tiền tôi nói là làm", "tôi", "em"),  # sau tên riêng
        ],
    )
    def test_meaning_changing_swaps_are_blocked(
        self, text: str, found: str, expected: str
    ) -> None:
        assert possessive_swap_is_safe(text, found, expected) is False


def _cues(first: int, count: int) -> str:
    return json.dumps(
        [{"id": first + i, "spk": "张依贝", "to": "宋钱"} for i in range(count)],
        ensure_ascii=False,
    )


class TestEndToEnd:
    def test_real_line_666_is_fixed(self) -> None:
        """Ca thật dòng 666: "nhà tôi" giữa cảnh Y Bối xưng "em" với Tống Tiền."""
        sources = {i: "我知道" for i in range(660, 667)}
        translations = {i: "em biết mà" for i in range(660, 666)}
        sources[666] = "就在我家诊所"
        translations[666] = "ở phòng khám nhà tôi"
        fixes = find_address_rule_fixes(sources, translations, _cues(660, 7), CHARACTERS)
        assert [(f.index, f.text_after) for f in fixes] == [(666, "ở phòng khám nhà em")]

    def test_ambiguous_kinship_possessive_is_left_alone(self) -> None:
        sources = {i: "我知道" for i in range(660, 667)}
        translations = {i: "em biết mà" for i in range(660, 666)}
        sources[666] = "我哥上班了"
        translations[666] = "anh tôi đi làm rồi"
        assert (
            find_address_rule_fixes(sources, translations, _cues(660, 7), CHARACTERS)
            == []
        )
