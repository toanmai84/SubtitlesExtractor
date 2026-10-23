"""[v3.23.582] Lưới số: họ "Triệu", "mức" ≠ "ức", bài đếm hàng, cụm chữ số gõ nhầm.

Chuẩn đoán thật v3.23.581 (phim "he thong sung the", 5540 dòng):

* Bản dịch thô lệch câu ở 662-663 ("为目标赵红娟 / 消费三十四点五万元" → "ký chủ tiêu
  345.000 tệ / cho mục tiêu Triệu Hồng Quyên,"). Họ 赵 = "Triệu" bị nhận là con số
  1.000.000, sai giá trị so với 345.000 → lưới đọc chuẩn của v3.23.581 thay họ người thành
  "Ba trăm bốn mươi lăm nghìn Hồng Quyên".
* Dòng 667 "三百四十五万": khâu văn phong gõ "3.455 vạn tệ..." (34,55 triệu) ngay dưới
  dòng 665 "345 vạn tệ" — sai giá trị nhưng không sửa được vì cụm viết bằng chữ số.
* Toàn bộ 15 lượt dịch thật của phim kia có "我愿意出百万年薪" → "mức lương năm triệu tệ"
  đều bị bỏ qua: "ức" trong "mức" được tính là cụm số thứ hai.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from subtitles_extractor.application.services.number_consistency_guard import (
    find_number_findings,
    find_number_fixes,
    format_vietnamese_digits,
)


class TestUnitWordAloneIsNotANumber:
    def test_real_surname_line_is_left_alone(self) -> None:
        # Dòng 656 cho phim một cách đọc đúng của 345.000 — đúng điều kiện đã làm
        # v3.23.581 thay họ người.
        sources = {
            656: "三十四点五万元", 662: "为目标赵红娟", 663: "消费三十四点五万元"
        }
        translations = {
            656: "là ba trăm bốn mươi lăm nghìn tệ.",
            662: "ký chủ tiêu 345.000 tệ",
            663: "cho mục tiêu Triệu Hồng Quyên,",
        }
        assert [fix.index for fix in find_number_fixes(sources, translations)] == []
        assert [item.index for item in find_number_findings(sources, translations)] == []

    def test_surname_without_any_film_reading(self) -> None:
        sources = {662: "为目标赵红娟", 663: "消费三十四点五万元"}
        translations = {
            662: "ký chủ tiêu 345.000 tệ", 663: "cho mục tiêu Triệu Hồng Quyên,"
        }
        assert find_number_fixes(sources, translations) == []

    def test_millionaire_is_not_a_number(self) -> None:
        findings = find_number_findings(
            {670: "也是百万富翁了"}, {670: "cũng là triệu phú rồi."}
        )
        assert findings == []


SALARY_SOURCES = {2047: "我愿意出百万年薪", 2050: "百万年薪"}


class TestWholeWordsOnly:
    def test_real_salary_line_is_fixed(self) -> None:
        translations = {
            2047: "Tôi sẵn sàng đưa ra mức lương năm triệu tệ,",
            2050: "lương một triệu một năm",
        }
        fixes = find_number_fixes(SALARY_SOURCES, translations)
        assert [(fix.index, fix.text_after) for fix in fixes] == [
            (2047, "Tôi sẵn sàng đưa ra mức lương một triệu tệ,")
        ]

    def test_real_salary_line_is_reported(self) -> None:
        findings = find_number_findings(
            SALARY_SOURCES, {2047: "Tôi sẵn sàng trả mức lương năm triệu tệ,"}
        )
        assert [(item.index, item.found_phrase, item.kind) for item in findings] == [
            (2047, "năm triệu", "mismatch")
        ]

    @pytest.mark.parametrize("word", ["mức", "sức", "chức", "tỉnh"])
    def test_unit_inside_a_word_is_ignored(self, word: str) -> None:
        findings = find_number_findings({1: "十万"}, {1: f"{word} mười vạn"})
        assert [item.kind for item in findings] == []


# Dòng 2305-2307 thật: nhân vật đếm hàng chữ số trên tờ séc.
CHANT_SOURCES = {
    284: "十万元已到账", 2305: "这到底是不是两千万", 2306: "个十百千万", 2307: "十万百千"
}
CHANT_TRANSLATIONS = {
    284: "Một trăm nghìn tệ đã vào tài khoản.",
    2305: "đây rốt cuộc có phải 20 triệu Tệ không!",
    2306: "Một, mười, trăm, nghìn, vạn...",
    2307: "Trăm vạn, nghìn vạn...",
}


class TestCountingChant:
    def test_chant_fragment_is_not_rewritten(self) -> None:
        fixes = find_number_fixes(CHANT_SOURCES, CHANT_TRANSLATIONS)
        assert [fix.index for fix in fixes if fix.index == 2307] == []

    def test_chant_fragment_is_not_reported(self) -> None:
        findings = find_number_findings(CHANT_SOURCES, CHANT_TRANSLATIONS)
        assert [item.index for item in findings if item.index == 2307] == []


TYPO_SOURCES = {665: "三百四十五万元", 666: "已到账", 667: "三百四十五万"}
TYPO_LITERAL = {
    665: "nhận về 3.450.000 tệ,", 666: "đã vào tài khoản.", 667: "3.450.000 tệ..."
}
TYPO_STYLE = {665: "345 vạn tệ", 666: "đã vào tài khoản.", 667: "3.455 vạn tệ..."}


class TestDigitPhraseTypo:
    def test_real_typo_is_fixed_in_the_film_writing(self) -> None:
        fixes = find_number_fixes(
            TYPO_SOURCES, TYPO_STYLE, prior_translations=TYPO_LITERAL
        )
        assert [(fix.index, fix.text_after, fix.reason) for fix in fixes] == [
            (667, "345 vạn tệ...", "sai giá trị")
        ]

    def test_decimal_rewrite_keeps_the_unit(self) -> None:
        fixes = find_number_fixes({1: "三百四十五万"}, {1: "tận 3,4 triệu tệ"})
        assert [fix.text_after for fix in fixes] == ["tận 3,45 triệu tệ"]

    def test_unwritable_value_is_left_alone(self) -> None:
        # 333.333 = 0,333333 triệu: không viết gọn bằng ≤ 3 chữ số thập phân → không đoán.
        sources = {1: "三十三万三千三百三十三"}
        assert find_number_fixes(sources, {1: "tận 3 triệu tệ"}) == []


class TestFormatVietnameseDigits:
    @pytest.mark.parametrize(
        ("amount", "written"),
        [
            (Fraction(345), "345"),
            (Fraction(3450), "3.450"),
            (Fraction(3_450_000), "3.450.000"),
            (Fraction(69, 20), "3,45"),
            (Fraction(1, 2), "0,5"),
        ],
    )
    def test_values(self, amount: Fraction, written: str) -> None:
        assert format_vietnamese_digits(amount) == written

    @pytest.mark.parametrize("amount", [Fraction(1, 3), Fraction(0), Fraction(-5)])
    def test_unwritable(self, amount: Fraction) -> None:
        assert format_vietnamese_digits(amount) is None
