"""[v3.23.588] Lối đọc số ('vạn' hay nghìn/triệu) là của CẢ PHIM, không của từng con số.

Đo trên bản cuối của 52 lượt dịch thật (ba phim): số DƯỚI một triệu đọc 'vạn' ở 395 dòng,
lối hiện đại ở 132 dòng; số TỪ một triệu đọc lối hiện đại ở 1057 dòng, thuần 'vạn' ở 10
dòng. "Năm mươi vạn tệ" nhưng "hai mươi triệu tệ" là quy ước tự nhiên của bản dịch — nên
lối đọc được xét riêng cho từng hạng độ lớn.

Chuẩn đoán thật v3.23.587 (phim "灭门惨案II", 1246 dòng). Bản dịch thô đọc tiền theo lối
"vạn" ở 9 dòng — "一萬" → "một vạn" (3), "兩萬" → "hai vạn" (1), "二十萬" → "hai mươi vạn"
(5) — riêng dòng 966 model viết "hai trăm nghìn". Lưới số xét RIÊNG con số 200.000: có một
cách đọc hiện đại nên năm dòng 'vạn' bị coi là "lệch lối đọc" và đổi hết sang "hai trăm
nghìn". Bản cuối: "một vạn tệ", "hai vạn tệ" nhưng "hai trăm nghìn" ở 6 dòng.

Cùng lỗi ở 3/13 lượt dịch phim này (13 dòng bị đổi); 10 lượt còn lại đọc 'vạn' trọn phim.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

from subtitles_extractor.application.services.number_consistency_guard import (
    find_number_findings,
    find_number_fixes,
)

SOURCES = {
    657: "這裡是一萬塊",
    658: "借種以後，我再給你一萬元",
    747: "拿兩萬塊錢來贖",
    788: "你給我二十萬",
    789: "有二十萬我兩父子",
    791: "二十萬？你們怎麼不去搶？",
    794: "二十萬買回你三個人的幸福",
    822: "二十萬",
    966: "這裡是二十萬，我老婆呢",
    1082: "這裡是一萬元",
}
# Bản dịch thô thật TRƯỚC khi lưới số chạy (dựng lại từ nhật ký: "788 (lệch lối đọc: hai
# mươi vạn → hai trăm nghìn), 789 (…), 791 (Hai mươi vạn → …), 794 (…), 822 (…)").
LITERAL = {
    657: "Đây là một vạn tệ.",
    658: "Sau khi mượn giống, tôi sẽ đưa thêm một vạn.",
    747: "thì mang hai vạn tệ đến chuộc.",
    788: "Mày đưa tao hai mươi vạn.",
    789: "Có hai mươi vạn, hai cha con tao",
    791: "Hai mươi vạn? Sao tụi mày không đi cướp đi?",
    794: "hai mươi vạn mua lại hạnh phúc ba người chúng mày,",
    822: "Hai mươi vạn.",
    966: "Đây là hai trăm nghìn, vợ tôi đâu?",
    1082: "Đây là một vạn tệ.",
}
TWO_HUNDRED_THOUSAND = (788, 789, 791, 794, 822, 966)


def _apply(
    translations: dict[int, str], prior: dict[int, str] | None = None
) -> dict[int, str]:
    fixed = dict(translations)
    for fix in find_number_fixes(SOURCES, translations, prior):
        fixed[fix.index] = fix.text_after
    return fixed


class TestFilmReadsInVan:
    def test_real_case_one_modern_line_follows_the_film(self) -> None:
        """Trước đây: 5 dòng 'vạn' bị đổi sang "hai trăm nghìn", dòng 966 giữ nguyên."""
        [fix] = find_number_fixes(SOURCES, LITERAL)
        assert (fix.index, fix.reason) == (966, "lệch lối đọc")
        assert fix.text_after == "Đây là hai mươi vạn, vợ tôi đâu?"

    def test_whole_film_ends_up_in_one_style(self) -> None:
        fixed = _apply(LITERAL)
        assert all("vạn" in fixed[index] for index in SOURCES)
        assert not any("nghìn" in text for text in fixed.values())

    def test_capital_letter_is_kept(self) -> None:
        translations = {**LITERAL, 822: "Hai trăm nghìn.", 966: "Đây là hai mươi vạn."}
        assert _apply(translations)[822] == "Hai mươi vạn."

    def test_digits_instead_of_words_is_not_a_style_error(self) -> None:
        """Bản địa hoá thật của lượt này viết "20 vạn" ở dòng 966 — vẫn là lối 'vạn'."""
        translations = {**_apply(LITERAL), 966: "Đây là 20 vạn, vợ tôi đâu?"}
        assert find_number_fixes(SOURCES, translations) == []

    def test_later_stage_cannot_flip_the_film(self) -> None:
        """Giai đoạn sau viết lại MỌI dòng 200.000 theo lối hiện đại: bản trước vẫn là
        cách đọc của phim nên cả phim giữ lối 'vạn'."""
        literal = _apply(LITERAL)
        style = {
            **literal,
            **{i: literal[i].replace("hai mươi vạn", "hai trăm nghìn")
               .replace("Hai mươi vạn", "Hai trăm nghìn") for i in TWO_HUNDRED_THOUSAND},
        }
        fixed = _apply(style, literal)
        assert {i: fixed[i] for i in TWO_HUNDRED_THOUSAND} == {
            i: literal[i] for i in TWO_HUNDRED_THOUSAND
        }

    def test_value_without_a_van_reading_gets_one_built_from_the_value(self) -> None:
        """[v3.23.590] Phim chưa đọc 300.000 theo lối 'vạn' lần nào: dựng từ giá trị.

        Bản 588 để nguyên dòng này ("không tự chế cách đọc") — hệ quả đo được: mỗi số
        tiền một lối trong 17/54 lượt dịch. Xem ``test_number_one_style_v323590``.
        """
        sources = {**SOURCES, 970: "再給你三十萬"}
        translations = {**_apply(LITERAL), 970: "Đưa thêm cho mày ba trăm nghìn."}
        [fix] = find_number_fixes(sources, translations)
        assert (fix.index, fix.text_after) == (970, "Đưa thêm cho mày ba mươi vạn.")


# Phim "he thong sung the" (lượt v3.23.579): tiền triệu đọc lối hiện đại ở hàng chục
# dòng — lối chủ đạo là hiện đại, lưới chạy như trước. Dòng 3440 là ca thật của bản
# 3.23.555; dòng 2875 dựng từ câu thật bằng cách thay "800 nghìn" bằng "80 vạn" (cách
# đọc mà nhật ký một lượt khác ghi lại: "80 vạn → 800 nghìn").
MODERN_SOURCES = {
    304: "一万元已到账",
    1979: "两千万",
    2074: "一千万我都不给啦",
    2132: "一千万我都不给了",
    2134: "我现在只给八百万",
    2873: "八十万",
    2875: "查个数据库要八十万",
    3077: "他给我转了一千万",
    3440: "给了我一千万",
}
MODERN = {
    304: "mười nghìn tệ đã được chuyển vào tài khoản.",
    1979: "Hai mươi triệu,",
    2074: "10 triệu tôi cũng không thèm mua đâu!",
    2132: "Mười triệu tôi cũng không thèm cho nữa!",
    2134: "bây giờ tao chỉ trả tám triệu tệ.",
    2873: "800 nghìn.",
    2875: "Tra cơ sở dữ liệu mà hết 80 vạn à?",
    3077: "em ấy vừa chuyển cho tớ mười triệu,",
    3440: "cho anh một nghìn vạn.",
}


def _modern_fixes(translations: dict[int, str]) -> dict[int, str]:
    fixes = find_number_fixes(MODERN_SOURCES, translations)
    return {fix.index: fix.text_after for fix in fixes}


class TestFilmReadsInModernUnits:
    def test_van_line_still_becomes_modern(self) -> None:
        assert _modern_fixes(MODERN) == {
            2875: "Tra cơ sở dữ liệu mà hết 800 nghìn à?",
            3440: "cho anh mười triệu.",
        }

    def test_modern_lines_are_not_turned_into_van(self) -> None:
        translations = {**MODERN, 2873: "80 vạn.", 2875: "hết 800 nghìn à?"}
        fixes = _modern_fixes(translations)
        assert fixes[2873] == "800 nghìn."
        assert 2875 not in fixes

    def test_tie_follows_the_convention_of_the_magnitude(self) -> None:
        """[v3.23.590] Bằng phiếu: dưới một triệu về 'vạn', từ một triệu về hiện đại.

        Bản 588 cho lối hiện đại thắng mọi trận hoà; lượt chạy thật kế tiếp hoà đúng 4-4
        ở hạng dưới một triệu. Xem ``test_number_one_style_v323590``.
        """
        sources = {1: "二十萬", 2: "二十萬"}
        translations = {1: "hai mươi vạn.", 2: "hai trăm nghìn."}
        [fix] = find_number_fixes(sources, translations)
        assert (fix.index, fix.text_after) == (2, "hai mươi vạn.")
        sources = {1: "兩百萬", 2: "兩百萬"}
        translations = {1: "hai trăm vạn.", 2: "hai triệu."}
        [fix] = find_number_fixes(sources, translations)
        assert (fix.index, fix.text_after) == (1, "hai triệu.")


# Phim "he thong sung the" (lượt v3.23.578, nhật ký ghi "126 (lệch lối đọc: hai vạn → 20
# nghìn), 357 (…), 419 (…)"): bản dịch thô TRƯỚC lưới số. Cả lượt có 16 dòng dưới một
# triệu đọc 'vạn', 11 dòng đọc lối hiện đại; tiền triệu thì đọc lối hiện đại hết.
MIXED_SOURCES = {
    126: "两万元已到账",
    168: "四十万元已到账",
    169: "四十万",
    284: "十万元已到账",
    357: "之前借我两万块钱",
    419: "就借林逸两万块钱",
    633: "四十万",
    825: "那两万块钱",
    835: "两万块钱",
    895: "一共九十六万",
    1979: "两千万",
    2134: "我现在只给八百万",
    3354: "五十万起吧",
    5097: "五十万的感谢费",
    5258: "这里有五十万",
}
MIXED_LITERAL = {
    126: "hai vạn tệ đã về tài khoản.",
    168: "Bốn mươi vạn tệ đã đến tài khoản.",
    169: "Bốn mươi vạn?",
    284: "Một trăm nghìn tệ đã được chuyển vào tài khoản.",
    357: "Hôm trước mượn mày hai vạn tệ,",
    419: "mà dám cho Lâm Dật vay hai vạn tệ hả?",
    633: "bốn mươi vạn.",
    825: "20 nghìn tệ kia",
    835: "20 nghìn tệ",
    895: "Tổng cộng là 960 nghìn tệ.",
    1979: "Hai mươi triệu,",
    2134: "bây giờ tao chỉ trả tám triệu.",
    3354: "cũng phải từ năm mươi vạn trở lên ấy nhỉ?",
    5097: "năm mươi vạn tiền cảm ơn",
    5258: "Ở đây có năm mươi vạn,",
}


class TestStyleIsJudgedPerMagnitude:
    def test_real_case_small_amounts_follow_the_van_majority(self) -> None:
        """Trước đây: ba dòng "hai vạn" (126, 357, 419) bị đổi sang "20 nghìn" trong khi
        "bốn mươi vạn", "năm mươi vạn" của cùng phim giữ nguyên."""
        fixes = find_number_fixes(MIXED_SOURCES, MIXED_LITERAL)
        # [v3.23.590] Kèm hai số tiền phim chưa đọc 'vạn' lần nào (284, 895).
        assert {fix.index: fix.text_after for fix in fixes} == {
            284: "Mười vạn tệ đã được chuyển vào tài khoản.",
            825: "hai vạn tệ kia",
            835: "hai vạn tệ",
            895: "Tổng cộng là 96 vạn tệ.",
        }

    def test_millions_do_not_vote_on_small_amounts(self) -> None:
        """Hàng chục dòng tiền triệu đọc lối hiện đại không biến phim thành "lối hiện
        đại" cho tiền vạn."""
        millions = {index: "两千万" for index in range(9000, 9030)}
        sources = {**MIXED_SOURCES, **millions}
        translations = {**MIXED_LITERAL, **dict.fromkeys(millions, "hai mươi triệu.")}
        fixed = {fix.index for fix in find_number_fixes(sources, translations)}
        assert fixed == {284, 825, 835, 895}

    def test_small_van_amounts_do_not_turn_millions_into_van(self) -> None:
        sources = {**MIXED_SOURCES, 3440: "给了我一千万"}
        translations = {**MIXED_LITERAL, 3440: "cho anh một nghìn vạn."}
        fixes = {f.index: f.text_after for f in find_number_fixes(sources, translations)}
        assert 1979 not in fixes
        # [v3.23.590] Phim chưa đọc 10 triệu theo lối hiện đại lần nào: dựng từ giá trị.
        assert fixes[3440] == "cho anh mười triệu."

    def test_built_reading_keeps_digits_as_digits_and_words_as_words(self) -> None:
        fixes = {f.index: f for f in find_number_fixes(MIXED_SOURCES, MIXED_LITERAL)}
        assert (fixes[284].phrase_before, fixes[284].phrase_after) == (
            "Một trăm nghìn", "Mười vạn"
        )
        assert (fixes[895].phrase_before, fixes[895].phrase_after) == (
            "960 nghìn", "96 vạn"
        )


class TestFindings:
    def test_modern_line_in_a_van_film_is_reported(self) -> None:
        [finding] = find_number_findings(SOURCES, LITERAL)
        assert (finding.index, finding.kind) == (966, "standard_unit")

    def test_consistent_van_film_has_no_findings(self) -> None:
        assert find_number_findings(SOURCES, _apply(LITERAL)) == []

    def test_van_line_in_a_modern_film_keeps_its_kind(self) -> None:
        kinds = {
            item.index: item.kind
            for item in find_number_findings(MODERN_SOURCES, MODERN)
        }
        assert kinds == {2875: "non_standard_unit", 3440: "non_standard_unit"}
