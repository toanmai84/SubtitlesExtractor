"""[v3.23.590] Một phim một lối đọc số: không còn "hai vạn tệ" bên cạnh "hai trăm nghìn".

Chuẩn đoán thật v3.23.589 (phim "灭门惨案II", phụ đề PHỒN THỂ). Bản dịch thô trước lưới số
(dựng lại từ nhật ký "822 (sai giá trị: Hai mươi nghìn → Hai trăm nghìn), 966 (lệch lối
đọc: hai mươi vạn → hai trăm nghìn)"):

* "一萬" → "một vạn" (3 dòng), "二十萬" → "hai mươi vạn" (dòng 966): 4 dòng lối 'vạn';
* "二十萬" → "hai trăm nghìn" (4 dòng): 4 dòng lối hiện đại;
* "二十萬" → "Hai mươi nghìn" (dòng 822) và "兩萬" → "hai mươi vạn tệ" (dòng 747): hai
  dòng SAI GIÁ TRỊ.

Ba lỗi của lưới số trên lượt đó:

1. ``兩`` (phồn thể của ``两``) không có trong bảng chữ số: dòng 747 đọc 200.000 thay cho
   20.000 mà lưới không thấy (giai đoạn sau tình cờ sửa lại).
2. Bằng phiếu 4-4 thì lối hiện đại thắng: dòng 966 đang đúng bị đổi thành "hai trăm
   nghìn". Bản cuối: "một vạn tệ", "hai vạn tệ" nhưng "hai trăm nghìn" ở 6 dòng.
3. Dòng lệch lối chỉ được sửa khi phim ĐÃ có cách đọc đúng lối cho chính con số đó, nên
   mỗi số tiền một lối. Đo trên bản cuối của 54 lượt dịch thật (ba phim): 17 lượt đọc lẫn
   hai lối trong hạng dưới một triệu, 107 dòng lệch.

Mọi dòng dữ liệu dưới đây chép nguyên văn từ file chuẩn đoán.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

from typing import Any

import pytest

from subtitles_extractor.application.services.number_consistency_guard import (
    chinese_number_value,
    find_number_findings,
    find_number_fixes,
    vietnamese_number_value,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

# Lượt v3.23.589: (câu gốc, bản dịch thô TRƯỚC lưới số).
RANSOM = {
    657: ("這裡是一萬塊", "Đây là một vạn tệ."),
    658: ("借種以後，我再給你一萬元", "Mượn giống xong, anh sẽ đưa thêm một vạn nữa."),
    747: ("拿兩萬塊錢來贖", "thì mang hai mươi vạn tệ đến chuộc."),
    752: (
        "一兩萬也沒有你當我白痴嗎",
        "Ngay cả một hai vạn mà cũng không có, mày tưởng tao là đồ ngốc chắc?",
    ),
    788: ("你給我二十萬", "Mày đưa tao hai trăm nghìn."),
    789: ("有二十萬我兩父子", "Có hai trăm nghìn, hai cha con tao"),
    791: ("二十萬？你們怎麼不去搶？", "Hai trăm nghìn? Sao tụi mày không đi cướp?"),
    794: (
        "二十萬買回你三個人的幸福",
        "hai trăm nghìn mua lại hạnh phúc ba người các mày,",
    ),
    822: ("二十萬", "Hai mươi nghìn."),
    966: ("這裡是二十萬，我老婆呢", "Đây là hai mươi vạn, vợ tôi đâu?"),
    1082: ("這裡是一萬元", "Đây là một vạn tệ."),
}

# Phim "he thong sung the" (5540 dòng), BẢN CUỐI của lượt v3.23.583: hạng dưới một triệu
# đọc 'vạn' ở 16 dòng, lối hiện đại ở 12 dòng — "bốn mươi vạn", "năm mươi vạn" nhưng "20
# nghìn tệ", "800 nghìn". Kèm vài dòng tiền triệu thật của cùng bản.
FACTORY = {
    126: ("两万元已到账", "20 nghìn tệ đã đến tài khoản."),
    168: ("四十万元已到账", "Bốn mươi vạn tệ đã đến tài khoản."),
    169: ("四十万", "Bốn mươi vạn?"),
    192: ("四十万", "Bốn mươi vạn,"),
    284: ("十万元已到账", "100 nghìn tệ đã được chuyển vào tài khoản."),
    304: ("一万元已到账", "10 nghìn tệ đã được chuyển vào tài khoản."),
    357: ("之前借我两万块钱", "Trước đó tao mượn mày 20 nghìn tệ,"),
    419: ("就借林逸两万块钱", "mà cho Lâm Dật vay 20 nghìn tệ à?"),
    614: ("二十万", "Hai mươi vạn?"),
    633: ("四十万", "bốn mươi vạn."),
    634: ("四十万", "Bốn mươi vạn."),
    656: ("三十四点五万元", "là ba mươi tư phẩy năm vạn tệ."),
    663: ("消费三十四点五万元", "34,5 vạn tệ,"),
    667: ("三百四十五万", "Ba trăm bốn mươi lăm vạn..."),
    799: ("这三万你拿着", "3 vạn này mày cầm lấy đi."),
    825: ("那两万块钱", "20 nghìn tệ đó"),
    835: ("两万块钱", "20 nghìn tệ"),
    840: ("三百八十万", "Ba triệu tám trăm nghìn á?"),
    895: ("一共九十六万", "Tổng cộng chín mươi sáu vạn."),
    1979: ("两千万", "Hai mươi triệu."),
    2074: ("一千万我都不给啦", "mười triệu tệ tôi cũng chẳng thèm bố thí."),
    2134: ("我现在只给八百万", "bây giờ tao chỉ mua lại với giá tám triệu."),
    2873: ("八十万", "800 nghìn."),
    2875: ("查个数据库要八十万", "Tra cơ sở dữ liệu mà mất 800 nghìn á?"),
    2898: ("就要八十万", "thôi mà đã mất 800 nghìn rồi."),
    3052: ("花了八十万", "đã tốn 800 nghìn."),
    3054: ("花了八十万", "Tốn 800 nghìn cơ à?"),
    3354: ("五十万起吧", "cũng phải năm mươi vạn chứ nhỉ?"),
    5097: ("五十万的感谢费", "năm mươi vạn tiền cảm ơn"),
    5258: ("这里有五十万", "Ở đây có năm mươi vạn,"),
    5276: ("要不收这五十万的话", "nếu mày không nhận năm mươi vạn này,"),
    5333: ("这五十万的赔偿金", "Năm mươi vạn tiền bồi thường này"),
    5420: ("这五十万", "năm mươi vạn này"),
}
FACTORY_MODERN_SMALL = (126, 284, 304, 357, 419, 825, 835, 2873, 2875, 2898, 3052, 3054)

# Phim "傻子落水" (3902 dòng), bản cuối của lượt v3.23.551: tiền triệu đọc lối hiện đại,
# riêng "一千万" ra "một nghìn vạn" ở MỌI dòng nên phim không có mẫu "mười triệu" nào.
WINDFALL = {
    1156: ("一百万", "cũng phải một triệu."),
    1615: (
        "银行到账二百万元",
        "Ngân hàng đã thông báo tài khoản nhận được hai triệu tệ.",
    ),
    2432: (
        "这车落地至少五百万",
        "Giá lăn bánh của chiếc xe này ít nhất cũng phải năm triệu.",
    ),
    2338: ("这张卡里有一千万", "Trong thẻ này có một nghìn vạn,"),
    3377: ("给了我一千万", "còn cho anh một nghìn vạn."),
    3378: ("一千万?", "Một nghìn vạn?"),
    3388: ("你挣到二百万就够多了", "em kiếm được hai trăm vạn là đã nhiều lắm rồi,"),
    3392: ("又多了一千万", "lại có thêm một nghìn vạn."),
}


def _split(rows: dict[int, tuple[str, str]]) -> tuple[dict[int, str], dict[int, str]]:
    sources = {index: source for index, (source, _text) in rows.items()}
    translations = {index: text for index, (_source, text) in rows.items()}
    return sources, translations


def _fixed(rows: dict[int, tuple[str, str]]) -> dict[int, str]:
    sources, translations = _split(rows)
    fixed = dict(translations)
    for fix in find_number_fixes(sources, translations):
        fixed[fix.index] = fix.text_after
    return fixed


class TestTraditionalTwo:
    @pytest.mark.parametrize(
        ("expression", "value"),
        [("兩萬", 20_000), ("兩千萬", 20_000_000), ("兩億", 200_000_000)],
    )
    def test_value(self, expression: str, value: int) -> None:
        assert chinese_number_value(expression) == value

    def test_real_wrong_amount_is_caught(self) -> None:
        """Dòng 747: "兩萬" (20.000) dịch thô ra "hai mươi vạn tệ" (200.000)."""
        sources, translations = _split(RANSOM)
        fix = next(f for f in find_number_fixes(sources, translations) if f.index == 747)
        assert fix.reason == "sai giá trị"
        assert fix.text_after == "thì mang hai vạn tệ đến chuộc."

    def test_two_inside_another_word_is_not_part_of_the_amount(self) -> None:
        """Dòng 789 "有二十萬我兩父子": chữ 兩 của "hai cha con" đứng ngoài cụm số."""
        fixed = _fixed(RANSOM)
        assert fixed[789] == "Có hai mươi vạn, hai cha con tao"


class TestApproximateNumbers:
    def test_real_line_is_left_alone(self) -> None:
        """Dòng 752 "一兩萬" → "một hai vạn": số ước lượng ở cả hai phía."""
        sources, translations = _split(RANSOM)
        assert _fixed(RANSOM)[752] == translations[752]
        assert 752 not in {f.index for f in find_number_findings(sources, translations)}

    @pytest.mark.parametrize(
        "source", ["一两万也没有", "七八万块", "三四十万", "十七八万"]
    )
    def test_adjacent_source_digits_are_not_an_exact_amount(self, source: str) -> None:
        """Trước đây "一两万" tính là 20.000 → "một hai vạn" bị coi là sai giá trị."""
        translations = {1: "một hai vạn cũng không có", 2: "hai vạn tệ"}
        sources = {1: source, 2: "两万块"}
        assert find_number_fixes(sources, translations) == []

    @pytest.mark.parametrize("phrase", ["một hai vạn", "bảy tám vạn", "ba bốn mươi vạn"])
    def test_adjacent_spoken_digits_have_no_exact_value(self, phrase: str) -> None:
        assert vietnamese_number_value(phrase) is None

    @pytest.mark.parametrize(
        ("phrase", "value"),
        [("hai mươi mốt vạn", 210_000), ("một trăm linh năm nghìn", 105_000),
         ("ba mươi bốn phẩy năm vạn", 345_000), ("mười lăm vạn", 150_000)],
    )
    def test_ordinary_readings_keep_their_value(self, phrase: str, value: int) -> None:
        assert vietnamese_number_value(phrase) == value

    def test_decimal_digits_are_not_adjacent_digits(self) -> None:
        sources = {1: "三十四点五五万"}
        translations = {1: "34,55 vạn tệ"}
        assert chinese_number_value("三十四点五五万") == 345_500
        assert find_number_findings(sources, translations) == []


class TestZeroHundreds:
    def test_zero_hundreds_reads_as_zero(self) -> None:
        """Trước đây "không trăm" tính như "một trăm": 1.020.000 đọc ra 1.120.000."""
        assert vietnamese_number_value("một triệu không trăm hai mươi nghìn") == 1_020_000

    def test_wrong_amount_gets_the_standard_reading(self) -> None:
        sources = {1: "一百零二万"}
        translations = {1: "một triệu hai trăm nghìn tệ"}
        [fix] = find_number_fixes(sources, translations)
        assert fix.text_after == "một triệu không trăm hai mươi nghìn tệ"


class TestRealTie:
    def test_film_ends_up_in_one_style(self) -> None:
        fixed = _fixed(RANSOM)
        assert {index: fixed[index] for index in (788, 791, 794, 822, 966)} == {
            788: "Mày đưa tao hai mươi vạn.",
            791: "Hai mươi vạn? Sao tụi mày không đi cướp?",
            794: "hai mươi vạn mua lại hạnh phúc ba người các mày,",
            822: "Hai mươi vạn.",
            966: "Đây là hai mươi vạn, vợ tôi đâu?",
        }
        assert not any("nghìn" in text for text in fixed.values())

    def test_correct_van_line_is_no_longer_flipped(self) -> None:
        """Bản 589 đổi dòng 966 thành "hai trăm nghìn" vì hoà phiếu."""
        sources, translations = _split(RANSOM)
        assert 966 not in {fix.index for fix in find_number_fixes(sources, translations)}

    def test_reasons(self) -> None:
        sources, translations = _split(RANSOM)
        reasons = {f.index: f.reason for f in find_number_fixes(sources, translations)}
        assert reasons == {
            747: "sai giá trị", 788: "lệch lối đọc", 789: "lệch lối đọc",
            791: "lệch lối đọc", 794: "lệch lối đọc", 822: "sai giá trị",
        }

    def test_second_pass_changes_nothing(self) -> None:
        sources, _ = _split(RANSOM)
        fixed = _fixed(RANSOM)
        assert find_number_fixes(sources, fixed) == []
        assert find_number_findings(sources, fixed) == []


class TestEveryAmountFollowsTheFilm:
    def test_amounts_without_a_van_reading_get_one(self) -> None:
        fixed = _fixed(FACTORY)
        assert {index: fixed[index] for index in FACTORY_MODERN_SMALL} == {
            126: "2 vạn tệ đã đến tài khoản.",
            284: "10 vạn tệ đã được chuyển vào tài khoản.",
            304: "1 vạn tệ đã được chuyển vào tài khoản.",
            357: "Trước đó tao mượn mày 2 vạn tệ,",
            419: "mà cho Lâm Dật vay 2 vạn tệ à?",
            825: "2 vạn tệ đó",
            835: "2 vạn tệ",
            2873: "80 vạn.",
            2875: "Tra cơ sở dữ liệu mà mất 80 vạn á?",
            2898: "thôi mà đã mất 80 vạn rồi.",
            3052: "đã tốn 80 vạn.",
            3054: "Tốn 80 vạn cơ à?",
        }

    def test_lines_already_in_the_film_style_are_untouched(self) -> None:
        _, translations = _split(FACTORY)
        fixed = _fixed(FACTORY)
        changed = {index for index in FACTORY if fixed[index] != translations[index]}
        assert changed == {*FACTORY_MODERN_SMALL, 667}

    def test_millions_keep_modern_units_and_lose_the_stray_van(self) -> None:
        fixed = _fixed(FACTORY)
        assert fixed[667] == "Ba triệu bốn trăm năm mươi nghìn..."
        assert fixed[1979] == "Hai mươi triệu."

    def test_mixed_reading_without_a_modern_model(self) -> None:
        """"一千万" đọc "một nghìn vạn" ở mọi dòng: nay đọc lại là "mười triệu"."""
        fixed = _fixed(WINDFALL)
        assert fixed[2338] == "Trong thẻ này có mười triệu,"
        assert fixed[3378] == "Mười triệu?"
        assert fixed[3392] == "lại có thêm mười triệu."
        assert fixed[3388] == "em kiếm được hai triệu là đã nhiều lắm rồi,"
        assert not any("vạn" in text for text in fixed.values())

    def test_second_pass_changes_nothing(self) -> None:
        for rows in (FACTORY, WINDFALL):
            sources, _ = _split(rows)
            fixed = _fixed(rows)
            assert find_number_fixes(sources, fixed) == []
            assert find_number_findings(sources, fixed) == []


class TestBuiltReadingBoundaries:
    """Biên DỰNG THÊM quanh các dòng thật (kho chưa có ca nào như vậy)."""

    def test_odd_amount_has_no_clean_van_reading_and_is_left_alone(self) -> None:
        text = "Đưa thêm một trăm hai mươi ba nghìn."
        rows = {**FACTORY, 9000: ("再给你十二点三万", text)}
        assert _fixed(rows)[9000] == text

    def test_half_van_is_read_as_ruoi(self) -> None:
        text = "Đưa thêm một trăm hai mươi lăm nghìn."
        rows = {**FACTORY, 9000: ("再给你十二点五万", text)}
        assert _fixed(rows)[9000] == "Đưa thêm mười hai vạn rưỡi."

    def test_digits_stay_digits(self) -> None:
        rows = {**FACTORY, 9000: ("再给你十二点三万", "Đưa thêm 123 nghìn.")}
        assert _fixed(rows)[9000] == "Đưa thêm 12,3 vạn."

    def test_capital_letter_is_kept(self) -> None:
        assert _fixed(RANSOM)[791].startswith("Hai mươi vạn?")

    def test_clear_majority_still_decides(self) -> None:
        """Phim đọc lối hiện đại rõ ràng thì "một vạn" mới là dòng lệch."""
        rows = {index: RANSOM[index] for index in (788, 789, 791, 794, 1082)}
        fixed = _fixed(rows)
        assert fixed[1082] == "Đây là mười nghìn tệ."
        assert fixed[788] == "Mày đưa tao hai trăm nghìn."


def _events(name: str) -> list[dict[str, Any]]:
    return [e for e in get_process_log().snapshot()["events"] if e["event"] == name]


class TestUseCase:
    def test_stage_output_is_normalised_and_recorded(self) -> None:
        get_process_log().clear()
        sources = [
            TranslationLine(index=i, start_ms=i, end_ms=i + 1, text=s, original_text=s)
            for i, (s, _t) in RANSOM.items()
        ]
        stage = [
            TranslationLine(index=i, start_ms=i, end_ms=i + 1, text=t, original_text=s)
            for i, (s, t) in RANSOM.items()
        ]
        result = TranslateSubtitlesUseCase._enforce_number_consistency(
            stage, sources, "Giai đoạn 2: Dịch thô sát nghĩa"
        )
        assert {line.index: line.text for line in result} == _fixed(RANSOM)
        [event] = _events("numbers_normalised")
        assert [fix["index"] for fix in event["fixes"]] == [747, 788, 789, 791, 794, 822]
        assert event["fixes"][0] == {
            "index": 747,
            "reason": "sai giá trị",
            "before": "hai mươi vạn",
            "after": "hai vạn",
        }
