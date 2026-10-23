"""[v3.23.587] Sửa con số sai giá trị theo CHÍNH lối đọc của phim, kể cả lối "vạn".

Chuẩn đoán thật v3.23.586 (phim "灭门惨案II", 1246 dòng). Phim đọc tiền theo lối "vạn"
từ đầu tới cuối: "一萬" → "một vạn", "兩萬" → "hai vạn", năm dòng "二十萬" → "hai mươi
vạn". Riêng dòng 966 bản dịch thô nhầm thành "hai trăm vạn" (2 triệu). Lưới số sửa đúng
giá trị nhưng bằng cách đọc chuẩn MỚI "hai trăm nghìn"; sang giai đoạn văn phong, chính
cụm mới đó thành "cách đọc chuẩn của phim" cho 200.000 và kéo cả năm dòng ĐÚNG đổi theo
("lệch lối đọc: hai mươi vạn → hai trăm nghìn" ở 6 dòng) — phim thành ra "hai vạn" chỗ
này, "hai trăm nghìn" chỗ kia.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

from subtitles_extractor.application.services.number_consistency_guard import (
    find_number_fixes,
)

# Câu gốc và bản dịch THÔ thật của lượt chạy (trước khi lưới số sửa dòng 966).
SOURCES = {
    657: "這裡是一萬塊",
    747: "拿兩萬塊錢來贖",
    788: "你給我二十萬",
    789: "有二十萬我兩父子",
    791: "二十萬？你們怎麼不去搶？",
    794: "二十萬買回你三個人的幸福",
    822: "二十萬",
    966: "這裡是二十萬，我老婆呢",
}
LITERAL = {
    657: "Đây là một vạn tệ.",
    747: "thì mang hai vạn tệ đến chuộc.",
    788: "Mày đưa tao hai mươi vạn.",
    789: "Có hai mươi vạn, hai cha con tao",
    791: "Hai mươi vạn? Sao tụi mày không đi cướp đi?",
    794: "hai mươi vạn mua lại hạnh phúc ba người các mày,",
    822: "hai mươi vạn.",
    966: "Đây là hai trăm vạn, vợ tôi đâu?",
}


def _apply(
    translations: dict[int, str], prior: dict[int, str] | None = None
) -> dict[int, str]:
    fixed = dict(translations)
    for fix in find_number_fixes(SOURCES, translations, prior):
        fixed[fix.index] = fix.text_after
    return fixed


class TestWrongValueFollowsTheFilm:
    def test_real_case_uses_the_films_own_reading(self) -> None:
        [fix] = find_number_fixes(SOURCES, LITERAL)
        assert (fix.index, fix.reason) == (966, "sai giá trị")
        # Trước đây: "Đây là hai trăm nghìn, vợ tôi đâu?"
        assert fix.text_after == "Đây là hai mươi vạn, vợ tôi đâu?"

    def test_next_stage_does_not_flip_the_correct_lines(self) -> None:
        """Giai đoạn văn phong giữ nguyên các cụm số: không còn dòng nào bị đổi lối."""
        literal = _apply(LITERAL)
        assert find_number_fixes(SOURCES, literal, literal) == []  # trước: 5 dòng bị đổi
        assert {literal[i] for i in (788, 822)} == {
            "Mày đưa tao hai mươi vạn.", "hai mươi vạn."
        }

    def test_later_stage_error_is_fixed_from_the_earlier_stage(self) -> None:
        literal = _apply(LITERAL)
        style = {**literal, 966: "Hai trăm vạn đây, vợ tôi đâu?"}
        assert _apply(style, literal)[966] == "Hai mươi vạn đây, vợ tôi đâu?"


# Lượt chạy thật khác (phim "he thong sung the"): phim viết 两万 cả hai kiểu.
TWO_WAN_SOURCES = {
    126: "两万元已到账", 357: "之前我借你两万", 419: "借林逸两万",
    825: "那两万", 835: "两万",
}
TWO_WAN_FILM = {
    357: "Trước đây tao mượn mày hai vạn tệ,",
    419: "mà dám cho Lâm Dật vay hai vạn tệ đấy à?",
    825: "2 vạn tệ đó",
    835: "2 vạn tệ",
}


class TestSameWritingFormIsPreferred:
    def test_words_are_fixed_with_words(self) -> None:
        translations = {**TWO_WAN_FILM, 126: "hai mươi vạn tệ đã đến tài khoản."}
        [fix] = find_number_fixes(TWO_WAN_SOURCES, translations)
        assert fix.text_after == "hai vạn tệ đã đến tài khoản."

    def test_digits_are_fixed_with_digits(self) -> None:
        translations = {**TWO_WAN_FILM, 126: "20 vạn tệ đã đến tài khoản."}
        [fix] = find_number_fixes(TWO_WAN_SOURCES, translations)
        assert fix.text_after == "2 vạn tệ đã đến tài khoản."

    def test_other_form_is_used_when_it_is_the_only_one(self) -> None:
        translations = {835: "2 vạn tệ", 126: "hai mươi vạn tệ đã đến tài khoản."}
        [fix] = find_number_fixes(TWO_WAN_SOURCES, translations)
        assert fix.text_after == "2 vạn tệ đã đến tài khoản."


class TestOtherCasesAreUnchanged:
    def test_minority_modern_reading_does_not_win(self) -> None:
        """[v3.23.588] Bản 587 cho cách đọc hiện đại của MỘT dòng kéo cả phim theo; lượt
        chạy thật kế tiếp gặp đúng ca đó. Xem ``test_number_unit_style_v323588``."""
        translations = {**LITERAL, 822: "hai trăm nghìn."}
        fixed = _apply(translations)
        assert fixed[966] == "Đây là hai mươi vạn, vợ tôi đâu?"
        assert fixed[788] == "Mày đưa tao hai mươi vạn."
        assert fixed[822] == "hai mươi vạn."

    def test_mixed_unit_reading_is_not_a_model(self) -> None:
        """Ca thật khác: "三百四十三万" đọc "ba triệu bốn mươi ba vạn" — đúng giá trị
        nhưng trộn hai lối; không lấy làm mẫu để sửa dòng sai.

        [v3.23.590] Chính cụm trộn hai lối đó cũng được đọc lại theo lối của phim (tiền
        triệu: lối hiện đại), nên hai dòng của cùng số tiền ra cùng một cách đọc.
        """
        sources = {1143: "三百四十三万", 1145: "三百四十三万"}
        translations = {
            1143: "ba triệu bốn mươi ba vạn", 1145: "ba triệu bốn trăm ba mươi vạn"
        }
        fixes = {f.index: f for f in find_number_fixes(sources, translations)}
        assert {index: fix.phrase_after for index, fix in fixes.items()} == {
            1143: "ba triệu bốn trăm ba mươi nghìn",
            1145: "ba triệu bốn trăm ba mươi nghìn",
        }
        assert (fixes[1143].reason, fixes[1145].reason) == ("lệch lối đọc", "sai giá trị")

    def test_no_correct_reading_in_the_film_falls_back_to_the_standard(self) -> None:
        sources = {665: "三百四十五万", 667: "三百四十五万"}
        translations = {
            665: "ba triệu bốn trăm năm mươi vạn", 667: "ba triệu bốn trăm năm mươi vạn"
        }
        fixes = find_number_fixes(sources, translations)
        assert {fix.phrase_after for fix in fixes} == {"ba triệu bốn trăm năm mươi nghìn"}
