"""[v3.23.556] Hồi quy cho lưới CON SỐ (chuẩn đoán thật v3.23.555, 43 dòng có 万/亿).

Hai lỗi thật, không cờ nào cũ bắt được vì câu vẫn trôi chảy và vẫn đủ dài:

* **Sai giá trị** — ``百万年薪`` (lương năm MỘT triệu) → "lương năm triệu tệ": người nghe
  hiểu thành 5 triệu.
* **Lệch lối đọc** — cùng ``一千万`` khi thì "mười triệu" (dòng 2338), khi thì "một nghìn
  vạn" (5 dòng ở đoạn 3377-3440).

Các lớp ``TestDangerousCases`` là những ca lưới SUÝT phá khi đo thử trên dữ liệu thật.
"""

from __future__ import annotations

from subtitles_extractor.application.services.number_consistency_guard import (
    chinese_number_value,
    find_number_findings,
    find_number_fixes,
    vietnamese_number_value,
)


class TestChineseParser:
    def test_big_units(self) -> None:
        assert chinese_number_value("一千万") == 10_000_000
        assert chinese_number_value("二百万") == 2_000_000
        assert chinese_number_value("十八万") == 180_000
        assert chinese_number_value("百万") == 1_000_000
        assert chinese_number_value("十亿") == 1_000_000_000

    def test_non_numeric_returns_none(self) -> None:
        assert chinese_number_value("好几") is None


class TestVietnameseParser:
    def test_standard_forms(self) -> None:
        assert vietnamese_number_value("mười triệu") == 10_000_000
        assert vietnamese_number_value("một trăm tám mươi nghìn") == 180_000
        assert vietnamese_number_value("tám mươi lăm nghìn") == 85_000
        assert vietnamese_number_value("một tỉ") == 1_000_000_000

    def test_nested_chinese_style_grouping(self) -> None:
        """'một nghìn vạn' = 1000 x 10^4 — lối đọc gây bản dịch không nhất quán."""
        assert vietnamese_number_value("một nghìn vạn") == 10_000_000
        assert vietnamese_number_value("mười tám vạn") == 180_000

    def test_approximate_has_no_value(self) -> None:
        assert vietnamese_number_value("hàng chục ức") is None
        assert vietnamese_number_value("mấy triệu") is None


class TestRealCases:
    def test_wrong_value_is_fixed(self) -> None:
        """Ca thật dòng 2047/2049: 百万年薪 → 'lương năm triệu' (gấp 5 lần)."""
        sources = {2047: "我愿意出百万年薪", 2049: "百万年薪?", 1156: "一百万"}
        translations = {
            2047: "Tôi sẵn sàng trả lương năm triệu tệ,",
            2049: "Lương năm triệu tệ á?",
            1156: "một triệu.",
        }
        fixes = find_number_fixes(sources, translations)
        assert [f.index for f in fixes] == [2047, 2049]
        assert fixes[0].text_after == "Tôi sẵn sàng trả lương một triệu tệ,"
        assert all(f.reason == "sai giá trị" for f in fixes)

    def test_inconsistent_reading_is_normalised(self) -> None:
        """Ca thật: 一千万 khi 'mười triệu', khi 'một nghìn vạn' — cùng một khoản tiền."""
        sources = {2338: "这张卡里有一千万", 3377: "给了我一千万", 3440: "一千万长什么样"}
        translations = {
            2338: "Trong thẻ này có mười triệu,",
            3377: "họ còn cho anh một nghìn vạn.",
            3440: "một nghìn vạn trông thế nào đã.",
        }
        fixes = find_number_fixes(sources, translations)
        assert [f.index for f in fixes] == [3377, 3440]
        assert fixes[0].text_after == "họ còn cho anh mười triệu."
        assert all(f.reason == "lệch lối đọc" for f in fixes)

    def test_capitalisation_is_preserved(self) -> None:
        sources = {2338: "这张卡里有一千万", 3378: "一千万?"}
        translations = {2338: "Trong thẻ này có mười triệu,", 3378: "Một nghìn vạn?"}
        fixes = find_number_fixes(sources, translations)
        assert fixes[0].text_after == "Mười triệu?"

    def test_findings_expose_both_kinds(self) -> None:
        sources = {
            2049: "百万年薪?", 1156: "一百万", 3440: "一千万长什么样", 2338: "有一千万",
        }
        translations = {
            2049: "Lương năm triệu tệ á?",
            1156: "một triệu.",
            3440: "một nghìn vạn trông thế nào.",
            2338: "có mười triệu,",
        }
        kinds = {f.index: f.kind for f in find_number_findings(sources, translations)}
        assert kinds == {2049: "mismatch", 3440: "non_standard_unit"}


class TestDangerousCases:
    """Những ca lưới SUÝT phá khi đo thử trên 7 bộ chuẩn đoán thật."""

    def test_adverbial_qianwan_is_never_touched(self) -> None:
        """``千万不能`` = 'tuyệt đối không', KHÔNG phải 10 triệu."""
        sources = {269: "千万不能亏待自己", 2885: "大家可千万别当真啊", 2338: "有一千万"}
        translations = {
            269: "tuyệt đối không được để bản thân chịu thiệt.",
            2885: "Mọi người ngàn lần đừng tưởng thật đấy.",
            2338: "có mười triệu,",
        }
        assert find_number_fixes(sources, translations) == []
        assert find_number_findings(sources, translations) == []

    def test_shorthand_with_trailing_digit_is_skipped(self) -> None:
        """``一万四`` = 14.000 chứ không phải 10.000 — cụm bắt được không đủ tin cậy."""
        sources = {952: "足足一万四", 1031: "这一万块钱"}
        translations = {952: "Tận mười bốn nghìn đấy!", 1031: "Một vạn tệ này"}
        assert [f.index for f in find_number_findings(sources, translations)] == []

    def test_digit_chant_is_not_a_number(self) -> None:
        """``个十百千万`` là bài đếm hàng ("Một, mười, một trăm, một nghìn…")."""
        sources = {1626: "个十百千万"}
        translations = {1626: "Một, mười, một trăm, một nghìn..."}
        assert find_number_findings(sources, translations) == []

    def test_approximate_source_is_skipped(self) -> None:
        """``身价数十亿`` = 'tài sản hàng tỉ' — ước lượng, không có trị để đối chiếu."""
        sources = {2995: "身价数十亿的"}
        translations = {2995: "tài sản hàng chục tỉ,"}
        assert find_number_findings(sources, translations) == []

    def test_style_only_difference_is_left_alone(self) -> None:
        """Cả phim đọc '一万' là 'một vạn' thì đó là VĂN PHONG, không phải lỗi."""
        sources = {1031: "这一万块钱", 1033: "一万?", 1047: "一万块啊"}
        translations = {
            1031: "Một vạn tệ này", 1033: "Một vạn?", 1047: "Một vạn tệ ấy à,",
        }
        assert find_number_fixes(sources, translations) == []
        assert find_number_findings(sources, translations) == []

    def test_line_with_two_numbers_is_skipped(self) -> None:
        sources = {10: "三万和五万"}
        translations = {10: "ba vạn và năm vạn"}
        assert find_number_findings(sources, translations) == []


class TestUseCaseWiring:
    def test_stage_output_is_patched(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationLine,
        )

        def line(index: int, text: str, original: str) -> TranslationLine:
            return TranslationLine(
                index=index, start_ms=index * 1000, end_ms=index * 1000 + 900,
                text=text, original_text=original,
            )

        source = [line(1156, "", "一百万"), line(2049, "", "百万年薪?")]
        stage = [line(1156, "một triệu.", ""), line(2049, "Lương năm triệu tệ?", "")]
        result = TranslateSubtitlesUseCase._enforce_number_consistency(
            stage, source, "Dịch thô"
        )
        assert result[1].text == "Lương một triệu tệ?"
        assert result[0].text == "một triệu."

    def test_no_numbers_means_no_change(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )

        stage: list = []
        assert TranslateSubtitlesUseCase._enforce_number_consistency(
            stage, [], "Dịch thô"
        ) is stage
