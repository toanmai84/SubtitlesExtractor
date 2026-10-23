"""[v3.23.591] Hồi quy: tên nhân vật bị đọc bằng âm Hán-Việt của một CHỮ KHÁC.

Chuẩn đoán thật v3.23.590 (phân tích bằng gemini-3.8-flash): nam chính ``甄劍聲`` được ghi
"Chân Kiếm Sanh" — 聲 đọc "thanh"; "sanh" là âm của 甥/笙. Âm sai lệch xa âm đúng nên quy
tắc "lệch một chút" (v3.23.567) chỉ báo mà không sửa: tên sai 16 chỗ trong bảng nhân vật,
53 chỗ trong tóm tắt.

Mọi dòng bảng nhân vật dưới đây chép NGUYÊN VĂN từ các bản phân tích thật trong kho chuẩn
đoán (ghi rõ phiên bản ứng dụng đã sinh ra nó).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.name_reading_check import (
    OTHER_CHARACTER_MIN_NAME_CHARS,
    NameReadingIssue,
    NameReadings,
    correct_context_names,
    find_name_reading_issues,
    name_corrections,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
)
from subtitles_extractor.infrastructure.translation.name_readings_loader import (
    HANVIET_READINGS_FILENAME,
    load_name_readings,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"

# ── v3.23.590, phim "灭门惨案II" (gemini-3.8-flash) ─────────────────────────────────────
KIEM_THANH_ROW = (
    "甄劍聲 (Chân Kiếm Sanh) [nam, ~35] — Chồng của Tần Khả Tú, cựu chiến binh thương "
    "tật mất khả năng sinh sản, yêu thương vợ tha thiết; sau khi phát hiện vợ con bị "
    "hãm hại đã vác súng đi báo thù nhưng bị nhà họ Hồng đánh lén sát hại; xưng hô: "
    "với Tần Khả Tú tự xưng 'anh' gọi 'em' (A Tú), với Chu Chính Quân tự xưng 'tôi' "
    "gọi 'cậu' (A Quân), với Hồng Chấn và Hồng Chí Quyền tự xưng 'tao' gọi 'mày', với "
    "bác sĩ tự xưng 'tôi' gọi 'bác sĩ', với nhân viên lữ quán tự xưng 'tôi' gọi 'anh'"
)
DOCTOR_ROW = (
    "醫生 (Bác sĩ) [nam, ~50] — bác sĩ trạm xá trấn Đông Long, tính cách bộc trực, "
    "khám thai cho Khả Tú rồi phát hiện cô nhiễm giang mai giai đoạn một và khuyên cô "
    "phá thai; xưng hô: với Chân Kiếm Sanh tự xưng 'tôi' gọi 'anh', với Tần Khả Tú tự "
    "xưng 'tôi' gọi 'cô', với Long Truyền Nhân tự xưng 'tôi' gọi 'đồng chí công an'"
)
OVERVIEW_590 = (
    "Lấy bối cảnh vùng quê Đông Long thời kỳ đầu cải cách mở cửa, bộ phim kể về bi "
    "kịch đẫm máu của đôi vợ chồng cựu binh Chân Kiếm Sanh và Tần Khả Tú. Do di chứng "
    "chiến tranh khiến Kiếm Sanh mất khả năng sinh sản, hai vợ chồng nhờ cựu đồng đội "
    "Chu Chính Quân thụ thai hộ ('mượn giống')."
)
GLOSSARY_590 = (
    "先夫 = người chồng quá cố\n甄劍聲 = Chân Kiếm Sanh\n西米露芒果 = xoài bột báng"
)

# ── v3.23.573, cùng phim (bản phân tích khác) ───────────────────────────────────────────
LAO_MAC_ROW = (
    "老莫 (Lão Mạc) [nam, trung niên] — Dân làng Đông Long: chồng của Mạc Sảo, vô cùng "
    "tức giận khi Long Truyền Nhân sàm sỡ vợ mình."
)
MAC_TAU_ROW = (
    "莫嫂 (Mạc Sảo) [nữ, trung niên] — Vợ Lão Mạc: người bị ngất xỉu giữa đường khiến "
    "đồn công an tưởng lầm là nạn nhân vụ án mạng."
)
HONG_DAI_TAU_ROW = (
    "洪大嫂 (Hồng Đại Sảo) [nữ, trung niên] — Vợ Hồng Chấn: thâm độc, giả nhân giả "
    "nghĩa. Bà ta lừa Tần Khả Tú giao kéo tự vệ ra, sau đó trở mặt dung túng cho chồng "
    "con cưỡng hiếp cô và hiến kế dùng vòi nước xịt rửa để xóa bằng chứng tinh dịch. "
    "Xưng hô: gọi Tần Khả Tú 'A Tú'."
)

# ── Tên dịch NGHĨA có chủ ý: không được sửa ─────────────────────────────────────────────
# v3.23.565, phim "傻子落水…": "quan" là âm Hán-Việt của 官/關 nhưng "Quan khách" là dịch.
GUESTS_ROW = (
    "宾客 (Quan khách) [nam, ~28] — Khách mời dự đám cưới, bức xúc vì cỗ cưới bị đổi "
    "thành cháo trắng cọng rau nên làm loạn đòi lại tiền mừng cưới; xưng hô: với nhà "
    "họ Từ tự xưng 'chúng tôi' gọi 'nhà họ Từ'"
)
# v3.23.577, phim "he thong sung the".
SISTER_MAI_ROW = (
    "王梅 / 梅姐 (Vương Mai / Chị Mai) [nữ, ~35] — nhân vật phụ: chị chủ tiệm tạp hóa "
    "hàng xóm, lúng liếng và mê mẩn vẻ đẹp trai của Lâm Dật"
)


@pytest.fixture(scope="module")
def readings() -> NameReadings:
    """Bảng âm Hán-Việt THẬT đi kèm ứng dụng."""
    table = load_name_readings(DATA_DIR / HANVIET_READINGS_FILENAME)
    assert table, "thiếu bảng âm Hán-Việt đi kèm ứng dụng"
    return table


def _roster_issues(roster: str, readings: NameReadings) -> list[NameReadingIssue]:
    return [
        issue
        for issue in find_name_reading_issues(roster, "", readings)
        if issue.source == "roster"
    ]


class TestBundledTableFacts:
    """Các dữ kiện của bảng âm mà quy tắc dựa vào."""

    def test_sheng_has_exactly_one_reading(self, readings: NameReadings) -> None:
        assert readings.suggestable("聲") == ("thanh",)

    def test_sanh_is_a_real_hanviet_syllable_of_other_characters(
        self, readings: NameReadings
    ) -> None:
        assert readings.is_hanviet_syllable("Sanh")
        assert "sanh" not in readings.accepted("聲")

    def test_sao_is_a_real_hanviet_syllable(self, readings: NameReadings) -> None:
        assert readings.is_hanviet_syllable("Sảo")
        assert readings.suggestable("嫂") == ("tẩu",)

    @pytest.mark.parametrize("word", ["Béo", "Sen", "Chị", "Chó", "Đen"])
    def test_native_words_are_not_hanviet_syllables(
        self, readings: NameReadings, word: str
    ) -> None:
        assert not readings.is_hanviet_syllable(word)

    def test_syllable_check_ignores_case_and_tone_position(
        self, readings: NameReadings
    ) -> None:
        assert readings.is_hanviet_syllable("hoà") == readings.is_hanviet_syllable("HÒA")
        assert readings.is_hanviet_syllable("thanh")


class TestRealCaseKiemThanh:
    def test_far_misreading_now_gets_a_fix(self, readings: NameReadings) -> None:
        (issue,) = _roster_issues(KIEM_THANH_ROW, readings)
        assert (issue.name, issue.char, issue.wrote) == ("甄劍聲", "聲", "Sanh")
        assert issue.suggestion == "Thanh"

    def test_corrections_cover_full_and_given_name(self, readings: NameReadings) -> None:
        issues = _roster_issues(KIEM_THANH_ROW, readings)
        assert name_corrections(issues, KIEM_THANH_ROW) == {
            "Chân Kiếm Sanh": "Chân Kiếm Thanh",
            "Kiếm Sanh": "Kiếm Thanh",
        }

    def test_whole_context_is_corrected(self, readings: NameReadings) -> None:
        context = SubtitleContextAnalysis(
            source_lang="zh",
            characters=f"{KIEM_THANH_ROW}\n{DOCTOR_ROW}",
            overview=OVERVIEW_590,
            glossary=GLOSSARY_590,
        )
        fixed, fixes = correct_context_names(context, readings)
        assert [fix.suggestion for fix in fixes] == ["Thanh"]
        for text in (fixed.characters, fixed.overview, fixed.glossary):
            assert "Sanh" not in text
        assert fixed.characters.startswith("甄劍聲 (Chân Kiếm Thanh) [nam, ~35]")
        assert "với Chân Kiếm Thanh tự xưng 'tôi' gọi 'anh'" in fixed.characters
        assert "cựu binh Chân Kiếm Thanh và Tần Khả Tú" in fixed.overview
        assert "khiến Kiếm Thanh mất khả năng sinh sản" in fixed.overview
        assert "甄劍聲 = Chân Kiếm Thanh" in fixed.glossary

    def test_corrected_context_is_clean(self, readings: NameReadings) -> None:
        context = SubtitleContextAnalysis(
            source_lang="zh",
            characters=KIEM_THANH_ROW,
            overview="",
            glossary=GLOSSARY_590,
        )
        fixed, _ = correct_context_names(context, readings)
        assert find_name_reading_issues(fixed.characters, fixed.glossary, readings) == []
        again, fixes = correct_context_names(fixed, readings)
        assert again is fixed
        assert fixes == []

    def test_glossary_entry_alone_is_still_only_reported(
        self, readings: NameReadings
    ) -> None:
        """Bảng thuật ngữ đầy cách dịch nghĩa — chỉ báo, không tự sửa (như v3.23.567)."""
        (issue,) = find_name_reading_issues("", GLOSSARY_590, readings)
        assert issue.source == "glossary"
        assert issue.suggestion == ""


class TestRealCaseSisterInLaw:
    def test_three_character_name_is_fixed(self, readings: NameReadings) -> None:
        (issue,) = _roster_issues(HONG_DAI_TAU_ROW, readings)
        assert (issue.name, issue.wrote, issue.suggestion) == ("洪大嫂", "Sảo", "Tẩu")

    def test_two_character_name_alone_is_only_reported(
        self, readings: NameReadings
    ) -> None:
        (issue,) = _roster_issues(MAC_TAU_ROW, readings)
        assert (issue.name, issue.wrote) == ("莫嫂", "Sảo")
        assert issue.suggestion == ""

    def test_two_character_name_follows_the_confirmed_misreading(
        self, readings: NameReadings
    ) -> None:
        """Cùng bảng đã có "Hồng Đại Sảo" đủ chắc → "Mạc Sảo" sửa theo."""
        roster = f"{LAO_MAC_ROW}\n{MAC_TAU_ROW}\n{HONG_DAI_TAU_ROW}"
        issues = _roster_issues(roster, readings)
        assert {(issue.name, issue.suggestion) for issue in issues} == {
            ("莫嫂", "Tẩu"),
            ("洪大嫂", "Tẩu"),
        }
        assert name_corrections(issues, roster) == {
            "Mạc Sảo": "Mạc Tẩu",
            "Hồng Đại Sảo": "Hồng Đại Tẩu",
            "Đại Sảo": "Đại Tẩu",
        }

    def test_other_rows_mentioning_the_name_are_fixed(
        self, readings: NameReadings
    ) -> None:
        context = SubtitleContextAnalysis(
            source_lang="zh",
            characters=f"{LAO_MAC_ROW}\n{MAC_TAU_ROW}\n{HONG_DAI_TAU_ROW}",
            overview="tinh dịch của anh trên thi thể Hồng Đại Sảo.",
            glossary="甄大嫂 = Chị dâu Chân",
        )
        fixed, _ = correct_context_names(context, readings)
        assert "chồng của Mạc Tẩu" in fixed.characters
        assert "莫嫂 (Mạc Tẩu)" in fixed.characters
        assert "洪大嫂 (Hồng Đại Tẩu)" in fixed.characters
        assert fixed.overview == "tinh dịch của anh trên thi thể Hồng Đại Tẩu."
        assert fixed.glossary == "甄大嫂 = Chị dâu Chân"


class TestMeaningTranslationsAreKept:
    def test_guests_translated_by_meaning_are_not_touched(
        self, readings: NameReadings
    ) -> None:
        """"quan" là âm Hán-Việt thật nhưng tên chỉ hai chữ → chưa đủ chắc."""
        (issue,) = _roster_issues(GUESTS_ROW, readings)
        assert (issue.name, issue.wrote) == ("宾客", "Quan")
        assert issue.suggestion == ""
        context = SubtitleContextAnalysis(
            source_lang="zh", characters=GUESTS_ROW, overview="", glossary=""
        )
        assert correct_context_names(context, readings)[0] is context

    def test_sister_mai_is_not_touched(self, readings: NameReadings) -> None:
        issues = _roster_issues(SISTER_MAI_ROW, readings)
        assert all(issue.suggestion == "" for issue in issues)

    def test_guests_do_not_follow_an_unrelated_fix(self, readings: NameReadings) -> None:
        """Sửa theo chỉ áp cho CÙNG chữ, CÙNG âm sai."""
        roster = f"{KIEM_THANH_ROW}\n{GUESTS_ROW}"
        suggestions = {
            issue.name: issue.suggestion for issue in _roster_issues(roster, readings)
        }
        assert suggestions == {"甄劍聲": "Thanh", "宾客": ""}


class TestSafetyGates:
    """Chốt an toàn, trên bảng âm nhỏ dựng tay để kiểm đúng từng điều kiện."""

    TABLE = NameReadings.from_encoded({
        "甄": "chân",
        "劍": "kiếm",
        "聲": "thanh",
        "笙": "sanh",
        "沈": "thẩm trầm",
        "明": "minh",
        "月": "nguyệt",
        "吴": "ngô",
        "大": "đại",
        "胖": "bàn",
        "莫": "mạc",
        "嫂": "tẩu|dâu",
        "稍": "sảo",
        "洪": "hồng",
    })

    def test_minimum_length_is_three_characters(self) -> None:
        assert OTHER_CHARACTER_MIN_NAME_CHARS == 3

    def test_native_word_in_a_long_name_is_not_fixed(self) -> None:
        """"Béo" không phải âm Hán-Việt của chữ nào → là dịch nghĩa có chủ ý."""
        (issue,) = find_name_reading_issues("吴大胖 (Ngô Đại Béo) [nam]", "", self.TABLE)
        assert issue.suggestion == ""

    def test_character_with_several_readings_is_not_guessed(self) -> None:
        """沈 đọc "thẩm" hoặc "trầm": không biết sửa "Sanh" về âm nào."""
        roster = "沈明月 (Sanh Minh Nguyệt) [nữ]"
        (issue,) = find_name_reading_issues(roster, "", self.TABLE)
        assert issue.char == "沈"
        assert issue.suggestion == ""

    def test_two_wrong_characters_are_not_fixed(self) -> None:
        roster = "甄劍聲笙 (Chân Kiếm Sảo Sảo) [nam]"
        issues = find_name_reading_issues(roster, "", self.TABLE)
        assert issues
        assert all(issue.suggestion == "" for issue in issues)

    def test_lowercase_syllable_is_not_a_name(self) -> None:
        """Âm lệch viết thường là từ thường trong mô tả, không phải tên riêng."""
        roster = "甄劍聲 (Chân Kiếm sanh) [nam]"
        assert find_name_reading_issues(roster, "", self.TABLE) == []

    def test_extra_reading_is_accepted_not_reported(self) -> None:
        roster = "洪大嫂 (Hồng Đại Dâu) [nữ]"
        assert find_name_reading_issues(roster, "", self.TABLE) == []

    def test_near_miss_rule_still_wins(self) -> None:
        """Lệch một chút (thiếu dấu) vẫn do quy tắc cũ xử lý, kể cả tên hai chữ."""
        (issue,) = find_name_reading_issues("莫嫂 (Mạc Tâu) [nữ]", "", self.TABLE)
        assert issue.suggestion == "Tẩu"

    def test_follow_needs_the_same_wrong_syllable(self) -> None:
        roster = "洪大嫂 (Hồng Đại Sảo) [nữ]\n莫嫂 (Mạc Sanh) [nữ]"
        suggestions = {
            issue.name: issue.suggestion
            for issue in find_name_reading_issues(roster, "", self.TABLE)
        }
        assert suggestions == {"洪大嫂": "Tẩu", "莫嫂": ""}

    def test_follow_keeps_the_written_case(self) -> None:
        roster = "洪大嫂 (Hồng Đại Sảo) [nữ]\n莫嫂 (Mạc Sảo) [nữ]"
        suggestions = {
            issue.name: issue.suggestion
            for issue in find_name_reading_issues(roster, "", self.TABLE)
        }
        assert suggestions == {"洪大嫂": "Tẩu", "莫嫂": "Tẩu"}

    def test_empty_table_checks_nothing(self) -> None:
        assert find_name_reading_issues(KIEM_THANH_ROW, "", NameReadings()) == []
