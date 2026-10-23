"""[v3.23.589] Hồi quy: bảng nhân vật gọi chức danh khác bảng thuật ngữ.

Chuẩn đoán thật v3.23.588, phim "灭门惨案II": bảng thuật ngữ ``所長 = Trưởng đồn công
an``, bảng nhân vật bốn quy tắc "gọi 'Sở trưởng'" (âm Hán-Việt của 所長). Dịch thô theo
bảng thuật ngữ 15/15 dòng; giai đoạn phong cách đổi lại "Sở trưởng" ở dòng của người nói
có quy tắc ấy. Bản cuối ba lượt chạy cùng phân tích này: 3, 0, 6 trên 15 dòng "Sở trưởng",
còn lại "Trưởng đồn".

Phim thứ hai (5540 dòng, 6 lượt chạy): ``厂长 = giám đốc xưởng`` với "gọi 'xưởng
trưởng'" — bản cuối 99 dòng "xưởng trưởng" / 50 dòng "giám đốc xưởng".

Chạy trên cả 18 phân tích khác nhau của kho chuẩn đoán: đúng 2 phân tích trên bị đổi, 16
phân tích còn lại (hai bảng đã khớp, hoặc không có thuật ngữ chức danh) giữ nguyên từng
ký tự. Mọi dòng bảng nhân vật và mục thuật ngữ dưới đây chép nguyên văn (hai dòng dài
龍傳人 và 朱正軍 cắt bớt các quy tắc xưng hô ở cuối dòng).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from subtitles_extractor.application.services.name_reading_check import NameReadings
from subtitles_extractor.application.services.roster_format import address_clauses
from subtitles_extractor.application.services.title_reconciliation import (
    TitleFix,
    describe_title_fix,
    reconcile_context_titles,
    reconcile_titles,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    TranslationContext,
    TranslationLine,
)
from subtitles_extractor.infrastructure.translation.name_readings_loader import (
    HANVIET_READINGS_FILENAME,
    load_name_readings,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"

# Phim "灭门惨案II", phân tích của lượt v3.23.586 tới v3.23.588.
POLICE_ROSTER = "\n".join([
    "龍傳人 (Long Truyền Nhân) [nam, ~35] — Trưởng đồn công an thôn Đông Long, tính tình "
    "khoe khoang, háo danh, giảo hoạt nhưng bản chất có lòng trắc ẩn; xưng hô: với Tần "
    "Khả Tú tự xưng 'tôi' gọi 'chị Chân' (danh xưng: Chân đại tẩu), với Dư Tiểu Liên tự "
    "xưng 'tôi' gọi 'cô Tiểu Liên', với Ái Quốc tự xưng 'tôi' gọi 'cậu'",
    "余小蓮 (Dư Tiểu Liên) [nữ, ~22] — nữ công an trẻ trung tại đồn Đông Long, ngây thơ, "
    "mắc chứng mẫn cảm bẩm sinh khi bị đụng chạm thân thể; xưng hô: với Long Truyền Nhân "
    "tự xưng 'tôi' gọi 'Sở trưởng', với Bác Sĩ tự xưng 'tôi' gọi 'bác sĩ'",
    "阿積 (A Tích) [nam, ~28] — công an viên đồn Đông Long phụ trách khám nghiệm mẫu vật "
    "và phòng thí nghiệm; xưng hô: với Long Truyền Nhân tự xưng 'tôi' gọi 'Sở trưởng'",
    "老表 (Lão Biểu) [nam, ~35] — anh họ của Long Truyền Nhân (xuất hiện qua ảnh chân "
    "dung vinh danh), sĩ quan công an phá đại án diệt môn và được thăng chức Cục trưởng",
    "朱正軍 / 阿軍 (Chu Chính Quân / A Quân) [nam, ~27] — cựu chiến binh làm nông, được "
    "thuê mượn giống rồi sinh lòng si mê Khả Tú mù quáng, chấp nhận làm mọi thứ vì cô; "
    "xưng hô: với Long Truyền Nhân tự xưng 'tôi' gọi 'Đại đội trưởng', với Tần Khả Tú "
    "trước khi bộc bạch xưng 'tôi' gọi 'chị dâu'",
    "愛國 (Ái Quốc) [nam, ~20] — công an viên đồn Đông Long, dáng người gầy, hay nịnh "
    "nọt cấp trên Long Truyền Nhân; xưng hô: với Long Truyền Nhân tự xưng 'tôi' gọi 'Sở "
    "trưởng'",
    "胖公安 (Công an béo) [nam, ~28] — công an viên đồn Đông Long, mập mạp, nhiều "
    "chuyện, thích đi hát và ngắm gái; xưng hô: với Long Truyền Nhân tự xưng 'tôi' gọi "
    "'Sở trưởng', với Chu Chính Quân tự xưng 'tao' gọi 'mày'",
    "老伯 (Lão bá cầm bô) [nam, ~70] — dân làng cao tuổi, nhiệt tình hưởng ứng việc lấy "
    "mẫu tinh dịch với chiếc bô tráng men; xưng hô: với Long Truyền Nhân tự xưng 'tôi' "
    "gọi 'Đội trưởng'",
    "鎮長 (Trấn trưởng) [nam, ~55] — quan chức đứng đầu thị trấn, từng được bác sĩ chữa "
    "bệnh kín và đề tặng hoành phi (nhắc tên)",
])
POLICE_GLOSSARY = "\n".join([
    "派出所 = đồn công an xã/phường",
    "所長 = Trưởng đồn công an",
    "老表 = anh họ / anh em chú bác",
])

# Phim 5540 dòng, phân tích của lượt v3.23.578 tới v3.23.583.
FACTORY_ROSTER = "\n".join([
    "付金奎 (Phó Kim Khuê) [nam, ~50] — phó giám đốc xưởng điện gia dụng Hồng Nhật: tham "
    "lam, biển thủ công quỹ, quỵt lương công nhân và dở thói dâm ô gạ gẫm Chu Tinh Tinh; "
    "xưng hô: với Vương Đại Phúc tự xưng 'tao' gọi 'thằng béo'; với Lâm Dật tự xưng "
    "'tao' gọi 'mày'; với Chu Tinh Tinh tự xưng 'tôi' gọi 'cô'; với Phùng Minh tự xưng "
    "'tôi' gọi 'xưởng trưởng'",
    "冯明 / 冯厂长 (Phùng Minh / Phùng xưởng trưởng) [nam, ~52] — giám đốc xưởng "
    "điện gia dụng Hồng Nhật: người cương trực, bảo bọc công nhân và kiên quyết "
    "vạch mặt kẻ tham ô; xưng hô: với Phó Kim Khuê tự xưng 'tôi' gọi 'anh'; với "
    "Lâm Dật tự xưng 'tôi' gọi 'Lâm tiên sinh'; với Chu Tinh Tinh tự xưng 'chú' "
    "gọi 'cháu'; với Chung Tuấn Đào tự xưng 'tôi' gọi 'cậu Chung'",
    "钟俊涛 (Chung Tuấn Đào) [nam, ~26] — thiếu gia công ty Điện gia dụng Chung Điểm: kẻ "
    "hống hách, âm mưu ép giá thôn tính xưởng Hồng Nhật nhưng bị Lâm Dật đánh bại; xưng "
    "hô: với Phùng Minh tự xưng 'tôi' gọi 'Phùng xưởng trưởng'; với Lâm Dật tự xưng "
    "'tao' gọi 'mày'; với Chu Tinh Tinh tự xưng 'tôi' gọi 'cô'",
    "周晶晶 / 晶晶 (Chu Tinh Tinh / Tinh Tinh) [nữ, ~29] — chị họ của Bàn Tử, người đứng "
    "tên và quản lý xưởng điện gia dụng Hồng Nhật: người phụ nữ sắc sảo, kiên cường, đã "
    "ly hôn và một mình nuôi 3 con; xưng hô: với Lâm Dật tự xưng 'chị' gọi 'Tiểu Dật'; "
    "với Vương Đại Phúc tự xưng 'chị' gọi 'em'; với Phùng Minh tự xưng 'cháu' gọi 'chú' "
    "(danh xưng: xưởng trưởng); với Phó Kim Khuê tự xưng 'tôi' gọi 'ông'; với Chung Tuấn "
    "Đào tự xưng 'tôi' gọi 'anh'",
])
FACTORY_GLOSSARY = "\n".join([
    "副厂长 = phó giám đốc xưởng",
    "厂长 = giám đốc xưởng",
    "保安队队长 = đội trưởng đội bảo vệ",
])

# Cùng phim "灭门惨案II", phân tích của lượt v3.23.574: hai bảng ĐÃ khớp nhau.
AGREEING_ROSTER = "\n".join([
    "愛國 (Ái Quốc) [nam, ~28] — Cảnh sát viên đồn Đông Long, thích hùa theo nịnh bợ cấp "
    "trên; xưng hô: với Long Truyền Nhân tự xưng 'tôi' gọi 'Sở trưởng'; với Tần Khả Tú "
    "tự xưng 'tôi' gọi 'chị'",
    "老伯 (Lão bá / Cụ già cầm bô) [nam, ~70] — Dân làng cao tuổi, hào hứng và dai sức "
    "nhất khi thực hiện lệnh nộp tinh dịch; xưng hô: với Long Truyền Nhân tự xưng 'tôi' "
    "gọi 'Đội trưởng'",
])
AGREEING_GLOSSARY = "所長 = sở trưởng (trưởng đồn)\n大隊長 = đội trưởng"


@pytest.fixture(scope="module")
def readings() -> NameReadings:
    return load_name_readings(DATA_DIR / HANVIET_READINGS_FILENAME)


class TestPoliceStation:
    def test_four_address_rules_follow_the_glossary(self, readings: NameReadings) -> None:
        roster, fixes = reconcile_titles(POLICE_ROSTER, POLICE_GLOSSARY, readings)
        assert fixes == [TitleFix("所長", "Sở trưởng", "Trưởng đồn công an", 4)]
        assert "Sở trưởng" not in roster
        assert roster.count("gọi 'Trưởng đồn công an'") == 4

    def test_nothing_else_in_the_roster_moves(self, readings: NameReadings) -> None:
        roster, _ = reconcile_titles(POLICE_ROSTER, POLICE_GLOSSARY, readings)
        assert roster == POLICE_ROSTER.replace("'Sở trưởng'", "'Trưởng đồn công an'")

    def test_titles_without_a_glossary_entry_are_kept(
        self, readings: NameReadings
    ) -> None:
        """大隊長 và 鎮長 không có trong bảng thuật ngữ: không có chuẩn nào để theo."""
        roster, _ = reconcile_titles(POLICE_ROSTER, POLICE_GLOSSARY, readings)
        for kept in ("gọi 'Đại đội trưởng'", "gọi 'Đội trưởng'", "鎮長 (Trấn trưởng)"):
            assert kept in roster

    def test_character_named_after_a_glossary_term_keeps_its_name(
        self, readings: NameReadings
    ) -> None:
        """老表 có trong bảng thuật ngữ ("anh họ") nhưng "Lão Biểu" là TÊN nhân vật."""
        roster, _ = reconcile_titles(POLICE_ROSTER, POLICE_GLOSSARY, readings)
        assert "老表 (Lão Biểu)" in roster

    def test_address_clauses_keep_their_listeners(self, readings: NameReadings) -> None:
        roster, _ = reconcile_titles(POLICE_ROSTER, POLICE_GLOSSARY, readings)
        before, after = address_clauses(POLICE_ROSTER), address_clauses(roster)
        assert [c.listeners for c in before] == [c.listeners for c in after]
        assert len(before) == len(after) > 0

    def test_second_pass_changes_nothing(self, readings: NameReadings) -> None:
        once, _ = reconcile_titles(POLICE_ROSTER, POLICE_GLOSSARY, readings)
        twice, fixes = reconcile_titles(once, POLICE_GLOSSARY, readings)
        assert twice is once and fixes == []


class TestFactory:
    def test_bare_title_title_note_and_surname_form(self, readings: NameReadings) -> None:
        roster, fixes = reconcile_titles(FACTORY_ROSTER, FACTORY_GLOSSARY, readings)
        assert sorted(fixes, key=lambda fix: fix.written) == [
            TitleFix("厂长", "Phùng xưởng trưởng", "giám đốc xưởng Phùng", 1),
            TitleFix("厂长", "xưởng trưởng", "giám đốc xưởng", 2),
        ]
        assert "tự xưng 'tôi' gọi 'giám đốc xưởng'" in roster
        assert "gọi 'chú' (danh xưng: giám đốc xưởng)" in roster
        # Tiếng Việt đặt họ SAU chức danh.
        assert "gọi 'giám đốc xưởng Phùng'" in roster

    def test_transliteration_after_the_name_is_kept(self, readings: NameReadings) -> None:
        """Còn là khoá tra người nghe của các quy tắc "với Phùng xưởng trưởng …"."""
        roster, _ = reconcile_titles(FACTORY_ROSTER, FACTORY_GLOSSARY, readings)
        assert "冯明 / 冯厂长 (Phùng Minh / Phùng xưởng trưởng)" in roster
        before, after = address_clauses(FACTORY_ROSTER), address_clauses(roster)
        assert [c.listeners for c in before] == [c.listeners for c in after]

    def test_prose_is_not_touched(self, readings: NameReadings) -> None:
        roster, _ = reconcile_titles(FACTORY_ROSTER, FACTORY_GLOSSARY, readings)
        assert "phó giám đốc xưởng điện gia dụng Hồng Nhật" in roster
        assert "giám đốc xưởng điện gia dụng Hồng Nhật: người cương trực" in roster


class TestNoConflict:
    def test_glossary_using_the_sino_vietnamese_reading(
        self, readings: NameReadings
    ) -> None:
        """Lượt v3.23.574: ``所長 = sở trưởng`` — hai bảng khớp, bản cuối 15/15."""
        roster, fixes = reconcile_titles(AGREEING_ROSTER, AGREEING_GLOSSARY, readings)
        assert roster is AGREEING_ROSTER and fixes == []

    def test_glossary_without_title_terms(self, readings: NameReadings) -> None:
        glossary = "派出所 = đồn công an xã/phường\n老表 = anh họ / anh em chú bác"
        roster, fixes = reconcile_titles(POLICE_ROSTER, glossary, readings)
        assert roster is POLICE_ROSTER and fixes == []

    @pytest.mark.parametrize("table", [None, NameReadings()])
    def test_without_a_reading_table(self, table: NameReadings | None) -> None:
        roster, fixes = reconcile_titles(POLICE_ROSTER, POLICE_GLOSSARY, table)
        assert roster is POLICE_ROSTER and fixes == []

    def test_empty_inputs(self, readings: NameReadings) -> None:
        assert reconcile_titles("", POLICE_GLOSSARY, readings) == ("", [])
        assert reconcile_titles(POLICE_ROSTER, "", readings) == (POLICE_ROSTER, [])


class TestBoundaries:
    """Biên DỰNG THÊM từ các mục thật (kho chưa có ca nào như vậy)."""

    def test_ordinary_word_spelled_like_a_title_is_safe(
        self, readings: NameReadings
    ) -> None:
        """"sở trường" (所長 đọc âm "trường") là từ thường; văn xuôi cũng không bị sửa."""
        roster = (
            "阿積 (A Tích) [nam, ~28] — sở trường là khám nghiệm, muốn làm sở trưởng; "
            "xưng hô: với Long Truyền Nhân tự xưng 'tôi' gọi 'sở trường'"
        )
        assert reconcile_titles(roster, POLICE_GLOSSARY, readings) == (roster, [])

    def test_only_the_matching_alternative_is_replaced(
        self, readings: NameReadings
    ) -> None:
        roster = (
            "愛國 (Ái Quốc) [nam, ~20] — công an viên; xưng hô: với Long Truyền Nhân tự "
            "xưng 'tôi' gọi 'Sở trưởng/anh Long'"
        )
        fixed, fixes = reconcile_titles(roster, POLICE_GLOSSARY, readings)
        assert "gọi 'Trưởng đồn công an/anh Long'" in fixed
        assert fixes == [TitleFix("所長", "Sở trưởng", "Trưởng đồn công an", 1)]

    def test_lowercase_glossary_term_takes_the_capital_of_the_address(
        self, readings: NameReadings
    ) -> None:
        roster = FACTORY_ROSTER.replace("gọi 'xưởng trưởng'", "gọi 'Xưởng trưởng'")
        fixed, _ = reconcile_titles(roster, FACTORY_GLOSSARY, readings)
        assert "gọi 'Giám đốc xưởng'" in fixed

    def test_surname_form_needs_a_roster_name_carrying_the_title(
        self, readings: NameReadings
    ) -> None:
        """Không có nhân vật 冯厂长 thì "Phùng xưởng trưởng" không chắc là họ + chức."""
        roster = FACTORY_ROSTER.replace("冯明 / 冯厂长", "冯明")
        fixed, fixes = reconcile_titles(roster, FACTORY_GLOSSARY, readings)
        assert "gọi 'Phùng xưởng trưởng'" in fixed
        assert fixes == [TitleFix("厂长", "xưởng trưởng", "giám đốc xưởng", 2)]

    def test_translation_that_would_break_the_roster_syntax_is_not_used(
        self, readings: NameReadings
    ) -> None:
        """Dấu nháy trong cách dịch sẽ cắt đôi quy tắc "gọi '…'"."""
        glossary = "所長 = sếp 'lớn'"
        fixed, fixes = reconcile_titles(POLICE_ROSTER, glossary, readings)
        assert fixed is POLICE_ROSTER and fixes == []

    def test_glossary_note_is_not_part_of_the_address(
        self, readings: NameReadings
    ) -> None:
        glossary = "所長 = trưởng đồn (cách gọi 'thân mật' của cấp dưới)"
        fixed, fixes = reconcile_titles(POLICE_ROSTER, glossary, readings)
        assert fixes == [TitleFix("所長", "Sở trưởng", "Trưởng đồn", 4)]
        assert len(address_clauses(fixed)) == len(address_clauses(POLICE_ROSTER))


class TestContext:
    def test_only_the_roster_field_changes(self, readings: NameReadings) -> None:
        context = TranslationContext(
            target_lang="Vietnamese",
            characters=POLICE_ROSTER,
            glossary=POLICE_GLOSSARY,
            overview="Trưởng đồn công an Long Truyền Nhân khao khát phá đại án.",
        )
        fixed, fixes = reconcile_context_titles(context, readings)
        assert len(fixes) == 1
        assert fixed == replace(
            context,
            characters=POLICE_ROSTER.replace("'Sở trưởng'", "'Trưởng đồn công an'"),
        )

    def test_unchanged_context_is_returned_as_is(self, readings: NameReadings) -> None:
        context = TranslationContext(
            target_lang="Vietnamese",
            characters=AGREEING_ROSTER,
            glossary=AGREEING_GLOSSARY,
        )
        assert reconcile_context_titles(context, readings) == (context, [])
        assert reconcile_context_titles(context, readings)[0] is context

    def test_description(self) -> None:
        fix = TitleFix("所長", "Sở trưởng", "Trưởng đồn công an", 4)
        assert describe_title_fix(fix) == (
            "所長: 'Sở trưởng' → 'Trưởng đồn công an' (4 chỗ)"
        )


class _FakeAnalyzer:
    def __init__(self, result: SubtitleContextAnalysis) -> None:
        self._result = result

    def analyze_global_context(self, **_kwargs: object) -> SubtitleContextAnalysis:
        return self._result


class TestFreshAnalysis:
    def test_new_analysis_is_reconciled_before_it_is_shown_or_saved(
        self, readings: NameReadings
    ) -> None:
        analysis = SubtitleContextAnalysis(
            source_lang="zh",
            characters=POLICE_ROSTER,
            overview="Tóm tắt.",
            glossary=POLICE_GLOSSARY,
        )
        use_case = AnalyzeSubtitleContextUseCase(
            translator=_FakeAnalyzer(analysis),  # type: ignore[arg-type]
            name_readings=readings,
        )
        lines = [TranslationLine(index=1, start_ms=0, end_ms=1, text="所長")]
        result = use_case.execute(lines, "Vietnamese")
        assert "Sở trưởng" not in result.characters
        assert result.characters.count("gọi 'Trưởng đồn công an'") == 4
        assert result.glossary == POLICE_GLOSSARY
