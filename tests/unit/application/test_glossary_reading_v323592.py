"""[v3.23.592] Bảng thuật ngữ: mục chỉ ĐỌC ÂM Hán-Việt, và cặp "gần trùng" bị báo nhầm.

Bản phân tích thật v3.23.591 (phim "灭门惨案II", gemini-3.8-flash) ghi:

    所長 = sở trưởng (trưởng đồn công an)
    性自樂 = tính tự nhạc (thuật ngữ hài hước chỉ việc tự sướng/thủ dâm)

Khâu dịch chép phần trước ngoặc: bản cuối của lượt đó gọi "Sở trưởng" ở 15/15 dòng có
"所長" và "tính tự nhạc" ở 4/4 dòng có "性自樂". File chuẩn đoán ngữ cảnh không nói gì về
hai mục này; cảnh báo duy nhất của nó về bảng thuật ngữ — "借種 ~ 配種 gần trùng" — lại là
báo nhầm.

Mọi mục thuật ngữ dưới đây chép nguyên văn từ các bản phân tích thật (ghi ở từng nhóm).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.context_diagnostics import (
    _near_duplicate_pairs,
    build_context_diagnostics,
)
from subtitles_extractor.application.services.glossary_reading_check import (
    ReadingOnlyTerm,
    find_reading_only_terms,
)
from subtitles_extractor.application.services.name_reading_check import NameReadings
from subtitles_extractor.application.services.roster_format import parse_glossary
from subtitles_extractor.infrastructure.translation.name_readings_loader import (
    HANVIET_READINGS_FILENAME,
    load_name_readings,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"

# Bản phân tích của lượt 0d20d817 (v3.23.591) — 15 trong 136 mục.
REAL_GLOSSARY = "\n".join([
    "借種 = mượn giống (thụ tinh nhân tạo/nhờ người phối giống)",
    "派出所 = đồn công an",
    "公安 = công an",
    "所長 = sở trưởng (trưởng đồn công an)",
    "局長 = cục trưởng",
    "老表 = anh họ / chú em họ",
    "性自樂 = tính tự nhạc (thuật ngữ hài hước chỉ việc tự sướng/thủ dâm)",
    "黑龍江 = Hắc Long Giang",
    "大隊長 = đại đội trưởng",
    "楊柳青 = Dương Liễu Thanh (dân ca/hành khúc quân đội)",
    "配種 = phối giống",
    "綠帽子 = nón xanh (bị cắm sừng)",
    "通姦 = thông dâm",
    "禽獸 = cầm thú (súc sinh)",
    "惡有惡報 = ác giả ác báo (ở ác gặp ác)",
])
# Mục thật của các bản phân tích khác: lượt 11dab90b (兴师问罪, 定), 5c6326e5 (通姦,
# 自瀆), 069d0c24 (前功尽弃, 义父), 07e94e55 (发财, 白富美, 土狗).
OTHER_REAL_ENTRIES = {
    "兴师问罪": "兴师问罪 = hưng sư vấn tội (đến hỏi tội)",
    "通姦": "通姦 = thông gian (ngoại tình)",
    "自瀆": "自瀆 = tự độc (thủ dâm)",
    "前功尽弃": "前功尽弃 = tiền công tận khí (công cốc / đổ sông đổ biển)",
    "义父": "义父 = nghĩa phụ (cha nuôi)",
    "发财": "发财 = phát tài (trúng mánh)",
    "白富美": "白富美 = bạch phú mỹ (cô gái giàu đẹp hoàn hảo)",
    "土狗": "土狗 = Thổ Cẩu (chó ta / cách Ngô Nhị Cẩu tự hạ mình xưng hô)",
    "定": "定 = định (thuật định thân)",
}


@pytest.fixture(scope="module")
def readings() -> NameReadings:
    """Bảng âm Hán-Việt THẬT đi kèm ứng dụng."""
    table = load_name_readings(DATA_DIR / HANVIET_READINGS_FILENAME)
    assert table, "thiếu bảng âm Hán-Việt đi kèm ứng dụng"
    return table


def _found(glossary: str, readings: NameReadings) -> list[str]:
    return [term.describe() for term in find_reading_only_terms(glossary, readings)]


class TestFindReadingOnlyTerms:
    def test_real_analysis_points_at_the_two_calques(
        self, readings: NameReadings
    ) -> None:
        assert _found(REAL_GLOSSARY, readings) == [
            "所長 = sở trưởng (trưởng đồn công an)",
            "性自樂 = tính tự nhạc (thuật ngữ hài hước chỉ việc tự sướng/thủ dâm)",
            "禽獸 = cầm thú (súc sinh)",
        ]

    def test_fields_keep_the_note_for_the_reader(self, readings: NameReadings) -> None:
        first = find_reading_only_terms(REAL_GLOSSARY, readings)[0]
        assert first == ReadingOnlyTerm(
            term="所長", reading="sở trưởng", note="trưởng đồn công an"
        )

    @pytest.mark.parametrize("term", ["兴师问罪", "通姦", "自瀆", "前功尽弃", "白富美"])
    def test_real_calques_from_other_analyses(
        self, term: str, readings: NameReadings
    ) -> None:
        entry = OTHER_REAL_ENTRIES[term]
        assert _found(entry, readings) == [entry]

    @pytest.mark.parametrize("term", ["义父", "发财"])
    def test_everyday_hanviet_words_are_listed_too(
        self, term: str, readings: NameReadings
    ) -> None:
        """"nghĩa phụ", "phát tài" là từ người Việt vẫn dùng — máy không phân biệt được
        với "sở trưởng", nên chỉ liệt kê cho người soát chứ không tự đổi."""
        entry = OTHER_REAL_ENTRIES[term]
        assert _found(entry, readings) == [entry]

    @pytest.mark.parametrize("term", ["公安", "局長", "大隊長", "黑龍江"])
    def test_reading_without_a_note_is_not_listed(
        self, term: str, readings: NameReadings
    ) -> None:
        """Không có chú thích thì không có gì cho thấy phần đọc âm là chưa đủ."""
        listed = [item.term for item in find_reading_only_terms(REAL_GLOSSARY, readings)]
        assert term not in listed

    def test_proper_names_are_read_and_that_is_right(
        self, readings: NameReadings
    ) -> None:
        listed = [item.term for item in find_reading_only_terms(REAL_GLOSSARY, readings)]
        assert "楊柳青" not in listed
        assert _found(OTHER_REAL_ENTRIES["土狗"], readings) == []

    @pytest.mark.parametrize("term", ["綠帽子", "惡有惡報", "借種", "派出所"])
    def test_translated_by_meaning_is_not_listed(
        self, term: str, readings: NameReadings
    ) -> None:
        listed = [item.term for item in find_reading_only_terms(REAL_GLOSSARY, readings)]
        assert term not in listed

    def test_single_character_term_is_left_alone(self, readings: NameReadings) -> None:
        assert _found(OTHER_REAL_ENTRIES["定"], readings) == []

    def test_entry_with_several_options_is_left_alone(
        self, readings: NameReadings
    ) -> None:
        assert _found("通姦 = thông gian / ngoại tình (tội)", readings) == []

    def test_entry_with_several_source_spellings_is_left_alone(
        self, readings: NameReadings
    ) -> None:
        assert _found("通姦 / 通奸 = thông gian (ngoại tình)", readings) == []

    def test_one_wrong_syllable_is_not_a_reading(self, readings: NameReadings) -> None:
        assert _found("所長 = sở tưởng (trưởng đồn công an)", readings) == []

    def test_no_reading_table_means_no_findings(self) -> None:
        assert find_reading_only_terms(REAL_GLOSSARY, NameReadings()) == []

    def test_empty_glossary(self, readings: NameReadings) -> None:
        assert find_reading_only_terms("", readings) == []


def _report(glossary: str, readings: NameReadings | None) -> dict:
    context = SimpleNamespace(
        target_lang="Vietnamese", source_lang="zh", overview="Tóm tắt. " * 400,
        characters="", glossary=glossary, visual_cues="",
    )
    return build_context_diagnostics(
        context=context, source_events=[], name_readings=readings
    )


def _flags(report: dict, code: str) -> list[dict]:
    return [flag for flag in report["quality_flags"] if flag["code"] == code]


class TestContextDiagnosticsReport:
    def test_real_analysis_gets_a_warning_naming_the_entries(
        self, readings: NameReadings
    ) -> None:
        report = _report(REAL_GLOSSARY, readings)
        assert report["glossary"]["reading_only"] == _found(REAL_GLOSSARY, readings)
        (flag,) = _flags(report, "glossary_reading_only")
        assert flag["severity"] == "warning"
        assert flag["message"].startswith("3 mục thuật ngữ chỉ ĐỌC ÂM Hán-Việt")
        assert "所長 = sở trưởng (trưởng đồn công an)" in flag["message"]
        assert "sửa tay" in flag["message"]

    def test_message_names_at_most_four_entries(self, readings: NameReadings) -> None:
        # Bỏ "通姦": bảng thật ở trên đã có mục của thuật ngữ này.
        others = [entry for term, entry in OTHER_REAL_ENTRIES.items() if term != "通姦"]
        glossary = "\n".join([REAL_GLOSSARY, *others])
        (flag,) = _flags(_report(glossary, readings), "glossary_reading_only")
        assert flag["message"].startswith("9 mục")
        assert "… và 5 mục nữa" in flag["message"]

    def test_clean_glossary_has_no_flag(self, readings: NameReadings) -> None:
        report = _report("派出所 = đồn công an\n所長 = trưởng đồn", readings)
        assert report["glossary"]["reading_only"] == []
        assert _flags(report, "glossary_reading_only") == []

    def test_without_the_reading_table_nothing_is_reported(self) -> None:
        report = _report(REAL_GLOSSARY, None)
        assert report["glossary"]["reading_only"] == []
        assert _flags(report, "glossary_reading_only") == []


def _pairs(*entries: str) -> list[tuple[str, str, bool]]:
    parsed = parse_glossary("\n".join(entries))
    return _near_duplicate_pairs([(item.term, item.translation) for item in parsed])


class TestNearDuplicateIgnoresNotes:
    def test_real_case_match_inside_the_note_is_not_a_duplicate(self) -> None:
        """"phối giống" chỉ có trong CHÚ THÍCH của mục 借種."""
        assert _pairs(
            "借種 = mượn giống (thụ tinh nhân tạo/nhờ người phối giống)",
            "配種 = phối giống",
        ) == []

    def test_real_analysis_no_longer_warns(self, readings: NameReadings) -> None:
        report = _report(REAL_GLOSSARY, readings)
        assert report["glossary"]["near_duplicates"] == []
        assert _flags(report, "glossary_near_duplicates") == []

    def test_real_case_option_contained_in_the_other_is_still_reported(self) -> None:
        # Lượt 3f424d20: hai cách viết gần nhau, một bên thêm phương án thứ hai.
        assert _pairs("邪门 = tà môn / quái dị", "邪性 = tà môn") == [
            ("邪门", "邪性", False)
        ]

    def test_real_case_translation_extended_by_the_other_is_still_reported(self) -> None:
        # Lượt 7ed55a62 và f52617e9.
        assert _pairs("大奔 = xe Mercedes", "大G = xe Mercedes G-Class (Đại G)") == [
            ("大奔", "大G", False)
        ]
        assert _pairs(
            "阶层跃升 = đổi đời", "阶级跃升 = đổi đời / nâng tầm giai cấp"
        ) == [("阶层跃升", "阶级跃升", False)]

    def test_identical_translations_with_notes_are_the_same_instruction(self) -> None:
        assert _pairs(
            "满月宴 = tiệc đầy tháng (lễ mừng)", "满月酒 = tiệc đầy tháng (lễ mừng)"
        ) == [("满月宴", "满月酒", True)]

    def test_opposites_are_still_ignored(self) -> None:
        assert _pairs("装货 = chất hàng", "卸货 = dỡ hàng") == []
