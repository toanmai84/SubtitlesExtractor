"""[v3.23.567] Hồi quy: phiên âm tên nhân vật sai âm Hán-Việt lan ra toàn bộ bản dịch.

Chuẩn đoán thật v3.23.566 (phân tích bằng gemini-3.6-flash): bảng nhân vật ghi
``柳慕云 (Liễu Mộc Vân)`` — 慕 đọc "mộ". Bản dịch chép theo bảng: 44/45 dòng nhắc tới cô
mang tên sai, tóm tắt sai 28 chỗ. Soát 40 bảng nhân vật thật khác:
``赵可可 (Triệu Kha Kha)`` (可 đọc "khả"); còn các tên Việt hoá có chủ ý
(``吴胖`` → "Ngô Béo") và thuật ngữ theo âm quy ước (``心包经`` → "Tâm Bào Kinh") thì
KHÔNG được sửa.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from subtitles_extractor.application.services.name_reading_check import (
    NameReadings,
    apply_name_corrections,
    correct_context_names,
    find_name_reading_issues,
    name_corrections,
    syllable_key,
)
from subtitles_extractor.application.services.roster_format import (
    merge_roster,
    parse_roster,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    TranslationContext,
    TranslationLine,
)
from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    SqliteTranslationSessionStore,
)
from subtitles_extractor.infrastructure.translation.name_readings_loader import (
    HANVIET_READINGS_FILENAME,
    load_name_readings,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"

READINGS = NameReadings.from_encoded({
    "柳": "liễu",
    "慕": "mộ",
    "云": "vân",
    "木": "mộc",
    "赵": "triệu",
    "可": "khả khắc",
    "吴": "ngô",
    "胖": "bàn",
    "心": "tâm",
    "包": "bao",
    "经": "kinh",
    "长": "trường trưởng",
    "张": "trương",
    "和": "hoà hoạ",
    "李": "lý",
    "任": "nhâm nhiệm|nhậm",
    "盈": "doanh",
    "宋": "tống",
    "钱": "tiền",
})

REAL_ROSTER = (
    "宋钱 / 宋神医 (Tống Tiền) [nam, ~23] — nhân vật chính. Xưng hô: với 柳慕云 tự xưng "
    "'tôi' gọi 'chị Mộc Vân' (danh xưng: Liễu tiểu thư)\n"
    "柳慕云 (Liễu Mộc Vân) [nữ, ~25] — Tổng giám đốc Tập đoàn Liễu thị. Xưng hô: "
    "với 宋钱 tự xưng 'chị' gọi 'em Tiền'\n"
    "赵可可 / 可可 (Triệu Kha Kha / Kha Kha) [nữ, ~22] — trợ lý của Liễu Mộc Vân"
)
REAL_OVERVIEW = (
    "Tống Tiền chữa khỏi bệnh cho Liễu Mộc Vân. Mộc Vân thầm yêu anh; bàn gỗ mộc trong "
    "phòng Mộc Vân là quà của Kha Kha."
)


def _analysis(**overrides: str) -> SubtitleContextAnalysis:
    fields = {
        "source_lang": "zh",
        "characters": REAL_ROSTER,
        "overview": REAL_OVERVIEW,
        "glossary": "",
        "visual_cues": '[{"id": 1, "spk": "柳慕云", "cue": "Liễu Mộc Vân mỉm cười"}]',
    }
    fields.update(overrides)
    return SubtitleContextAnalysis(**fields)


class TestSyllableKey:
    def test_tone_mark_position_is_ignored(self) -> None:
        assert syllable_key("Hoà") == syllable_key("hòa")

    def test_final_y_equals_i(self) -> None:
        assert syllable_key("Lý") == syllable_key("lí")

    def test_different_tones_differ(self) -> None:
        assert syllable_key("Mộ") != syllable_key("Mô")


class TestRealCase:
    def test_wrong_reading_is_found_with_a_fix(self) -> None:
        issues = find_name_reading_issues(REAL_ROSTER, "", READINGS)
        by_char = {issue.char: issue for issue in issues}
        assert by_char["慕"].wrote == "Mộc"
        assert by_char["慕"].suggestion == "Mộ"
        assert by_char["可"].suggestion == "Khả"

    def test_name_is_fixed_everywhere(self) -> None:
        fixed, fixes = correct_context_names(_analysis(), READINGS)
        assert fixes
        for text in (fixed.characters, fixed.overview, fixed.visual_cues):
            assert "Mộc Vân" not in text
        assert "chị Mộ Vân" in fixed.characters
        assert "(Liễu Mộ Vân)" in fixed.characters
        assert "(Triệu Khả Khả / Khả Khả)" in fixed.characters
        assert "quà của Khả Khả" in fixed.overview
        assert "Liễu Mộ Vân mỉm cười" in fixed.visual_cues

    def test_lone_common_word_is_untouched(self) -> None:
        fixed, _ = correct_context_names(_analysis(), READINGS)
        assert "bàn gỗ mộc" in fixed.overview

    def test_fixed_context_is_clean(self) -> None:
        fixed, _ = correct_context_names(_analysis(), READINGS)
        assert not [
            issue
            for issue in find_name_reading_issues(fixed.characters, "", READINGS)
            if issue.suggestion
        ]

    def test_translation_context_is_supported(self) -> None:
        context = TranslationContext(
            target_lang="Vietnamese", characters=REAL_ROSTER, overview=REAL_OVERVIEW
        )
        fixed, _ = correct_context_names(context, READINGS)
        assert isinstance(fixed, TranslationContext)
        assert "Liễu Mộ Vân" in fixed.overview


class TestNoFalseFix:
    def test_intentional_localized_nickname_is_only_reported(self) -> None:
        issues = find_name_reading_issues("吴胖 (Ngô Béo) — bạn thân", "", READINGS)
        assert [issue.char for issue in issues] == ["胖"]
        assert issues[0].suggestion == ""

    def test_glossary_is_never_auto_fixed(self) -> None:
        glossary = "心包经 = Tâm Bào Kinh"
        issues = find_name_reading_issues("", glossary, READINGS)
        assert issues and not any(issue.suggestion for issue in issues)
        context = _analysis(characters="", overview="Tâm Bào Kinh", glossary=glossary)
        assert correct_context_names(context, READINGS) == (context, [])

    def test_ambiguous_nearest_reading_is_not_guessed(self) -> None:
        issues = find_name_reading_issues("张长 (Trương Trương) — ông", "", READINGS)
        assert issues and issues[0].suggestion == ""

    def test_extra_readings_are_accepted_but_never_suggested(self) -> None:
        assert find_name_reading_issues("任盈 (Nhậm Doanh) — nữ", "", READINGS) == []

    def test_translated_title_is_skipped(self) -> None:
        roster = "柳慕云 (Tổng giám đốc) — sếp"
        assert find_name_reading_issues(roster, "", READINGS) == []

    def test_other_character_phrase_is_protected(self) -> None:
        roster = "柳慕云 (Liễu Mộc Vân) — nữ\n柳木 (Liễu Mộc) — anh trai"
        issues = find_name_reading_issues(roster, "", READINGS)
        fixes = [issue for issue in issues if issue.suggestion]
        corrections = name_corrections(fixes, roster)
        assert "Liễu Mộc" not in corrections
        assert corrections["Liễu Mộc Vân"] == "Liễu Mộ Vân"

    def test_nothing_to_fix_returns_same_object(self) -> None:
        context = _analysis(characters="宋钱 (Tống Tiền) — chính", overview="Tống Tiền.")
        assert correct_context_names(context, READINGS)[0] is context

    def test_no_table_means_no_check(self) -> None:
        context = _analysis()
        assert correct_context_names(context, None) == (context, [])
        assert correct_context_names(context, NameReadings()) == (context, [])


class TestApplyCorrections:
    def test_whole_words_only(self) -> None:
        text = apply_name_corrections("XMộc Vân và Mộc Vân", {"Mộc Vân": "Mộ Vân"})
        assert text == "XMộc Vân và Mộ Vân"

    def test_decomposed_text_is_matched(self) -> None:
        import unicodedata

        text = unicodedata.normalize("NFD", "chị Mộc Vân")
        assert apply_name_corrections(text, {"Mộc Vân": "Mộ Vân"}) == "chị Mộ Vân"


class TestBundledTable:
    def test_real_table_knows_the_real_case(self) -> None:
        readings = load_name_readings(DATA_DIR / HANVIET_READINGS_FILENAME)
        assert "mộ" in readings.suggestable("慕")
        assert "khả" in readings.suggestable("可")
        issues = find_name_reading_issues(REAL_ROSTER, "", readings)
        assert {(i.char, i.suggestion) for i in issues} == {("慕", "Mộ"), ("可", "Khả")}

    def test_real_table_accepts_established_readings(self) -> None:
        readings = load_name_readings(DATA_DIR / HANVIET_READINGS_FILENAME)
        roster = (
            "宋钱 (Tống Tiền) — chính\n任盈盈 (Nhậm Doanh Doanh) — nữ\n"
            "诸葛亮 (Gia Cát Lượng) — quân sư\n吴胖 (Ngô Béo) — bạn"
        )
        issues = find_name_reading_issues(roster, "", readings)
        fixes = [issue for issue in issues if issue.suggestion]
        assert fixes == []

    def test_missing_file_gives_empty_table(self, tmp_path: Path) -> None:
        assert not load_name_readings(tmp_path / "none.json")

    def test_wrong_schema_gives_empty_table(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema": "other", "readings": {}}), encoding="utf-8")
        assert not load_name_readings(path)

    def test_corrupt_file_gives_empty_table(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{", encoding="utf-8")
        assert not load_name_readings(path)


class _FakeTranslator:
    def __init__(self, result: SubtitleContextAnalysis) -> None:
        self._result = result

    def analyze_global_context(self, **_kwargs: object) -> SubtitleContextAnalysis:
        return self._result


class TestAnalyzeUseCase:
    def test_analysis_result_is_corrected(self) -> None:
        use_case = AnalyzeSubtitleContextUseCase(
            translator=_FakeTranslator(_analysis()),  # type: ignore[arg-type]
            name_readings=READINGS,
        )
        lines = [TranslationLine(index=1, start_ms=0, end_ms=1, text="我叫柳慕云")]
        result = use_case.execute(lines, "Vietnamese")
        assert "Liễu Mộ Vân" in result.characters
        assert "Mộc Vân" not in result.overview

    def test_without_table_result_is_untouched(self) -> None:
        analysis = _analysis()
        use_case = AnalyzeSubtitleContextUseCase(
            translator=_FakeTranslator(analysis)  # type: ignore[arg-type]
        )
        lines = [TranslationLine(index=1, start_ms=0, end_ms=1, text="我叫柳慕云")]
        assert use_case.execute(lines, "Vietnamese") is analysis


class TestContextDiagnostics:
    @staticmethod
    def _report(characters: str, glossary: str = "") -> dict:
        context = SimpleNamespace(
            target_lang="Vietnamese",
            source_lang="zh",
            overview="Tóm tắt.",
            characters=characters,
            glossary=glossary,
            visual_cues="",
            enable_tags=False,
            include_desc=True,
        )
        return build_context_diagnostics(
            context=context, source_events=[], name_readings=READINGS
        )

    def test_roster_mismatch_is_a_warning_with_details(self) -> None:
        report = self._report(REAL_ROSTER)
        flags = {flag["code"]: flag for flag in report["quality_flags"]}
        assert flags["roster_name_reading"]["severity"] == "warning"
        assert "Mộc" in flags["roster_name_reading"]["message"]
        issues = report["roster"]["name_reading_issues"]
        assert {"char": "慕", "auto_fix": "Mộ"}.items() <= issues[0].items()

    def test_glossary_mismatch_is_info_only(self) -> None:
        report = self._report("宋钱 (Tống Tiền) — chính", "心包经 = Tâm Bào Kinh")
        flags = {flag["code"]: flag for flag in report["quality_flags"]}
        assert flags["glossary_name_reading"]["severity"] == "info"
        assert "roster_name_reading" not in flags

    def test_clean_roster_has_no_flag(self) -> None:
        report = self._report("宋钱 (Tống Tiền) — chính")
        codes = {flag["code"] for flag in report["quality_flags"]}
        assert not {"roster_name_reading", "glossary_name_reading"} & codes


class TestStoreKeepsAnalysisValid:
    def test_save_analysis_texts_keeps_hash(self, tmp_path: Path) -> None:
        store = SqliteTranslationSessionStore(tmp_path / "s.db")
        store.save_analysis(
            "video.mp4",
            characters="柳慕云 (Liễu Mộc Vân)",
            overview="Liễu Mộc Vân.",
            source_lang="zh",
            target_lang="Vietnamese",
            input_hash="h1",
            glossary="g",
            visual_cues="[]",
        )
        store.save_analysis_texts(
            "video.mp4",
            characters="柳慕云 (Liễu Mộ Vân)",
            overview="Liễu Mộ Vân.",
            glossary="g",
            visual_cues="[]",
        )
        session = store.get("video.mp4")
        assert session is not None
        assert session.analysis_characters == "柳慕云 (Liễu Mộ Vân)"
        assert session.analysis_overview == "Liễu Mộ Vân."
        assert session.analysis_source_lang == "zh"
        assert session.has_valid_analysis("h1")
        store.close()

    def test_empty_key_is_rejected(self, tmp_path: Path) -> None:
        store = SqliteTranslationSessionStore(tmp_path / "s.db")
        with pytest.raises(ValueError):
            store.save_analysis_texts(
                "", characters="", overview="", glossary="", visual_cues=""
            )
        store.close()


class TestRosterMergeDuplicates:
    """Ca thật v3.23.566: bảng 30 dòng có 3 nhân vật lặp sau khi gộp các đoạn video."""

    def test_reversed_aliases_are_one_character(self) -> None:
        merged = merge_roster(
            "柳杰 / 柳河 (Liễu Kiệt) [nam, ~22] — em trai",
            "柳河 / 柳杰 (Liễu Hà / Liễu Kiệt) [nam, ~22] — em trai, bắt cóc Đào Hoa",
        )
        entries = parse_roster(merged)
        assert len(merged.splitlines()) == 1
        assert entries[0].aka == "Liễu Kiệt / Liễu Hà"
        assert "bắt cóc" in entries[0].role

    def test_subset_alias_is_merged(self) -> None:
        merged = merge_roster(
            "春花 / 许春花 (Hứa Xuân Hoa) [nữ, ~28] — vợ Trương Thiết Trụ",
            "许春花 (Hứa Xuân Hoa) [nữ, ~28] — vợ Thiết Trụ",
        )
        assert len(merged.splitlines()) == 1

    def test_distinct_people_stay_apart(self) -> None:
        merged = merge_roster("柳杰 (Liễu Kiệt) — em", "柳慕云 (Liễu Mộ Vân) — chị")
        assert len(merged.splitlines()) == 2
