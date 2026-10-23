"""[v3.23.484] Test chuẩn đoán ngữ cảnh toàn cục (hàm thuần)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)


@dataclass
class _Ctx:
    target_lang: str = "Vietnamese"
    source_lang: str = "zh"
    overview: str = ""
    characters: str = ""
    glossary: str = ""
    visual_cues: str = ""
    enable_tags: bool = False
    include_desc: bool = True


@dataclass
class _Interval:
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass
class _Event:
    index: int
    text: str
    interval: _Interval = field(default_factory=_Interval)


def _events(*texts: str) -> list[_Event]:
    return [_Event(i, t) for i, t in enumerate(texts, start=1)]


def _codes(report: dict) -> set[str]:
    return {f["code"] for f in report["quality_flags"]}


class TestRoster:
    def test_detects_names_not_in_subtitles(self) -> None:
        ctx = _Ctx(characters="林逸: nam chính, 王总: sếp, Tên Bịa: không có")
        report = build_context_diagnostics(
            context=ctx, source_events=_events("林逸 đi làm", "王总 quát")
        )
        assert report["roster"]["not_found_in_subtitles"] == ["Tên Bịa"]
        assert "roster_names_not_in_subtitles" in _codes(report)

    def test_roster_names_stripped_of_role_description(self) -> None:
        ctx = _Ctx(characters="林逸 (Lâm Dật): nam chính\n王总 - sếp")
        report = build_context_diagnostics(context=ctx, source_events=_events("林逸"))
        assert report["roster"]["names"] == ["林逸", "王总"]

    def test_empty_roster_is_error(self) -> None:
        report = build_context_diagnostics(context=_Ctx(overview="x"), source_events=_events("a"))
        assert "roster_empty" in _codes(report)
        assert report["counts"]["errors"] >= 1

    def test_duplicate_names_deduped(self) -> None:
        ctx = _Ctx(characters="林逸, 林逸, 王总")
        report = build_context_diagnostics(context=ctx, source_events=_events("林逸 王总"))
        assert report["roster"]["names"] == ["林逸", "王总"]


class TestGlossary:
    def test_detects_duplicates(self) -> None:
        ctx = _Ctx(glossary="修炼 = tu luyện\n灵石 = linh thạch\n修炼 = tu luyen")
        report = build_context_diagnostics(context=ctx, source_events=_events("a"))
        assert report["glossary"]["duplicates"] == ["修炼"]
        assert "glossary_duplicates" in _codes(report)

    def test_empty_glossary_is_info_only(self) -> None:
        report = build_context_diagnostics(context=_Ctx(), source_events=_events("a"))
        flags = {f["code"]: f["severity"] for f in report["quality_flags"]}
        assert flags.get("glossary_empty") == "info"


class TestVisualCues:
    def test_detects_out_of_range_indices(self) -> None:
        ctx = _Ctx(visual_cues=json.dumps({"1": "cận cảnh", "9999": "sai"}))
        report = build_context_diagnostics(context=ctx, source_events=_events("a", "b"))
        assert report["visual_cues"]["out_of_range_indices"] == [9999]
        assert "cues_out_of_range" in _codes(report)

    def test_accepts_list_of_records(self) -> None:
        ctx = _Ctx(visual_cues=json.dumps([{"index": 1, "cue": "x"}, {"index": 2, "cue": "y"}]))
        report = build_context_diagnostics(context=ctx, source_events=_events("a", "b"))
        assert report["visual_cues"]["covered_lines"] == 2
        assert report["visual_cues"]["out_of_range_indices"] == []

    def test_invalid_json_does_not_raise(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(visual_cues="{không phải json"), source_events=_events("a")
        )
        assert report["visual_cues"]["parsed"] is False


class TestOverview:
    def test_thin_overview_for_long_movie(self) -> None:
        ctx = _Ctx(overview="Tóm tắt ngắn.", characters="A, B, C")
        report = build_context_diagnostics(
            context=ctx, source_events=_events("A B C"), duration_sec=9435.0
        )
        assert "overview_thin" in _codes(report)

    def test_empty_overview_is_error(self) -> None:
        report = build_context_diagnostics(context=_Ctx(characters="A"), source_events=_events("A"))
        assert "overview_empty" in _codes(report)


class TestPrivacyAndFormat:
    def test_only_file_name_not_full_path(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x"), source_events=_events("a"),
            video_path=r"D:\phim rieng tu\he thong.mp4",
        )
        assert report["video_name"] == "he thong.mp4"
        assert "phim rieng tu" not in json.dumps(report, ensure_ascii=False)

    def test_no_full_subtitle_text_included(self) -> None:
        secret = "DÒNG PHỤ ĐỀ BÍ MẬT KHÔNG ĐƯỢC XUẤT"
        report = build_context_diagnostics(
            context=_Ctx(overview="x", characters="A"), source_events=_events(secret)
        )
        assert secret not in json.dumps(report, ensure_ascii=False)

    def test_overview_excerpt_capped(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 5000), source_events=_events("a")
        )
        assert len(report["summary"]["overview_excerpt"]) <= 280

    def test_report_is_json_serializable(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x", characters="A", glossary="a = b"),
            source_events=_events("A"), app_version="3.23.484",
            analysis_settings={"model": "gemini-3.6-flash", "fps": 1.0},
        )
        json.dumps(report, ensure_ascii=False)  # không raise
        assert report["schema"].startswith("subtitles_extractor.context_diagnostics/")
        assert report["analysis_settings"]["fps"] == 1.0

    def test_no_events_does_not_crash(self) -> None:
        report = build_context_diagnostics(context=_Ctx(overview="x", characters="A"))
        assert report["summary"]["subtitle_lines"] == 0


class TestRosterParsingRealFormat:
    """[v3.23.485] Roster thật: "Tên (phiên âm) [giới, tuổi]: mô tả, mô tả." — mô tả CHỨA dấu phẩy.

    v3.23.484 tách theo dấu phẩy trước → cắt vụn mô tả thành "tên bịa" (23 dương tính giả trên
    dữ liệu thật). Nay phải bóc đúng phần tên.
    """

    REAL = (
        "林逸 (Lâm Dật) [nam, ~24 tuổi]: nhân vật chính, quyết đoán, mưu trí, hiếu thảo., "
        "宋欣爽 (Tống Hân Sảng) [nữ, ~26 tuổi]: vợ, dịu dàng, nhân hậu., "
        "胖子 / 三胖子 (Mập): bạn thân, tung hứng., 医生 (Bác sĩ): vai phụ."
    )

    def test_extracts_only_names_not_description_fragments(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(characters=self.REAL, overview="x" * 3000),
            source_events=_events("林逸", "宋欣爽", "胖子", "医生"),
        )
        assert report["roster"]["names"] == ["林逸", "宋欣爽", "胖子 / 三胖子", "医生"]
        assert report["roster"]["not_found_in_subtitles"] == []
        assert "roster_names_not_in_subtitles" not in _codes(report)

    def test_alias_name_matches_any_alias(self) -> None:
        # Phụ đề chỉ dùng 胖子, roster ghi "胖子 / 三胖子" → KHÔNG được báo bịa tên.
        report = build_context_diagnostics(
            context=_Ctx(characters="胖子 / 三胖子 (Mập): bạn thân.", overview="x" * 3000),
            source_events=_events("胖子 nói"),
        )
        assert report["roster"]["not_found_in_subtitles"] == []

    def test_dash_style_entry_still_parsed(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(characters="林逸 (Lâm Dật): nam chính\n王总 - sếp", overview="x" * 3000),
            source_events=_events("林逸", "王总"),
        )
        assert report["roster"]["names"] == ["林逸", "王总"]

    def test_genuine_fabricated_name_still_flagged(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(characters="林逸: chính, mạnh mẽ\nNhân Vật Bịa: vai phụ", overview="x" * 3000),
            source_events=_events("林逸 đi làm"),
        )
        assert report["roster"]["not_found_in_subtitles"] == ["Nhân Vật Bịa"]


class TestNewQualityFlags:
    """[v3.23.485] Cờ mới từ dữ liệu thật: cues rỗng dù đính video; glossary mỏng."""

    def test_flags_empty_cues_when_video_attached(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 3000, characters="林逸: chính", visual_cues="{}"),
            source_events=_events("林逸"),
            analysis_settings={"attach_video": True},
        )
        assert "cues_empty_despite_video" in _codes(report)

    def test_no_cue_flag_when_video_not_attached(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 3000, characters="林逸: chính", visual_cues="{}"),
            source_events=_events("林逸"),
            analysis_settings={"attach_video": False},
        )
        assert "cues_empty_despite_video" not in _codes(report)

    def test_thin_glossary_flagged_for_long_subtitle(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 9000, characters="林逸: chính", glossary="a = b\nc = d"),
            source_events=_events(*[f"dòng {i}" for i in range(1500)]),
        )
        assert "glossary_thin" in _codes(report)


class TestRosterSeparatorRobustness:
    """[v3.23.487] Roster có thể ngăn entry bằng dấu CHẤM, phẩy, chấm phẩy, xuống dòng hoặc gạch
    đầu dòng; mô tả còn chứa dấu hai chấm THỨ HAI ("Tên [nam]: vai chính: mô tả").

    v3.23.485 chỉ nhận dấu phẩy/xuống dòng → roster ngăn bằng dấu chấm cho ra tên rác
    ("hiếu thảo. 宋欣爽") hoặc sót gần hết (dữ liệu thật: 12 nhân vật → báo cáo chỉ 1).
    """

    BASE = (
        "林逸 (Lâm Dật) [nam, ~24 tuổi]: nhân vật chính: nam thanh niên sở hữu hệ thống, hiếu thảo{S}"
        "宋欣爽 (Tống Hân Sảng) [nữ, ~26 tuổi]: chị dâu: dịu dàng, nhân hậu{S}"
        "秦月 (Tần Nguyệt) [nữ]: phản diện: tham vọng."
    )
    EXPECTED = ["林逸", "宋欣爽", "秦月"]

    @pytest.mark.parametrize("separator", [". ", ", ", "; ", ".\n"])
    def test_all_separators(self, separator: str) -> None:
        from subtitles_extractor.application.services.context_diagnostics import _split_roster
        assert _split_roster(self.BASE.format(S=separator)) == self.EXPECTED

    def test_bullet_list_format(self) -> None:
        from subtitles_extractor.application.services.context_diagnostics import _split_roster
        text = "- 林逸 (Lâm Dật): chính.\n- 宋欣爽: vợ.\n- 秦月: phản diện."
        assert _split_roster(text) == self.EXPECTED

    def test_second_colon_in_description_not_taken_as_name(self) -> None:
        from subtitles_extractor.application.services.context_diagnostics import _split_roster
        names = _split_roster("林逸 (Lâm Dật): nhân vật chính: nam thanh niên nghèo.")
        assert names == ["林逸"]


class TestCueKeySchemas:
    """[v3.23.488] Cues của ứng dụng dùng khoá RÚT GỌN ``id`` (và có thể bọc ``{"cues": [...]}``).

    v3.23.484 chỉ tìm ``index/line/idx`` → đọc ra 0 dòng và báo nhầm "Visual Cues RỖNG" dù log
    ghi rõ đã gộp 709 cues.
    """

    def test_short_key_id_recognized(self) -> None:
        cues = json.dumps([{"id": 1, "spk": "林逸", "to": "宋欣爽"}, {"id": 2, "spk": "秦月"}])
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 3000, characters="林逸: chính", visual_cues=cues),
            source_events=_events("a", "b"),
            analysis_settings={"attach_video": True},
        )
        assert report["visual_cues"]["covered_lines"] == 2
        assert "cues_empty_despite_video" not in _codes(report)

    def test_wrapped_cues_object_recognized(self) -> None:
        cues = json.dumps({"cues": [{"id": 1, "cue": "cận cảnh"}]})
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 3000, characters="林逸: chính", visual_cues=cues),
            source_events=_events("a"),
            analysis_settings={"attach_video": True},
        )
        assert report["visual_cues"]["covered_lines"] == 1

    def test_genuinely_empty_cues_still_flagged(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 3000, characters="林逸: chính", visual_cues="{}"),
            source_events=_events("a"),
            analysis_settings={"attach_video": True},
        )
        assert "cues_empty_despite_video" in _codes(report)


class TestVideoIdentifiedCharacters:
    """[v3.23.494] Nhân vật model nhận ra QUA HÌNH ẢNH (không được gọi tên trong thoại) là hợp lệ.

    Dữ liệu thật báo 15 tên "nghi AI bịa", phần lớn là nhãn mô tả từ video (绿发路人, 粉裙女同学)
    — dương tính giả. Có mặt trong Visual Cues chính là bằng chứng model thật sự thấy họ.
    """

    ROSTER = (
        "林逸 (Lâm Dật) [nam] — chính\n"
        "绿发路人 [nữ] — người qua đường tóc xanh\n"
        "Nhân Vật Bịa [nam] — không tồn tại"
    )
    CUES = json.dumps([{"id": 1, "spk": "林逸"}, {"id": 2, "spk": "绿发路人"}])

    def _report(self):
        return build_context_diagnostics(
            context=_Ctx(overview="x" * 9000, characters=self.ROSTER, visual_cues=self.CUES),
            source_events=_events("林逸 đi làm", "chuyện khác"),
            analysis_settings={"attach_video": True},
        )

    def test_video_only_label_not_flagged_as_fabricated(self) -> None:
        report = self._report()
        assert report["roster"]["identified_from_video"] == ["绿发路人"]
        assert "绿发路人" not in report["roster"]["not_found_in_subtitles"]

    def test_genuine_fabrication_still_flagged(self) -> None:
        report = self._report()
        assert report["roster"]["not_found_in_subtitles"] == ["Nhân Vật Bịa"]
        assert "roster_names_not_in_subtitles" in _codes(report)

    def test_addressee_label_also_counts(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 9000, characters="粉裙女同学 [nữ] — bạn học",
                         visual_cues=json.dumps([{"id": 1, "to": "粉裙女同学"}])),
            source_events=_events("chuyện khác"),
        )
        assert report["roster"]["identified_from_video"] == ["粉裙女同学"]


class TestGlossaryNearDuplicates:
    """[v3.23.494] Cùng thực thể ghi hai cách (福安私立医院 / 福安私人医院) → chỉ thị mâu thuẫn."""

    def test_detects_one_char_variants(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 9000, characters="林逸 — chính",
                         glossary="福安私立医院 = BV Phúc An\n福安私人医院 = BV Phúc An"),
            source_events=_events("林逸"),
        )
        assert report["glossary"]["near_duplicates"] == ["福安私立医院 ~ 福安私人医院"]
        assert "glossary_near_duplicates" in _codes(report)

    def test_distinct_terms_not_flagged(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 9000, characters="林逸 — chính",
                         glossary="宠妻系统 = Hệ thống\n黑金卡 = thẻ đen"),
            source_events=_events("林逸"),
        )
        assert report["glossary"]["near_duplicates"] == []


class TestNearDuplicateNeedsSameMeaning:
    """[v3.23.496] Chỉ báo gần trùng khi BẢN DỊCH cũng giống — tránh báo nhầm cặp TRÁI NGHĨA."""

    def _near(self, glossary: str) -> list[str]:
        report = build_context_diagnostics(
            context=_Ctx(overview="x" * 9000, characters="林逸 — chính", glossary=glossary),
            source_events=_events("林逸"),
        )
        return report["glossary"]["near_duplicates"]

    def test_same_translation_flagged(self) -> None:
        assert self._near("满月宴 = tiệc đầy tháng\n满月酒 = tiệc đầy tháng") == ["满月宴 ~ 满月酒"]

    def test_opposite_meaning_not_flagged(self) -> None:
        # 装货 (chất hàng) vs 卸货 (dỡ hàng): khác 1 ký tự nhưng NGƯỢC nghĩa.
        assert self._near("装货 = chất hàng\n卸货 = dỡ hàng") == []

    def test_missing_translation_not_flagged(self) -> None:
        assert self._near("装货\n卸货 = dỡ hàng") == []


class TestRosterNameLanguage:
    """[v3.23.536] Tên nhân vật phải là tên GỐC, không phải bản dịch.

    Dữ liệu thật vi phạm: roster có "Cậu bé lanh lợi (Cậu bé lanh lợi)" — cả hai vế đều tiếng Việt.
    Khi đó bảng quy đổi tên gửi cho khâu dịch không có gì để tra, nhãn người nói trong Visual Cues
    (vốn ghi bằng tên gốc) không khớp, và mốc neo căn dòng mất tác dụng với nhân vật đó.
    """

    def test_vietnamese_name_flagged(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(
                overview="x" * 9000,
                characters="林逸 (Lâm Dật) — chính\nCậu bé lanh lợi (Cậu bé lanh lợi) — phụ",
                source_lang="zh",
            ),
            source_events=_events("林逸"),
        )
        assert "roster_names_not_in_source_language" in _codes(report)

    def test_all_original_names_not_flagged(self) -> None:
        report = build_context_diagnostics(
            context=_Ctx(
                overview="x" * 9000,
                characters="林逸 (Lâm Dật) — chính\n王梅 (Vương Mai) — hàng xóm",
                source_lang="zh",
            ),
            source_events=_events("林逸"),
        )
        assert "roster_names_not_in_source_language" not in _codes(report)

    def test_latin_source_language_not_flagged(self) -> None:
        """Phim tiếng Anh thì tên Latin là ĐÚNG — không được báo nhầm."""
        report = build_context_diagnostics(
            context=_Ctx(
                overview="x" * 9000,
                characters="John Smith [nam] — bác sĩ",
                source_lang="en",
            ),
            source_events=_events("John"),
        )
        assert "roster_names_not_in_source_language" not in _codes(report)

    def test_alias_with_original_name_accepted(self) -> None:
        """Bí danh đầu là tên gốc thì hợp lệ."""
        report = build_context_diagnostics(
            context=_Ctx(
                overview="x" * 9000,
                characters="胖子 / Mập (Mập) — bạn thân",
                source_lang="zh",
            ),
            source_events=_events("胖子"),
        )
        assert "roster_names_not_in_source_language" not in _codes(report)
