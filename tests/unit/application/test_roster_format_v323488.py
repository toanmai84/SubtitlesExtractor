"""[v3.23.488] Quy cách CỐ ĐỊNH cho bảng nhân vật & thuật ngữ.

Gốc vấn đề: schema cũ bắt AI trả chuỗi TỰ DO nên mỗi lần một kiểu → parser luôn sai. Nay AI trả
mảng đối tượng, chuẩn hoá về văn bản mỗi mục MỘT DÒNG → đọc lại tất định.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.roster_format import (
    GlossaryEntry,
    RosterEntry,
    name_of_line,
    parse_glossary,
    parse_roster,
    render_glossary,
    render_roster,
    roster_names,
)

AI_ARRAY = [
    {"name": "林逸", "aka": "Lâm Dật", "gender": "nam", "age": "~24 tuổi",
     "role": "nhân vật chính: sở hữu hệ thống, hiếu thảo"},
    {"name": "宋欣爽", "aka": "Tống Hân Sảng", "gender": "nữ", "age": "~26 tuổi",
     "role": "chị dâu: dịu dàng"},
    {"name": "胖子 / 三胖子", "aka": "Mập", "role": "bạn thân"},
]


class TestRoundTrip:
    def test_array_to_canonical_text_one_line_each(self) -> None:
        text = render_roster(parse_roster(AI_ARRAY))
        assert len(text.splitlines()) == 3
        assert text.splitlines()[0] == "林逸 (Lâm Dật) [nam, ~24 tuổi] — nhân vật chính: sở hữu hệ thống, hiếu thảo"

    def test_canonical_text_reads_back_deterministically(self) -> None:
        text = render_roster(parse_roster(AI_ARRAY))
        assert roster_names(text) == ["林逸", "宋欣爽", "胖子 / 三胖子"]

    def test_description_colon_does_not_break_name(self) -> None:
        # Mô tả chứa dấu hai chấm THỨ HAI — không được nhầm thành tên.
        text = render_roster(parse_roster(AI_ARRAY))
        assert "nhân vật chính" not in roster_names(text)

    def test_duplicate_names_deduped(self) -> None:
        entries = parse_roster([{"name": "林逸"}, {"name": "林逸"}, {"name": "王总"}])
        assert [e.name for e in entries] == ["林逸", "王总"]


class TestLegacyFallback:
    """Dữ liệu CŨ (chuỗi tự do) và dữ liệu gõ tay vẫn phải đọc được."""

    @pytest.mark.parametrize("separator", [". ", ", ", "; ", "\n"])
    def test_freeform_separators(self, separator: str) -> None:
        text = (
            f"林逸 (Lâm Dật) [nam, ~24 tuổi]: chính: mạnh mẽ{separator}"
            f"宋欣爽 (Tống Hân Sảng) [nữ]: vợ: dịu dàng."
        )
        assert roster_names(text) == ["林逸", "宋欣爽"]

    def test_simple_comma_list(self) -> None:
        assert roster_names("林逸, 王总, Tên Bịa") == ["林逸", "王总", "Tên Bịa"]

    def test_bullet_list(self) -> None:
        assert roster_names("- 林逸: chính\n- 王总: sếp") == ["林逸", "王总"]


class TestNameOfLine:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("林逸 (Lâm Dật) [nam, ~24 tuổi] — nhân vật chính", "林逸"),
            ("林逸 (Lâm Dật): chính", "林逸"),
            ("王总 - sếp", "王总"),
            ("胖子 / 三胖子 (Mập) — bạn thân", "胖子 / 三胖子"),
            ("", ""),
        ],
    )
    def test_cases(self, line: str, expected: str) -> None:
        assert name_of_line(line) == expected


class TestGlossary:
    def test_array_round_trip(self) -> None:
        text = render_glossary(parse_glossary(
            [{"term": "宠妻系统", "translation": "Hệ thống Cưng Chiều Vợ"},
             {"term": "黑金卡", "translation": "thẻ đen"}]
        ))
        assert text.splitlines() == ["宠妻系统 = Hệ thống Cưng Chiều Vợ", "黑金卡 = thẻ đen"]
        assert [e.term for e in parse_glossary(text)] == ["宠妻系统", "黑金卡"]

    @pytest.mark.parametrize("separator", ["=", "=>", "->", "→", ":", "|"])
    def test_legacy_separators(self, separator: str) -> None:
        entries = parse_glossary(f"修炼 {separator} tu luyện")
        assert entries == [GlossaryEntry(term="修炼", translation="tu luyện")]

    def test_term_without_translation_kept(self) -> None:
        assert render_glossary(parse_glossary(["黑金卡"])) == "黑金卡"


class TestRenderEdgeCases:
    def test_entry_without_optional_fields(self) -> None:
        assert RosterEntry(name="林逸").render() == "林逸"

    def test_empty_names_dropped(self) -> None:
        assert render_roster([RosterEntry(name="  "), RosterEntry(name="林逸")]) == "林逸"

    def test_empty_input(self) -> None:
        assert parse_roster(None) == []
        assert parse_roster("") == []
        assert render_roster([]) == ""


class TestSequentialMerge:
    """[v3.23.489] Tích luỹ ngữ cảnh qua nhiều đoạn — trước đây đoạn CUỐI ghi đè tất cả."""

    def test_roster_accumulates_across_segments(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_roster
        merged = merge_roster("林逸 (Lâm Dật) [nam] — chính", "宋欣爽 (Tống Hân Sảng) [nữ] — vợ")
        assert roster_names(merged) == ["林逸", "宋欣爽"]

    def test_same_character_keeps_richer_description(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_roster
        merged = merge_roster("林逸 — chính", "林逸 (Lâm Dật) [nam, ~24 tuổi] — nhân vật chính, sở hữu hệ thống")
        assert len(merged.splitlines()) == 1
        assert "Lâm Dật" in merged and "sở hữu hệ thống" in merged

    def test_glossary_accumulates_and_dedupes(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_glossary
        merged = merge_glossary("宠妻系统 = Hệ thống", "宠妻系统 = Hệ thống\n黑金卡 = thẻ đen")
        assert merged.splitlines() == ["宠妻系统 = Hệ thống", "黑金卡 = thẻ đen"]

    def test_overview_appends_in_order(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_overview
        merged = merge_overview("Phần đầu phim.", "Phần cuối phim.")
        assert "Phần đầu phim." in merged and "Phần cuối phim." in merged

    def test_overview_no_duplicate_when_model_expands(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_overview
        assert merge_overview("Phần A.", "Phần A. Phần B.") == "Phần A. Phần B."
        assert merge_overview("Phần A. Phần B.", "Phần A.") == "Phần A. Phần B."

    def test_overview_capped_keeps_head_and_tail(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_overview
        merged = merge_overview("ĐẦU" + "x" * 5000, "y" * 5000 + "CUỐI", max_chars=2000)
        assert merged.startswith("ĐẦU") and merged.endswith("CUỐI")
        assert "lược bớt phần giữa" in merged
        assert len(merged) <= 2200

    def test_empty_sides(self) -> None:
        from subtitles_extractor.application.services.roster_format import (
            merge_glossary, merge_overview, merge_roster,
        )
        assert merge_overview("", "A") == "A"
        assert merge_overview("A", "") == "A"
        assert roster_names(merge_roster("", "林逸 — chính")) == ["林逸"]
        assert merge_glossary("", "a = b") == "a = b"


class TestAdapterSequentialAccumulation:
    """Kiểm ở tầng adapter: 23 đoạn phải tích luỹ đủ, không bị đoạn cuối ghi đè."""

    def test_23_segments_accumulate(self) -> None:
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleContextAnalysis,
        )
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )
        accumulated = None
        for i in range(1, 24):
            segment = SubtitleContextAnalysis(
                source_lang="zh",
                characters=f"NV{i} (Nhân {i}) [nam] — vai {i}",
                overview=f"Đoạn {i}: diễn biến phần {i}.",
                glossary=f"thuat_ngu_{i} = dịch {i}",
            )
            accumulated = GeminiSubtitleTranslator._merge_sequential_analysis(accumulated, segment)
        assert len(accumulated.characters.splitlines()) == 23
        assert len(accumulated.glossary.splitlines()) == 23
        assert "Đoạn 1:" in accumulated.overview
        assert "Đoạn 23:" in accumulated.overview


class TestFullWidthPunctuation:
    """[v3.23.492] Văn bản CJK dùng dấu câu FULL-WIDTH — không quy đổi thì tên dính cả phiên âm."""

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("林逸（Lâm Dật）[nam] — chính", ["林逸"]),
            ("宋欣爽（Tống Hân Sảng）［nữ］— vợ", ["宋欣爽"]),
            ("王梅（Vương Mai）［nữ, trung niên］— mẹ", ["王梅"]),
            ("林逸 (Lâm Dật) [nam] — chính", ["林逸"]),
        ],
    )
    def test_names_parsed(self, line: str, expected: list[str]) -> None:
        assert roster_names(line) == expected

    def test_aka_extracted_from_fullwidth(self) -> None:
        entries = parse_roster("林逸（Lâm Dật）[nam, ~24 tuổi] — chính")
        assert entries[0].aka == "Lâm Dật"
        assert entries[0].gender == "nam"


class TestGlossarySeparatorPriority:
    """[v3.23.492] Cả thuật ngữ lẫn bản dịch đều có thể chứa dấu ngăn — phải tách đúng chỗ."""

    @pytest.mark.parametrize(
        ("line", "term", "translation"),
        [
            # Bản dịch chứa ":" — phải tách ở "=" (chính ví dụ prompt nêu).
            ("米水比例 = tỉ lệ gạo:nước (gạo trước, nước sau)",
             "米水比例", "tỉ lệ gạo:nước (gạo trước, nước sau)"),
            # Thuật ngữ chứa ":" — vẫn tách ở "=".
            ("tỉ lệ gạo:nước = cách gọi", "tỉ lệ gạo:nước", "cách gọi"),
            # Thuật ngữ chứa "=" (công thức) — tách ở " = " có khoảng trắng.
            ("C=O = liên kết đôi", "C=O", "liên kết đôi"),
            # Các dạng thông thường.
            ("修炼 = tu luyện", "修炼", "tu luyện"),
            ("修炼=tu luyện", "修炼", "tu luyện"),
            ("FBI => FBI (Cục Điều tra)", "FBI", "FBI (Cục Điều tra)"),
            ("宠妻系统：Hệ thống Cưng Chiều Vợ", "宠妻系统", "Hệ thống Cưng Chiều Vợ"),
        ],
    )
    def test_split_priority(self, line: str, term: str, translation: str) -> None:
        entries = parse_glossary(line)
        assert entries[0].term == term
        assert entries[0].translation == translation


class TestOverviewDedupe:
    """[v3.23.493] Mỗi đoạn thường tóm tắt lại phần trước — nối thẳng qua 23 đoạn sẽ tích tụ
    nội dung TRÙNG, phình prompt và làm model rối."""

    def test_repeated_long_sentence_dropped(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_overview
        merged = merge_overview(
            "Phần A dài đủ để khử trùng lặp nội dung. Phần B cũng dài đủ để khử trùng.",
            "Phần B cũng dài đủ để khử trùng. Phần C hoàn toàn mới và dài đủ.",
        )
        assert merged.count("Phần B cũng dài đủ để khử trùng.") == 1
        assert "Phần C hoàn toàn mới" in merged

    def test_short_repeated_sentence_kept(self) -> None:
        # Câu ngắn có thể lặp hợp lệ ở nhiều đoạn — không được xoá.
        from subtitles_extractor.application.services.roster_format import merge_overview
        merged = merge_overview(
            "Anh đi. Nội dung dài đủ để khử trùng lặp ở đây nhé.",
            "Anh đi. Nội dung mới hoàn toàn khác biệt và dài.",
        )
        assert merged.count("Anh đi.") == 2

    def test_nothing_new_returns_previous(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_overview
        text = "Nội dung dài đủ để khử trùng lặp nội dung nhé."
        assert merge_overview(text, text) == text


class TestMergeWithMixedFormats:
    """Hai đoạn có thể ghi cùng nhân vật ở dạng khác nhau (full-width vs ASCII)."""

    def test_no_duplicate_entry_across_formats(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_roster
        merged = merge_roster(
            "林逸（Lâm Dật）[nam] — chính",
            "林逸 (Lâm Dật) [nam, ~24 tuổi] — nhân vật chính, sở hữu hệ thống",
        )
        assert len(merged.splitlines()) == 1
        assert "~24 tuổi" in merged and "sở hữu hệ thống" in merged

    def test_first_transliteration_wins(self) -> None:
        # Giữ phiên âm ĐẦU để tên không "trôi" giữa các đoạn.
        from subtitles_extractor.application.services.roster_format import merge_roster
        merged = merge_roster("林逸 (Lâm Dật) [nam] — chính", "林逸 (Lam Dat) [nam] — chính")
        assert "Lâm Dật" in merged and "Lam Dat" not in merged

    def test_glossary_first_translation_wins(self) -> None:
        from subtitles_extractor.application.services.roster_format import merge_glossary
        merged = merge_glossary("宠妻系统 = Hệ thống Cưng Chiều Vợ", "宠妻系统 = Hệ thống Sủng Thê")
        assert merged == "宠妻系统 = Hệ thống Cưng Chiều Vợ"


class TestAliasCoalescing:
    """[v3.23.496] Cùng một người bị model tách hai mục có bí danh chồng nhau.

    Dữ liệu thật: "胖子 / 三胖子" và "胖子 / 三胖子 / 王大福" — chia sẻ bí danh nên chắc chắn là
    một người; để hai mục sẽ khiến bảng quy đổi tên gửi cho khâu dịch bị nhiễu.
    """

    def test_entries_sharing_alias_merged(self) -> None:
        text = (
            "胖子 / 三胖子 (Mập) [nam] — bạn thân\n"
            "胖子 / 三胖子 / 王大福 (Vương Đại Phúc) [nam] — bạn thân, chủ tiệm\n"
            "林逸 (Lâm Dật) [nam] — chính"
        )
        names = roster_names(text)
        assert len(names) == 2
        assert "王大福" in names[0] and "三胖子" in names[0]

    def test_alias_union_preserves_order(self) -> None:
        names = roster_names("胖子 (Mập) — bạn\n三胖子 / 胖子 (Mập) — bạn thân")
        assert names == ["胖子 / 三胖子"]

    def test_distinct_people_not_merged(self) -> None:
        names = roster_names("林逸 — chính\n王梅 / 梅姐 — hàng xóm")
        assert names == ["林逸", "王梅 / 梅姐"]

    def test_richer_description_kept(self) -> None:
        from subtitles_extractor.application.services.roster_format import parse_roster
        entries = parse_roster("胖子 — bạn\n胖子 / 王大福 (Vương Đại Phúc) [nam] — bạn thân, chủ tiệm")
        assert entries[0].aka == "Vương Đại Phúc"
        assert "chủ tiệm" in entries[0].role
