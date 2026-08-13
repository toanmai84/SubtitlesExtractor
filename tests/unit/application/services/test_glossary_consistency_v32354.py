"""Test [v3.23.54] kiểm tra nhất quán bảng thuật ngữ trong bản dịch."""

from __future__ import annotations

from subtitles_extractor.application.services.glossary_consistency import (
    check_glossary_consistency,
    parse_glossary,
)


class TestParseGlossary:
    def test_parses_arrow_separator(self) -> None:
        entries = parse_glossary("灵力 => linh lực\n林恒 => Lâm Hằng")
        assert [(e.source, e.target) for e in entries] == [
            ("灵力", "linh lực"), ("林恒", "Lâm Hằng"),
        ]

    def test_strips_parenthetical_note(self) -> None:
        entries = parse_glossary("FBI => FBI (Cục Điều tra Liên bang)")
        assert entries[0].target == "FBI"

    def test_skips_blank_and_invalid(self) -> None:
        entries = parse_glossary("\n  \n林恒 => Lâm Hằng\nkhông-có-dấu-phân-tách\n")
        assert len(entries) == 1

    def test_dedup_source(self) -> None:
        entries = parse_glossary("X => A\nX => B")
        assert len(entries) == 1  # giữ mục đầu

    def test_multiple_separators(self) -> None:
        for sep in ("=>", "->", "→", ":", "="):
            entries = parse_glossary(f"abc {sep} xyz")
            assert entries and entries[0].source == "abc"
            assert entries[0].target == "xyz"


class TestConsistencyCheck:
    def test_no_violation_when_consistent(self) -> None:
        v = check_glossary_consistency(
            "林恒 => Lâm Hằng",
            ["林恒走进来", "你好"],
            ["Lâm Hằng đi vào", "Xin chào"],
        )
        assert v == []

    def test_detects_inconsistent_name(self) -> None:
        v = check_glossary_consistency(
            "林恒 => Lâm Hằng",
            ["林恒走进来"],
            ["Lin Heng đi vào"],
        )
        assert len(v) == 1
        assert v[0].line_index == 1
        assert v[0].source_term == "林恒"
        assert v[0].expected_target == "Lâm Hằng"

    def test_case_insensitive(self) -> None:
        v = check_glossary_consistency(
            "FBI => fbi",
            ["The FBI agent"],
            ["Đặc vụ FBI"],  # FBI khớp fbi (không phân biệt hoa thường)
        )
        assert v == []

    def test_only_checks_lines_with_source_term(self) -> None:
        v = check_glossary_consistency(
            "林恒 => Lâm Hằng",
            ["你好", "再见"],  # không có 林恒
            ["Xin chào", "Tạm biệt"],
        )
        assert v == []

    def test_empty_glossary_no_violations(self) -> None:
        assert check_glossary_consistency("", ["a"], ["b"]) == []

    def test_handles_length_mismatch(self) -> None:
        v = check_glossary_consistency(
            "X => Y", ["X here", "X again"], ["no match"],
        )
        # Chỉ kiểm cặp đầu (min length).
        assert len(v) == 1
        assert v[0].line_index == 1
