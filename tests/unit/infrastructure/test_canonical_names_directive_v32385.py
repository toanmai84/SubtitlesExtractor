"""[v3.23.85] Test chỉ thị 'force name compliance' chèn vào system prompt.

Chỉ thị "force name compliance" chèn vào system prompt: trích tên chuẩn từ roster,
buộc model dùng tên nhân vật nhất quán xuyên tập.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator as G,
)

_ROSTER = (
    "Hứa Phượng Niên (许凤年): nhân vật phụ: con trai Trấn Bắc Vương\n"
    "Hoàng đế (朕/皇帝): nhân vật chính: vua Đại Chu\n"
    "Thái hậu (太后): nhân vật phụ: mẹ vua\n"
)


def test_empty_roster_returns_empty() -> None:
    assert G._canonical_names_directive("") == ""


def test_lists_canonical_vietnamese_names() -> None:
    out = G._canonical_names_directive(_ROSTER)
    assert "DANH SÁCH TÊN CHUẨN:" in out
    assert "Hứa Phượng Niên" in out
    assert "Hoàng đế" in out
    assert "Thái hậu" in out
    # KHÔNG được chèn nguyên ký tự CJK alias vào danh sách tên chuẩn.
    assert "许凤年" not in out
    assert "皇帝" not in out


def test_directive_has_consistency_instruction() -> None:
    out = G._canonical_names_directive(_ROSTER)
    assert "BẮT BUỘC NHẤT QUÁN" in out


def test_legacy_cjk_first_format_extracts_vietnamese() -> None:
    # Định dạng "CJK (Việt)" vẫn lấy đúng phần Việt làm tên chuẩn.
    out = G._canonical_names_directive("林昆 (Lâm Côn): vai phụ")
    assert "Lâm Côn" in out
    assert "林昆" not in out


def test_names_deduplicated() -> None:
    roster = "Thái hậu (太后): a\nThái hậu (太后): b\n"
    out = G._canonical_names_directive(roster)
    assert out.count("Thái hậu") == 1
