"""[v3.23.611] Tên gọi thân mật hai chữ (阿/小/老 + tên) đọc bằng âm của chữ khác.

Chuẩn đoán thật v3.23.609 (phim "灭门惨案II", phân tích bằng gemini-3.8-flash)::

    黃秋生 / 阿彪 (A Tiêu / Cục trưởng Bưu) [nam, ~35] — ...

彪 chỉ có một âm Hán-Việt là "bưu" — chính dòng đó cũng viết "Cục trưởng Bưu"; "tiêu" là
âm của 標/鏢 (đồng âm *biāo*). Quy tắc v3.23.591 chỉ tự sửa tên từ ba chữ, nên file chuẩn
đoán báo mà không sửa. Tiền tố gọi thân mật cho biết chắc chữ sau là TÊN RIÊNG đang được
phiên âm (không phải dịch nghĩa) — trừ khi chữ sau là danh xưng quan hệ ("阿叔" → "A
Chú", "老闆" → "Lão Chủ"): những cụm đó dịch nghĩa có chủ ý.

Đo trên 26 bảng nhân vật thật của kho chuẩn đoán: chỉ "A Tiêu" được sửa thêm; "Mạc Sảo"
(莫嫂, không tiền tố), "Quan khách" (宾客), "Chị Mai" (王梅) vẫn như cũ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.name_reading_check import (
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

# Chép nguyên văn từ bản phân tích thật (translation_debug 20261013_193904).
A_BIU_ROW = (
    "黃秋生 / 阿彪 (A Tiêu / Cục trưởng Bưu) [nam, ~35] — cảnh sát phá đại án phần "
    "trước được thăng chức cục trưởng, anh họ của Long Truyền Nhân (xuất hiện qua ảnh)"
)
MAC_TAU_ROW = (
    "莫嫂 (Mạc Sảo) [nữ, trung niên] — Vợ Lão Mạc: người bị ngất xỉu giữa đường khiến "
    "đồn công an tưởng lầm là nạn nhân vụ án mạng."
)


@pytest.fixture(scope="module")
def readings() -> NameReadings:
    table = load_name_readings(DATA_DIR / HANVIET_READINGS_FILENAME)
    assert table, "thiếu bảng âm Hán-Việt đi kèm ứng dụng"
    return table


def _roster_issues(roster: str, readings: NameReadings) -> list[NameReadingIssue]:
    return [
        issue
        for issue in find_name_reading_issues(roster, "", readings)
        if issue.source == "roster"
    ]


class TestRealCaseABiu:
    def test_bundled_table_facts(self, readings: NameReadings) -> None:
        assert readings.suggestable("彪") == ("bưu",)
        assert readings.is_hanviet_syllable("Tiêu")

    def test_nickname_misreading_gets_a_fix(self, readings: NameReadings) -> None:
        (issue,) = _roster_issues(A_BIU_ROW, readings)
        assert (issue.name, issue.written, issue.wrote) == ("阿彪", "A Tiêu", "Tiêu")
        assert issue.suggestion == "Bưu"

    def test_corrections(self, readings: NameReadings) -> None:
        issues = _roster_issues(A_BIU_ROW, readings)
        assert name_corrections(issues, A_BIU_ROW) == {"A Tiêu": "A Bưu"}

    def test_whole_context_is_corrected(self, readings: NameReadings) -> None:
        context = SubtitleContextAnalysis(
            source_lang="zh",
            characters=A_BIU_ROW,
            overview="Ảnh A Tiêu treo trên tường đồn.",
            glossary="",
        )
        fixed, fixes = correct_context_names(context, readings)
        assert [fix.suggestion for fix in fixes] == ["Bưu"]
        assert fixed.characters.startswith("黃秋生 / 阿彪 (A Bưu / Cục trưởng Bưu)")
        assert fixed.overview == "Ảnh A Bưu treo trên tường đồn."


class TestOtherTwoCharacterNamesUnchanged:
    def test_name_without_nickname_prefix_is_only_reported(
        self, readings: NameReadings
    ) -> None:
        (issue,) = _roster_issues(MAC_TAU_ROW, readings)
        assert issue.suggestion == ""

    @pytest.mark.parametrize(
        ("roster", "char"),
        [
            ("阿叔 (A Chú) [nam]", "叔"),
            ("阿哥 (A Anh) [nam]", "哥"),
            ("老頭 (Lão Già) [nam]", "頭"),
            ("老闆 (Lão Chủ) [nam]", "闆"),
        ],
    )
    def test_kinship_title_translated_by_meaning_is_kept(
        self, readings: NameReadings, roster: str, char: str
    ) -> None:
        """Âm viết là âm Hán-Việt thật của chữ khác, chữ chỉ một âm — vẫn không sửa."""
        (issue,) = _roster_issues(roster, readings)
        assert issue.char == char
        assert issue.suggestion == ""

    def test_prefix_itself_must_be_transliterated(self, readings: NameReadings) -> None:
        """"Ông" cho 老 là dịch nghĩa → cả cụm không phải phiên âm."""
        assert all(
            issue.suggestion == ""
            for issue in _roster_issues("老彪 (Ông Tiêu) [nam]", readings)
        )
