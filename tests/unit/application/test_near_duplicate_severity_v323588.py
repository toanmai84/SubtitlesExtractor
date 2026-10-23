"""[v3.23.588] "Thuật ngữ gần trùng" chỉ là CẢNH BÁO khi hai mục có bản dịch khác nhau.

Chuẩn đoán phân tích thật của hai lượt liền (v3.23.586, v3.23.587) đều chỉ có một cảnh
báo: ``一回生二回熟 ~ 一回生兩回熟`` — "khâu dịch sẽ nhận hai chỉ thị mâu thuẫn". Hai mục
đó có bản dịch GIỐNG HỆT ("lần đầu lạ lần sau quen"), tức là cùng một chỉ thị. Trên 17
bảng thuật ngữ thật: 14 cặp gần trùng, 11 cặp có bản dịch giống hệt.
"""

from __future__ import annotations

from typing import Any

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from tests.unit.application.test_context_diagnostics_v323484 import _Ctx, _events

# Cặp thật của lượt này: bản dịch giống hệt.
SAME_TRANSLATION = "\n".join([
    "一回生二回熟 = lần đầu lạ lần sau quen",
    "一回生兩回熟 = lần đầu lạ lần sau quen",
])
# Cặp thật của phim khác: một bản dịch chứa bản kia.
DIFFERENT_TRANSLATION = "\n".join([
    "阶层跃升 = đổi đời",
    "阶级跃升 = đổi đời / nâng tầm giai cấp",
])


def _report(glossary: str) -> dict[str, Any]:
    return build_context_diagnostics(
        context=_Ctx(overview="x" * 9000, characters="林逸 — chính", glossary=glossary),
        source_events=_events("林逸"),
    )


def _flag(report: dict[str, Any]) -> dict[str, str]:
    flags = report["quality_flags"]
    [flag] = [item for item in flags if item["code"] == "glossary_near_duplicates"]
    return flag


class TestSameTranslationIsNotAWarning:
    def test_real_pair_is_listed_but_only_as_a_note(self) -> None:
        report = _report(SAME_TRANSLATION)
        assert report["glossary"]["near_duplicates"] == ["一回生二回熟 ~ 一回生兩回熟"]
        flag = _flag(report)
        assert flag["severity"] == "info"  # trước đây "warning"
        assert "mâu thuẫn" not in flag["message"]
        assert "không ảnh hưởng bản dịch" in flag["message"]

    def test_pair_adds_no_warning_to_the_count(self) -> None:
        """Bản phân tích thật của lượt này: 0 lỗi, 1 cảnh báo — chính là cặp này."""
        one_entry = SAME_TRANSLATION.splitlines()[0]
        assert _report(SAME_TRANSLATION)["counts"] == _report(one_entry)["counts"]


class TestDifferentTranslationsStillWarn:
    def test_real_pair_is_a_warning(self) -> None:
        flag = _flag(_report(DIFFERENT_TRANSLATION))
        assert flag["severity"] == "warning"
        assert "阶层跃升 ~ 阶级跃升" in flag["message"]

    def test_warning_names_only_the_conflicting_pairs(self) -> None:
        report = _report(f"{SAME_TRANSLATION}\n{DIFFERENT_TRANSLATION}")
        assert len(report["glossary"]["near_duplicates"]) == 2
        flag = _flag(report)
        assert flag["severity"] == "warning"
        assert flag["message"].startswith("1 cặp")
        assert "一回生二回熟" not in flag["message"]
