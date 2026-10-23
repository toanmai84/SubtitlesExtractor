"""[v3.23.585] Địa danh quy ước không bị báo là "phiên âm khác âm Hán-Việt".

Chuẩn đoán thật v3.23.584 (phim "灭门惨案II", nhân vật là cựu binh chiến tranh biên giới):
bảng thuật ngữ ghi ``諒山 = Lạng Sơn`` — đúng tên tỉnh — nhưng file chuẩn đoán báo
``諒 "Lạng" (đọc lượng)``. Người soát phải bỏ thời gian kiểm một mục vốn đúng.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.name_reading_check import (
    NameReadings,
    find_name_reading_issues,
)
from subtitles_extractor.infrastructure.translation.name_readings_loader import (
    HANVIET_READINGS_FILENAME,
    load_name_readings,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"


@pytest.fixture(scope="module")
def readings() -> NameReadings:
    return load_name_readings(DATA_DIR / HANVIET_READINGS_FILENAME)


@pytest.mark.parametrize("term", ["諒山", "谅山"])
def test_lang_son_is_accepted(readings: NameReadings, term: str) -> None:
    assert find_name_reading_issues("", f"{term} = Lạng Sơn", readings) == []


def test_place_reading_is_accepted_but_never_suggested(readings: NameReadings) -> None:
    """Tên người có chữ 諒 vẫn được đề xuất âm chuẩn "lượng", không phải "lạng"."""
    assert readings.suggestable("諒") == ("lượng",)
    assert readings.suggestable("谅") == ("lượng",)


def test_a_wrong_reading_of_the_same_character_is_still_reported(
    readings: NameReadings,
) -> None:
    issues = find_name_reading_issues("", "諒山 = Lãng Sơn", readings)
    assert [(issue.char, issue.wrote) for issue in issues] == [("諒", "Lãng")]
