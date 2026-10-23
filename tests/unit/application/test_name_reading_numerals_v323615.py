"""[v3.23.615] Số đếm đọc bằng số từ thuần Việt trong tên/nhãn không phải đọc nhầm.

Chẩn đoán ngữ cảnh thật 14/10 (gemini-3.8-flash): ``两位长老 (Hai vị trưởng lão)`` bị báo
"两 đọc lưỡng/lượng/lạng" — đây là dịch nghĩa đúng của nhãn "hai vị trưởng lão", và là
cảnh báo duy nhất của lượt phân tích đó. Đo trên 28 bảng nhân vật thật: chỉ mục này đổi.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.name_reading_check import (
    find_name_reading_issues,
)
from subtitles_extractor.infrastructure.translation.name_readings_loader import (
    load_name_readings,
)

DATA = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/data/hanviet_readings.json"
)


@pytest.fixture(scope="module")
def readings():
    return load_name_readings(DATA)


def _issues(roster: str, readings) -> list[tuple[str, str]]:
    return [(i.char, i.wrote) for i in find_name_reading_issues(roster, "", readings)]


def test_real_label_is_not_flagged(readings) -> None:
    roster = (
        "两位长老 (Hai vị trưởng lão) [nam, cao tuổi] — Các trưởng lão Linh Tiêu Tông."
    )
    assert _issues(roster, readings) == []


def test_numeral_in_a_name_read_natively_is_accepted(readings) -> None:
    assert _issues("三胖子 (Ba Bàn Tử) [nam] — bạn thân.", readings) == []


def test_real_misreading_is_still_reported(readings) -> None:
    assert _issues("柳慕云 (Liễu Mộc Vân) [nữ] — nữ chính.", readings) == [("慕", "Mộc")]


def test_wrong_native_numeral_is_still_reported(readings) -> None:
    """"Ba" không phải cách đọc của 两."""
    assert _issues("两位长老 (Ba vị trưởng lão) [nam] — trưởng lão.", readings) == [
        ("两", "Ba")
    ]
