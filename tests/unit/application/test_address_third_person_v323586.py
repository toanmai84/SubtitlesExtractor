"""[v3.23.586] Cờ "xưng hô không nhất quán" không được đếm ngôi ba, số nhiều, độc thoại.

Chuẩn đoán thật v3.23.585 (phim "灭门惨案II"): cặp ``醫生 → 龍傳人`` bị báo dùng lẫn bốn
cách xưng hô ``{"tôi": 6, "ông": 4, "cô": 4, "cậu": 3}``. Soát từng dòng: cả 4 lượt 'cô'
là "cô ấy" (bệnh nhân), 2 lượt 'ông' là "ông ấy" (trấn trưởng) — bác sĩ chỉ xưng "tôi" và
gọi "cậu". Trên 50 chuẩn đoán, 70 cặp bị báo oan kiểu này (phần lớn do 'ta' của "anh ta",
"chúng ta"), đồng thời 35 cặp lộn xộn THẬT bị che vì các lượt ngôi ba làm loãng tỉ lệ.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from subtitles_extractor.application.services.translation_diagnostics import (
    address_inconsistencies,
    address_pair_stats,
)

DOCTOR = "醫生"
CHIEF = "龍傳人"

# (dòng, bản dịch cuối) — các dòng có từ xưng hô của cặp 醫生 → 龍傳人 trong chuẩn đoán.
DOCTOR_LINES = [
    (219, "Tôi không thể tùy tiện tiết lộ thông tin bệnh nhân."),
    (221, "Tôi là bác sĩ"),
    (223, "Ông bao che nghi phạm,"),
    (224, "tôi sợ ông gánh không nổi đâu."),
    (225, "Cậu định dùng thế lực đè tôi à?"),
    (226, "Cậu không thấy mấy bức thư pháp này sao?"),
    (231, "Nếu không phải tôi chữa cái bút ấy cho ông ấy,"),
    (232, "thì ông ấy sao chịu hạ bút đề chữ cho tôi chứ?"),
    (252, "Cô ấy mắc chứng quá mẫn cảm bẩm sinh."),
    (254, "Nhưng cô ấy không bị bệnh giang mai đâu."),
    (258, "Sao cậu cứ chắc chắn cô ấy bị bệnh giang mai thế?"),
    (262, "Bởi vì cô ấy vẫn là gái còn trinh mà."),
    (319, "Cậu có bản lĩnh thì cứ hỏi đi."),
    (324, "Tôi chẳng phải không hỏi được khẩu cung sao?"),
]


def _cues(rows: list[tuple[int, str]], speaker: str, addressee: str) -> str:
    return json.dumps([{"id": i, "spk": speaker, "to": addressee} for i, _ in rows])


def _events(rows: list[tuple[int, str]]) -> list[SimpleNamespace]:
    return [SimpleNamespace(index=i, text=text) for i, text in rows]


def test_third_person_is_not_an_address_variant() -> None:
    findings, examined = address_pair_stats(
        _cues(DOCTOR_LINES, DOCTOR, CHIEF), _events(DOCTOR_LINES)
    )
    assert findings == []
    assert examined == 1  # cặp vẫn đủ dữ liệu để xét, chỉ là không lộn xộn


def test_plural_forms_are_not_counted() -> None:
    rows = [
        (1, "Chúng ta đi thôi, tôi đợi cậu."),
        (2, "Bọn tao không sợ đâu, tôi nói cậu nghe."),
        (3, "Tụi mày lui ra, tôi cần cậu giúp."),
        (4, "Chúng ta thắng rồi!"),
        (5, "Bọn tao đến đây."),
        (6, "Tụi mày nhớ đấy."),
    ]
    assert address_inconsistencies(_cues(rows, "A", "B"), _events(rows)) == []


def test_monologue_is_not_a_pair() -> None:
    rows = [
        (1, "Tôi phải làm sao đây, anh ơi."),
        (2, "Em sai rồi, tao biết mà."),
        (3, "Mày ngu lắm, cậu ạ."),
        (4, "Chị nói gì thế hả cháu."),
    ]
    assert address_pair_stats(_cues(rows, "A", "A"), _events(rows)) == ([], 0)


def test_real_mixing_is_still_reported() -> None:
    """Ca thật 春花 → 宋钱 (v3.23.524): một người nghe mà "chị", "em", "cậu", "tôi"."""
    rows = [
        (1, "Chị bảo cậu rồi mà."),
        (2, "Em xin lỗi anh."),
        (3, "Cậu nghe chị nói này."),
        (4, "Tôi không tin anh đâu."),
        (5, "Em nhớ anh lắm."),
        (6, "Tôi đã nói với cậu rồi."),
    ]
    [finding] = address_inconsistencies(_cues(rows, "春花", "宋钱"), _events(rows))
    assert set(finding["variants"]) == {"chị", "em", "cậu", "tôi", "anh"}
