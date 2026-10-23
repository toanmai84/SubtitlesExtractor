"""[v3.23.633] Lưới ĐẢO NGÔI: giai đoạn tinh chỉnh biến "ngươi" (你) thành "ta" (hoặc ngược lại).

Chấm mù 3 tập phim mẫu (6 bản chấm): chuỗi dịch thô → phong cách → bản địa hoá có lỗi nặng
(đảo vai) 8-14, trong khi chỉ dịch thô chỉ 2. Cả 4 + 4 dòng nặng ở tập 1 cùng một kiểu::

    当年你是天纵之资   thô: "Năm xưa ngươi có tư chất thiên tài"
                       phong cách: "Năm xưa ta là thiên tài hiếm có,"      (你 → ta)
    才会给你们缔结婚约 thô: "mới cho các người đính ước"
                       phong cách: "mới hứa hôn cho hai ta."                (你们 → hai ta)

Nhãn người nói/cue (máy tự gán, sai ở nhóm đệ tử: "叶然 — tự nhủ") đẩy giai đoạn tinh chỉnh tin
"đây là độc thoại" nên đổi 你 sang ngôi thứ nhất. Lưới xưng hô cũ (``address_regression_guard``)
chỉ chạy khi bảng nhân vật có quy tắc cho cặp người nói/nghe — cặp này không có.

Lưới này dùng CHỮ GỐC làm chuẩn, không cần nhãn nào: câu gốc có 你/您/你们 mà không có 我/咱/我们
là lời nói VỚI người nghe (ngôi thứ hai); bản dịch chỉ có đại từ ngôi thứ nhất (ta/tôi/tớ/tao) mà
không có ngôi thứ hai là đảo ngôi. Dòng nào bị khâu sau làm đảo mà bản trước không đảo thì lùi về
bản trước. Chỉ dùng đại từ KHÔNG nhập nhằng (huynh/muội/anh/em/con vừa là ngôi một vừa là ngôi hai
nên không tính). Hàm thuần: không I/O.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = ["find_person_flips", "flipped_person", "source_person"]

_SECOND_SRC = ("你", "您", "妳")
_FIRST_SRC = ("我", "咱", "吾", "本座", "老子", "朕", "本宫", "本尊")

#: Ngôi thứ nhất không nhập nhằng. "người ta" (ngôi ba) bị loại trước khi dò.
_FIRST_VI = re.compile(r"\b(?:ta|tôi|tớ|tao|bổn tọa|bản tọa|bổn cung|bản cung|trẫm|bổn tôn)\b")
#: Ngôi thứ hai không nhập nhằng. "các người"/"người" KHÔNG tính: kính ngữ/danh từ.
_SECOND_VI = re.compile(r"\b(?:ngươi|mày|cậu|bạn|các ngươi|bọn ngươi|bọn mày|lũ ngươi)\b")
_THIRD_TA = re.compile(r"\bngười ta\b|\bcậu ta\b|\bhắn ta\b|\bnàng ta\b|\bcô ta\b|\bbà ta\b|\bông ta\b|\bnó ta\b")


def source_person(source: str) -> int | None:
    """Ngôi của câu gốc: 2 (chỉ 你…), 1 (chỉ 我…), ``None`` nếu cả hai/không có."""
    has_second = any(mark in source for mark in _SECOND_SRC)
    has_first = any(mark in source for mark in _FIRST_SRC)
    if has_second and not has_first:
        return 2
    if has_first and not has_second:
        return 1
    return None


def _vi_persons(text: str) -> set[int]:
    lowered = _THIRD_TA.sub(" ", text.lower())
    found: set[int] = set()
    if _FIRST_VI.search(lowered):
        found.add(1)
    if _SECOND_VI.search(lowered):
        found.add(2)
    return found


def flipped_person(source: str, translation: str) -> bool:
    """Bản dịch chỉ mang ngôi NGƯỢC với câu gốc (gốc: 你, dịch: chỉ ta/tôi; hoặc ngược lại)."""
    expected = source_person(source)
    if expected is None:
        return False
    found = _vi_persons(translation)
    opposite = 3 - expected
    return opposite in found and expected not in found


def find_person_flips(
    sources: Mapping[int, str],
    previous: Mapping[int, str],
    current: Mapping[int, str],
) -> set[int]:
    """Các dòng khâu hiện tại làm ĐẢO NGÔI mà bản trước không đảo (nên lùi về bản trước).

    Args:
        sources: ``số dòng → câu gốc``.
        previous: ``số dòng → bản dịch của khâu trước``.
        current: ``số dòng → bản dịch của khâu hiện tại``.
    """
    flips: set[int] = set()
    for index, text in current.items():
        source = sources.get(index, "")
        before = previous.get(index)
        if not source or before is None or before == text:
            continue
        if flipped_person(source, text) and not flipped_person(source, before):
            flips.add(index)
    return flips
