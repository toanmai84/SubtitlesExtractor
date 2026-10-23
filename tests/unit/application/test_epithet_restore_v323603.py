# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.603] Lưới khôi phục xưng hô không coi "con" của lời chửi "con khốn" là đại từ.

Chuẩn đoán thật 11/10 (灭门惨案II) dòng 911 ``洪志權 → 秦可秀`` (bảng nhân vật: xưng
'tao' gọi 'con khốn'), câu gốc "讓我給你嚐點厲害吧": bản thô "Để tao cho con khốn này
biết thế nào là lợi hại", bản tinh chỉnh "Để tao cho mày nếm mùi lợi hại". Lưới thấy "con"
trong bản thô là đại từ đúng quy tắc nên "khôi phục" → "Để tao cho con nếm mùi lợi hại"
(giọng cha mẹ nói với con), ở CẢ giai đoạn tinh chỉnh lẫn bản địa hoá.

Trên mọi file chuẩn đoán thật, lưới khôi phục vẫn ra đúng 279 dòng như trước — chỉ ca này
khác.
"""

from __future__ import annotations

import json

from subtitles_extractor.application.services.address_regression_guard import (
    allowed_address_forms,
    find_address_regressions,
)

ROSTER = "\n".join([
    "秦可秀 (Tần Khả Tú) [nữ, ~25] — vợ của Chân Kiếm Thanh; xưng hô: với Hồng Chấn và "
    "Hồng Chí Quyền tự xưng 'tôi' gọi 'các người'",
    "洪震 (Hồng Chấn) [nam, ~45-50] — ác bá thôn quê; xưng hô: với Tần Khả Tú xưng 'tao' "
    "gọi 'con đàn bà dâm đãng', với Hồng đại tẩu tự xưng 'tôi' gọi 'bà nó'",
    "洪志權 (Hồng Chí Quyền) [nam, ~20-25] — con trai Hồng Chấn; xưng hô: với Hồng Chấn "
    "tự xưng 'con' gọi 'ba', với Tần Khả Tú xưng 'tao' gọi 'con khốn'",
    "洪大嫂 (Hồng đại tẩu) [nữ, ~45-50] — vợ Hồng Chấn; xưng hô: với Hồng Chấn tự xưng "
    "'tôi' gọi 'ông nó', với Hồng Chí Quyền tự xưng 'mẹ' gọi 'con'",
])
CUES = json.dumps([
    {"id": 911, "spk": "洪志權", "to": "秦可秀"},
    {"id": 1101, "spk": "洪震", "to": "洪大嫂"},
], ensure_ascii=False)
SOURCES = {911: "讓我給你嚐點厲害吧", 1101: "我今晚不去打牌， ，你放心吧"}


def test_loi_chui_khong_cho_phep_dai_tu_con() -> None:
    assert allowed_address_forms("xưng 'tao' gọi 'con khốn'") == set()
    assert allowed_address_forms("xưng 'tao' gọi 'con đàn bà dâm đãng'") == set()


def test_con_goi_con_cai_van_la_dai_tu() -> None:
    assert allowed_address_forms("tự xưng 'mẹ' gọi 'con'") == {"con"}
    assert allowed_address_forms("tự xưng 'ba' gọi 'con trai'") == {"con"}


def test_dong_911_khong_bi_doi_may_thanh_con() -> None:
    previous = {911: "Để tao cho con khốn này biết thế nào là lợi hại."}
    current = {911: "Để tao cho mày nếm mùi lợi hại."}
    assert find_address_regressions(SOURCES, previous, current, CUES, ROSTER) == []


def test_khoi_phuc_xung_ho_dung_van_hoat_dong() -> None:
    # Dòng 1101 thật: giai đoạn bản địa hoá đổi 'bà' (đúng bảng: gọi 'bà nó') thành 'cô'.
    previous = {1101: "Tôi tối nay không đi đánh bài đâu, bà yên tâm đi."}
    current = {1101: "Tôi tối nay không đi đánh bài đâu, cô yên tâm đi."}
    found = find_address_regressions(SOURCES, previous, current, CUES, ROSTER)
    assert [(item.index, item.text_after) for item in found] == [
        (1101, "Tôi tối nay không đi đánh bài đâu, bà yên tâm đi."),
    ]
