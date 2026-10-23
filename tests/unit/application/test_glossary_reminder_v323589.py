"""[v3.23.589] Khối nhắc thuật ngữ theo lô: mục nhiều phương án vẫn "must_translate_as".

Khối ``<must_use_terms>`` đặt cạnh lô dịch không có lời giải thích nào: TÊN KHOÁ chính là
chỉ thị cho model. Ba lượt chạy thật của phim "灭门惨案II", cùng bảng thuật ngữ, cùng
model (gemini-3.5-flash-lite), chỉ khác tên khoá của mục nhiều phương án:

* v3.23.586 — ``"must_translate_as": "tự sướng / thủ dâm"``: 性自樂 ra "tự sướng" ở cả 4
  dòng (82, 83, 84, 86); mục nhiều phương án sai 1/40 lượt ở dịch thô. Nhược điểm đã sửa
  ở v3.23.587: model chép cả chuỗi "xã/phường" vào lời thoại.
* v3.23.587 và v3.23.588 — ``"must_translate_as_one_of": ["tự sướng", "thủ dâm"]``: cả
  hai lượt 性自樂 ra "tính tự lạc" (đọc âm Hán-Việt, vô nghĩa với người nghe) ở cả 4
  dòng; mục nhiều phương án sai 6/40 lượt ở dịch thô, 7/40 và 9/40 ở bản cuối.

Mục MỘT phương án (khoá không đổi qua ba lượt) sai 17, 11, 10 trên 120 — không xấu đi.
Nay phương án đầu mang khoá "must_translate_as" như mục một phương án; các phương án còn
lại tách riêng trong "also_acceptable" nên không còn chuỗi "A / B" để chép.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

import json
from types import SimpleNamespace

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _batch_glossary_block,
)

# Các mục chép nguyên từ bảng thuật ngữ của lượt chạy thật.
GLOSSARY = "\n".join([
    "派出所 = đồn công an xã/phường",
    "所長 = Trưởng đồn công an",
    "性自樂 = tự sướng / thủ dâm",
    "補品 = thuốc bổ / đồ tẩm bổ",
    "四個現代化 = bốn hiện đại hóa",
    "犯賤 = đê tiện / không biết điều / rước nhục vào thân",
])


def _block(*sources: str) -> list[dict[str, object]]:
    batch = [SimpleNamespace(text=text) for text in sources]
    return _batch_glossary_block(batch, GLOSSARY)


class TestMultiOptionTerm:
    def test_real_case_first_option_is_the_instruction(self) -> None:
        assert _block("你們性自樂之後，從速交貨", "大隊長，甚麼是性自樂呀") == [
            {
                "term": "性自樂",
                "must_translate_as": "tự sướng",
                "also_acceptable": ["thủ dâm"],
            }
        ]

    def test_weaker_key_is_gone(self) -> None:
        payload = json.dumps(_block("常常會為她弄些補品", "叫所有男村民明天去派出所"))
        assert "must_translate_as_one_of" not in payload
        assert payload.count('"must_translate_as"') == 2

    def test_no_slashed_string_for_the_model_to_copy(self) -> None:
        """Điều v3.23.587 đã sửa vẫn giữ: không giá trị nào còn là chuỗi "A / B"."""
        for entry in _block("本派出所女公安", "你們性自樂之後", "你犯賤"):
            values = [entry["must_translate_as"], *entry.get("also_acceptable", [])]
            assert all("/" not in str(value) for value in values)

    def test_three_options_keep_their_order(self) -> None:
        assert _block("你犯賤") == [
            {
                "term": "犯賤",
                "must_translate_as": "đê tiện",
                "also_acceptable": ["không biết điều", "rước nhục vào thân"],
            }
        ]


class TestSingleOptionTerm:
    def test_shape_is_unchanged(self) -> None:
        assert _block("性自樂即是四個現代化的叫法")[-1] == {
            "term": "四個現代化",
            "must_translate_as": "bốn hiện đại hóa",
        }

    def test_every_entry_uses_the_same_instruction_key(self) -> None:
        block = _block("所長，派出所的補品", "性自樂即是四個現代化的叫法")
        assert len(block) == 5
        assert all("must_translate_as" in entry for entry in block)
