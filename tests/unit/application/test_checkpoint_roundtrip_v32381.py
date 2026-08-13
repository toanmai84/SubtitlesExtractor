"""[v3.23.81] Test round-trip checkpoint dịch giữ ĐỦ trường.

Bug trước: ``_lines_to_json`` bỏ ``original_text`` (mỏ neo bản gốc chống tam-sao-thất
-bản) và ``addressee`` (chọn đại từ/kính ngữ). Khi RESUME, STYLE/LOCALIZE mất mỏ neo
nguồn -> trôi nghĩa. Sửa: round-trip đủ trường, tương thích ngược checkpoint cũ.
"""

from __future__ import annotations

from subtitles_extractor.application.use_cases.translate_subtitles import (
    _json_to_lines,
    _lines_to_json,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


def test_round_trip_preserves_original_text_and_addressee() -> None:
    lines = [
        TranslationLine(
            index=1, start_ms=0, end_ms=1000,
            text="Chào ngài", speaker="Lâm", description="(nhẹ nhàng)",
            original_text="Hello sir", addressee="nhà vua",
        ),
    ]
    restored = _json_to_lines(_lines_to_json(lines))
    assert restored[0].original_text == "Hello sir"
    assert restored[0].addressee == "nhà vua"
    assert restored[0].text == "Chào ngài"
    assert restored[0].speaker == "Lâm"
    assert restored[0].description == "(nhẹ nhàng)"


def test_backward_compatible_with_old_checkpoint_missing_fields() -> None:
    # Checkpoint CŨ không có khoá "o"/"a" → mặc định rỗng, không lỗi.
    old_rows = [{"i": 1, "s": 0, "e": 500, "t": "Xin chào", "sp": "", "d": ""}]
    restored = _json_to_lines(old_rows)
    assert restored[0].original_text == ""
    assert restored[0].addressee == ""
    assert restored[0].text == "Xin chào"
