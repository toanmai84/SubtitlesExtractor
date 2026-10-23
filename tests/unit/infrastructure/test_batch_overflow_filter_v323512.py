"""[v3.23.512] Model trả kèm dòng NGỮ CẢNH thì lọc bỏ, đừng bỏ cuộc.

Log thật (5540 dòng): 51 cảnh báo "Model trả batch sai cấu trúc" và 6 lô THẤT BẠI HẲN. Nội dung:
"Model trả DƯ 26 dòng (kỳ vọng 1)". Khi lô bị chia nhỏ còn 1 dòng, model dịch luôn ~25 dòng ngữ
cảnh kèm theo. Ứng dụng coi là hỏng → thử lại 5 lần → chia nhỏ tiếp → bỏ cuộc GIỮ NGUYÊN VĂN BẢN
GỐC, dù dòng cần dịch nằm sẵn trong kết quả trả về.

Các dòng ngữ cảnh luôn có ``line_no`` NẰM NGOÀI khoảng của lô, nên lọc theo line_no là tất định.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _BatchValidationError,
)


@pytest.fixture
def translator() -> GeminiSubtitleTranslator:
    return GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)


def _payload_with_context(target_line: int, context_count: int = 25) -> dict:
    context = [
        {"line_no": n, "text": f"ngữ cảnh {n}"}
        for n in range(target_line - context_count, target_line)
    ]
    return {"subtitles": [*context, {"line_no": target_line, "text": "Bản dịch dòng cần."}]}


class TestOverflowFiltering:
    def test_context_lines_filtered_out(self, translator) -> None:
        payload = _payload_with_context(100)
        translator._validate_batch(payload, expected_count=1, start_idx=99)
        assert payload["subtitles"] == [{"line_no": 100, "text": "Bản dịch dòng cần."}]

    def test_multi_line_batch_filtered(self, translator) -> None:
        payload = {
            "subtitles": [
                {"line_no": n, "text": f"ngữ cảnh {n}"} for n in range(70, 100)
            ]
            + [{"line_no": 100, "text": "A"}, {"line_no": 101, "text": "B"}]
        }
        translator._validate_batch(payload, expected_count=2, start_idx=99)
        assert [item["line_no"] for item in payload["subtitles"]] == [100, 101]

    def test_missing_target_still_raises(self, translator) -> None:
        """Lọc xong vẫn thiếu dòng cần → phải báo hỏng để chia nhỏ/thử lại như cũ."""
        payload = {"subtitles": [{"line_no": n, "text": f"x{n}"} for n in range(75, 100)]}
        with pytest.raises(_BatchValidationError):
            translator._validate_batch(payload, expected_count=1, start_idx=99)

    def test_blank_target_text_not_accepted(self, translator) -> None:
        """Dòng cần có line_no nhưng text RỖNG = chưa dịch → không được coi là đủ."""
        payload = _payload_with_context(100)
        payload["subtitles"][-1]["text"] = "   "
        with pytest.raises(_BatchValidationError):
            translator._validate_batch(payload, expected_count=1, start_idx=99)


class TestNoRegression:
    def test_exact_count_untouched(self, translator) -> None:
        payload = {"subtitles": [{"line_no": 100, "text": "A"}, {"line_no": 101, "text": "B"}]}
        before = list(payload["subtitles"])
        translator._validate_batch(payload, expected_count=2, start_idx=99)
        assert payload["subtitles"] == before

    def test_small_overflow_still_tolerated(self, translator) -> None:
        """Dư 1-2 dòng vẫn được dung thứ như trước (không đi vào nhánh lọc)."""
        payload = {
            "subtitles": [
                {"line_no": 100, "text": "A"},
                {"line_no": 101, "text": "B"},
                {"line_no": 102, "text": "C"},
            ]
        }
        translator._validate_batch(payload, expected_count=2, start_idx=99)
