"""[v3.23.77] Unit test cho các hàm THUẦN trong pipeline dựng phụ đề từ OCR.

Phủ trực tiếp các hàm thuần lõi thuật toán vốn trước đây chỉ được kiểm gián tiếp qua
orchestrator/builder:
- ``box_filters._cjk_char_ratio`` — tỉ lệ ký tự CJK (Hán/Kana/Hangul).
- ``box_filters.is_latin_gibberish`` — phát hiện chuỗi Latin rác (watermark/logo).
- ``voting.select_anchor_text`` — chọn anchor cho vote ROVER.

Chỉ khẳng định các trường hợp xác định rõ từ đặc tả/code (tránh phụ thuộc ngưỡng
nội bộ chưa chắc chắn).
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.box_filters import (
    _cjk_char_ratio,
    is_latin_gibberish,
)
from subtitles_extractor.application.services.subtitle_pipeline.voting import (
    select_anchor_text,
)


class TestCjkCharRatio:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("", 0.0),
            ("   ", 0.0),  # toàn whitespace → bỏ qua hết
            ("abc123", 0.0),  # Latin/digit thuần
            ("你好", 1.0),  # Hán thuần
            ("こんにちは", 1.0),  # Hiragana
            ("한국어", 1.0),  # Hangul
        ],
    )
    def test_pure_ratios(self, text: str, expected: float) -> None:
        assert _cjk_char_ratio(text) == expected

    def test_mixed_cjk_latin(self) -> None:
        # 2 ký tự CJK / 5 ký tự không-whitespace.
        assert _cjk_char_ratio("你好abc") == pytest.approx(2 / 5)

    def test_whitespace_is_ignored_in_denominator(self) -> None:
        # Khoảng trắng không tính vào mẫu số → "你 好" vẫn là 100% CJK.
        assert _cjk_char_ratio("你 好") == 1.0


class TestIsLatinGibberish:
    def test_empty_text_is_not_gibberish(self) -> None:
        assert is_latin_gibberish("", 0.1, 20) is False

    def test_overlong_text_is_not_gibberish(self) -> None:
        # > 12 ký tự → coi như văn bản thật, không phải watermark ngắn.
        assert is_latin_gibberish("a" * 13, 0.1, 20) is False

    def test_high_confidence_low_framecount_is_exempt(self) -> None:
        # fc < 7 và confidence >= 0.85 → văn bản thật, không drop.
        assert is_latin_gibberish("HELLO", 0.9, 3) is False

    def test_non_uppercase_high_framecount_is_exempt(self) -> None:
        # fc >= 7 và KHÔNG phải chuỗi viết hoa bất thường → không drop.
        assert is_latin_gibberish("hello", 0.30, 10) is False

    def test_consonant_only_uppercase_low_conf_is_gibberish(self) -> None:
        # 'LKTR' (docstring liệt kê là rác): toàn phụ âm, conf thấp, fc nhỏ → drop.
        assert is_latin_gibberish("LKTR", 0.40, 2) is True


class TestSelectAnchorText:
    def test_empty_returns_empty_string(self) -> None:
        assert select_anchor_text([]) == ""

    def test_single_candidate_returned_as_is(self) -> None:
        assert select_anchor_text([("xin chào", 0.42)]) == "xin chào"

    def test_frequency_outweighs_slightly_higher_confidence(self) -> None:
        # Cùng độ dài (z-score length = 0) → tần suất (trọng số 0.6) thắng confidence.
        # 'aaa': score = 0.6*(2/3) + 0.4*0.50 = 0.600
        # 'bbb': score = 0.6*(1/3) + 0.4*0.99 = 0.596
        candidates = [("aaa", 0.50), ("aaa", 0.50), ("bbb", 0.99)]
        assert select_anchor_text(candidates) == "aaa"
