"""Test gợi ý ``max_chars`` theo thời lượng để dịch súc tích cho TTS.

[v3.23.222] Công thức đổi từ "tốc độ đọc HẰNG SỐ 16 ký tự/giây" sang mô hình VẬT LÝ đo
được ở tầng TTS (``timing_math.readable_char_budget``: mỗi câu có chi phí cố định ~0.3s).
Các con số kỳ vọng dưới đây cập nhật theo mô hình mới — xem
``test_char_budget_v32622.py`` để biết bằng chứng đo đạc.
"""

from __future__ import annotations

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MIN_CHAR_BUDGET,
    readable_syllable_budget,
    syllable_budget_to_chars,
)


def _ngan_sach(khung_s: float) -> int:
    """[v3.23.238] Ngân sách nay tính theo ÂM TIẾT rồi quy sang ký tự (xem v32638).

    Trước đây các test này khoá cứng con số của mô hình theo KÝ TỰ — mô hình đó đã bị bác
    bỏ bằng dữ liệu (R²=0.07 so với 0.89 của mô hình âm tiết biên dưới).
    """
    return max(
        MIN_CHAR_BUDGET,
        syllable_budget_to_chars(readable_syllable_budget(khung_s)),
    )


class TestLengthHint:
    def test_short_line_small_hint(self) -> None:
        line = TranslationLine(index=1, start_ms=0, end_ms=500, text="这")
        # Khung 0.5s: ngân sách nhỏ (trừ chi phí cố định ~0.3s) nhưng không bóp về 0.
        hint = GeminiSubtitleTranslator._length_hint(line)
        assert hint == _ngan_sach(0.5)
        assert MIN_CHAR_BUDGET <= hint < 20

    def test_long_line_larger_hint(self) -> None:
        line = TranslationLine(index=1, start_ms=0, end_ms=3000, text="x")
        # Khung dài: ngân sách RỘNG HƠN công thức cũ (48) — cũ ép cắt nghĩa vô ích.
        hint = GeminiSubtitleTranslator._length_hint(line)
        assert hint == _ngan_sach(3.0)
        assert hint > 48

    def test_no_timing_no_hint(self) -> None:
        line = TranslationLine(index=1, start_ms=0, end_ms=0, text="x")
        assert GeminiSubtitleTranslator._length_hint(line) == 0

    def test_minimum_floor(self) -> None:
        # Dòng cực ngắn vẫn cho tối thiểu MIN_CHAR_BUDGET ký tự (tránh ép cắt ý).
        line = TranslationLine(index=1, start_ms=0, end_ms=100, text="x")
        assert GeminiSubtitleTranslator._length_hint(line) == MIN_CHAR_BUDGET

    def test_hint_tang_theo_khung(self) -> None:
        ngan = TranslationLine(index=1, start_ms=0, end_ms=1000, text="x")
        dai = TranslationLine(index=2, start_ms=0, end_ms=2500, text="x")
        assert GeminiSubtitleTranslator._length_hint(ngan) < GeminiSubtitleTranslator._length_hint(dai)

    def test_payload_includes_hint(self) -> None:
        line = TranslationLine(index=1, start_ms=0, end_ms=1000, text="你好")
        payload = GeminiSubtitleTranslator._line_to_payload(line)
        assert payload["max_chars"] == _ngan_sach(1.0)
        assert payload["line_no"] == 1
        assert payload["text"] == "你好"

    def test_dual_payload_includes_hint(self) -> None:
        line = TranslationLine(index=1, start_ms=0, end_ms=2000, text="bản nháp")
        payload = GeminiSubtitleTranslator._line_to_dual_payload(line)
        assert payload["max_chars"] == _ngan_sach(2.0)

    def test_no_hint_field_when_no_timing(self) -> None:
        line = TranslationLine(index=1, start_ms=5, end_ms=5, text="x")
        assert "max_chars" not in GeminiSubtitleTranslator._line_to_payload(line)
