"""[v3.23.439] Test dịch CPS-aware: siết max_chars theo tốc độ đọc, toggle được."""
from __future__ import annotations
from subtitles_extractor.infrastructure.tts.timing_math import reading_char_budget
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


def _line(start_ms, end_ms, text="x"):
    return TranslationLine(index=1, start_ms=start_ms, end_ms=end_ms, text=text)


def test_reading_char_budget_pure():
    assert reading_char_budget(2.0, 21) == 42
    assert reading_char_budget(2.0, 0) == 0     # tắt
    assert reading_char_budget(0.0, 21) == 0    # khung 0


def test_cps_off_keeps_tts_budget():
    """reading_cps_budget=0 → dùng ngân sách TTS cũ (lỏng)."""
    adapter = GeminiSubtitleTranslator(api_key="k", reading_cps_budget=0.0)
    hint = adapter._length_hint(_line(0, 2000), next_start_ms=None)
    assert hint > 42  # TTS ~34 CPS → >42 cho 2s


def test_cps_on_tightens_budget():
    """reading_cps_budget=21 → siết về ~21 CPS (chặt hơn TTS)."""
    adapter = GeminiSubtitleTranslator(api_key="k", reading_cps_budget=21.0)
    hint = adapter._length_hint(_line(0, 2000), next_start_ms=None)
    assert hint == 42  # min(TTS~67, reading 42) = 42


def test_cps_on_respects_min_floor():
    """Khung siêu ngắn: vẫn giữ sàn MIN_CHAR_BUDGET, không về 0."""
    from subtitles_extractor.infrastructure.tts.timing_math import MIN_CHAR_BUDGET
    adapter = GeminiSubtitleTranslator(api_key="k", reading_cps_budget=21.0)
    hint = adapter._length_hint(_line(0, 200), next_start_ms=None)  # 0.2s
    assert hint >= MIN_CHAR_BUDGET


def test_cps_never_exceeds_tts():
    """CPS-aware chỉ SIẾT (min), không nới quá ngân sách TTS."""
    adapter_off = GeminiSubtitleTranslator(api_key="k", reading_cps_budget=0.0)
    adapter_on = GeminiSubtitleTranslator(api_key="k", reading_cps_budget=21.0)
    for dur in (1000, 2000, 3000, 5000):
        off = adapter_off._length_hint(_line(0, dur), next_start_ms=None)
        on = adapter_on._length_hint(_line(0, dur), next_start_ms=None)
        assert on <= off
