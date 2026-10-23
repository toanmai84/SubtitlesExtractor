"""[v3.23.615] Tiếng "tách" lẻ sau lời thoại phải bị cắt cùng khoảng lặng (mọi engine).

FLAC lồng tiếng thật 14/10 (VieNeu, 95 câu, phim "天骄女友退婚" tập 1): câu "Hôm nay muội
đến đây" nói xong ở 0.78s, im 305ms rồi có một tiếng tạp 30ms (-24 dB) ngay cuối đoạn; câu
"Hả?!" im 230ms rồi tạp âm 20ms. Bộ cắt lặng cũ lấy khung có tiếng CUỐI CÙNG làm mốc nên
giữ cả khoảng lặng lẫn tiếng tách: người nghe thấy một tiếng "click" sau câu thoại. Áp bộ
cắt mới lên đúng 95 đoạn của FLAC thật: chỉ hai câu đó đổi (ngắn đi 228ms và 152ms).
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.audio_utils import (
    speech_span,
    trim_edge_silence,
)
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter

SR = 48_000


def _tone(seconds: float, amplitude: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _speech_then_click(gap_s: float = 0.305, click_s: float = 0.03) -> np.ndarray:
    """Dạng sóng của câu 57 thật: lời 0.78s (-10 dB), lặng, tạp âm (-24 dB)."""
    return np.concatenate([
        _silence(0.05), _tone(0.78, 0.3), _silence(gap_s), _tone(click_s, 0.06, 1800.0),
    ])


class TestSpeechSpan:
    def test_short_tail_island_after_long_gap_is_dropped(self) -> None:
        voiced = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
        assert speech_span(voiced, frame_s=0.02) == (1, 4)

    def test_short_head_island_before_long_gap_is_dropped(self) -> None:
        voiced = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
        assert speech_span(voiced, frame_s=0.02) == (9, 12)

    def test_pause_inside_the_line_is_kept(self) -> None:
        """Từ ngắn ≥ 80ms ("À,") rồi ngắt dài là lời, không phải tạp âm."""
        voiced = np.array(
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=bool
        )
        assert speech_span(voiced, frame_s=0.02) == (0, 17)

    def test_island_close_to_speech_is_kept(self) -> None:
        voiced = np.array([1, 1, 1, 1, 0, 0, 1], dtype=bool)
        assert speech_span(voiced, frame_s=0.02) == (0, 6)

    def test_single_island_is_never_dropped(self) -> None:
        voiced = np.array([0, 0, 1, 0, 0], dtype=bool)
        assert speech_span(voiced, frame_s=0.02) == (2, 2)

    def test_no_voice(self) -> None:
        assert speech_span(np.zeros(5, dtype=bool), frame_s=0.02) is None


class TestTrimming:
    @pytest.mark.parametrize(("gap_s", "click_s"), [(0.305, 0.03), (0.23, 0.02)])
    def test_vieneu_gemini_trim_drops_the_click(
        self, gap_s: float, click_s: float
    ) -> None:
        trimmed = trim_edge_silence(_speech_then_click(gap_s, click_s), SR, adaptive=True)
        assert len(trimmed) / SR < 0.05 + 0.78 + 0.1

    def test_edge_trim_drops_the_click(self) -> None:
        trimmed = EdgeTTSAdapter._trim_silence(_speech_then_click(), SR)
        assert len(trimmed) / SR < 0.05 + 0.78 + 0.1

    def test_trailing_short_word_survives(self) -> None:
        """"Đi… à!" — từ cuối 120ms sau khoảng ngắt vẫn là lời."""
        audio = np.concatenate([
            _silence(0.05), _tone(0.5, 0.3), _silence(0.25), _tone(0.12, 0.3),
        ])
        assert len(trim_edge_silence(audio, SR, adaptive=True)) / SR > 0.5 + 0.25 + 0.1
        assert len(EdgeTTSAdapter._trim_silence(audio, SR)) / SR > 0.5 + 0.25 + 0.1
