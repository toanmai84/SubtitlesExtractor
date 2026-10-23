"""[v3.23.616] Độ dài đảo tạp âm ở biên đo trên LÕI của nó, không trên ngưỡng có tiếng.

FLAC lồng tiếng thật 14/10 21:45 (VieNeu, 95 câu, phim "天骄女友退婚" tập 1): câu
86 "nhưng đường tu tiên" còn một tiếng 40-50ms (-13 dB, có cao độ 224 Hz như giọng
đọc) sau 210ms lặng ở cuối — v3.23.615 không cắt được. Bộ cắt chạy TRƯỚC khi nén tốc độ
(1.3x), nên ở đó tiếng này dài ~52ms; nhưng ngưỡng "có tiếng" tự dò rất thấp (sát sàn
nhiễu -85 dB) nên phần hiện/tắt dần cũng được đếm và hai mép cửa sổ 20ms chạm vào: đảo
thành 80ms, quá ngưỡng 60ms. Ca 57 "Hôm nay muội đến đây" của FLAC 15:48 (lý do có
v3.23.615) cũng KHÔNG được cắt khi chạy đúng thứ tự trước-nén. Đo lại trên 190 đoạn
(bản sau nén và bản trước nén xấp xỉ) của hai FLAC thật: chỉ đúng hai câu này đổi.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.audio_utils import (
    EDGE_ISLAND_CORE_DB,
    speech_span,
    trim_edge_silence,
)

SR = 48_000
NOISE_FLOOR = 3e-5  # -90 dB: sàn nhiễu đo giữa các câu trong FLAC thật


def _voice(seconds: float, amplitude: float, freq: float = 224.0) -> np.ndarray:
    """Âm có cao độ, bật/tắt dần như tiếng người (đuôi xuống tới sàn nhiễu)."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    envelope = np.ones(n)
    ramp = min(n // 3, int(0.02 * SR))
    fade = np.geomspace(NOISE_FLOOR / amplitude, 1.0, ramp)
    envelope[:ramp] = fade
    envelope[n - ramp:] = fade[::-1]
    return (amplitude * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _floor(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (NOISE_FLOOR * rng.standard_normal(int(seconds * SR))).astype(np.float32)


#: Đường bao (dB, mỗi 5ms) của tiếng lẻ câu 86 đo trên FLAC thật (sau nén 1.3x).
REAL_BURST_DB = [
    -80.0, -74.2, -67.4, -61.1, -46.6, -28.6, -19.2, -16.7, -15.3, -14.1, -13.9, -13.0,
    -18.2, -38.3, -44.8, -56.1, -69.4, -79.7, -88.0,
]


def _real_burst(stretch: float = 1.3) -> np.ndarray:
    """Tiếng lẻ thật, kéo về độ dài TRƯỚC nén (bộ cắt chạy trước bước nén)."""
    hop_s = 0.005 * stretch
    n = int(len(REAL_BURST_DB) * hop_s * SR)
    t = np.arange(n) / SR
    envelope_db = np.interp(t, np.arange(len(REAL_BURST_DB)) * hop_s, REAL_BURST_DB)
    amplitude = np.sqrt(2.0) * 10.0 ** (envelope_db / 20.0)
    return (amplitude * np.sin(2 * np.pi * 224.0 * t)).astype(np.float32)


def _line_with_tail_burst(burst: np.ndarray, gap_s: float = 0.27) -> np.ndarray:
    """Câu 86 trước nén: lời 0.83s (-5 dB), lặng 270ms, rồi ``burst``."""
    return np.concatenate([
        _floor(0.01, 1), _voice(0.83, 0.55), _floor(gap_s, 2), burst, _floor(0.005, 3),
    ])


class TestRealTailBurst:
    def test_real_burst_is_cut_with_its_gap(self) -> None:
        audio = _line_with_tail_burst(_real_burst())
        trimmed = trim_edge_silence(audio, SR, adaptive=True)
        # Lời 0.84s + đệm đuôi 80ms; trước đây giữ nguyên cả lặng lẫn tiếng lẻ (~1.25s).
        assert len(trimmed) / SR < 0.97

    def test_short_word_after_a_pause_is_kept(self) -> None:
        """Từ ngắn thật ("À," ~120ms, to như lời) sau khoảng ngắt vẫn là lời."""
        audio = _line_with_tail_burst(_voice(0.12, 0.3))
        trimmed = trim_edge_silence(audio, SR, adaptive=True)
        assert len(trimmed) / SR > 1.15


class TestSpeechSpanLevels:
    def test_faint_fade_frames_do_not_count_as_island_length(self) -> None:
        voiced = np.array([1] * 40 + [0] * 12 + [1, 1, 1, 1], dtype=bool)
        core_ratio = 10.0 ** (EDGE_ISLAND_CORE_DB / 20.0)
        fade = 0.2 * core_ratio / 4
        levels = np.array([0.3] * 40 + [0.0] * 12 + [fade, 0.2, 0.2, fade])
        assert speech_span(voiced, 0.02) == (0, 55)  # cách cũ: đảo 80ms là lời
        assert speech_span(voiced, 0.02, levels=levels) == (0, 39)

    def test_loud_long_island_is_still_speech(self) -> None:
        voiced = np.array([1] * 40 + [0] * 12 + [1] * 5, dtype=bool)
        levels = np.array([0.3] * 40 + [0.0] * 12 + [0.2] * 5)
        assert speech_span(voiced, 0.02, levels=levels) == (0, 56)
