"""[v3.23.182] Test đồng bộ DSP dùng chung (audio_mastering) với các fix v177-v181.

Module ``audio_mastering`` được DÙNG CHUNG cho mọi engine TTS (Edge, F5, và VieNeu sắp
tích hợp). Trước đây nó chạy DSP CŨ, thiếu các fix chất lượng v177 (LUFS gating), v178
(gain cap), v179 (true-peak overlap), v181 (soft-clip liên tầng) mà ``edge_tts_adapter``
đã có -> F5-TTS không được hưởng cải tiến. Nay đồng bộ bằng cách tái dùng chính các hàm
thuần đã kiểm chứng. Test này khoá tính nhất quán giữa hai module.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts import audio_mastering as am

_SR = 24000


def _tone(duration_s: float, amplitude: float = 0.2, freq: float = 300.0) -> np.ndarray:
    samples = np.arange(int(_SR * duration_s))
    return (amplitude * np.sin(2 * np.pi * freq * samples / _SR)).astype(np.float32)


def test_measure_lufs_has_gating() -> None:
    # Fix v177: im lặng không kéo LUFS xuống sai (chênh < 1.5 LU, trước ~7 LU).
    speech = _tone(1.0)
    silence = np.zeros(_SR * 4, dtype=np.float32)
    lufs_speech = am.measure_lufs(speech, _SR)
    lufs_mixed = am.measure_lufs(np.concatenate([speech, silence]), _SR)
    assert abs(lufs_speech - lufs_mixed) < 1.5


def test_measure_lufs_empty() -> None:
    assert am.measure_lufs(np.array([]), _SR) == float("-inf")


def test_true_peak_boundary_overlap() -> None:
    # Fix v179: đỉnh liên-mẫu vắt ranh giới chunk không bị bỏ sót.
    chunk = _SR * 30
    signal = np.full(_SR * 31, 0.3, dtype=np.float32)
    signal[chunk - 1] = 0.9
    signal[chunk] = -0.9
    from scipy.signal import resample_poly
    reference = float(np.max(np.abs(resample_poly(signal.astype(np.float64), 4, 1))))
    assert abs(am.measure_true_peak(signal, _SR) - reference) < 0.01


def test_normalize_gain_capped() -> None:
    # Fix v178: master rất nhỏ không bị khuếch đại quá +15dB.
    very_quiet = _tone(2.0, amplitude=0.01)
    out = am.normalize_to_lufs(very_quiet, _SR, target_lufs=-14.0)
    applied_gain = float(np.max(np.abs(out))) / max(float(np.max(np.abs(very_quiet))), 1e-9)
    # +15dB = ~5.62x; cho phép sai số nhỏ do peak-limit.
    assert applied_gain <= 5.7


def test_voice_clarity_no_hard_clip_artifacts() -> None:
    # Fix v181: giọng to qua voice_clarity không bị hard-clip (đỉnh <= 1.0, hữu hạn).
    t = np.arange(_SR)
    loud = (0.9 * (0.6 * np.sin(2 * np.pi * 300 * t / _SR)
                   + 0.5 * np.sin(2 * np.pi * 3000 * t / _SR))).astype(np.float32)
    loud = (loud / np.max(np.abs(loud)) * 0.95).astype(np.float32)
    out = am.voice_clarity(loud, _SR)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.001


def test_consistency_with_edge_adapter() -> None:
    # measure_lufs của hai module phải cho cùng kết quả (dùng chung hàm thuần gating).
    from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter
    signal = _tone(2.0, amplitude=0.3)
    lufs_am = am.measure_lufs(signal, _SR)
    lufs_edge = EdgeTTSAdapter._measure_lufs(signal, _SR)
    assert abs(lufs_am - lufs_edge) < 0.01
