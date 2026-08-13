"""[v3.23.184] Test DC blocker high-pass (loại được DC drift chậm ở audio dài).

Bug: ``master_finalize`` dùng ``audio - mean(audio)`` làm DC blocker — chỉ loại DC KHÔNG
ĐỔI. Audio dài (cả phim) từ TTS neural thường có DC DRIFT chậm (offset thay đổi theo
thời gian); trừ một hằng số không loại được (đo thực: mỗi đoạn vẫn lệch ±0.09) -> chiếm
headroom, méo bass. Fix: hàm thuần ``dc_block`` dùng high-pass 20Hz — loại sạch DC+drift
mà không suy giảm giọng (>80Hz).
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.audio_mastering import (
    dc_block,
    master_finalize,
)

_SR = 24000


def _max_segment_dc(signal: np.ndarray, segment_samples: int) -> float:
    """DC offset tuyệt đối lớn nhất trên các đoạn liên tiếp (đo hiệu quả khử DC)."""
    return max(
        abs(float(np.mean(signal[i:i + segment_samples])))
        for i in range(0, len(signal), segment_samples)
    )


def test_removes_slow_dc_drift() -> None:
    t = np.arange(_SR * 4)
    voice = 0.3 * np.sin(2 * np.pi * 300 * t / _SR)
    drift = 0.1 * np.cos(2 * np.pi * 0.25 * t / _SR)  # DC drift 0.25Hz
    signal = (voice + drift).astype(np.float32)
    out = dc_block(signal, _SR)
    # Mọi đoạn 0.5s gần như hết DC (trước fix: tới 0.09).
    assert _max_segment_dc(out, _SR // 2) < 0.01


def test_preserves_voice_band() -> None:
    t = np.arange(_SR)
    voice = (0.3 * np.sin(2 * np.pi * 300 * t / _SR)).astype(np.float32)
    out = dc_block(voice, _SR)
    rms_before = float(np.sqrt(np.mean(voice ** 2)))
    rms_after = float(np.sqrt(np.mean(out ** 2)))
    assert abs(rms_after - rms_before) < 0.01  # giọng 300Hz không suy giảm


def test_removes_constant_dc() -> None:
    signal = (0.2 * np.sin(2 * np.pi * 200 * np.arange(_SR) / _SR) + 0.15).astype(np.float32)
    out = dc_block(signal, _SR)
    assert abs(float(np.mean(out))) < 0.005


def test_short_signal_uses_mean() -> None:
    signal = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    out = dc_block(signal, _SR)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_empty_signal() -> None:
    out = dc_block(np.array([], dtype=np.float32), _SR)
    assert out.size == 0


def test_master_finalize_still_works() -> None:
    t = np.arange(_SR * 2)
    signal = (0.3 * np.sin(2 * np.pi * 300 * t / _SR) + 0.1).astype(np.float32)
    out = master_finalize(signal, _SR, target_lufs=-16.0)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0
    # DC đã được loại trong chuỗi master.
    assert abs(float(np.mean(out))) < 0.01
