"""[v3.23.660] Master TTS phải đạt đúng độ to mục tiêu sau khi ghìm đỉnh (đo thật Edge: −16,9 LUFS khi mục tiêu −16)."""

from __future__ import annotations

import numpy as np
import pytest

pyln = pytest.importorskip("pyloudnorm")

from subtitles_extractor.infrastructure.tts.audio_mastering import master_finalize


def _speechy(sr: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * 1.5)) / sr
    parts = []
    for _ in range(20):
        f0 = 120 + 60 * rng.random()
        s = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
        s *= 1 + 3 * (rng.random(len(t)) > 0.997)  # vài đỉnh nhọn như phụ âm bật
        parts += [0.1 * s, np.zeros(int(sr * 0.7))]
    return np.concatenate(parts).astype(np.float32)


@pytest.mark.parametrize("target", [-16.0, -14.0, -20.0])
def test_master_hits_target_loudness(target: float) -> None:
    sr = 24000
    out = master_finalize(_speechy(sr), sr, target_lufs=target)
    lufs = pyln.Meter(sr).integrated_loudness(out.astype(np.float64))
    assert abs(lufs - target) <= 0.3, lufs
    assert np.abs(out).max() <= 0.951


def test_silence_stays_silent() -> None:
    out = master_finalize(np.zeros(24000 * 3, np.float32), 24000)
    assert not np.any(out)
