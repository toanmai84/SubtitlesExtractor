"""[v3.23.662] Edge TTS có đường master RIÊNG (không qua ``master_finalize``) → bản sửa bù độ to v3.23.660 không tới.

Đo thật tập 1–2 (bench 28/10): Edge vẫn ra −16,91 LUFS khi mục tiêu −16 (Gemini/VieNeu −16,0).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pyln = pytest.importorskip("pyloudnorm")

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import edge_tts_adapter

_SR = 24000


def _speechy(seed: int) -> np.ndarray:
    """Giọng nhiều đỉnh nhọn (phụ âm bật) — kiểu tín hiệu làm limiter cắt hụt độ to."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(_SR * 1.4)) / _SR
    f0 = 120 + 60 * rng.random()
    s = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    s *= 1 + 3 * (rng.random(len(t)) > 0.997)
    return (0.1 * s).astype(np.float32)


@pytest.mark.parametrize("target", [-16.0, -14.0])
def test_edge_master_hits_target_loudness(tmp_path: Path, monkeypatch, target: float) -> None:
    adapter = edge_tts_adapter.EdgeTTSAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    async def fake_generate(text, voice, rate):
        return _speechy(len(text)), _SR

    monkeypatch.setattr(adapter, "_async_generate", staticmethod(fake_generate))
    events = [SimpleNamespace(index=i + 1, text="x" * (i + 3), start_sec=2.0 * i, end_sec=2.0 * i + 1.8)
              for i in range(15)]
    request = TTSRequest(events=events, language="vi-VN", speaker="vi-VN-HoaiMyNeural", normalize=True,
                         target_lufs=target, double_pass=False, media_duration_s=31.0, output_format="wav")
    adapter.generate(request, tmp_path / "out.wav")
    audio, sr = sf.read(str(tmp_path / "out.wav"), dtype="float64")
    lufs = pyln.Meter(sr).integrated_loudness(audio)
    assert abs(lufs - target) <= 0.3, lufs
    assert np.abs(audio).max() <= 0.951


# ── Nguyên nhân gốc: bộ đo LUFS của app dùng hệ số K-weighting 48 kHz cho cả audio 24 kHz ─────────────

def _program(sr: int, seconds: float) -> np.ndarray:
    reps = int(seconds / 2.0) + 1
    out = np.concatenate([np.concatenate([_speechy(i), np.zeros(int(0.6 * sr), np.float32)]) for i in range(reps)])
    if sr != _SR:
        from scipy.signal import resample_poly

        out = resample_poly(out, sr, _SR).astype(np.float32)
    return out[: int(seconds * sr)]


@pytest.mark.parametrize("sr", [24000, 44100, 48000])
def test_app_meter_matches_bs1770_at_any_rate(sr: int) -> None:
    from subtitles_extractor.infrastructure.tts.audio_mastering import measure_lufs

    audio = _program(sr, 30.0)
    reference = pyln.Meter(sr).integrated_loudness(audio.astype(np.float64))
    # Trước đây ở 24 kHz lệch +0,9 LU (đo thật tập 1: app −15,1, pyloudnorm −16,0).
    assert abs(measure_lufs(audio, sr) - reference) <= 0.15


def test_long_episode_master_hits_target() -> None:
    """Tập > 5 phút đi đường đo theo khối của app (không qua pyloudnorm) — trước đây thiếu ~0,9 LU ở 24 kHz."""
    from subtitles_extractor.infrastructure.tts.audio_mastering import master_finalize

    audio = _program(_SR, 320.0)
    out = master_finalize(audio, _SR, target_lufs=-16.0, apply_clarity=False)
    lufs = pyln.Meter(_SR).integrated_loudness(out.astype(np.float64))
    assert abs(lufs + 16.0) <= 0.3, lufs


@pytest.mark.parametrize("samples", [100, 5000, 9599])
def test_master_shorter_than_one_block_does_not_crash(samples: int) -> None:
    """Rà độc lập: pyloudnorm ném ValueError với audio < 400 ms → master 1 câu ngắn làm hỏng lượt TTS."""
    from subtitles_extractor.infrastructure.tts.audio_mastering import compensate_lufs_shortfall, master_finalize

    audio = (0.01 * np.random.default_rng(1).standard_normal(samples)).astype(np.float32)
    assert np.isfinite(compensate_lufs_shortfall(audio.copy(), _SR, -16.0)).all()
    assert np.isfinite(master_finalize(audio, _SR, apply_clarity=False)).all()


@pytest.mark.parametrize("sr", [11025, 22050])
def test_meter_odd_rates(sr: int) -> None:
    from subtitles_extractor.infrastructure.tts.audio_mastering import measure_lufs

    audio = _program(sr, 20.0)
    assert abs(measure_lufs(audio, sr) - pyln.Meter(sr).integrated_loudness(audio.astype(np.float64))) <= 0.3
