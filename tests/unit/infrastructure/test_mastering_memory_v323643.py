"""[v3.23.643] Chuỗi master cho phim dài: kết quả y hệt bản cũ, RAM đỉnh thấp hơn nhiều.

Đo trên sandbox: chuỗi master cũ của audio 900 s ở 48 kHz đỉnh 989 MB (5,7× kích thước audio); riêng
``dc_block`` 3,2× — suy ra ~9 GB cho phim 2h19 ở 48 kHz.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from subtitles_extractor.infrastructure.tts import audio_mastering as am

SR = 16_000


def _signal(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(t.size) + 0.05
    tone[int(1.5 * SR):int(1.52 * SR)] *= 6.0          # đỉnh để soft_limit có việc làm
    return tone.astype(np.float32)


def _peak_mb(fn, *args) -> float:
    tracemalloc.start()
    try:
        fn(*args)
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


@pytest.fixture(autouse=True)
def _small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(am, "_MASTER_BLOCK_SAMPLES", 50_000)   # nhiều khối cả ở tín hiệu ngắn


def test_dc_block_equals_the_single_pass_filter() -> None:
    audio = _signal(20.0)
    sos = butter(2, 20.0 / (SR * 0.5), btype="highpass", output="sos")
    reference = sosfilt(sos, audio).astype(np.float32)
    assert np.array_equal(am.dc_block(audio, SR), reference)


def test_soft_limit_equals_the_reference_formula() -> None:
    audio = _signal(20.0)
    abs_a = np.abs(audio)
    over = abs_a > 0.85
    reference = audio.copy()
    room = 0.92 - 0.85
    reference[over] = np.sign(audio[over]) * (0.85 + room * np.tanh((abs_a[over] - 0.85) / room))
    got = am.soft_limit(audio, 0.85, 0.92)
    assert over.any() and np.array_equal(got, reference.astype(np.float32))
    assert audio.max() > 0.92                                  # đầu vào không bị sửa tại chỗ


def test_soft_limit_returns_input_when_nothing_is_over() -> None:
    quiet = (_signal(5.0) * 0.1).astype(np.float32)
    assert am.soft_limit(quiet) is quiet


def test_master_finalize_unchanged_result_for_short_audio() -> None:
    audio = _signal(30.0)
    out = am.master_finalize(audio.copy(), SR, target_lufs=-16.0)
    assert out.dtype == np.float32 and out.shape == audio.shape
    assert float(np.max(np.abs(out))) <= 1.0


def test_dc_block_peak_memory_is_about_one_array() -> None:
    audio = _signal(200.0)
    size_mb = audio.nbytes / 1e6
    peak = _peak_mb(am.dc_block, audio, SR)
    assert peak < 1.6 * size_mb, (peak, size_mb)               # cũ: ~3.2×


def test_soft_limit_peak_memory_is_at_most_two_arrays() -> None:
    audio = _signal(200.0)
    size_mb = audio.nbytes / 1e6
    peak = _peak_mb(am.soft_limit, audio)
    assert peak < 2.2 * size_mb, (peak, size_mb)               # cũ: ~3×


def test_streaming_write_matches_one_shot_write(tmp_path: Path) -> None:
    audio = _signal(10.0) * 1.7                                # vượt ±1 để thử chặn biên
    for fmt, subtype in (("wav", "PCM_16"), ("flac", "PCM_24"), ("wav", "FLOAT")):
        got = am.write_audio(audio, SR, tmp_path / f"a_{subtype}", fmt=fmt, subtype=subtype)
        expected = tmp_path / f"b_{subtype}.{fmt}"
        sf.write(str(expected), am.clip_for_output(audio), SR, subtype=subtype)
        left, _ = sf.read(str(got), dtype="float32")
        right, _ = sf.read(str(expected), dtype="float32")
        assert np.array_equal(left, right), (fmt, subtype)
