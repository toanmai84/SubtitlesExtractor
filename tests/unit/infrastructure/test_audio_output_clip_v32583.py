"""[v3.23.183] Test chặn biên độ trước khi ghi file (nhất quán mọi định dạng).

Bug: ``write_audio`` ghi WAV/FLAC bằng soundfile KHÔNG chặn biên trước. Với subtype
FLOAT, soundfile GIỮ NGUYÊN mẫu vượt ±1.0 -> méo/clip khi phát lại hoặc encode tiếp (vd
lồng vào video qua AAC). Path ffmpeg đã clip tường minh nhưng path soundfile thì không
-> KHÔNG NHẤT QUÁN. Fix: hàm thuần ``clip_for_output`` áp cho mọi path ghi.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from subtitles_extractor.infrastructure.tts.audio_mastering import (
    clip_for_output,
    write_audio,
)

_SR = 24000


# ── Hàm thuần clip_for_output ────────────────────────────────────────────


def test_clips_values_above_one() -> None:
    signal = np.array([0.5, 1.5, -1.8, 2.0], dtype=np.float32)
    result = clip_for_output(signal)
    assert np.max(np.abs(result)) <= 1.0
    assert result[0] == 0.5  # mẫu hợp lệ giữ nguyên


def test_does_not_mutate_input() -> None:
    signal = np.array([1.5, -1.5], dtype=np.float32)
    original = signal.copy()
    clip_for_output(signal)
    assert np.array_equal(signal, original)


def test_custom_ceiling() -> None:
    signal = np.array([0.8, 0.95, -0.99], dtype=np.float32)
    result = clip_for_output(signal, ceiling=0.9)
    assert np.max(np.abs(result)) <= 0.9


def test_returns_float32() -> None:
    result = clip_for_output(np.array([0.5, 2.0], dtype=np.float64))
    assert result.dtype == np.float32


def test_within_range_unchanged() -> None:
    signal = np.array([0.3, -0.7, 0.9], dtype=np.float32)
    result = clip_for_output(signal)
    assert np.array_equal(result, signal)


# ── Tích hợp write_audio ─────────────────────────────────────────────────


def test_wav_float_no_overflow() -> None:
    # WAV subtype FLOAT trước đây giữ mẫu vượt biên -> nay được clip.
    signal = np.array([0.5, 1.5, -1.8, 2.0], dtype=np.float32)
    tmp_dir = tempfile.mkdtemp()
    try:
        path = write_audio(
            signal, _SR, Path(tmp_dir) / "out", fmt="wav", subtype="FLOAT"
        )
        readback, _ = sf.read(str(path), dtype="float32")
        assert np.max(np.abs(readback)) <= 1.0
    finally:
        for file in Path(tmp_dir).iterdir():
            file.unlink()
        os.rmdir(tmp_dir)


def test_flac_pcm24_no_overflow() -> None:
    signal = np.array([0.5, 1.2, -1.5, 0.3], dtype=np.float32)
    tmp_dir = tempfile.mkdtemp()
    try:
        path = write_audio(
            signal, _SR, Path(tmp_dir) / "out", fmt="flac", subtype="PCM_24"
        )
        readback, _ = sf.read(str(path), dtype="float32")
        assert np.max(np.abs(readback)) <= 1.0
    finally:
        for file in Path(tmp_dir).iterdir():
            file.unlink()
        os.rmdir(tmp_dir)
