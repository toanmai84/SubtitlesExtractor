"""[v3.23.209] Test NÉN ÊM (soft-limit) thay clip cứng khi tắt "Chuẩn hoá loudness".

Rà soát nhánh ``normalize=False``: master cộng dồn chồng tiếng có đỉnh vượt 1.0 (2
giọng chồng đo được ~1.5-1.8) -> ``np.clip`` cắt phẳng gây MÉO nghe rõ ở chính đoạn
chồng. Fix: VieNeu + Gemini dùng ``audio_mastering.soft_limit`` (nén êm vùng vượt,
trong suốt phần còn lại); ``write_audio`` vẫn giữ clip lưới an toàn cuối. Nhánh
normalize=True không đổi (master_finalize đã có true-peak limiter).
"""

from __future__ import annotations

import pathlib

import numpy as np

from subtitles_extractor.infrastructure.tts.audio_mastering import soft_limit

_SR = 24000


def _two_voices_overlapped() -> np.ndarray:
    # Mô phỏng đoạn 2 giọng chồng: tổng đỉnh ~1.7 (vượt full-scale).
    t = np.arange(_SR) / _SR
    return (0.9 * np.sin(2 * np.pi * 200 * t) + 0.8 * np.sin(2 * np.pi * 310 * t)).astype(
        np.float32
    )


def test_soft_limit_tames_overlapped_peaks() -> None:
    mixed = _two_voices_overlapped()
    assert float(np.abs(mixed).max()) > 1.3  # tiền đề: thật sự vượt biên
    out = soft_limit(mixed)
    assert float(np.abs(out).max()) <= 1.0  # không còn vượt full-scale


def test_soft_limit_flattens_less_than_hard_clip() -> None:
    # Méo clip = vùng bị CẮT PHẲNG (đạo hàm ~0 kéo dài). Soft phải ít hơn hẳn hard.
    mixed = _two_voices_overlapped()
    hard = np.clip(mixed, -1.0, 1.0)
    soft = soft_limit(mixed)
    flat_hard = int((np.abs(np.diff(hard)) < 1e-7).sum())
    flat_soft = int((np.abs(np.diff(soft)) < 1e-7).sum())
    assert flat_soft < flat_hard * 0.6


def test_soft_limit_transparent_for_normal_audio() -> None:
    # Audio bình thường (đỉnh dưới knee) đi qua gần như nguyên vẹn.
    quiet = (0.5 * np.sin(2 * np.pi * 220 * np.arange(_SR) / _SR)).astype(np.float32)
    out = soft_limit(quiet)
    assert float(np.max(np.abs(out - quiet))) < 0.02


def test_vieneu_and_gemini_use_soft_limit_when_normalize_off() -> None:
    vieneu = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
    ).read_text(encoding="utf-8")
    gemini = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "master = mastering.soft_limit(master)" in vieneu
    assert "master = _am.soft_limit(master)" in gemini
    # Không còn hard clip trực tiếp ở nhánh tắt chuẩn hoá.
    assert "np.clip(master, -1.0, 1.0, out=master)" not in gemini
