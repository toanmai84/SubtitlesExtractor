"""[v3.23.177] Test đo LUFS có GATING theo EBU R128 (im lặng không kéo loudness sai).

Bug: ``_measure_lufs`` lấy mean-square TOÀN CỤC gồm cả khoảng lặng giữa câu -> master
thưa câu bị đo LUFS THẤP hơn thực -> gain chuẩn hoá đẩy cao oan -> giọng to quá mức, và
độ to không nhất quán giữa phim dày/thưa thoại. Đo thực: 4s im lặng kéo -17.6 xuống
-24.6 LUFS (lệch 7 LU). Fix: hàm thuần ``_gated_loudness_from_kweighted`` áp absolute
gate (-70 LUFS) + relative gate (-10 LU) như chuẩn R128.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    EdgeTTSAdapter,
    _gated_loudness_from_kweighted,
)

_SR = 24000


def _tone(duration_s: float, amplitude: float = 0.2, freq: float = 300.0) -> np.ndarray:
    samples = np.arange(int(_SR * duration_s))
    return (amplitude * np.sin(2 * np.pi * freq * samples / _SR)).astype(np.float32)


# ── Hàm thuần _gated_loudness_from_kweighted ─────────────────────────────


def test_silence_blocks_excluded() -> None:
    # Tín hiệu K-weighted giả: 1s "giọng" + 4s im lặng. Gating loại im lặng ->
    # loudness gần với chỉ-giọng, không bị kéo xuống.
    speech = _tone(1.0)
    silence = np.zeros(_SR * 4, dtype=np.float32)
    lufs_speech = _gated_loudness_from_kweighted(speech, _SR)
    lufs_mixed = _gated_loudness_from_kweighted(
        np.concatenate([speech, silence]), _SR
    )
    assert abs(lufs_speech - lufs_mixed) < 1.5  # trước fix: ~7 LU


def test_empty_returns_neg_inf() -> None:
    assert _gated_loudness_from_kweighted(np.array([]), _SR) == float("-inf")


def test_all_silence_returns_neg_inf() -> None:
    assert _gated_loudness_from_kweighted(
        np.zeros(_SR, dtype=np.float32), _SR
    ) == float("-inf")


def test_short_signal_below_one_block() -> None:
    # Ngắn hơn 400ms -> đo trực tiếp (không gating), vẫn trả giá trị hữu hạn.
    short = _tone(0.1)
    result = _gated_loudness_from_kweighted(short, _SR)
    assert np.isfinite(result)


def test_louder_signal_higher_lufs() -> None:
    quiet = _gated_loudness_from_kweighted(_tone(2.0, amplitude=0.1), _SR)
    loud = _gated_loudness_from_kweighted(_tone(2.0, amplitude=0.4), _SR)
    assert loud > quiet


# ── Tích hợp _measure_lufs ───────────────────────────────────────────────


def test_measure_lufs_robust_to_silence() -> None:
    speech = _tone(1.0)
    silence = np.zeros(_SR * 4, dtype=np.float32)
    lufs_speech = EdgeTTSAdapter._measure_lufs(speech, _SR)
    lufs_master = EdgeTTSAdapter._measure_lufs(
        np.concatenate([speech, silence]), _SR
    )
    # Im lặng chỉ được phép làm lệch < 1.5 LU (trước fix: 7 LU).
    assert abs(lufs_speech - lufs_master) < 1.5


def test_measure_lufs_empty() -> None:
    assert EdgeTTSAdapter._measure_lufs(np.array([]), _SR) == float("-inf")
