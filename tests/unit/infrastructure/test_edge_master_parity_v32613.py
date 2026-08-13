"""[v3.23.213] Test Edge dùng DC blocker chuẩn + bộ đo LUFS dùng chung (parity + RAM).

Rà chéo pipeline master hoá RIÊNG của Edge (không dùng ``master_finalize``):
1. **DC blocker lỗi thời**: Edge trừ MEAN toàn cục (``master -= mean``) — chỉ loại DC
   KHÔNG ĐỔI. TTS neural sinh DC DRIFT CHẬM trên audio dài; đo thực với drift 0.02Hz:
   Edge để sót DC gấp **~1449x** so với ``dc_block`` (high-pass 20Hz, fix v184 đã áp cho
   VieNeu/Gemini) -> ăn **23%% headroom** -> limiter kích hoạt oan, nén tiếng vô cớ.
2. **Bản sao ``_measure_lufs`` dính bug RAM v212**: tích luỹ list float64 + concatenate
   (~9x RAM audio) -> phim dài ngốn hàng GB. Delegate sang ``audio_mastering.
   measure_lufs`` (CÙNG hệ số K-weighting — đã đối chiếu trùng khít — và cùng gating,
   nhưng đã tối ưu RAM).
"""

from __future__ import annotations

import pathlib
import tracemalloc

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.audio_mastering import dc_block, measure_lufs
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter

_SR = 24000
_EDGE_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/edge_tts_adapter.py"
).read_text(encoding="utf-8")


def _voice_with_dc_drift(duration_s: float = 30.0) -> np.ndarray:
    t = np.arange(int(_SR * duration_s)) / _SR
    voice = 0.25 * np.sin(2 * np.pi * 180 * t)
    drift = 0.08 * np.sin(2 * np.pi * 0.02 * t)  # DC drift chậm (TTS neural)
    return (voice + drift).astype(np.float32)


def _residual_dc(audio: np.ndarray, window_s: float = 5.0) -> float:
    win = int(window_s * _SR)
    kernel = np.ones(win // 100) / (win // 100)
    smooth = np.convolve(audio[::100].astype(np.float64), kernel, mode="valid")
    return float(np.abs(smooth).max())


# ── DC blocker ───────────────────────────────────────────────────────────


def test_dc_block_beats_mean_subtraction_on_drift() -> None:
    sig = _voice_with_dc_drift()
    old_way = (sig - np.float32(sig.mean())).astype(np.float32)
    new_way = dc_block(sig, _SR)
    # Trừ mean để sót DC drift; dc_block loại gần sạch (ít nhất 10x tốt hơn).
    assert _residual_dc(new_way) < _residual_dc(old_way) / 10.0


def test_dc_block_frees_headroom() -> None:
    sig = _voice_with_dc_drift()
    old_way = (sig - np.float32(sig.mean())).astype(np.float32)
    new_way = dc_block(sig, _SR)
    # Đỉnh thấp hơn = còn headroom cho limiter -> không nén tiếng oan.
    assert float(np.abs(new_way).max()) < float(np.abs(old_way).max())


def test_dc_block_preserves_speech_band() -> None:
    # Không được ăn vào dải thoại (300-4kHz) — chỉ loại hạ âm/DC.
    from scipy.signal import welch

    t = np.arange(int(_SR * 5.0)) / _SR
    speech = (0.25 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
    out = dc_block(speech, _SR)
    freqs, p_in = welch(speech, _SR, nperseg=4096)
    _, p_out = welch(out, _SR, nperseg=4096)
    band = (freqs >= 300) & (freqs < 4000)
    loss = (p_in[band].sum() - p_out[band].sum()) / p_in[band].sum()
    assert abs(loss) < 0.01  # < 1% -> không nghe được


def test_edge_uses_shared_dc_block() -> None:
    assert "master = dc_block(master, sr)" in _EDGE_SRC
    assert "master -= np.float32(np.mean(master))" not in _EDGE_SRC  # hết cách cũ


# ── LUFS: delegate về nguồn chung ────────────────────────────────────────


def test_edge_lufs_matches_shared_module() -> None:
    audio = _voice_with_dc_drift(20.0)
    assert EdgeTTSAdapter._measure_lufs(audio, _SR) == pytest.approx(
        measure_lufs(audio, _SR), abs=0.001
    )


def test_edge_lufs_copy_removed() -> None:
    # Bản sao dính bug RAM v212 phải biến mất khỏi Edge (một nguồn sự thật).
    assert "kweighted_parts" not in _EDGE_SRC
    assert "audio_mastering import measure_lufs" in _EDGE_SRC


def test_edge_lufs_ram_bounded() -> None:
    audio = _voice_with_dc_drift(120.0)
    tracemalloc.start()
    EdgeTTSAdapter._measure_lufs(audio, _SR)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / audio.nbytes < 8.0  # trước ~11.8x, sau tối ưu ~6.7x
