"""[v3.23.199] Test pedalboard vào module time_stretch dùng chung + Gemini gap-aware.

Rà soát tiếp phát hiện: Gemini ``_time_stretch`` dùng ``time_stretch_preserve_pitch``
(module time_stretch.py) — KHÔNG qua ``_time_stretch_vocal`` của Edge nên KHÔNG được
hưởng pedalboard v197. Fix: chuyển ``stretch_with_pedalboard`` vào module dùng chung,
``time_stretch_preserve_pitch`` ưu tiên nó -> mọi caller (Gemini, Edge, VieNeu) cùng
hưởng Rubber Band. Edge giữ wrapper delegate (một nguồn sự thật, DRY). Đồng thời Gemini
nhận gap-aware stretch (đồng bộ VieNeu v194 — giảm nén mạnh -55%) ở CẢ 2 đường.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts import time_stretch as ts_mod
from subtitles_extractor.infrastructure.tts.time_stretch import (
    stretch_with_pedalboard,
    time_stretch_preserve_pitch,
)

_SR = 24000
_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
)


def _tone(duration_s: float) -> np.ndarray:
    n = int(_SR * duration_s)
    return (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / _SR)).astype(np.float32)


# ── pedalboard trong module dùng chung ───────────────────────────────────


def test_preserve_pitch_uses_pedalboard_when_available() -> None:
    pytest.importorskip("pedalboard")
    out = time_stretch_preserve_pitch(_tone(1.0), _SR, 2.0)
    # Rubber Band trả độ dài CHÍNH XÁC (0.500s); WSOLA chỉ xấp xỉ -> đây là bằng chứng
    # đường pedalboard được chọn.
    assert out.ndim == 1
    assert abs(out.size / _SR - 0.5) < 0.02


def test_preserve_pitch_falls_back_when_pedalboard_missing(monkeypatch) -> None:
    # Chặn pedalboard -> vẫn stretch được (WSOLA/librosa), không crash.
    monkeypatch.setattr(ts_mod, "stretch_with_pedalboard", lambda *a, **k: None)
    out = time_stretch_preserve_pitch(_tone(1.0), _SR, 1.5)
    assert out.size > 0
    assert abs(out.size / _SR - (1.0 / 1.5)) < 0.15  # xấp xỉ (WSOLA)


def test_stretch_with_pedalboard_returns_none_when_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "pedalboard":
            raise ImportError("No module named 'pedalboard'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert stretch_with_pedalboard(_tone(0.5), _SR, 2.0) is None


def test_edge_wrapper_delegates_to_shared_module() -> None:
    # Edge giữ tên _stretch_with_pedalboard (tương thích) nhưng phải delegate sang
    # module dùng chung — không còn bản sao logic (DRY).
    edge_src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/edge_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "from subtitles_extractor.infrastructure.tts.time_stretch import" in edge_src
    assert edge_src.count("from pedalboard import") == 0  # logic thật ở time_stretch


# ── Gemini gap-aware (đồng bộ VieNeu v194) ───────────────────────────────


def test_gemini_uses_effective_available_both_paths() -> None:
    source = _GEMINI_SRC.read_text(encoding="utf-8")
    # Cả đường thường lẫn batch đều dùng khung hiệu dụng (>=2 lần gọi).
    assert source.count("effective_available_seconds(") >= 2
    # Stretch/skip/fit đều theo khung hiệu dụng, không còn so khung gốc.
    assert "audio_dur - effective_available" in source
    assert source.count("effective_available, request.max_overlap_ms") >= 2


def test_gemini_passes_next_start_to_standard_path() -> None:
    source = _GEMINI_SRC.read_text(encoding="utf-8")
    assert "next_start_sec: float | None = None" in source
    assert "next_start_sec=next_start" in source
