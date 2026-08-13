"""[v3.23.210] Test đồng bộ silent-check (v205) + trim im lặng (v204) sang Gemini/Edge.

Kỷ luật engine parity: hai fix chất lượng lớn nhất chỉ có ở VieNeu. Rà chéo phát hiện
Gemini (2 đường: standard retry + native async) và Edge (2 retry loop) đều chỉ kiểm
``len(audio) > 0`` — CÙNG lỗ hổng v205: engine sinh audio CÓ độ dài nhưng toàn im lặng
-> mất thoại âm thầm mà vẫn báo OK (đo thực ở VieNeu: 4 câu/video). Đồng thời Gemini
thiếu ``trim_edge_silence`` (v204) -> tiếng có thể trễ so với mốc phụ đề.

Fix: dùng chung hàm thuần (``is_effectively_silent``, ``trim_edge_silence``) — Gemini
import trực tiếp (tiền lệ stretch_ratio_cap/fit_limit_samples), Edge qua wrapper
delegate ``_is_silent_audio``. Edge KHÔNG trim (elastic đo duration ở Pass 1 — trim ở
đó có rủi ro lệch timing; ghi backlog).
"""

from __future__ import annotations

import pathlib

import numpy as np

from subtitles_extractor.infrastructure.tts import edge_tts_adapter as edge_mod
from subtitles_extractor.infrastructure.tts import gemini_tts_adapter as gemini_mod

_SR = 24000

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")
_EDGE_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/edge_tts_adapter.py"
).read_text(encoding="utf-8")


def _voice() -> np.ndarray:
    return (0.2 * np.sin(2 * np.pi * 220 * np.arange(_SR) / _SR)).astype(np.float32)


def _silence() -> np.ndarray:
    return np.zeros(_SR, dtype=np.float32)


# ── Edge: wrapper delegate ───────────────────────────────────────────────


def test_edge_silent_detector_delegates() -> None:
    assert edge_mod._is_silent_audio(_silence())
    assert not edge_mod._is_silent_audio(_voice())


def test_edge_retry_loops_check_silence() -> None:
    # CẢ HAI retry loop (async + probe) phải coi im lặng là thất bại.
    assert _EDGE_SRC.count("not _is_silent_audio(audio)") >= 2
    assert _EDGE_SRC.count("toàn im lặng") >= 2  # log phân biệt rỗng vs im lặng


# ── Gemini: cả 2 đường ───────────────────────────────────────────────────


def test_gemini_imports_shared_pure_functions() -> None:
    assert "is_effectively_silent" in _GEMINI_SRC
    assert "trim_edge_silence" in _GEMINI_SRC
    assert gemini_mod.is_effectively_silent(_silence())
    assert not gemini_mod.is_effectively_silent(_voice())


def test_gemini_standard_retry_rejects_silence() -> None:
    assert "if len(audio) > 0 and not is_effectively_silent(audio):" in _GEMINI_SRC


def test_gemini_native_path_rejects_silence() -> None:
    assert "if len(audio) == 0 or is_effectively_silent(audio):" in _GEMINI_SRC


def test_gemini_trims_before_dialog_pause_both_paths() -> None:
    # Trim phải đứng TRƯỚC pause hội thoại (pause có kiểm soát không bị cắt ngược).
    assert _GEMINI_SRC.count("trim_edge_silence(") >= 2  # standard + native/batch
    # [v3.23.217] Pause chèn SAU nén; trim vẫn phải đứng TRƯỚC (không cắt nhầm pause).
    # [v3.23.241] trim nay bật adaptive=True (ngưỡng tự dò).
    trim_pos = _GEMINI_SRC.index("trim_edge_silence(audio, sr, adaptive=True)")
    pause_pos = _GEMINI_SRC.index("if pause_s > 0.0:")
    assert trim_pos < pause_pos


def test_no_circular_import() -> None:
    # Gemini import VieNeu ở top-level — 3 engine phải cùng nạp được.
    import subtitles_extractor.infrastructure.tts.vieneu_tts_adapter as vieneu_mod

    assert vieneu_mod.is_effectively_silent is gemini_mod.is_effectively_silent
