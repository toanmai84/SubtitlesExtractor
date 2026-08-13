"""[v3.23.197] Test tôn trọng "Cho phép chồng tiếng" + time-stretch pedalboard.

Hai cải tiến từ yêu cầu người dùng:
1. Bật "Cho phép chồng tiếng" nhưng VieNeu vẫn cắt: bug — nhánh cắt chỉ đọc
   ``max_overlap_ms`` (=0 -> cắt khít khung), bỏ qua ``allow_audio_overlap``. Fix:
   ``fit_limit_samples`` trả None (không cắt, audio tràn tự nhiên vào master cộng dồn)
   khi cho phép chồng tiếng -> giữ TRỌN nội dung thoại.
2. Pedalboard (Rubber Band, formant-preserving) làm đường time-stretch chất lượng cao
   nhất — optional dependency (GPL/dual-license, không bundle), fallback librosa/WSOLA.
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _stretch_with_pedalboard,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    _MASTER_TAIL_PAD_S,
    fit_limit_samples,
)

_SR = 24000


# ── fit_limit_samples: tôn trọng chồng tiếng ─────────────────────────────


def test_allow_overlap_means_no_truncation() -> None:
    # Bật chồng tiếng -> None = KHÔNG cắt (giữ trọn nội dung, tràn tự nhiên).
    assert fit_limit_samples(2.0, 0, True, _SR) is None


def test_allow_overlap_ignores_max_overlap_ms() -> None:
    # Bật chồng tiếng: max_overlap_ms=0 (ca cấu hình người dùng) vẫn KHÔNG cắt.
    assert fit_limit_samples(1.0, 0, True, _SR) is None
    assert fit_limit_samples(1.0, 500, True, _SR) is None


def test_disallow_overlap_limits_by_frame_plus_overlap() -> None:
    # Tắt chồng tiếng -> giới hạn = (khung + lấn) * sr.
    assert fit_limit_samples(2.0, 500, False, _SR) == int(2.5 * _SR)


def test_disallow_overlap_zero_ms_cuts_at_frame() -> None:
    assert fit_limit_samples(2.0, 0, False, _SR) == int(2.0 * _SR)


def test_master_tail_pad_covers_last_event_extension() -> None:
    # Đệm đuôi master phải >= trần nới câu cuối (max_gap_use 2s) để không chặt biên.
    assert _MASTER_TAIL_PAD_S >= 2.0


# ── _stretch_with_pedalboard ─────────────────────────────────────────────


def test_pedalboard_returns_none_when_missing(monkeypatch) -> None:
    # Giả lập pedalboard chưa cài -> None (caller fallback librosa/WSOLA).
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "pedalboard":
            raise ImportError("No module named 'pedalboard'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    tone = np.zeros(_SR, dtype=np.float32)
    assert _stretch_with_pedalboard(tone, _SR, 2.0) is None


def test_pedalboard_real_stretch_if_installed() -> None:
    # Chạy thật nếu pedalboard có trong môi trường (sandbox CI có).
    pytest.importorskip("pedalboard")
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(_SR) / _SR)).astype(np.float32)
    out = _stretch_with_pedalboard(tone, _SR, 2.0)
    assert out is not None
    assert out.ndim == 1  # ép về mono 1D dù pedalboard trả (1, n)
    assert out.dtype == np.float32
    # 1.0s nén 2x -> ~0.5s (dung sai 10%).
    assert abs(out.size / _SR - 0.5) < 0.05
    # Năng lượng formant giữ được (Rubber Band) — peak không sụp đổ.
    assert float(np.abs(out).max()) > 0.15
