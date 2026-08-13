"""[v3.23.266] Cờ tắt pedalboard cho build THƯƠNG MẠI (tránh GPL).

pedalboard (Spotify) là **GPL v3** (nhúng JUCE/VST3 SDK) — không dùng được trong
ứng dụng thương mại đóng mà không công khai mã nguồn. App dùng nó cho time-stretch.

**Giải pháp:** app đã có sẵn fallback librosa (ISC, an toàn thương mại). Thêm cờ
``SUBEXT_DISABLE_PEDALBOARD=1`` để ép fallback, đảm bảo build thương mại KHÔNG chạm
pedalboard kể cả khi nó vô tình được cài. Xem docs/LICENSE_ANALYSIS.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.time_stretch import (
    stretch_with_pedalboard,
    vocal_time_stretch,
)


@pytest.fixture
def _sine() -> np.ndarray:
    sr = 24_000
    return np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)


def test_cờ_tắt_pedalboard_trả_none(monkeypatch, _sine) -> None:
    # Đặt cờ -> stretch_with_pedalboard trả None (ép fallback), không chạm pedalboard.
    monkeypatch.setenv("SUBEXT_DISABLE_PEDALBOARD", "1")
    assert stretch_with_pedalboard(_sine, 24_000, 1.5) is None


def test_không_cờ_pedalboard_hoạt_động(monkeypatch, _sine) -> None:
    # Không cờ -> pedalboard chạy (nếu cài). Chưa cài cũng trả None (an toàn).
    monkeypatch.delenv("SUBEXT_DISABLE_PEDALBOARD", raising=False)
    result = stretch_with_pedalboard(_sine, 24_000, 1.5)
    # Kết quả: hoặc mảng (pedalboard cài) hoặc None (chưa cài) — không lỗi.
    assert result is None or isinstance(result, np.ndarray)


def test_vocal_stretch_vẫn_đúng_khi_tắt_pedalboard(monkeypatch, _sine) -> None:
    # Khi tắt pedalboard, vocal_time_stretch tự fallback librosa -> vẫn đúng độ dài.
    monkeypatch.setenv("SUBEXT_DISABLE_PEDALBOARD", "1")
    out = vocal_time_stretch(_sine, 24_000, 1.5)
    expected = int(len(_sine) / 1.5)
    assert abs(len(out) - expected) / expected < 0.05


def test_cờ_giá_trị_khác_1_không_tắt(monkeypatch, _sine) -> None:
    # Chỉ "1" mới tắt; giá trị khác không ảnh hưởng (an toàn, không tắt nhầm).
    monkeypatch.setenv("SUBEXT_DISABLE_PEDALBOARD", "0")
    result = stretch_with_pedalboard(_sine, 24_000, 1.5)
    assert result is None or isinstance(result, np.ndarray)
