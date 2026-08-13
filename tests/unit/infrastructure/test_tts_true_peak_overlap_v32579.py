"""[v3.23.179] Test đo true-peak theo chunk CÓ overlap (không bỏ sót đỉnh ở ranh giới).

Bug: ``_measure_true_peak`` chia master thành chunk 30s rồi ``resample_poly`` từng chunk
RỜI NHAU. Đỉnh liên-mẫu VẮT QUA ranh giới chunk bị bỏ sót (chunk không thấy mẫu lân cận
ở chunk kế) -> đo THẤP hơn thực (đo thực: báo 0.90 trong khi đỉnh thật 1.05) -> limiter
tưởng an toàn -> đỉnh lọt qua -> méo khi encode lossy. Fix: hàm thuần
``_true_peak_chunked_overlap`` lấy thêm overlap mẫu ở mỗi biên chunk.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    EdgeTTSAdapter,
    _true_peak_chunked_overlap,
)

_SR = 24000


def _global_true_peak(signal: np.ndarray, oversample: int = 4) -> float:
    """True-peak tham chiếu tính TOÀN CỤC (không chia chunk) làm chuẩn so sánh."""
    upsampled = resample_poly(signal.astype(np.float64), oversample, 1)
    return float(np.max(np.abs(upsampled)))


def test_peak_at_chunk_boundary_not_missed() -> None:
    chunk = _SR * 30
    signal = np.full(_SR * 31, 0.3, dtype=np.float32)
    # Đỉnh liên-mẫu cao (đổi dấu nhanh) vắt qua ranh giới chunk 30s.
    signal[chunk - 1] = 0.9
    signal[chunk] = -0.9
    measured = _true_peak_chunked_overlap(signal, _SR)
    reference = _global_true_peak(signal)
    assert abs(measured - reference) < 0.01  # trước fix lệch ~0.15


def test_matches_global_for_short_signal() -> None:
    rng = np.random.default_rng(7)
    signal = (rng.uniform(-0.8, 0.8, size=_SR * 2)).astype(np.float32)
    measured = _true_peak_chunked_overlap(signal, _SR)
    reference = _global_true_peak(signal)
    assert abs(measured - reference) < 0.005


def test_empty_returns_zero() -> None:
    assert _true_peak_chunked_overlap(np.array([], dtype=np.float32), _SR) == 0.0


def test_tiny_signal_returns_sample_peak() -> None:
    signal = np.array([0.1, -0.5, 0.3], dtype=np.float32)
    assert abs(_true_peak_chunked_overlap(signal, _SR) - 0.5) < 1e-6


def test_true_peak_above_sample_peak() -> None:
    # Sóng có đỉnh liên-mẫu > đỉnh mẫu rời rạc (đặc trưng cần oversample để bắt).
    t = np.arange(_SR)
    signal = (0.9 * np.sin(2 * np.pi * 6000 * t / _SR)).astype(np.float32)
    measured = _true_peak_chunked_overlap(signal, _SR)
    assert measured >= float(np.max(np.abs(signal))) - 1e-6


def test_measure_true_peak_integration_boundary() -> None:
    chunk = _SR * 30
    signal = np.full(_SR * 31, 0.3, dtype=np.float32)
    signal[chunk - 1] = 0.9
    signal[chunk] = -0.9
    measured = EdgeTTSAdapter._measure_true_peak(signal, _SR)
    assert abs(measured - _global_true_peak(signal)) < 0.01
