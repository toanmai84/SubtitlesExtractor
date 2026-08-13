"""[v3.23.176] Test kẹp gain sau làm mượt trong lookahead limiter (chống lọt méo đỉnh).

Bug: làm mượt đường gain (uniform_filter1d) tránh pumping nhưng KÉO gain tại đỉnh
transient đơn lẻ LÊN CAO (vì lân cận gain ~1.0) -> đỉnh không được ghìm đủ -> lọt méo
(đo thực: đỉnh 2.5 chỉ giảm còn 1.747, vẫn > 1.0). Fix: hàm thuần ``_clamp_smoothed_gain``
lấy min(mượt, thô) theo mẫu -> làm mượt chỉ được GIẢM gain, không nới lỏng mức ghìm.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    EdgeTTSAdapter,
    _clamp_smoothed_gain,
)

_SR = 24000


# ── Hàm thuần _clamp_smoothed_gain ───────────────────────────────────────


def test_clamp_takes_minimum_per_sample() -> None:
    smoothed = np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float32)
    raw = np.array([1.0, 0.5, 0.9, 0.6], dtype=np.float32)
    result = _clamp_smoothed_gain(smoothed, raw)
    # Mỗi mẫu = min(mượt, thô).
    assert np.allclose(result, [1.0, 0.5, 0.8, 0.6])


def test_clamp_never_exceeds_raw_gain() -> None:
    rng = np.random.default_rng(42)
    smoothed = rng.uniform(0.0, 1.5, size=1000).astype(np.float32)
    raw = rng.uniform(0.0, 1.0, size=1000).astype(np.float32)
    result = _clamp_smoothed_gain(smoothed, raw)
    assert np.all(result <= raw + 1e-6)


def test_clamp_preserves_smoothed_when_lower() -> None:
    # Khi mượt đã thấp hơn thô -> giữ nguyên mượt (làm mượt có hiệu lực).
    smoothed = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    raw = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    result = _clamp_smoothed_gain(smoothed, raw)
    assert np.allclose(result, 0.3)


# ── Tích hợp _lookahead_soft_limiter ─────────────────────────────────────


def test_lookahead_limiter_ghim_dinh_nhon() -> None:
    # Đỉnh transient đơn lẻ 2.5 giữa nền 0.3: sau limiter KHÔNG còn vượt xa 1.0
    # (trước fix: 1.747). Nền giữ nguyên (không ghìm oan).
    master = np.full(int(_SR * 0.1), 0.3, dtype=np.float32)
    master[len(master) // 2] = 2.5
    out = EdgeTTSAdapter._lookahead_soft_limiter(master, _SR, threshold=0.95)
    assert np.max(np.abs(out)) <= 1.01  # trước fix: 1.747
    assert np.allclose(out[:100], 0.3, atol=0.01)  # nền không bị ghìm


def test_lookahead_no_op_when_under_threshold() -> None:
    # Toàn bộ dưới threshold -> trả nguyên vẹn.
    master = np.full(int(_SR * 0.05), 0.5, dtype=np.float32)
    out = EdgeTTSAdapter._lookahead_soft_limiter(master, _SR, threshold=0.95)
    assert np.array_equal(out, master)
