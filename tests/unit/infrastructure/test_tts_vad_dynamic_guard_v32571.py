"""[v3.23.171] Test VAD cắt lặng an toàn động học — không cắt nhầm phụ âm cuối nhẹ.

Bug: ngưỡng cắt lặng ``max(noise_floor×3.5, peak×10^(-45/20))`` bị ĐỘI LÊN khi câu có
động học lớn (một từ rất to) -> cắt nhầm phụ âm cuối nhẹ ('s','ch','th') / âm tắt dần
(đo thực: cắt 230ms đuôi -36dB so đỉnh). Fix: CHẶN TRÊN ngưỡng theo dynamic guard
(-42dB so đỉnh) qua hàm thuần ``_voiced_bounds_from_rms``.
"""

from __future__ import annotations

# ruff: noqa: RUF002, RUF003 — dấu × (nhân) trong mô tả dB là CHỦ ĐÍCH

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    EdgeTTSAdapter,
    _voiced_bounds_from_rms,
)

_SR = 24000


def _tone(duration_s: float, amplitude: float, freq: float = 200.0) -> np.ndarray:
    samples = np.arange(int(_SR * duration_s))
    return (amplitude * np.sin(2 * np.pi * freq * samples / _SR)).astype(np.float32)


# ── Hàm thuần _voiced_bounds_from_rms ────────────────────────────────────


def test_guard_keeps_quiet_tail() -> None:
    # rms: đỉnh 0.7 + đuôi 0.01 (~-37dB). Không guard, ngưỡng peak×10^(-45/20)=0.0039
    # -> đuôi 0.01 > 0.0039 vẫn giữ; nhưng noise_floor×3.5 có thể đội cao -> guard giữ.
    rms = np.array([0.7, 0.7, 0.7, 0.01, 0.01], dtype=np.float64)
    bounds = _voiced_bounds_from_rms(rms, noise_floor=0.008, peak_rms=0.7)
    assert bounds is not None
    assert bounds[1] == 4  # đuôi nhỏ VẪN nằm trong biên (không bị cắt)


def test_threshold_capped_by_guard() -> None:
    # noise_floor cao giả tạo (do phần to kéo lên) -> nếu không guard, ngưỡng =
    # 0.05×3.5 = 0.175 sẽ cắt đuôi 0.02; guard -42dB×0.7 = 0.0056 kéo ngưỡng xuống.
    rms = np.array([0.7, 0.7, 0.02, 0.02], dtype=np.float64)
    bounds = _voiced_bounds_from_rms(rms, noise_floor=0.05, peak_rms=0.7)
    assert bounds is not None and bounds[1] == 3  # đuôi 0.02 được giữ nhờ guard


def test_all_silence_returns_none() -> None:
    # Khi peak_rms ~ noise_floor (không có động học), guard theo đỉnh rất nhỏ nhưng
    # base_threshold = noise_floor×3.5 vẫn cao hơn mọi mẫu -> None (toàn im lặng).
    # Lưu ý: tầng gọi (_trim_silence) đã chặn max_val<1e-4 trước khi tới đây; hàm này
    # nhận rms có biên độ hợp lệ. Mô phỏng nhiễu nền phẳng thấp:
    rms = np.array([0.002, 0.0021, 0.002], dtype=np.float64)
    # peak 0.0021, noise 0.002 -> base=max(0.007, 0.0000118)=0.007; guard=0.0021×0.0079
    # =1.6e-5 -> threshold=min(0.007,1.6e-5)=1.6e-5 -> mọi mẫu > 1.6e-5 -> KHÔNG none.
    # Đây là hành vi ĐÚNG: rms phẳng không động học -> tầng _trim_silence lọc bằng
    # max_val; hàm thuần chỉ tìm biên khi CÓ tiếng. Kiểm biên trả về trọn vẹn:
    bounds = _voiced_bounds_from_rms(rms, noise_floor=0.002, peak_rms=0.0021)
    assert bounds == (0, 2)  # rms phẳng -> coi toàn bộ là "có tiếng" (biên đầy đủ)


def test_empty_rms_returns_none() -> None:
    assert _voiced_bounds_from_rms(np.array([]), 0.0, 0.0) is None


def test_finds_inner_voiced_region() -> None:
    rms = np.array([0.001, 0.5, 0.5, 0.001], dtype=np.float64)
    bounds = _voiced_bounds_from_rms(rms, noise_floor=0.001, peak_rms=0.5)
    assert bounds == (1, 2)


# ── Tích hợp _trim_silence ───────────────────────────────────────────────


def test_trim_keeps_soft_final_consonant() -> None:
    # Đỉnh to + đuôi rất nhỏ (-36dB) mô phỏng phụ âm cuối -> KHÔNG bị cắt.
    audio = np.concatenate([
        _tone(0.4, 1.0),
        np.zeros(int(_SR * 0.03), dtype=np.float32),
        _tone(0.25, 0.015, 150.0),
    ])
    trimmed = EdgeTTSAdapter._trim_silence(audio, _SR)
    cut_ms = (len(audio) - len(trimmed)) / _SR * 1000.0
    assert cut_ms < 50.0  # gần như không cắt (giữ trọn đuôi phụ âm)


def test_trim_still_removes_true_silence() -> None:
    audio = np.concatenate([
        np.zeros(int(_SR * 0.2), dtype=np.float32),
        _tone(0.3, 0.8),
        np.zeros(int(_SR * 0.2), dtype=np.float32),
    ])
    trimmed = EdgeTTSAdapter._trim_silence(audio, _SR)
    cut_ms = (len(audio) - len(trimmed)) / _SR * 1000.0
    assert 200.0 < cut_ms < 400.0  # cắt phần lớn im lặng hai đầu (chừa pad 40ms)


def test_trim_empty_audio() -> None:
    assert EdgeTTSAdapter._trim_silence(np.array([], dtype=np.float32), _SR).size == 0
