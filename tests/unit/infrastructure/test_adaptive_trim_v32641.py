"""[v3.23.241] Ngưỡng cắt im lặng TỰ DÒ — đưa lợi ích VAD của Edge sang VieNeu/Gemini.

**Phát hiện khi rà engine-parity:** Edge dùng ``_trim_silence`` với ngưỡng ADAPTIVE
percentile (tự dò sàn nhiễu = p5 của RMS, chặn trên theo đỉnh để không lẹm phụ âm cuối).
VieNeu và Gemini lại dùng ``trim_edge_silence`` với ngưỡng CỐ ĐỊNH tuyệt đối 0.008.

Ngưỡng cố định có một điểm yếu: câu model sinh biên độ thấp (RMS < 0.008) thì KHÔNG cắt
được im lặng đầu -> tiếng vang trễ so với mốc phụ đề. Đo trên câu tổng hợp::

    biên độ đỉnh   ngưỡng cố định 0.008   ngưỡng adaptive
    0.30           cắt 90ms lặng          cắt 90ms lặng
    0.015          cắt 110ms lặng         cắt 90ms lặng
    0.008          GIỮ NGUYÊN (bỏ sót)    cắt 90ms lặng

Trên FLAC Gemini thật, câu nhỏ nhất có RMS thô ~0.018-0.046 — VẪN trên 0.008, nên bug này
CHƯA kích hoạt trong dữ liệu hiện có. Nhưng nó là bất đối xứng chất lượng thật: Edge có
đường xử lý im lặng bền vững hơn mà hai engine kia chưa hưởng. Đưa về hàm dùng chung
(``adaptive=True``) là engine-parity đúng nghĩa, phòng khi gặp câu giọng nhỏ/thì thầm.

**Mặc định GIỮ NGUYÊN** (``adaptive=False``) để không hồi quy hành vi đã nghiệm thu.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.audio_utils import (
    _adaptive_voice_threshold,
    trim_edge_silence,
)

_SR = 24_000


def _cau(amp: float, lead_s: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(1)
    sil = (rng.standard_normal(int(lead_s * _SR)) * 0.0005).astype(np.float32)
    t = np.arange(int(0.5 * _SR)) / _SR
    voice = (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    return np.concatenate([sil, voice])


def _da_cat_lead(audio_goc: np.ndarray, audio_trim: np.ndarray) -> bool:
    return len(audio_trim) < len(audio_goc) - int(0.05 * _SR)


# ── Mặc định KHÔNG đổi (chống hồi quy) ──────────────────────────────────────
def test_mac_dinh_van_dung_nguong_co_dinh() -> None:
    # Câu RMS < 0.008: ngưỡng cố định bỏ sót -> giữ nguyên. Phải y như trước v241.
    a = _cau(0.008)
    assert _da_cat_lead(a, trim_edge_silence(a, _SR)) is False


def test_cau_bien_do_thuong_ca_hai_che_do_deu_cat() -> None:
    a = _cau(0.30)
    assert _da_cat_lead(a, trim_edge_silence(a, _SR)) is True
    assert _da_cat_lead(a, trim_edge_silence(a, _SR, adaptive=True)) is True


# ── Adaptive cứu được câu biên độ thấp ──────────────────────────────────────
def test_adaptive_cat_duoc_cau_bien_do_thap() -> None:
    a = _cau(0.008)
    # Cố định bó tay, adaptive cắt được.
    assert _da_cat_lead(a, trim_edge_silence(a, _SR)) is False
    assert _da_cat_lead(a, trim_edge_silence(a, _SR, adaptive=True)) is True


# ── Adaptive không lẹm khi toàn nhiễu (không phân biệt được tiếng) ──────────
def test_adaptive_giu_nguyen_khi_khong_co_tieng_ro() -> None:
    # Tín hiệu quá yếu, lẫn nhiễu -> không cắt liều (tránh lẹm).
    a = (np.random.default_rng(2).standard_normal(int(0.6 * _SR)) * 0.004).astype(
        np.float32
    )
    out = trim_edge_silence(a, _SR, adaptive=True)
    # Không được cắt mạnh tay khi không chắc đâu là tiếng.
    assert len(out) >= len(a) - int(0.2 * _SR)


# ── Ngưỡng adaptive: giữa sàn nhiễu và đỉnh ─────────────────────────────────
def test_nguong_adaptive_nam_giua_san_va_dinh() -> None:
    rms = np.array([0.001, 0.001, 0.001, 0.2, 0.2, 0.2], dtype=np.float32)
    thr = _adaptive_voice_threshold(rms)
    assert thr > float(np.percentile(rms, 5))  # trên sàn nhiễu
    assert thr < float(np.max(rms))  # dưới đỉnh (không bỏ sót tiếng)


def test_nguong_adaptive_chan_tren_theo_dinh() -> None:
    # Câu động học lớn: ngưỡng không được vượt 15% đỉnh (kẻo lẹm phụ âm cuối nhẹ).
    rms = np.array([0.05, 0.05, 1.0, 1.0], dtype=np.float32)
    thr = _adaptive_voice_threshold(rms)
    assert thr <= 1.0 * 0.15 + 1e-9


def test_nguong_adaptive_mang_rong() -> None:
    assert _adaptive_voice_threshold(np.array([], dtype=np.float32)) == 0.008
