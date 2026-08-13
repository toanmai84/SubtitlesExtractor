"""[v3.23.256->260] VieNeu v3 Turbo sample rate: giữ nguyên sr GỐC của engine.

**Lịch sử:**
- v256: phát hiện v3 Turbo xuất 48kHz nhưng ``infer`` trả array THUẦN (không kèm sr). App
  mặc định 24kHz -> audio 48kHz bị coi 24kHz -> méo tiếng + tính sai thời lượng. Fix đầu:
  resample 48kHz -> 24kHz (pipeline cố định 24kHz).
- v260: Toan chỉ ra file xuất 24kHz làm MẤT chất lượng (đo FLAC thật: năng lượng >10kHz =
  0%, dải tần 12-24kHz bị cắt). Quyết định đúng: **giữ nguyên sr GỐC của engine** — engine
  48kHz lưu 48kHz, engine 24kHz lưu 24kHz. ``_to_mono_pipeline_rate`` chỉ ép mono, KHÔNG
  resample; ``generate`` đặt pipeline sr = ``engine.sample_rate``.

Nhờ v260, cả hai vấn đề giải quyết cùng lúc: không méo tiếng (sr đúng) VÀ không mất chất
lượng (giữ 48kHz).
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter


def test_array_48khz_giữ_nguyên() -> None:
    # [v3.23.260] 48000 mẫu @ 48kHz -> GIỮ NGUYÊN 48000 mẫu (không hạ về 24kHz).
    audio_48k = np.ones(48_000, dtype=np.float32)
    out = VieNeuTtsAdapter._to_mono_pipeline_rate(audio_48k, 48_000)
    assert out.size == 48_000


def test_array_24khz_giữ_nguyên() -> None:
    # Engine 24kHz (standard/GGUF): giữ nguyên 24000 mẫu.
    audio_24k = np.ones(24_000, dtype=np.float32)
    out = VieNeuTtsAdapter._to_mono_pipeline_rate(audio_24k, 24_000)
    assert out.size == 24_000


def test_tuple_audio_sr_ép_mono_giữ_mẫu() -> None:
    # Bản trả (audio, sr): ép mono, giữ nguyên số mẫu.
    audio_48k = np.ones(48_000, dtype=np.float32)
    out = VieNeuTtsAdapter._to_mono_pipeline_rate((audio_48k, 48_000), None)
    assert out.size == 48_000


def test_stereo_48khz_ép_mono_giữ_sr() -> None:
    # Stereo 48kHz -> mono, giữ nguyên 48000 mẫu (không resample).
    stereo = np.ones((48_000, 2), dtype=np.float32)
    out = VieNeuTtsAdapter._to_mono_pipeline_rate(stereo, 48_000)
    assert out.ndim == 1
    assert out.size == 48_000


def test_không_còn_hạ_chất_lượng_48_xuống_24() -> None:
    # Cốt lõi v260: audio 48kHz KHÔNG bị hạ số mẫu (giữ dải tần cao).
    audio_48k = np.ones(96_000, dtype=np.float32)  # 2s @ 48kHz
    out = VieNeuTtsAdapter._to_mono_pipeline_rate(audio_48k, 48_000)
    # Nếu bị hạ về 24kHz sẽ còn 48000 mẫu; giữ nguyên phải là 96000.
    assert out.size == 96_000
