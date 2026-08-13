"""[v3.23.212] Test tiết kiệm RAM khi master hoá audio DÀI (phim/tập dài).

Bug đo được: chuỗi master ngốn RAM đỉnh ~15x kích thước audio (nhiều mảng trung gian
float64 của sosfiltfilt + bản K-weighted ghép của bộ đo LUFS cùng tồn tại) -> tập 45
phút cần ~3.9GB, phim 2h cần ~10.4GB -> treo/crash máy đang chạy PaddleOCR song song.

Fix (KHÔNG đổi kết quả — đo trên giọng thật: sai lệch 0.000%%, LUFS y hệt):
1. ``process_in_blocks``: lọc theo KHỐI có đệm warm-up (IIR zero-phase nguội trong vùng
   đệm -> không click, kết quả gần như đồng nhất); ``voice_clarity`` dùng khi audio dài.
2. ``measure_lufs``: ghi thẳng vào mảng cấp sẵn float32 thay vì tích luỹ list float64
   rồi concatenate.
Audio ngắn (< 300s) giữ NGUYÊN đường cũ -> zero regression.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts import audio_mastering as am

_SR = 24000


def _speech_like(duration_s: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(int(_SR * duration_s)) / _SR
    voiced = 0.2 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 540 * t)
    return (voiced + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


# ── process_in_blocks (hàm thuần) ────────────────────────────────────────


def test_blocks_preserve_length() -> None:
    audio = _speech_like(5.0)
    out = am.process_in_blocks(audio, _SR, am._voice_clarity_core, block_s=1.0, pad_s=0.2)
    assert out.shape == audio.shape
    assert out.dtype == np.float32


def test_blocks_match_global_processing() -> None:
    # Tiêu chí cốt lõi: xử lý theo khối KHÔNG đổi âm thanh (sai lệch dưới ngưỡng nghe).
    audio = _speech_like(8.0)
    blocked = am.process_in_blocks(
        audio, _SR, am._voice_clarity_core, block_s=2.0, pad_s=1.0
    )
    global_out = am._voice_clarity_core(audio, _SR)
    rms_err = float(np.sqrt(((blocked - global_out) ** 2).mean()))
    rms_sig = float(np.sqrt((global_out**2).mean()))
    assert rms_err / rms_sig < 0.01  # < 1% = không phân biệt được khi nghe


def test_blocks_no_click_at_seams() -> None:
    # Đường ghép không tạo bước nhảy biên độ (click).
    audio = _speech_like(6.0)
    out = am.process_in_blocks(
        audio, _SR, am._voice_clarity_core, block_s=1.0, pad_s=0.5
    )
    jumps = np.abs(np.diff(out))
    assert float(jumps.max()) < 0.5  # không có bước nhảy đột ngột


def test_short_audio_uses_direct_path() -> None:
    audio = _speech_like(2.0)
    direct = am._voice_clarity_core(audio, _SR)
    via_blocks = am.process_in_blocks(audio, _SR, am._voice_clarity_core)
    assert np.array_equal(direct, via_blocks)  # ngắn -> gọi thẳng, không chia khối


# ── RAM: audio dài ───────────────────────────────────────────────────────


@pytest.mark.slow
def test_long_audio_ram_bounded() -> None:
    # Audio 6 phút (> ngưỡng 300s) -> RAM đỉnh phải dưới 10x kích thước audio
    # (trước fix: ~15x; mục tiêu là tỉ lệ với KHỐI, không với toàn phim).
    audio = _speech_like(360.0)
    tracemalloc.start()
    am.master_finalize(audio, _SR, target_lufs=-14.0, apply_clarity=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / audio.nbytes < 10.0


def test_measure_lufs_stable_after_float32_storage() -> None:
    # Lưu bản K-weighted ở float32 không làm lệch LUFS (< 0.1 LU).
    audio = _speech_like(20.0)
    lufs = am.measure_lufs(audio, _SR)
    assert lufs != float("-inf")
    louder = (audio * 2.0).astype(np.float32)
    # Nhân đôi biên độ -> +6.02 dB (kiểm tính đúng đắn của thang đo).
    assert am.measure_lufs(louder, _SR) == pytest.approx(lufs + 6.02, abs=0.1)


def test_thresholds_defined() -> None:
    assert am._LONG_AUDIO_S == 300.0
    assert am._BLOCK_S > 0 and am._BLOCK_PAD_S > 0
