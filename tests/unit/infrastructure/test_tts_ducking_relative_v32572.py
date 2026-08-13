"""[v3.23.172] Test ducking dùng ngưỡng TƯƠNG ĐỐI theo đỉnh (không bỏ sót giọng nhỏ).

Bug: độ dài vùng chồng dùng ngưỡng cứng 0.01 (-40dBFS). Khi câu trước NHỎ (thì thầm,
hoặc tắt chuẩn hoá nên giữ biên gốc Edge ~0.008), mọi mẫu dưới ngưỡng -> coi như KHÔNG
chồng -> KHÔNG duck -> hai giọng cộng cùng độ to -> đục tiếng. Fix: ngưỡng lấy MAX
(min thực ra) giữa sàn tuyệt đối và mức -40dB SO ĐỈNH region -> luôn bắt đúng vùng chồng
dù câu trước to hay nhỏ, đồng thời KHÔNG tính đuôi fade cực nhỏ của câu xa (giữ ưu điểm
bản cũ).
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _overlap_length_samples,
)


def test_detects_overlap_for_quiet_previous_voice() -> None:
    # Câu trước biên 0.008 (< 0.01 cứng cũ) -> nay VẪN phát hiện chồng.
    region = np.concatenate([
        np.full(6000, 0.008, dtype=np.float32),
        np.zeros(6000, dtype=np.float32),
    ])
    assert _overlap_length_samples(region) == 6000


def test_detects_overlap_for_loud_previous_voice() -> None:
    region = np.concatenate([
        np.full(6000, 0.8, dtype=np.float32),
        np.zeros(6000, dtype=np.float32),
    ])
    assert _overlap_length_samples(region) == 6000


def test_ignores_tiny_fade_tail_of_distant_voice() -> None:
    # Đỉnh 0.8 + đuôi fade cực nhỏ 1e-5: ngưỡng tương đối cao -> KHÔNG tính đuôi
    # (giữ ưu điểm V12.1: không thổi phồng độ dài chồng).
    region = np.concatenate([
        np.full(3000, 0.8, dtype=np.float32),
        np.full(9000, 1e-5, dtype=np.float32),
    ])
    assert _overlap_length_samples(region) == 3000


def test_silence_returns_zero() -> None:
    assert _overlap_length_samples(np.zeros(12000, dtype=np.float32)) == 0


def test_empty_region_returns_zero() -> None:
    assert _overlap_length_samples(np.array([], dtype=np.float32)) == 0


def test_overlap_at_region_end() -> None:
    # Tín hiệu tới tận cuối region -> ov_end = độ dài region.
    region = np.full(5000, 0.5, dtype=np.float32)
    assert _overlap_length_samples(region) == 5000


def test_relative_threshold_scales_with_peak() -> None:
    # Hai region cùng HÌNH DẠNG nhưng khác biên độ -> cùng độ dài chồng (tương đối).
    shape = np.concatenate([
        np.full(4000, 1.0, dtype=np.float32),
        np.zeros(4000, dtype=np.float32),
    ])
    loud = shape * 0.9
    quiet = shape * 0.02
    assert _overlap_length_samples(loud) == _overlap_length_samples(quiet)
