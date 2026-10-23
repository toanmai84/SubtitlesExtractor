"""[v3.23.450] Ghép lại master audio từ các segment RIÊNG của từng dòng (hàm thuần).

Dùng sau khi TTS LẠI một số dòng: thay vì render lại toàn bộ, ta chỉ tổng hợp lại các dòng đã đổi
rồi GHÉP LẠI master từ tất cả segment WAV (dòng cũ tái dùng file cũ, dòng mới dùng file vừa ghi).
Mỗi segment được đặt tại mốc ĐỌC THẬT (``adjusted_start_sec``) của dòng đó.

Tách khỏi I/O ghi file để test được: hàm nhận sẵn mảng audio từng segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentPlacement:
    """Một segment audio + mốc đặt vào timeline."""

    start_sec: float
    audio: np.ndarray  # mono float32, đã ở đúng sample rate


def remerge_master(
    placements: list[SegmentPlacement],
    sample_rate: int,
    total_length_samples: int,
) -> np.ndarray:
    """Ghép các segment vào một master mới (cộng dồn tại mốc đọc thật của từng dòng).

    Args:
        placements: Danh sách (mốc bắt đầu giây, mảng audio) của từng dòng còn tiếng.
        sample_rate: Tần số lấy mẫu (Hz).
        total_length_samples: Độ dài master (mẫu) — thường bằng master gốc để giữ nguyên
            tổng thời lượng/đồng bộ video.

    Returns:
        Mảng master ``float32`` độ dài ``total_length_samples``.

    Raises:
        ValueError: Nếu ``sample_rate`` hoặc ``total_length_samples`` không hợp lệ.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate phải > 0")
    if total_length_samples <= 0:
        raise ValueError("total_length_samples phải > 0")

    master = np.zeros(total_length_samples, dtype=np.float32)
    for placement in placements:
        audio = placement.audio
        if audio is None or len(audio) == 0:
            continue
        start_sample = max(0, int(placement.start_sec * sample_rate))
        if start_sample >= total_length_samples:
            continue
        end_sample = min(start_sample + len(audio), total_length_samples)
        placed = end_sample - start_sample
        master[start_sample:end_sample] += audio[:placed].astype(np.float32)
    return master


__all__ = ["SegmentPlacement", "remerge_master"]
