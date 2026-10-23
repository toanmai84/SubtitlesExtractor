"""[v3.23.455] Đệm im lặng đầu audio để nghe thử per-line ổn định.

Thiết bị âm thanh có ĐỘ TRỄ KHỞI ĐỘNG (cold-start): vài chục–vài trăm ms đầu khi bắt đầu phát bị
"nuốt"/rè ("tẹt tẹt"), nên đầu câu bị mất. Chèn một đoạn im lặng ngắn TRƯỚC tiếng nói cho thiết bị
"nóng máy" xong rồi mới tới lời thoại → nghe đủ và sạch đầu câu.

Chỉ cho NGHE THỬ (preview) — KHÔNG ảnh hưởng audio xuất ra (segment gốc + master giữ nguyên).
"""

from __future__ import annotations

import numpy as np


def prepend_silence(audio: np.ndarray, sample_rate: int, silence_s: float) -> np.ndarray:
    """Thêm ``silence_s`` giây im lặng vào TRƯỚC ``audio`` (mono float32).

    Args:
        audio: Mảng audio (mono). Nếu là stereo sẽ lấy kênh trái.
        sample_rate: Tần số lấy mẫu (Hz).
        silence_s: Độ dài im lặng cần chèn (giây, ≥ 0).

    Returns:
        Mảng ``float32`` = [im lặng][audio]. Nếu ``silence_s`` ≤ 0 trả về audio (float32) như cũ.

    Raises:
        ValueError: Nếu ``sample_rate`` không dương.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate phải > 0")
    mono = audio[:, 0] if audio.ndim > 1 else audio
    mono = mono.astype(np.float32, copy=False)
    if silence_s <= 0:
        return mono
    pad = np.zeros(int(silence_s * sample_rate), dtype=np.float32)
    return np.concatenate([pad, mono])


__all__ = ["prepend_silence"]
