"""[v3.23.416] Gợi ý batch size TTS GPU theo VRAM của card NVIDIA.

Batch càng lớn càng nhanh nhưng tốn VRAM (giữ nhiều audio/kích hoạt cùng lúc). VieNeu GPU chạy
tiến trình riêng chia sẻ VRAM với OCR/decode video, nên chọn thận trọng theo dung lượng card.
"""

from __future__ import annotations

import subprocess

from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs

#: Ánh xạ VRAM (GB) → batch size an toàn. Ngưỡng thận trọng vì VieNeu còn chia sẻ VRAM với
#: OCR/NVENC khi chạy song song. Card 8GB (vd RTX 3060 Ti) → 16; 12GB → 24; 16GB+ → 32.
_VRAM_BATCH_TABLE: tuple[tuple[float, int], ...] = (
    (16.0, 32),
    (12.0, 24),
    (8.0, 16),
    (6.0, 8),
    (0.0, 4),
)

#: Batch mặc định khi KHÔNG dò được VRAM (an toàn cho đa số card phổ thông 8GB).
DEFAULT_AUTO_BATCH_SIZE = 16


def batch_size_for_vram_gb(vram_gb: float) -> int:
    """Trả batch size an toàn cho lượng VRAM (GB) cho trước.

    Args:
        vram_gb: Tổng VRAM của card (GB).

    Returns:
        Batch size gợi ý (≥2).
    """
    for threshold_gb, batch in _VRAM_BATCH_TABLE:
        if vram_gb >= threshold_gb:
            return batch
    return 4


def detect_total_vram_gb() -> float | None:
    """Dò tổng VRAM (GB) của GPU NVIDIA đầu tiên qua ``nvidia-smi``.

    Returns:
        Tổng VRAM tính bằng GB, hoặc ``None`` nếu không dò được (không có nvidia-smi / lỗi).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_line = (result.stdout or "").strip().splitlines()
    if not first_line:
        return None
    try:
        mib = float(first_line[0].strip())
    except ValueError:
        return None
    return mib / 1024.0  # MiB → GiB


def suggest_auto_batch_size() -> int:
    """Gợi ý batch size theo VRAM thực tế (dò được), hoặc mặc định an toàn.

    Returns:
        Batch size gợi ý (≥2).
    """
    vram = detect_total_vram_gb()
    if vram is None:
        return DEFAULT_AUTO_BATCH_SIZE
    return batch_size_for_vram_gb(vram)


#: [v3.23.421] Ánh xạ VRAM (GB) → số CHUNK/forward GPU (tham số ``batch_size`` của
#: ``infer_batch``). LƯU Ý: mặc định v3turbo ĐÃ là 32 → phải VƯỢT 32 mới đổi hành vi. Chunk
#: audio ngắn nên hoạt hoá RẤT nhỏ (~0.01GB/chunk); model ~3GB chiếm phần lớn VRAM → tăng
#: chunk-batch giúp GIẢM số forward (ít overhead launch) hơn là để "đầy VRAM". 8GB→64 (2× mặc
#: định), tăng dần theo VRAM. Nếu OOM → worker tự lùi per-câu.
_VRAM_CHUNK_BATCH_TABLE: tuple[tuple[float, int], ...] = (
    (24.0, 192),
    (16.0, 128),
    (12.0, 96),
    (8.0, 64),
    (6.0, 32),
    (0.0, 16),
)

#: Chunk-batch mặc định khi không dò được VRAM (vẫn vượt mặc định 32 của v3turbo).
DEFAULT_GPU_CHUNK_BATCH = 64


def gpu_chunk_batch_for_vram_gb(vram_gb: float) -> int:
    """Số chunk/forward GPU an toàn cho lượng VRAM cho trước.

    Args:
        vram_gb: Tổng VRAM (GB).

    Returns:
        Số chunk mỗi forward (≥4).
    """
    for threshold_gb, chunks in _VRAM_CHUNK_BATCH_TABLE:
        if vram_gb >= threshold_gb:
            return chunks
    return 8


def suggest_gpu_chunk_batch() -> int:
    """Gợi ý số chunk/forward GPU theo VRAM (dò được), hoặc mặc định an toàn.

    Returns:
        Số chunk mỗi forward (≥4).
    """
    vram = detect_total_vram_gb()
    if vram is None:
        return DEFAULT_GPU_CHUNK_BATCH
    return gpu_chunk_batch_for_vram_gb(vram)


__all__ = [
    "DEFAULT_AUTO_BATCH_SIZE",
    "DEFAULT_GPU_CHUNK_BATCH",
    "batch_size_for_vram_gb",
    "detect_total_vram_gb",
    "gpu_chunk_batch_for_vram_gb",
    "suggest_auto_batch_size",
    "suggest_gpu_chunk_batch",
]
