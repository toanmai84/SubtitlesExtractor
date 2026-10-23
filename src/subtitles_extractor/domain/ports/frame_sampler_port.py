"""Hợp đồng lấy mẫu khung hình từ video."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.value_objects.roi import Roi

if TYPE_CHECKING:
    import numpy as np

@dataclass(frozen=True, slots=True)
class SampledFrame:
    frame_index: int
    timestamp_sec: float
    image_rgb: np.ndarray
    is_duplicate: bool = False
    is_error: bool = False
    #: [v3.23.644] Khung KHÔNG có chữ (bộ lọc chữ trắng): kết quả OCR là RỖNG, không cần chạy OCR và
    #: KHÔNG được sao chép kết quả khung trước như khung trùng.
    is_blank: bool = False

@dataclass(frozen=True, slots=True)
class FrameSamplingConfig:
    sample_step_sec: float = 0.25
    phash_distance_threshold: int = 5
    pixel_diff_threshold: float = 0.01
    skip_intro_sec: float = 0.0
    skip_outro_sec: float = 0.0

    apply_median_blend: bool = False
    median_blend_frames: int = 3

    #[CẢI TIẾN]: Ủy quyền toàn bộ Tiền xử lý sang VRAM (CuPy)
    vram_upscale_small_text: bool = False
    vram_upscale_target_height_px: int = 96
    vram_add_border: bool = False
    vram_border_thickness_px: int = 8
    vram_sharpen: bool = False
    vram_contrast_factor: float = 1.0

    #: [v3.23.644] Lọc khung theo CHỮ TRẮNG: chỉ OCR khi các điểm ảnh sáng (chữ phụ đề trắng) trong ROI
    #: thật sự đổi; khung không có chữ bỏ qua. Mặc định TẮT (giả định phụ đề trắng/sáng).
    white_text_gate: bool = False
    #: Điểm ảnh coi là "chữ" khi độ chói (0-255) vượt ngưỡng này. 200 là mặc định an toàn: đo trên tập 1 phim
    #: mẫu, ngưỡng 220 làm rơi một phụ đề xám nhạt 2 chữ, ngưỡng 200 giữ đủ 95/95 câu mà vẫn giảm số khung OCR
    #: từ 72% xuống ~31%.
    white_text_luma: int = 200

#: [v3.23.615] Dung sai so mốc lấy mẫu (giây): lớn hơn sai số dấu phẩy động của
#: ``pts * time_base`` nhưng nhỏ hơn rất nhiều khoảng cách hai khung thật (≥ 4ms ở
#: 240fps).
SAMPLE_TIME_TOLERANCE_SEC = 1e-4


def is_sample_due(timestamp_sec: float, next_target_sec: float) -> bool:
    """[v3.23.615] Khung ở ``timestamp_sec`` đã tới lượt lấy mẫu chưa (hàm thuần).

    Bộ lấy mẫu đặt mốc kế tiếp ``= mốc khung vừa lấy + bước``. Với bước đúng bằng một
    khung (0.04s ở 25fps), ``pts * time_base`` của khung kế tiếp có thể nhỏ hơn mốc đó
    một phần tỉ giây do làm tròn số thực, và phép so ``<`` bỏ oan khung đó. Đo trên video
    thật 14/10 (HEVC 25fps, 4421 khung, time_base 1/12800): bộ lấy mẫu NVDEC chỉ lấy 3700
    khung — bỏ 721 khung (16%), đều đặn 1 trong 2-3 khung từ giây 64 trở đi. Đo bằng điểm
    ảnh: 14/15 câu bắt đầu trễ 40ms và 20/22 câu kết thúc sớm thêm 40ms đều rơi đúng khung
    bị bỏ.

    Args:
        timestamp_sec: Mốc của khung đang xét.
        next_target_sec: Mốc sớm nhất được lấy mẫu tiếp.

    Returns:
        True nếu khung đã tới (hoặc chỉ hụt trong dung sai làm tròn).
    """
    return timestamp_sec + SAMPLE_TIME_TOLERANCE_SEC >= next_target_sec


@runtime_checkable
class FrameSamplerPort(Protocol):
    def iter_frames(
        self, metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig,
    ) -> Iterator[SampledFrame]: ...

__all__ = [
    "SAMPLE_TIME_TOLERANCE_SEC",
    "FrameSamplerPort",
    "FrameSamplingConfig",
    "SampledFrame",
    "is_sample_due",
]
