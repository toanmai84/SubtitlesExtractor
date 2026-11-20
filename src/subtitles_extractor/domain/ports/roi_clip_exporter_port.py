"""[v3.23.728] Cổng xuất VIDEO VÙNG ROI đi kèm dữ liệu OCR thô (``.seraw.json.gz``) để gỡ lỗi.

Thay cho xuất ảnh JPG từng khung (hai ảnh mỗi khung có chữ — phim nối 86 tập ≈ 240 nghìn ảnh, nhiều GB): một tệp video
cắt đúng ROI, đủ mọi khung, đủ rõ để đọc lại chữ / đo lại điểm sáng (đo tập 24: CRF 26 HEVC 1,4 MB / 64 s, PSNR độ
chói 39,5 dB so với khung gốc).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.value_objects.roi import Roi


@dataclass(frozen=True, slots=True)
class RoiClipResult:
    """Kết quả xuất video ROI.

    Attributes:
        clip_path: Tệp video đã ghi.
        sidecar_path: Tệp mô tả (``.roi.json``): vùng cắt thật, mốc đầu phim nguồn, bộ mã hoá.
        crop_xywh: Vùng cắt THẬT trong khung gốc (đã làm chẵn cho bộ mã hoá — có thể rộng hơn ROI 1 px).
        encoder: Bộ mã hoá đã dùng (vd ``hevc_nvenc``, ``libx265``).
        pipeline: Cách chạy: ``gpu_zero_copy`` (NVDEC cắt trên VRAM → NVENC), ``gpu_hybrid`` hoặc ``cpu``.
        elapsed_sec: Thời gian xuất.
    """

    clip_path: Path
    sidecar_path: Path
    crop_xywh: tuple[int, int, int, int]
    encoder: str
    pipeline: str
    elapsed_sec: float


@runtime_checkable
class RoiClipExporterPort(Protocol):
    """Cắt vùng ROI của phim thành một video nhỏ (giữ đủ mọi khung, mốc thời gian quy đổi được)."""

    def export(
        self, video_path: Path, roi: Roi | None, output_path: Path, *, prefer_gpu: bool,
        progress: Callable[[float], bool] | None = None, quality: int | None = None,
    ) -> RoiClipResult | None:
        """Ghi ``output_path`` (+ tệp mô tả cạnh nó). ``progress(tỉ lệ 0..1)`` trả ``False`` để huỷ.

        ``quality``: [v3.23.738] thang CRF x264 (Cài đặt "Chất lượng video vùng ROI"); ``None`` = mặc định 24.

        Trả ``None`` nếu không xuất được (thiếu ffmpeg, mọi cách mã hoá đều lỗi, bị huỷ).
        """
        ...


__all__ = ["RoiClipExporterPort", "RoiClipResult"]
