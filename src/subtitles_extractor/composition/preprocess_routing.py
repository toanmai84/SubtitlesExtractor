"""[v3.23.723] Một nguồn duy nhất quyết định tiền xử lý ảnh OCR chạy ở đâu: VRAM (CUDA) hay CPU.

Trước bản này trang Trích xuất và Chỉnh sửa (ReOCR) mỗi nơi tự chép tay phép tách cờ, với điều kiện "có NVDEC" là
``backend == "pynvvideocodec" and có cupy`` — KHÁC điều kiện container dùng để chọn bộ giải mã (cần cả PyNvVideoCodec).
Máy có CuPy mà thiếu PyNvVideoCodec: container giải mã bằng PyAV nhưng cờ tiền xử lý lại giao cho VRAM → PyAV bỏ qua,
bước CPU bị tắt → tiền xử lý mất sạch, không báo gì. Nay cả hai dùng :func:`nvdec_sampler_selected` (đúng điều kiện
của ``ApplicationContainer._create_frame_sampler``) và :func:`route_preprocessing`.

Quy tắc của chủ dự án: nhánh GPU (giải mã NVDEC) thì MỌI bước tiền xử lý chạy CUDA trên VRAM, chỉ chép về RAM một lần.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from subtitles_extractor.domain.ports.ocr_engine_port import PreprocessConfig


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cupy_usable() -> bool:
    """[v3.23.724] CuPy NẠP ĐƯỢC và thấy GPU (không chỉ "có gói"): bản 3.23.723 có gói mà ``import cupy`` ném lỗi trong
    bản đóng gói → vẫn chọn NVDEC rồi sập."""
    if not _module_available("cupy"):
        return False
    try:
        from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters
    except Exception:  # noqa: BLE001 — bất kỳ lỗi nạp nào cũng nghĩa là không dùng được
        return False
    return bool(gpu_image_filters.HAS_CUPY)


def nvdec_sampler_selected(snapshot: Any) -> bool:
    """Container có chọn bộ giải mã NVDEC (PyNvVideoCodec + CuPy dùng được) với cài đặt này không."""
    backend = str(getattr(snapshot.hardware, "frame_decoder_backend", "") or "")
    return backend == "pynvvideocodec" and _module_available("PyNvVideoCodec") and cupy_usable()


def route_preprocessing(
    pre: Any, *, on_gpu: bool, overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], PreprocessConfig]:
    """Tách cài đặt tiền xử lý thành (trường ``vram_*`` của ``FrameSamplingConfig``, ``PreprocessConfig`` cho CPU).

    ``on_gpu``: giải mã NVDEC → mọi bước ở VRAM, bước CPU tắt hết; ngược lại mọi bước ở CPU (bộ OCR), VRAM tắt.
    ``overrides``: tinh chỉnh riêng của ReOCR (khoá ``upscale``, ``upscale_h``, ``border`` = độ dày px, ``sharpen``,
    ``contrast``, ``contrast_factor``, ``clahe``, ``clahe_clip``, ``clahe_tile``).
    """
    tweaks = overrides or {}
    upscale = bool(tweaks.get("upscale", pre.upscale_small_text))
    target = int(tweaks.get("upscale_h", pre.upscale_target_height_px))
    if "border" in tweaks:
        thickness = int(tweaks["border"])
        border = thickness > 0
    else:
        thickness = int(pre.border_thickness_px)
        border = bool(pre.add_white_border)
    sharpen = bool(tweaks.get("sharpen", pre.apply_sharpen))
    contrast = bool(tweaks.get("contrast", pre.apply_contrast_boost))
    factor = float(tweaks.get("contrast_factor", pre.contrast_factor))
    clahe = bool(tweaks.get("clahe", pre.apply_clahe))
    clip = float(tweaks.get("clahe_clip", pre.clahe_clip_limit))
    tile = int(tweaks.get("clahe_tile", pre.clahe_tile_size))

    vram = {
        "vram_upscale_small_text": upscale and on_gpu,
        "vram_upscale_target_height_px": target,
        "vram_add_border": border and on_gpu,
        "vram_border_thickness_px": thickness,
        "vram_sharpen": sharpen and on_gpu,
        "vram_contrast_factor": factor if (contrast and on_gpu) else 1.0,
        "vram_apply_clahe": clahe and on_gpu,
        "vram_clahe_clip_limit": clip,
        "vram_clahe_tile_size": tile,
    }
    cpu = PreprocessConfig(
        upscale_small_text=upscale and not on_gpu, upscale_target_height_px=target,
        add_white_border=border and not on_gpu, border_thickness_px=thickness,
        apply_sharpen=sharpen and not on_gpu,
        apply_contrast_boost=contrast and not on_gpu, contrast_factor=factor,
        apply_clahe=clahe and not on_gpu, clahe_clip_limit=clip, clahe_tile_size=tile,
    )
    return vram, cpu


__all__ = ["nvdec_sampler_selected", "route_preprocessing"]
