"""Hàm tiện ích tiền xử lý ảnh trên VRAM (GPU) dùng CuPy và Custom CUDA Kernels.

BẢN CẬP NHẬT v3.28 (GPU UNIFICATION):
    Tách biệt toàn bộ logic xử lý ảnh GPU ra khỏi NVDEC Decoder.
    Tạo ra sự đối xứng hoàn hảo:
        - CPU -> apply_cpu_preprocessing_pipeline (numpy + cv2)
        - GPU -> apply_gpu_preprocessing_pipeline (cupy + CUDA C++)
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import warnings

# CuPy in cảnh báo "CUDA path could not be detected" ngay khi import nếu máy
# không có CUDA. Chặn cảnh báo đó và kiểm tra GPU CÓ THỰC SỰ khả dụng hay không
# → fallback CPU sạch sẽ, không làm nhiễu console của người dùng.
HAS_CUPY = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp  # type: ignore
    try:
        HAS_CUPY = bool(cp.cuda.is_available())
    except Exception:  # noqa: BLE001 — cupy raise nhiều loại lỗi runtime CUDA khác nhau
        HAS_CUPY = False
    if not HAS_CUPY:
        logger.info(
            "CuPy có nhưng không truy cập được GPU/CUDA — dùng CPU cho tiền xử lý "
            "ảnh OCR. Để bật GPU: cài CUDA Toolkit và đặt biến môi trường CUDA_PATH."
        )
except ImportError:
    HAS_CUPY = False


# ==============================================================================
# CUSTOM CUDA KERNELS (MÃ C++ CHẠY TRỰC TIẾP TRÊN HÀNG NGÀN LÕI GPU)
# ==============================================================================

@functools.lru_cache(maxsize=1)
def _get_preprocess_kernel() -> Any:
    if not HAS_CUPY: return None
    return cp.RawKernel(r'''
    extern "C" __global__
    void preprocess_rgb(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int width, int height, float contrast_factor, bool do_sharpen
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int row_stride = width * 3;
        int cx = x * 3;
        int center_idx = y * row_stride + cx;

        if (do_sharpen) {
            int top_idx   = max(y - 1, 0) * row_stride + cx;
            int bot_idx   = min(y + 1, height - 1) * row_stride + cx;
            int left_idx  = center_idx - (x > 0 ? 3 : 0);
            int right_idx = center_idx + (x < width - 1 ? 3 : 0);
            #pragma unroll
            for(int c = 0; c < 3; c++) {
                int val = 5 * src[center_idx + c] - src[top_idx + c] - src[bot_idx + c] - src[left_idx + c] - src[right_idx + c];
                float fval = (val - 127.5f) * contrast_factor + 127.5f;
                dst[center_idx + c] = (unsigned char)fminf(fmaxf(fval, 0.0f), 255.0f);
            }
        } else {
            #pragma unroll
            for(int c = 0; c < 3; c++) {
                float fval = (src[center_idx + c] - 127.5f) * contrast_factor + 127.5f;
                dst[center_idx + c] = (unsigned char)fminf(fmaxf(fval, 0.0f), 255.0f);
            }
        }
    }
    ''', 'preprocess_rgb')

@functools.lru_cache(maxsize=1)
def _get_resize_bilinear_kernel() -> Any:
    if not HAS_CUPY: return None
    return cp.RawKernel(r'''
    extern "C" __global__
    void resize_bilinear_rgb(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int src_w, int src_h, int dst_w, int dst_h
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= dst_w || y >= dst_h) return;

        float src_x = x * ((float)(src_w - 1) / (float)(max(1, dst_w - 1)));
        float src_y = y * ((float)(src_h - 1) / (float)(max(1, dst_h - 1)));
        int x1 = (int)src_x;
        int y1 = (int)src_y;
        int x2 = min(x1 + 1, src_w - 1);
        int y2 = min(y1 + 1, src_h - 1);

        float x_diff = src_x - x1;
        float y_diff = src_y - y1;
        float w1 = (1.0f - x_diff) * (1.0f - y_diff);
        float w2 = x_diff * (1.0f - y_diff);
        float w3 = (1.0f - x_diff) * y_diff;
        float w4 = x_diff * y_diff;

        int src_stride = src_w * 3;
        int row1 = y1 * src_stride, row2 = y2 * src_stride;
        int p1_idx = row1 + x1 * 3, p2_idx = row1 + x2 * 3;
        int p3_idx = row2 + x1 * 3, p4_idx = row2 + x2 * 3;
        int dst_idx = (y * dst_w + x) * 3;

        #pragma unroll
        for (int c = 0; c < 3; c++) {
            float val = src[p1_idx + c] * w1 + src[p2_idx + c] * w2 + src[p3_idx + c] * w3 + src[p4_idx + c] * w4;
            dst[dst_idx + c] = (unsigned char)(val + 0.5f);
        }
    }
    ''', 'resize_bilinear_rgb')


# ==============================================================================
# PIPELINE TIỀN XỬ LÝ TRÊN VRAM
# ==============================================================================

def apply_gpu_preprocessing_pipeline(
    gpu_array: Any,
    vram_buffers: dict[str, Any],
    upscale_small_text: bool,
    upscale_target_height_px: int,
    sharpen: bool,
    contrast_factor: float,
    add_border: bool,
    border_thickness_px: int
) -> Any:
    """Hợp nhất toàn bộ Pipeline xử lý ảnh trên GPU (Zero-Copy).

    Sử dụng Dict `vram_buffers` truyền từ ngoài vào để tái sử dụng vùng nhớ,
    triệt tiêu hoàn toàn độ trễ cấp phát bộ nhớ (Memory Allocation Overhead).
    """
    if not HAS_CUPY:
        return gpu_array

    # 1. Upscale Bilinear (Mượt hơn Nearest Neighbor)
    if upscale_small_text:
        h, w = gpu_array.shape[:2]
        target_h = upscale_target_height_px
        if h < max(target_h // 2, 32) and h < target_h:
            scale = min(3.0, target_h / h)
            target_h = int(h * scale)
            target_w = int(w * scale)

            key = f"resize_{target_w}_{target_h}"
            if key not in vram_buffers:
                vram_buffers[key] = cp.empty((target_h, target_w, 3), dtype=cp.uint8)
            resized_gpu = vram_buffers[key]

            block = (32, 32)
            grid = (math.ceil(target_w / 32), math.ceil(target_h / 32))
            _get_resize_bilinear_kernel()(grid, block, (gpu_array, resized_gpu, w, h, target_w, target_h))
            gpu_array = resized_gpu

    # 2. Contrast & Sharpen (Gộp chung vào 1 Kernel C++ để chạy trong 1 chu kỳ Clock)
    if contrast_factor != 1.0 or sharpen:
        h, w = gpu_array.shape[:2]
        key = f"enhanced_{w}_{h}"
        if key not in vram_buffers:
            vram_buffers[key] = cp.empty_like(gpu_array)
        enhanced_gpu = vram_buffers[key]

        block = (32, 32)
        grid = (math.ceil(w / 32), math.ceil(h / 32))
        _get_preprocess_kernel()(grid, block, (gpu_array, enhanced_gpu, w, h, cp.float32(contrast_factor), sharpen))
        gpu_array = enhanced_gpu

    # 3. Add Border (Padding)
    if add_border and border_thickness_px > 0:
        b = border_thickness_px
        # cp.pad tự động tạo bộ nhớ mới nhưng thao tác này rất rẻ trên VRAM
        gpu_array = cp.pad(gpu_array, ((b, b), (b, b), (0, 0)), mode='constant', constant_values=0)

    return gpu_array

__all__ = ["apply_gpu_preprocessing_pipeline"]
