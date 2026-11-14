"""Adapter FrameSamplerPort dùng PyNvVideoCodec — GPU-First / Zero-Copy Mode.

BẢN CẬP NHẬT v3.28 (GPU UNIFICATION):
    Đã tách bạch hoàn toàn Logic xử lý ảnh (CUDA Kernels) sang `gpu_image_filters.py`.
    Tuân thủ tuyệt đối quy tắc S.O.L.I.D và Clean Architecture.
"""
from __future__ import annotations

import contextlib
import ctypes
import functools
import gc
import logging
import math
import os
import site
import struct
from collections.abc import Iterator
from typing import Any

import numpy as np

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import VideoDecodeError, VideoNotFoundError
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    SampledFrame,
    is_sample_due,
)
from subtitles_extractor.domain.value_objects.boundary_track import BoundaryTrack
from subtitles_extractor.domain.value_objects.roi import Roi

# [V3.28] Import Pipeline GPU mới tạo
from subtitles_extractor.infrastructure.ocr.preprocessing.gpu_image_filters import (
    apply_gpu_preprocessing_pipeline,
    color_matrix_for,
    is_full_range,
    nv12_roi_to_rgb,
)
from subtitles_extractor.infrastructure.video.decoders.boundary_probe import (
    BoundaryProber,
    expand_studio_luma,
    probe_av_frame,
    probing_enabled,
)
from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import (
    _KEYFRAME_ONLY_MIN_STEP_SEC,
    _next_target,
)
from subtitles_extractor.infrastructure.video.frame_dedupe import (
    FrameDeduper,
    FrameVerdict,
)

logger = logging.getLogger(__name__)

# [v3.23.724] Dùng chung bộ nạp CuPy an toàn của ``gpu_image_filters`` (bắt MỌI lỗi khởi tạo — trước đây chỉ bắt
# ImportError nên FileNotFoundError của CuPy trong bản đóng gói làm sập lượt trích xuất).
from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters as _gpu  # noqa: E402

cp = _gpu.cp
HAS_CUPY = bool(_gpu.HAS_CUPY)

_CUDA_D2H_FLAG: int = 2
EMPTY_ARRAY = np.empty(0, dtype=np.uint8)


def _get_av_errors() -> tuple[type[Exception], ...]:
    errors: list[type[Exception]] = [OSError, ValueError, RuntimeError]
    try:
        import av
        if hasattr(av, "AVError"):
            errors.append(av.AVError)
        if hasattr(av, "error") and hasattr(av.error, "FFmpegError"):
            errors.append(av.error.FFmpegError)
        if hasattr(av, "error") and hasattr(av.error, "InvalidDataError"):
            errors.append(av.error.InvalidDataError)
    except ImportError:
        pass
    return tuple(errors)

# ==============================================================================
# QUẢN LÝ KERNEL CHUYỂN MÀU & TRỘN ẢNH BẢO VỆ CHỐNG RUNG (MEDIAN BLEND)
# ==============================================================================

# [v3.23.723] Nhân CUDA ``nv12_to_rgb`` (đổi CẢ khung bằng công thức dải đầy BT.601) đã bỏ: thay bằng
# ``gpu_image_filters.nv12_roi_to_rgb`` — chỉ vùng ROI, trùng khít PyAV/swscale.

@functools.lru_cache(maxsize=1)
def _get_median3_kernel() -> Any:
    if not HAS_CUPY: return None
    return cp.RawKernel(r'''
    extern "C" __global__
    void median3_rgb(
        const unsigned char* __restrict__ f1,
        const unsigned char* __restrict__ f2,
        const unsigned char* __restrict__ f3,
        unsigned char* __restrict__ out,
        int width, int height
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int base_idx = (y * width + x) * 3;
        #pragma unroll
        for(int c = 0; c < 3; c++) {
            int idx = base_idx + c;
            unsigned char a = f1[idx];
            unsigned char b = f2[idx];
            unsigned char c_val = f3[idx];
            out[idx] = max(min(a, b), min(max(a, b), c_val));
        }
    }
    ''', 'median3_rgb')

@functools.lru_cache(maxsize=1)
def _get_shift_rgb_kernel() -> Any:
    if not HAS_CUPY: return None
    return cp.RawKernel(r'''
    extern "C" __global__
    void shift_rgb(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int width, int height, int dx, int dy
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int src_x = x - dx;
        int src_y = y - dy;
        int row_stride = width * 3;
        int dst_idx = y * row_stride + x * 3;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            int src_idx = src_y * row_stride + src_x * 3;
            #pragma unroll
            for(int c = 0; c < 3; c++) dst[dst_idx + c] = src[src_idx + c];
        } else {
            #pragma unroll
            for(int c = 0; c < 3; c++) dst[dst_idx + c] = 0;
        }
    }
    ''', 'shift_rgb')


@functools.lru_cache(maxsize=8)
def _get_fft_caches(h: int, w: int) -> tuple[Any, Any]:
    weights = cp.array([0.299, 0.587, 0.114], dtype=cp.float32)
    wy = cp.hanning(h)[:, None]
    wx = cp.hanning(w)[None, :]
    hann = (wy * wx).astype(cp.float32)
    return weights, hann

def _align_and_shift_cupy(src_img: Any, target_img: Any, vram_buffers: dict[str, Any]) -> Any:
    try:
        h, w = src_img.shape[:2]
        gray_weights, hann_window = _get_fft_caches(h, w)

        src_gray = cp.dot(src_img.astype(cp.float32), gray_weights)
        tgt_gray = cp.dot(target_img.astype(cp.float32), gray_weights)

        src_gray = (src_gray - cp.mean(src_gray)) * hann_window
        tgt_gray = (tgt_gray - cp.mean(tgt_gray)) * hann_window

        G_src = cp.fft.fft2(src_gray)
        G_tgt = cp.fft.fft2(tgt_gray)

        R = G_src * cp.conj(G_tgt)
        R /= (cp.abs(R) + 1e-8)

        r = cp.real(cp.fft.ifft2(R))
        idx = cp.argmax(r)
        dy, dx = cp.unravel_index(idx, r.shape)

        if dy > h // 2: dy -= h
        if dx > w // 2: dx -= w
        dy, dx = int(dy), int(dx)

        if dx == 0 and dy == 0:
            return src_img

        buffer_key = f"shifted_{w}_{h}"
        if buffer_key not in vram_buffers:
            vram_buffers[buffer_key] = cp.zeros_like(src_img)

        shifted = vram_buffers[buffer_key]
        block, grid = (32, 32), (math.ceil(w / 32), math.ceil(h / 32))
        _get_shift_rgb_kernel()(grid, block, (src_img, shifted, w, h, dx, dy))

        return shifted.copy()
    except (RuntimeError, ValueError, AttributeError, MemoryError) as exc:
        logger.debug("Bỏ qua Optical Alignment do lỗi: %s.", exc)
        return src_img

def _wrap_cuda_array(obj: Any) -> Any:
    if hasattr(obj, '__cuda_array_interface__'):
        cuda_interface = obj.__cuda_array_interface__
        pointer = cuda_interface['data'][0]
        shape = tuple(cuda_interface['shape'])
        type_str = cuda_interface.get('typestr', '|u1')
        memory_obj = cp.cuda.UnownedMemory(pointer, 0, obj)
        memory_pointer = cp.cuda.MemoryPointer(memory_obj, 0)
        return cp.ndarray(shape, dtype=np.dtype(type_str), memptr=memory_pointer)
    if hasattr(obj, '__dlpack__'):
        return cp.from_dlpack(obj)
    raise AttributeError(f"Đối tượng {type(obj)} không có __cuda_array_interface__ hoặc __dlpack__")


# ==============================================================================
# HÀM SETUP & CODEC PARSERS
# ==============================================================================

_dll_initialized: bool = False
def _setup_dll() -> None:
    global _dll_initialized
    if _dll_initialized or os.name != "nt":
        _dll_initialized = True
        return
    _dll_initialized = True
    for site_func in (lambda: [site.getusersitepackages()], lambda: site.getsitepackages()):
        try:
            for site_path in site_func():
                for sub_dir in ("nvidia/cuda_runtime/bin", "PyNvVideoCodec"):
                    full_path = os.path.join(site_path, sub_dir)
                    if os.path.isdir(full_path):
                        with contextlib.suppress(OSError):
                            os.add_dll_directory(full_path)  # type: ignore[attr-defined]
        except (OSError, AttributeError) as exc:
            logger.debug("Bỏ qua lỗi tìm DLL site-packages: %s.", exc)

_cudart_lib: Any | None = None
_memcpy_func: Any | None = None
def _get_cudart() -> Any | None:
    global _cudart_lib, _memcpy_func
    if _cudart_lib is not None: return _cudart_lib
    candidate_dlls = ("cudart64_12.dll", "cudart64_110.dll", "cudart64_120.dll", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so")
    for dll_name in candidate_dlls:
        try:
            lib = ctypes.CDLL(dll_name)
            lib.cudaMemcpy.restype = ctypes.c_int
            lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            _cudart_lib = lib; _memcpy_func = lib.cudaMemcpy
            return _cudart_lib
        except OSError: pass
    return None

def _is_annexb(data: bytes) -> bool:
    return data[:4] == b'\x00\x00\x00\x01' or data[:3] == b'\x00\x00\x01'

def _h264_extract_extradata(extra_data: bytes) -> tuple[bytes, int]:
    if not extra_data or len(extra_data) < 7: return b'', 4
    nal_length_size = ((extra_data[4] & 0x03) + 1)
    res = bytearray()
    pos = 6
    for _ in range(extra_data[5] & 0x1F):
        if pos + 2 > len(extra_data): break
        nal_size = struct.unpack('>H', extra_data[pos:pos+2])[0]
        pos += 2
        res.extend(b'\x00\x00\x00\x01')
        res.extend(extra_data[pos:pos+nal_size])
        pos += nal_size
    if pos < len(extra_data):
        num_pps = extra_data[pos]
        pos += 1
        for _ in range(num_pps):
            if pos + 2 > len(extra_data): break
            nal_size = struct.unpack('>H', extra_data[pos:pos+2])[0]
            pos += 2
            res.extend(b'\x00\x00\x00\x01')
            res.extend(extra_data[pos:pos+nal_size])
            pos += nal_size
    return bytes(res), nal_length_size

def _hevc_extract_extradata(extra_data: bytes) -> tuple[bytes, int]:
    if not extra_data or len(extra_data) < 23: return b'', 4
    nal_length_size = ((extra_data[21] & 0x03) + 1)
    pos, num_arrays, res = 23, extra_data[22], bytearray()
    for _ in range(num_arrays):
        if pos + 3 > len(extra_data): break
        pos += 1
        num_nalus = struct.unpack('>H', extra_data[pos:pos+2])[0]
        pos += 2
        for _ in range(num_nalus):
            if pos + 2 > len(extra_data): break
            nal_size = struct.unpack('>H', extra_data[pos:pos+2])[0]
            pos += 2
            res.extend(b'\x00\x00\x00\x01')
            res.extend(extra_data[pos:pos+nal_size])
            pos += nal_size
    return bytes(res), nal_length_size

def _convert_packet_to_annexb(data: bytes, nal_length_size: int) -> bytes:
    res, pos = bytearray(), 0
    while pos + nal_length_size <= len(data):
        nal_size = int.from_bytes(data[pos:pos+nal_length_size], 'big')
        pos += nal_length_size
        if nal_size == 0 or pos + nal_size > len(data): break
        res.extend(b'\x00\x00\x00\x01')
        res.extend(data[pos:pos+nal_size])
        pos += nal_size
    return bytes(res)

def _create_packet_data(data: bytes, av_packet: Any) -> tuple[Any, np.ndarray]:
    from PyNvVideoCodec import PacketData
    packet_data = PacketData()
    numpy_array = np.frombuffer(data, dtype=np.uint8)
    packet_data.bsl_data = numpy_array.ctypes.data
    packet_data.bsl = len(data)
    if getattr(av_packet, 'pts', None) is not None: packet_data.pts = int(av_packet.pts)
    if getattr(av_packet, 'dts', None) is not None: packet_data.dts = int(av_packet.dts)
    packet_data.key = bool(getattr(av_packet, 'is_keyframe', False))
    return packet_data, numpy_array

def nvdec_frame_time(pts_val: Any, time_base: float, fallback_sec: float, stream_has_pts: bool) -> float:
    """[v3.23.655] Bug #16: mốc giờ của khung NVDEC trả về.

    Bản cũ ``float(pts) * tb if pts else <mốc gói>``: PTS = 0 (khung ĐẦU phim) bị coi là "không có" → lấy mốc của gói
    vừa nạp (thứ tự GIẢI MÃ, lệch với khung B). Chỉ dùng mốc dự phòng khi thật sự không có PTS.
    """
    if pts_val is None or (not stream_has_pts and not pts_val):
        return fallback_sec
    return float(pts_val) * time_base


def error_frame_time(packet_sec: float, last_sec: float, start_sec: float) -> float | None:
    """[v3.23.655] Bug #16: mốc của "khung lỗi" khi NVDEC giải mã một gói thất bại; ``None`` = không phát khung lỗi.

    Bản cũ gán mốc của GÓI (thứ tự giải mã: có thể -1 khi gói không có PTS, hoặc lùi/tiến so với khung đã phát) rồi dời
    mốc lấy mẫu theo đó → khung rỗng chen sai chỗ (cắt đôi câu phụ đề) và lịch lấy mẫu nhảy. Nay mốc không bao giờ lùi.
    """
    if packet_sec < 0 or packet_sec < start_sec:
        return None
    return max(packet_sec, last_sec)


@functools.lru_cache(maxsize=1)
def _get_nv_codec_map() -> dict[str, Any]:
    try:
        from PyNvVideoCodec import cudaVideoCodec
        return {
            'h264': cudaVideoCodec.H264, 'hevc': cudaVideoCodec.HEVC,
            'vp9': cudaVideoCodec.VP9, 'av1': cudaVideoCodec.AV1,
            'mpeg2video': cudaVideoCodec.MPEG2, 'mpeg4': cudaVideoCodec.MPEG4,
            'vp8': cudaVideoCodec.VP8,
        }
    except ImportError: return {}



# ==============================================================================
# [v3.23.723] XỬ LÝ MỘT KHUNG: CHUNG CHO ĐƯỜNG NVDEC, XẢ EOS, VIDEO 10-BIT/HDR VÀ KHI THIẾU CUPY
# ==============================================================================


def _split_nv12(buffer: Any, video_height: int) -> tuple[Any, Any]:
    """Tách bộ đệm NV12 (``H·3/2`` hàng × pitch, hoặc phẳng 1 chiều) thành mặt phẳng Y và UV xen kẽ."""
    chroma_rows = (video_height + 1) // 2                # [v3.23.723] chiều cao lẻ: thêm một hàng sắc độ
    if buffer.ndim == 1:
        pitch = buffer.size // (video_height + chroma_rows)
        buffer = buffer.reshape((video_height + chroma_rows, pitch))
    elif buffer.ndim == 3:
        buffer = buffer.squeeze()
    return buffer[:video_height], buffer[video_height:video_height + chroma_rows]


def _crop_nv12(y_plane: Any, uv_plane: Any, video_width: int, video_height: int) -> tuple[Any, Any]:
    """[v3.23.723] Cắt bỏ phần đệm pitch. Bề rộng / chiều cao LẺ: UV giữ ``(W+1)//2`` cặp và ``(H+1)//2`` hàng — trước đây
    cắt ``H//2`` hàng × ``W`` cột → ROI chạm mép phải / dưới đọc sắc độ ngoài mảng (NumPy: IndexError → cả lượt lùi
    PyAV; CuPy: chỉ số âm thầm quấn về cột / hàng 0)."""
    return y_plane[:video_height, :video_width], uv_plane[: (video_height + 1) // 2, : ((video_width + 1) // 2) * 2]


def vram_preprocessing_enabled(config: FrameSamplingConfig) -> bool:
    """Cấu hình lấy mẫu có giao bước tiền xử lý nào cho VRAM không."""
    return bool(
        config.vram_upscale_small_text or config.vram_add_border or config.vram_sharpen
        or abs(float(config.vram_contrast_factor) - 1.0) >= 1e-4 or getattr(config, "vram_apply_clahe", False)
    )


class _FrameProcessor:
    """Quyết định lấy mẫu + chuyển màu + khử trùng + tiền xử lý cho TỪNG khung giải mã — đúng như ``PyAvFrameSampler``.

    Lỗi cũ của nhánh NVDEC (đã đo trên khung thật phim 86 tập):

    * Đổi màu cả khung bằng công thức dải đầy → ảnh lệch PyAV TB 7–13 mức xám, "lõi chữ trắng" ≥ 240 ít đi 6–20 lần →
      bộ khử trùng (``white_text_gate``) chọn khung OCR khác hẳn nhánh CPU.
    * Đổi màu MỌI khung giải mã, kể cả khung chưa tới lượt lấy mẫu; đổi cả khung 720×1280 rồi mới cắt ROI.
    * Bộ khử trùng xét ảnh ĐÃ tiền xử lý (phóng / làm nét / tương phản) trong khi nhánh PyAV xét ảnh gốc → ngưỡng chói
      200 / 240 cho kết quả khác nhau giữa hai nhánh; khung trùng cũng bị tiền xử lý vô ích.
    * Mốc lấy mẫu kế tiếp không theo luật "chỉ khung khoá" của PyAV khi bước thưa.

    Nay: chỉ khung tới lượt mới đổi màu, chỉ vùng ROI; khử trùng trên ảnh gốc; chỉ khung cần OCR mới tiền xử lý trên VRAM
    rồi chép về RAM đúng một lần. ``xp`` = CuPy (GPU) hoặc NumPy (máy thiếu CuPy — cùng công thức).
    """

    def __init__(
        self, *, config: FrameSamplingConfig, roi: Roi | None, video_width: int, video_height: int, matrix: str,
        full_range: bool, prober: BoundaryProber | None, xp: Any, start_sec: float,
    ) -> None:
        self._config = config
        self.xp = xp
        self._matrix = matrix
        self._full_range = full_range
        self._prober = prober
        box = roi.clip_to(video_width, video_height) if roi is not None else None
        self._box = (box.x, box.y, box.x2, box.y2) if box is not None else (0, 0, video_width, video_height)
        self._roi = roi
        self.step_sec = max(0.001, config.sample_step_sec)
        self.keyframe_only = self.step_sec >= _KEYFRAME_ONLY_MIN_STEP_SEC
        self.start_sec = start_sec
        self.next_target_sec = start_sec
        self.last_emitted_sec = start_sec
        self.frame_index = 0
        self._deduper = FrameDeduper(config)
        self._preprocess = vram_preprocessing_enabled(config)
        self._buffers: dict[str, Any] = {}
        self._blend = max(3, min(7, config.median_blend_frames)) if config.apply_median_blend else 1
        self._history: Any = None
        self._history_count = 0

    # ── khung NV12 (NVDEC) ──────────────────────────────────────────────────
    def handle_nv12(self, y_plane: Any, uv_plane: Any, timestamp_sec: float) -> SampledFrame | None:
        due = is_sample_due(timestamp_sec, self.next_target_sec)
        blending = self._blend > 1
        if self._prober is not None:
            self._probe_luma(y_plane, timestamp_sec, due and (not blending or self._history_count + 1 >= self._blend))
        if not due and not blending:
            return None                                   # không đổi màu khung chưa tới lượt
        x0, y0, x1, y1 = self._box
        raw = nv12_roi_to_rgb(y_plane, uv_plane, x0, y0, x1, y1, matrix=self._matrix, full_range=self._full_range,
                              xp=self.xp)
        if blending:
            raw = self._blend_history(raw, due)
            if raw is None:
                return None
        return self._finish(raw, timestamp_sec)

    # ── khung RGB đã có trên RAM (video 10-bit / HDR giải mã bằng PyAV) ─────
    def handle_rgb(self, rgb_full: np.ndarray, timestamp_sec: float) -> SampledFrame | None:
        if not is_sample_due(timestamp_sec, self.next_target_sec):
            return None
        if self._roi is not None:
            box = self._roi.clip_to(rgb_full.shape[1], rgb_full.shape[0])
            rgb_full = rgb_full[box.y:box.y2, box.x:box.x2]
        raw = self.xp.asarray(np.ascontiguousarray(rgb_full))
        return self._finish(raw, timestamp_sec)

    def error_frame(self, packet_sec: float, end_sec: float) -> SampledFrame | None:
        """Khung lỗi khi NVDEC giải mã một gói thất bại (mốc không lùi — v3.23.655)."""
        error_sec = error_frame_time(packet_sec, self.last_emitted_sec, self.start_sec)
        if error_sec is None or error_sec > end_sec:
            return None
        frame = SampledFrame(frame_index=self.frame_index, timestamp_sec=error_sec, image_rgb=EMPTY_ARRAY, is_error=True)
        self.frame_index += 1
        self.last_emitted_sec = error_sec
        self.next_target_sec = max(self.next_target_sec, error_sec + self.step_sec)
        return frame

    # ── nội bộ ─────────────────────────────────────────────────────────────
    def _finish(self, raw: Any, timestamp_sec: float) -> SampledFrame:
        xp = self.xp
        raw_host = raw if xp is np else xp.asnumpy(raw)
        verdict = self._deduper.verdict(raw_host, timestamp_sec)
        needs_ocr = verdict is FrameVerdict.OCR
        image = EMPTY_ARRAY
        if needs_ocr:
            image = raw_host
            if self._preprocess:
                config = self._config
                prepared = apply_gpu_preprocessing_pipeline(
                    raw, self._buffers,
                    upscale_small_text=config.vram_upscale_small_text,
                    upscale_target_height_px=config.vram_upscale_target_height_px,
                    sharpen=config.vram_sharpen,
                    contrast_factor=config.vram_contrast_factor,
                    add_border=config.vram_add_border,
                    border_thickness_px=config.vram_border_thickness_px,
                    apply_clahe=bool(getattr(config, "vram_apply_clahe", False)),
                    clahe_clip_limit=float(getattr(config, "vram_clahe_clip_limit", 3.0)),
                    clahe_tile_size=int(getattr(config, "vram_clahe_tile_size", 8)),
                    xp=xp,
                )
                image = prepared if xp is np else xp.asnumpy(prepared)      # chép về RAM đúng một lần
        frame = SampledFrame(
            frame_index=self.frame_index,
            timestamp_sec=timestamp_sec,
            image_rgb=np.ascontiguousarray(image) if needs_ocr else EMPTY_ARRAY,
            is_duplicate=verdict is FrameVerdict.DUPLICATE,
            is_blank=verdict is FrameVerdict.BLANK,
            continues_text=needs_ocr and self._deduper.continues_text,
        )
        if needs_ocr:
            self._deduper.accept(raw_host, timestamp_sec)
        self.frame_index += 1
        self.last_emitted_sec = max(self.last_emitted_sec, timestamp_sec)
        self.next_target_sec = _next_target(self.next_target_sec, timestamp_sec, self.step_sec, self.keyframe_only)
        return frame

    def _probe_luma(self, y_plane: Any, timestamp_sec: float, sampled: bool) -> None:
        prober = self._prober
        if prober is None:
            return
        try:                               # lỗi ở đây chỉ bỏ đầu dò của khung này, KHÔNG làm mất khung OCR
            x0, y0, x1, y1 = self._box
            luma = y_plane[y0:y1, x0:x1]
            luma = luma if self._full_range else expand_studio_luma(luma)
            host = luma if self.xp is np else self.xp.asnumpy(luma)
            (prober.sampled if sampled else prober.skipped)(np.ascontiguousarray(host), timestamp_sec)
        except (RuntimeError, ValueError, AttributeError, IndexError, TypeError) as exc:
            logger.debug("Bỏ đầu dò ranh giới (NVDEC): %s", exc)

    def _blend_history(self, raw: Any, due: bool) -> Any | None:
        """Trộn trung vị nhiều khung liền nhau (có căn chỉnh dịch chuyển) — chỉ trên GPU như trước."""
        xp = self.xp
        if self._history is None:
            self._history = xp.empty((self._blend, *raw.shape), dtype=raw.dtype)
        if self._history.shape[1:] != raw.shape:
            self._history = xp.empty((self._blend, *raw.shape), dtype=raw.dtype)
            self._history_count = 0
        self._history[self._history_count % self._blend] = raw
        self._history_count += 1
        if not due or self._history_count < self._blend:
            return None
        count = self._history_count
        target_idx = (count - self._blend // 2 - 1) % self._blend
        target = self._history[target_idx]
        if xp is np:
            return xp.median(self._history, axis=0).astype(np.uint8)
        aligned = []
        for i in range(self._blend):
            idx = (count - self._blend + i) % self._blend
            frame_i = self._history[idx]
            aligned.append(frame_i if idx == target_idx else _align_and_shift_cupy(frame_i, target, self._buffers))
        if self._blend == 3:
            key = f"blended_{target.shape[1]}_{target.shape[0]}"
            if key not in self._buffers:
                self._buffers[key] = cp.empty_like(target)
            blended = self._buffers[key]
            h, w = target.shape[:2]
            block, grid = (32, 32), (math.ceil(w / 32), math.ceil(h / 32))
            _get_median3_kernel()(grid, block, (aligned[0], aligned[1], aligned[2], blended, w, h))
            return blended.copy()
        return cp.median(cp.stack(aligned, axis=0), axis=0).astype(cp.uint8)


# ==============================================================================
# CLASS CHÍNH
# ==============================================================================

class PyNvVideoCodecFrameSampler:
    def __init__(self, *, gpu_id: int = 0) -> None:
        self._gpu_id = gpu_id
        if not HAS_CUPY:
            _setup_dll()
            _get_cudart()

    #: [v3.23.680] Dải ranh giới của lượt vừa xong (xem ``PyAvFrameSampler.last_boundary_track``).
    last_boundary_track: BoundaryTrack | None = None

    def iter_frames(self, metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig) -> Iterator[SampledFrame]:
        self.last_boundary_track = None
        if not metadata.path.exists():
            raise VideoNotFoundError(f"Không tìm thấy video: {metadata.path}")

        try:
            import av
            from PyNvVideoCodec import EOS, CreateDecoder, PacketData
        except ImportError as exc:
            raise VideoDecodeError("PyNvVideoCodec chưa được cài đặt.") from exc

        codec_map = _get_nv_codec_map()
        try:
            av_container = av.open(str(metadata.path))
        except _get_av_errors() as exc:
            raise VideoDecodeError(f"PyAV lỗi mở: {exc}") from exc

        video_streams = [s for s in av_container.streams if s.type == 'video']
        if not video_streams:
            av_container.close()
            raise VideoDecodeError("Không tìm thấy luồng video.")

        av_stream = video_streams[0]
        codec_name = av_stream.codec_context.name
        nv_codec_enum = codec_map.get(codec_name)

        if nv_codec_enum is None:
            av_container.close()
            raise VideoDecodeError(f"NVIDIA NVDEC không hỗ trợ codec: '{codec_name}'.")

        video_width = av_stream.width
        video_height = av_stream.height
        time_base = float(av_stream.time_base) if av_stream.time_base else 1.0/90000.0
        pixel_format = av_stream.codec_context.pix_fmt or ''
        is_16bit = '10' in pixel_format or 'p010' in pixel_format.lower() or 'p016' in pixel_format.lower()
        # [v3.23.24] Kernel NV12→RGB của NVDEC chỉ đúng cho 8-bit NV12 (BT.601). Với
        # video 10-bit (Main 10, P010/P016), việc tự suy pitch + dịch bit gây NHÂN ĐÔI
        # KHUNG ngang và sai màu (đặc biệt HDR BT.2020/PQ). Giải pháp chắc chắn: với MỌI
        # video 10-bit, UỶ QUYỀN giải mã sang PyAV — libswscale xử lý pitch/màu đúng;
        # nếu HDR còn thêm tonemap PQ/BT.2020 → BT.709 SDR. Video 8-bit (đa số) vẫn dùng
        # đường NVDEC GPU nhanh như cũ (không hồi quy).
        is_hdr = self._detect_hdr_transfer(av_stream)
        if is_16bit or is_hdr:
            logger.info(
                "Video %s (pix_fmt=%s, trc=%s) — giải mã bằng PyAV%s để pitch/màu đúng, "
                "tránh nhân đôi khung.",
                "HDR" if is_hdr else "10-bit SDR", pixel_format,
                getattr(av_stream.codec_context, "color_trc", "?"),
                " kèm tonemap HDR→SDR" if is_hdr else "",
            )
            av_container.close()
            hdr_prober = BoundaryProber(roi) if probing_enabled(roi, config.sample_step_sec) else None
            yield from self._iter_frames_hdr_via_pyav(metadata, roi, config, is_hdr=is_hdr, prober=hdr_prober)
            if hdr_prober is not None:
                self.last_boundary_track = hdr_prober.finish()
            return
        extra_data_bytes = bytes(av_stream.codec_context.extradata) if av_stream.codec_context.extradata else b''

        needs_annexb_conversion = False
        nal_length_size = 4
        annexb_header = b''

        if codec_name in ('hevc', 'h264') and extra_data_bytes and not _is_annexb(extra_data_bytes):
            extract_func = _hevc_extract_extradata if codec_name == 'hevc' else _h264_extract_extradata
            annexb_header, nal_length_size = extract_func(extra_data_bytes)
            needs_annexb_conversion = True

        logger.info("NVDEC GPU-First: Codec=%s, VRAM_Preprocess=%s, CuPy=%s, KeyframeOnly=%s",
                    codec_name, vram_preprocessing_enabled(config), HAS_CUPY,
                    max(0.001, config.sample_step_sec) >= _KEYFRAME_ONLY_MIN_STEP_SEC)

        try:
            nv_decoder = CreateDecoder(gpuid=self._gpu_id, codec=nv_codec_enum, usedevicememory=True)
        except RuntimeError as exc:
            av_container.close()
            raise VideoDecodeError(f"Khởi tạo CreateDecoder thất bại: {exc}") from exc

        # [v3.23.680] Mặt phẳng Y vùng phụ đề được cắt sẵn trên GPU → bộ dò không cắt lại (roi=None). Không CuPy (đường
        # CPU dự phòng chỉ đổi khung mẫu) → không ghi.
        prober = BoundaryProber(None) if HAS_CUPY and probing_enabled(roi, config.sample_step_sec) else None
        try:
            yield from self._decode_loop(
                av_container=av_container, av_stream=av_stream, nv_decoder=nv_decoder,
                video_width=video_width, video_height=video_height, time_base=time_base,
                is_16bit=is_16bit, is_hdr=is_hdr, needs_annexb_conversion=needs_annexb_conversion,
                nal_length_size=nal_length_size, annexb_header=annexb_header,
                enum_eos=EOS, class_packet_data=PacketData, metadata=metadata,
                roi=roi, config=config, prober=prober,
            )
            if prober is not None:
                self.last_boundary_track = prober.finish()
        finally:
            # [v3.7 bugfix REOCR-LEAK]: Giải phóng TƯỜNG MINH NVDEC session + bộ
            # nhớ GPU sau mỗi lần quét. Trước đây finally chỉ đóng av_container,
            # còn nv_decoder (phiên giải mã NVDEC, giới hạn 2–8 session đồng thời
            # trên GPU consumer) và các buffer CuPy chỉ được thả khi GC chạy —
            # không xác định. Qua nhiều lần Re-OCR → tích lũy session NVDEC chưa
            # giải phóng + VRAM creep (CuPy pool giữ block, không trả về driver).
            # Đây là một tác nhân cộng dồn gây cạn tài nguyên GPU.
            #
            # _decode_loop đã StopIteration/close trước khi finally này chạy nên
            # các buffer GPU local của nó (rgb_gpu_buffer, vram_buffers,
            # history_ring_buffer) đã hết tham chiếu → free_all_blocks thu hồi được.
            nv_decoder = None  # bỏ tham chiếu để finalizer NVDEC chạy ở gc.collect()
            gc.collect()  # ép finalizer giải phóng session NVDEC ngay (không chờ GC định kỳ)
            if HAS_CUPY:
                # Trả VRAM trong pool CuPy về driver. Cleanup phải KHÔNG ném lỗi
                # (CUDARuntimeError kế thừa RuntimeError; AttributeError nếu API pool khác).
                with contextlib.suppress(RuntimeError, AttributeError):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
            with contextlib.suppress(*_get_av_errors(), AttributeError):
                av_container.close()

    def _iter_frames_hdr_via_pyav(
        self, metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig,
        is_hdr: bool = True,
        prober: BoundaryProber | None = None,
    ) -> Iterator[SampledFrame]:
        """Giải mã video 10-bit/HDR bằng PyAV (libswscale xử lý pitch/màu đúng).

        Dùng cho video 10-bit (Main 10) thay vì kernel NVDEC tự viết (gây nhân đôi
        khung + sai màu). Nếu HDR (PQ/HLG/BT.2020): thêm chuỗi filter tonemap
        zscale=t=linear → tonemap=hable → zscale=p=bt709:t=bt709:m=bt709 → rgb24 để
        ra BT.709 SDR đúng màu. Nếu 10-bit SDR: chỉ reformat rgb24 (libswscale tự lo).
        """
        import av

        try:
            container = av.open(str(metadata.path))
        except _get_av_errors() as exc:
            raise VideoDecodeError(f"PyAV lỗi mở video 10-bit/HDR: {exc}") from exc

        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            container.close()
            raise VideoDecodeError("Không tìm thấy luồng video (10-bit/HDR path).")
        stream.thread_type = "AUTO"

        graph, graph_sink = (None, None)
        if is_hdr:
            graph, graph_sink = self._build_hdr_tonemap_graph(stream)
        start_sec = max(0.0, config.skip_intro_sec)
        end_sec = (metadata.duration_sec - config.skip_outro_sec
                   if config.skip_outro_sec > 0 else metadata.duration_sec)
        # [v3.23.723] Cùng bộ xử lý khung với đường NVDEC: khử trùng ``FrameDeduper`` (trước đây băm thô và BỎ HẲN khung
        # trùng → mất mốc "chữ còn hiện", lọc chữ trắng không có tác dụng), tiền xử lý VRAM như cài đặt (trước đây bị bỏ
        # qua hoàn toàn cho video 10-bit/HDR), mốc lấy mẫu theo khung thật thay vì cộng dồn lưới.
        processor = _FrameProcessor(
            config=config, roi=roi, video_width=int(stream.width or 0), video_height=int(stream.height or 0),
            matrix="bt709", full_range=True, prober=None, xp=cp if HAS_CUPY else np, start_sec=start_sec,
        )

        if processor.keyframe_only:              # [v3.23.723] bước thưa: chỉ giải mã khung khoá như PyAV
            with contextlib.suppress(AttributeError, ValueError, TypeError):
                stream.codec_context.skip_frame = "NONKEY"

        try:
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                ts = float(frame.pts * stream.time_base)
                if ts < start_sec:
                    continue
                if ts > end_sec:
                    break
                due = is_sample_due(ts, processor.next_target_sec)
                if prober is not None:
                    probe_av_frame(prober.sampled if due else prober.skipped, frame, ts)
                if not due:
                    continue
                rgb = self._tonemap_frame_to_rgb(frame, graph, graph_sink)
                if rgb is None:
                    processor.next_target_sec = ts + processor.step_sec
                    continue
                try:
                    sampled = processor.handle_rgb(rgb, ts)
                except Exception as exc:  # noqa: BLE001 — lỗi GPU: làm tiếp bằng NumPy, GIỮ tonemap HDR
                    if processor.xp is np:
                        raise VideoDecodeError(f"Xử lý khung 10-bit/HDR thất bại ở {ts:.3f} s: {exc}") from exc
                    logger.warning("GPU lỗi khi xử lý khung 10-bit/HDR (%s) — làm tiếp trên CPU, giữ tonemap.", exc)
                    processor.xp = np
                    sampled = processor.handle_rgb(rgb, ts)
                if sampled is not None:
                    yield sampled
        finally:
            with contextlib.suppress(*_get_av_errors(), AttributeError):
                container.close()

    @staticmethod
    def _build_hdr_tonemap_graph(stream: Any) -> tuple[Any, Any]:
        """Dựng filter graph PyAV tonemap HDR→SDR. Trả (graph, sink) hoặc (None, None)."""
        try:
            import av.filter
            graph = av.filter.Graph()
            src = graph.add_buffer(template=stream)
            chain = graph.add(
                "zscale", "t=linear:npl=100"
            )
            tonemap = graph.add("tonemap", "hable")
            back = graph.add("zscale", "p=bt709:t=bt709:m=bt709:r=tv")
            fmt = graph.add("format", "rgb24")
            sink = graph.add("buffersink")
            src.link_to(chain); chain.link_to(tonemap); tonemap.link_to(back)
            back.link_to(fmt); fmt.link_to(sink)
            graph.configure()
            return graph, sink
        except Exception as exc:  # noqa: BLE001 — thiếu filter → hạ cấp reformat
            logger.warning(
                "Không dựng được filter tonemap HDR (%s) — hạ cấp reformat rgb24 "
                "(màu HDR có thể nhạt nhưng không nhân đôi khung).", exc,
            )
            return None, None

    @staticmethod
    def _tonemap_frame_to_rgb(frame: Any, graph: Any, sink: Any) -> np.ndarray | None:
        """Đưa frame qua graph tonemap → ndarray RGB; fallback reformat nếu không có graph."""
        try:
            if graph is not None and sink is not None:
                graph.push(frame)
                out = sink.pull()
                return out.to_ndarray(format="rgb24")
            return frame.to_ndarray(format="rgb24")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tonemap frame lỗi: %s", exc)
            try:
                return frame.to_ndarray(format="rgb24")
            except Exception:  # noqa: BLE001
                return None

    def _decode_loop(
        self, *, av_container: Any, av_stream: Any, nv_decoder: Any,
        video_width: int, video_height: int, time_base: float,
        is_16bit: bool, is_hdr: bool = False, needs_annexb_conversion: bool,
        nal_length_size: int,
        annexb_header: bytes, enum_eos: Any, class_packet_data: Any,
        metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig,
        prober: BoundaryProber | None = None,
    ) -> Iterator[SampledFrame]:
        start_sec = max(0.0, config.skip_intro_sec)
        end_sec = metadata.duration_sec - config.skip_outro_sec if config.skip_outro_sec > 0 else metadata.duration_sec
        codec_context = av_stream.codec_context
        processor = _FrameProcessor(
            config=config, roi=roi, video_width=video_width, video_height=video_height,
            matrix=color_matrix_for(getattr(codec_context, "colorspace", None)),
            full_range=is_full_range(getattr(codec_context, "color_range", None), getattr(codec_context, "pix_fmt", "")),
            prober=prober, xp=cp if HAS_CUPY else np, start_sec=start_sec,
        )

        if start_sec > 0:
            try:
                # [BUG FIX v2.9+]: Đồng nhất với seek_worker.py — dùng int(sec / float(tb))
                # thay vì * denominator để đúng với mọi loại time_base (kể cả VFR).
                tb_float = float(av_stream.time_base) if av_stream.time_base else 1.0 / 90000.0
                seek_timestamp = int(max(0.0, start_sec - 5.0) / tb_float)
                av_container.seek(seek_timestamp, backward=True, stream=av_stream)
            except (RuntimeError, ValueError, AttributeError, OSError) as exc:
                logger.warning("Bỏ qua lỗi PyAV khi Seek: %s.", exc)

        stream_has_pts = False   # [v3.23.655] có gói mang PTS → PTS 0 của khung là mốc thật, không phải "thiếu"
        for av_packet in av_container.demux(av_stream):
            if av_packet.size == 0: continue
            # [v3.23.723] Bước thưa (tự dò vùng) → chỉ nạp khung khoá cho NVDEC, như PyAV (``skip_frame = NONKEY``).
            if processor.keyframe_only and not av_packet.is_keyframe: continue

            stream_has_pts = stream_has_pts or av_packet.pts is not None
            packet_timestamp_sec = float(av_packet.pts * time_base) if av_packet.pts is not None else -1.0
            if packet_timestamp_sec > end_sec + 2.0: break

            raw_bytes = bytes(av_packet)
            if needs_annexb_conversion and raw_bytes:
                try:
                    converted_bytes = _convert_packet_to_annexb(raw_bytes, nal_length_size)
                    raw_bytes = annexb_header + converted_bytes if av_packet.is_keyframe else converted_bytes
                except ValueError: pass

            try:
                packet_data, _ = _create_packet_data(raw_bytes, av_packet)
                decoded_frames = nv_decoder.Decode(packet_data)
            except RuntimeError as exc:
                logger.debug("NVDEC Decode thất bại: %s", exc)
                error = processor.error_frame(packet_timestamp_sec, end_sec)
                if error is not None:
                    yield error
                continue

            for device_frame in decoded_frames or ():
                frame_timestamp_sec = nvdec_frame_time(
                    device_frame.getPTS(), time_base, packet_timestamp_sec, stream_has_pts,
                )
                if frame_timestamp_sec < start_sec: continue
                if frame_timestamp_sec > end_sec: return
                sampled = self._handle_device_frame(processor, device_frame, video_width, video_height, frame_timestamp_sec)
                if sampled is not None:
                    yield sampled

        # Flush EOS
        try:
            packet_eos = class_packet_data()
            eos_dummy = np.zeros(1, dtype=np.uint8)
            packet_eos.bsl_data = eos_dummy.ctypes.data
            packet_eos.bsl = 0; packet_eos.decode_flag = enum_eos
            flushed = nv_decoder.Decode(packet_eos) or []
        except RuntimeError:
            flushed = []
        for device_frame in flushed:
            frame_timestamp_sec = nvdec_frame_time(device_frame.getPTS(), time_base, end_sec, stream_has_pts)
            if frame_timestamp_sec > end_sec: break
            if frame_timestamp_sec < start_sec: continue
            sampled = self._handle_device_frame(processor, device_frame, video_width, video_height, frame_timestamp_sec)
            if sampled is not None:
                yield sampled

    def _handle_device_frame(
        self, processor: _FrameProcessor, device_frame: Any, video_width: int, video_height: int, timestamp_sec: float,
    ) -> SampledFrame | None:
        """[v3.23.723] Một khung NVDEC → ``SampledFrame``. Lỗi GPU (CuPy/NVRTC/CUDA) KHÔNG còn bị nuốt im lặng (trước đây
        trả ``None`` → khung bị bỏ, phụ đề mất mà không báo): ném ``VideoDecodeError`` để bộ lấy mẫu dự phòng
        (``FallbackFrameSampler``) chuyển sang PyAV từ đúng mốc này."""
        try:
            planes = (self._nv12_planes_device(device_frame, video_width, video_height) if HAS_CUPY
                      else self._nv12_planes_host(device_frame, video_width, video_height))
            if planes is None:
                raise VideoDecodeError("Không đọc được mặt phẳng NV12 của khung NVDEC.")
            return processor.handle_nv12(planes[0], planes[1], timestamp_sec)
        except VideoDecodeError:
            raise
        except Exception as exc:  # noqa: BLE001 — CuPy/NVRTC/CUDA ném nhiều loại lỗi khác nhau
            raise VideoDecodeError(f"Xử lý khung NVDEC trên GPU thất bại ở {timestamp_sec:.3f} s: {exc}") from exc

    @staticmethod
    def _nv12_planes_device(device_frame: Any, video_width: int, video_height: int) -> tuple[Any, Any] | None:
        """Mặt phẳng Y và UV (xen kẽ) của khung NVDEC trên VRAM — không chép (DLPack / CUDA array interface)."""
        interleaved = None
        y_plane = uv_plane = None
        if hasattr(device_frame, "__dlpack__"):
            interleaved = cp.from_dlpack(device_frame)
        else:
            cuda_data = device_frame.cuda()
            if hasattr(cuda_data, "__dlpack__"):
                interleaved = cp.from_dlpack(cuda_data)
            elif hasattr(cuda_data, "__cuda_array_interface__"):
                interleaved = cp.asarray(cuda_data)
            elif isinstance(cuda_data, (list, tuple)):
                if len(cuda_data) >= 2:
                    y_plane, uv_plane = _wrap_cuda_array(cuda_data[0]), _wrap_cuda_array(cuda_data[1])
                elif len(cuda_data) == 1:
                    plane = cuda_data[0]
                    if hasattr(plane, "__dlpack__"):
                        interleaved = cp.from_dlpack(plane)
                    elif hasattr(plane, "__cuda_array_interface__"):
                        interleaved = cp.asarray(plane)
                    else:
                        interleaved = _wrap_cuda_array(plane)
            else:
                interleaved = _wrap_cuda_array(cuda_data)
        if interleaved is not None:
            y_plane, uv_plane = _split_nv12(interleaved, video_height)
        if y_plane is None or uv_plane is None:
            return None
        return _crop_nv12(y_plane, uv_plane, video_width, video_height)

    @staticmethod
    def _nv12_planes_host(device_frame: Any, video_width: int, video_height: int) -> tuple[Any, Any] | None:
        """Máy thiếu CuPy: chép khung NV12 về RAM (DLPack / ``__array__`` / ``cudaMemcpy``) rồi tách Y, UV."""
        host_buffer = None
        if hasattr(device_frame, "to_ndarray"):
            with contextlib.suppress(RuntimeError, ValueError, AttributeError):
                host_buffer = np.asarray(device_frame.to_ndarray())
        if host_buffer is None and hasattr(np, "from_dlpack") and hasattr(device_frame, "__dlpack__"):
            host_buffer = np.from_dlpack(device_frame)
        if host_buffer is None and hasattr(device_frame, "__array__"):
            host_buffer = np.asarray(device_frame)
        if host_buffer is None and hasattr(device_frame, "cuda"):
            cuda_data = device_frame.cuda()
            plane_obj = cuda_data[0] if isinstance(cuda_data, (list, tuple)) else cuda_data
            if hasattr(np, "from_dlpack") and hasattr(plane_obj, "__dlpack__"):
                host_buffer = np.from_dlpack(plane_obj)
            elif hasattr(plane_obj, "__array__"):
                host_buffer = np.asarray(plane_obj)
            elif _memcpy_func is not None:
                cuda_interface = getattr(plane_obj, "__cuda_array_interface__", {})
                strides, shape = cuda_interface.get("strides"), cuda_interface.get("shape")
                if "data" in cuda_interface and (strides or shape):
                    pitch_bytes = int(strides[0]) if strides else int(shape[1])
                    buffer = np.empty((video_height + video_height // 2, pitch_bytes), dtype=np.uint8)
                    copied = _memcpy_func(
                        ctypes.c_void_p(buffer.ctypes.data), ctypes.c_void_p(int(cuda_interface["data"][0])),
                        ctypes.c_size_t(buffer.nbytes), ctypes.c_int(_CUDA_D2H_FLAG),
                    )
                    if copied == 0:
                        host_buffer = buffer
        if host_buffer is None:
            return None
        y_plane, uv_plane = _split_nv12(host_buffer, video_height)
        return _crop_nv12(y_plane, uv_plane, video_width, video_height)

    @staticmethod
    def _detect_hdr_transfer(av_stream: Any) -> bool:
        """True nếu luồng video là HDR (PQ/HLG/HDR Vivid/BT.2020).

        Bắt theo NHIỀU dấu hiệu vì 'HDR Vivid' (CUVV, chuẩn TQ) đôi khi không khai
        báo color_trc chuẩn:
        - color_trc: 'smpte2084'/'pq' (HDR10/HDR Vivid), 'arib-std-b67'/'hlg' (HLG).
        - color_primaries 'bt2020' + 10-bit → coi là HDR (an toàn: tonemap không hại).
        """
        try:
            ctx = av_stream.codec_context
            trc = getattr(ctx, "color_trc", None)
            trc_name = str(getattr(trc, "name", trc) or "").lower()
            prim = getattr(ctx, "color_primaries", None)
            prim_name = str(getattr(prim, "name", prim) or "").lower()
            pix = str(getattr(ctx, "pix_fmt", "") or "").lower()
        except (AttributeError, ValueError):
            return False
        if trc_name in {"smpte2084", "pq", "arib-std-b67", "hlg", "smptest2084"}:
            return True
        # BT.2020 + 10-bit (Main 10) → gần như chắc chắn HDR (HDR Vivid/HDR10).
        is_10bit = "10" in pix or "p010" in pix or "p016" in pix
        if is_10bit and ("2020" in prim_name or "2020" in trc_name):
            return True
        return False

__all__ = ["PyNvVideoCodecFrameSampler"]
