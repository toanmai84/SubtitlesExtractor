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

import cv2
import numpy as np

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import VideoDecodeError, VideoNotFoundError
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    SampledFrame,
)
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.video.perceptual_hash import (
    compute_phash,
    hamming_distance,
    pixel_diff_ratio,
)

# [V3.28] Import Pipeline GPU mới tạo
from subtitles_extractor.infrastructure.ocr.preprocessing.gpu_image_filters import apply_gpu_preprocessing_pipeline

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    logger.debug(
        "Thư viện 'cupy' chưa được cài đặt — backend PyNvVideoCodec sẽ dùng "
        "fallback CPU (chậm hơn). Cài bằng 'pip install cupy-cuda12x' nếu cần."
    )

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

@functools.lru_cache(maxsize=1)
def _get_cuda_nv12_to_rgb_kernel() -> Any:
    if not HAS_CUPY: return None
    return cp.RawKernel(r'''
    extern "C" __global__
    void nv12_to_rgb(
        const unsigned char* __restrict__ Y,
        const unsigned char* __restrict__ UV,
        unsigned char* __restrict__ RGB,
        int width, int height, int pitch
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int y_idx = y * pitch + x;
        int uv_idx = (y >> 1) * pitch + (x & ~1);
        int y_scaled = Y[y_idx] << 13;
        int u_val = UV[uv_idx] - 128;
        int v_val = UV[uv_idx + 1] - 128;

        int r = (y_scaled + 11485 * v_val) >> 13;
        int g = (y_scaled - 2819 * u_val - 5850 * v_val) >> 13;
        int b = (y_scaled + 14516 * u_val) >> 13;

        int rgb_idx = (y * width + x) * 3;
        RGB[rgb_idx]     = (unsigned char)max(0, min(255, r));
        RGB[rgb_idx + 1] = (unsigned char)max(0, min(255, g));
        RGB[rgb_idx + 2] = (unsigned char)max(0, min(255, b));
    }
    ''', 'nv12_to_rgb')

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
# CLASS CHÍNH
# ==============================================================================

class PyNvVideoCodecFrameSampler:
    def __init__(self, *, gpu_id: int = 0) -> None:
        self._gpu_id = gpu_id
        if not HAS_CUPY:
            _setup_dll()
            _get_cudart()

    def iter_frames(self, metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig) -> Iterator[SampledFrame]:
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
            yield from self._iter_frames_hdr_via_pyav(metadata, roi, config, is_hdr=is_hdr)
            return
        extra_data_bytes = bytes(av_stream.codec_context.extradata) if av_stream.codec_context.extradata else b''

        needs_annexb_conversion = False
        nal_length_size = 4
        annexb_header = b''

        if codec_name in ('hevc', 'h264') and extra_data_bytes and not _is_annexb(extra_data_bytes):
            extract_func = _hevc_extract_extradata if codec_name == 'hevc' else _h264_extract_extradata
            annexb_header, nal_length_size = extract_func(extra_data_bytes)
            needs_annexb_conversion = True

        logger.info("NVDEC GPU-First: Codec=%s, VRAM_Preprocess=%s, CuPy=%s",
                    codec_name, config.vram_sharpen or config.vram_upscale_small_text, HAS_CUPY)

        try:
            nv_decoder = CreateDecoder(gpuid=self._gpu_id, codec=nv_codec_enum, usedevicememory=True)
        except RuntimeError as exc:
            av_container.close()
            raise VideoDecodeError(f"Khởi tạo CreateDecoder thất bại: {exc}") from exc

        try:
            yield from self._decode_loop(
                av_container=av_container, av_stream=av_stream, nv_decoder=nv_decoder,
                video_width=video_width, video_height=video_height, time_base=time_base,
                is_16bit=is_16bit, is_hdr=is_hdr, needs_annexb_conversion=needs_annexb_conversion,
                nal_length_size=nal_length_size, annexb_header=annexb_header,
                enum_eos=EOS, class_packet_data=PacketData, metadata=metadata,
                roi=roi, config=config,
            )
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
        step_sec = max(0.001, config.sample_step_sec)
        start_sec = max(0.0, config.skip_intro_sec)
        end_sec = (metadata.duration_sec - config.skip_outro_sec
                   if config.skip_outro_sec > 0 else metadata.duration_sec)
        next_target_sec = start_sec
        frame_idx = 0
        last_hash: int | None = None

        try:
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                ts = float(frame.pts * stream.time_base)
                if ts < start_sec:
                    continue
                if ts > end_sec:
                    break
                if ts + 1e-6 < next_target_sec:
                    continue
                next_target_sec += step_sec

                rgb = self._tonemap_frame_to_rgb(frame, graph, graph_sink)
                if rgb is None:
                    continue
                if roi is not None:
                    clipped = roi.clip_to(rgb.shape[1], rgb.shape[0])
                    rgb = rgb[clipped.y:clipped.y2, clipped.x:clipped.x2]

                # Khử trùng frame liền kề (đỡ OCR thừa) — hash thô theo mẫu điểm ảnh.
                h = int(rgb[::16, ::16, 0].sum()) if rgb.size else 0
                if h == last_hash:
                    frame_idx += 1
                    continue
                last_hash = h

                yield SampledFrame(
                    image_rgb=np.ascontiguousarray(rgb),
                    frame_index=frame_idx,
                    timestamp_sec=ts,
                )
                frame_idx += 1
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

    def _apply_vram_preprocessing(self, gpu_array: Any, config: FrameSamplingConfig, vram_buffers: dict[str, Any]) -> Any:
        """Sử dụng Pipeline Đồng Bộ từ gpu_image_filters.py"""
        if not HAS_CUPY: return gpu_array

        return apply_gpu_preprocessing_pipeline(
            gpu_array, vram_buffers,
            upscale_small_text=config.vram_upscale_small_text,
            upscale_target_height_px=config.vram_upscale_target_height_px,
            sharpen=config.vram_sharpen,
            contrast_factor=config.vram_contrast_factor,
            add_border=config.vram_add_border,
            border_thickness_px=config.vram_border_thickness_px
        )

    def _decode_loop(
        self, *, av_container: Any, av_stream: Any, nv_decoder: Any,
        video_width: int, video_height: int, time_base: float,
        is_16bit: bool, is_hdr: bool = False, needs_annexb_conversion: bool,
        nal_length_size: int,
        annexb_header: bytes, enum_eos: Any, class_packet_data: Any,
        metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig
    ) -> Iterator[SampledFrame]:

        step_sec = max(0.001, config.sample_step_sec)
        start_sec = max(0.0, config.skip_intro_sec)
        end_sec = metadata.duration_sec - config.skip_outro_sec if config.skip_outro_sec > 0 else metadata.duration_sec

        last_hash_val: int | None = None
        last_image_arr: np.ndarray | None = None
        next_target_sec = start_sec
        frame_counter_idx = 0

        vram_buffers: dict[str, Any] = {}
        rgb_gpu_buffer = cp.empty((video_height, video_width, 3), dtype=cp.uint8) if HAS_CUPY else None

        history_ring_buffer = None
        history_count = 0
        num_blend_frames = max(3, min(7, config.median_blend_frames)) if config.apply_median_blend else 1

        if start_sec > 0:
            try:
                # [BUG FIX v2.9+]: Đồng nhất với seek_worker.py — dùng int(sec / float(tb))
                # thay vì * denominator để đúng với mọi loại time_base (kể cả VFR).
                tb_float = float(av_stream.time_base) if av_stream.time_base else 1.0 / 90000.0
                seek_timestamp = int(max(0.0, start_sec - 5.0) / tb_float)
                av_container.seek(seek_timestamp, backward=True, stream=av_stream)
            except (RuntimeError, ValueError, AttributeError, OSError) as exc:
                logger.warning("Bỏ qua lỗi PyAV khi Seek: %s.", exc)

        for av_packet in av_container.demux(av_stream):
            if av_packet.size == 0: continue

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
                yield SampledFrame(frame_index=frame_counter_idx, timestamp_sec=packet_timestamp_sec, image_rgb=EMPTY_ARRAY, is_error=True)
                frame_counter_idx += 1; next_target_sec = packet_timestamp_sec + step_sec
                continue

            if not decoded_frames: continue

            for device_frame in decoded_frames:
                pts_val = device_frame.getPTS()
                frame_timestamp_sec = float(pts_val) * time_base if pts_val else packet_timestamp_sec

                if frame_timestamp_sec < start_sec: continue
                if frame_timestamp_sec > end_sec: return

                if HAS_CUPY:
                    cropped_gpu = self._convert_and_crop_cupy(device_frame, video_height, video_width, is_16bit, is_hdr, roi, rgb_gpu_buffer)
                    if cropped_gpu is None: continue

                    if config.apply_median_blend:
                        if history_ring_buffer is None:
                            history_ring_buffer = cp.empty((num_blend_frames, *cropped_gpu.shape), dtype=cropped_gpu.dtype)

                        insert_idx = history_count % num_blend_frames
                        history_ring_buffer[insert_idx] = cropped_gpu
                        history_count += 1

                        if frame_timestamp_sec < next_target_sec or history_count < num_blend_frames:
                            continue

                        target_idx = (history_count - num_blend_frames // 2 - 1) % num_blend_frames
                        target_frame = history_ring_buffer[target_idx]

                        aligned_frames = []
                        for i in range(num_blend_frames):
                            idx = (history_count - num_blend_frames + i) % num_blend_frames
                            frame_i = history_ring_buffer[idx]
                            aligned_frames.append(frame_i if idx == target_idx else _align_and_shift_cupy(frame_i, target_frame, vram_buffers))

                        if num_blend_frames == 3:
                            blended_key = f"blended_{target_frame.shape[1]}_{target_frame.shape[0]}"
                            if blended_key not in vram_buffers: vram_buffers[blended_key] = cp.empty_like(target_frame)
                            blended_gpu = vram_buffers[blended_key]

                            h, w = target_frame.shape[:2]
                            block, grid = (32, 32), (math.ceil(w / 32), math.ceil(h / 32))
                            _get_median3_kernel()(grid, block, (aligned_frames[0], aligned_frames[1], aligned_frames[2], blended_gpu, w, h))
                        else:
                            stacked = cp.stack(aligned_frames, axis=0)
                            blended_gpu = cp.median(stacked, axis=0).astype(cp.uint8)

                        blended_gpu = self._apply_vram_preprocessing(blended_gpu, config, vram_buffers)
                        final_rgb_numpy = cp.asnumpy(blended_gpu)

                    else:
                        if frame_timestamp_sec < next_target_sec: continue
                        cropped_gpu = self._apply_vram_preprocessing(cropped_gpu, config, vram_buffers)
                        final_rgb_numpy = cp.asnumpy(cropped_gpu)
                else:
                    if frame_timestamp_sec < next_target_sec: continue
                    final_rgb_numpy = self._convert_and_crop_cpu_fallback(device_frame, video_height, video_width, is_16bit, is_hdr, roi)
                    if final_rgb_numpy is None: continue

                current_hash_val = compute_phash(final_rgb_numpy)
                is_dup = not self._should_keep_frame(final_rgb_numpy, current_hash_val, last_hash_val, last_image_arr, config)

                yield SampledFrame(
                    frame_index=frame_counter_idx,
                    timestamp_sec=frame_timestamp_sec,
                    image_rgb=EMPTY_ARRAY if is_dup else final_rgb_numpy,
                    is_duplicate=is_dup
                )

                if not is_dup:
                    last_hash_val = current_hash_val
                    last_image_arr = final_rgb_numpy

                frame_counter_idx += 1
                next_target_sec = frame_timestamp_sec + step_sec

        # Flush EOS
        try:
            packet_eos = class_packet_data()
            eos_dummy = np.zeros(1, dtype=np.uint8)
            packet_eos.bsl_data = eos_dummy.ctypes.data
            packet_eos.bsl = 0; packet_eos.decode_flag = enum_eos

            for device_frame in nv_decoder.Decode(packet_eos) or []:
                pts_val = device_frame.getPTS()
                frame_timestamp_sec = float(pts_val) * time_base if pts_val else end_sec
                if frame_timestamp_sec > end_sec: break
                if frame_timestamp_sec < next_target_sec: continue

                if HAS_CUPY:
                    cropped_gpu = self._convert_and_crop_cupy(device_frame, video_height, video_width, is_16bit, is_hdr, roi, rgb_gpu_buffer)
                    if cropped_gpu is None: continue
                    cropped_gpu = self._apply_vram_preprocessing(cropped_gpu, config, vram_buffers)
                    final_rgb_numpy = cp.asnumpy(cropped_gpu)
                else:
                    final_rgb_numpy = self._convert_and_crop_cpu_fallback(device_frame, video_height, video_width, is_16bit, is_hdr, roi)
                    if final_rgb_numpy is None: continue

                current_hash_val = compute_phash(final_rgb_numpy)
                is_dup = not self._should_keep_frame(final_rgb_numpy, current_hash_val, last_hash_val, last_image_arr, config)

                yield SampledFrame(
                    frame_index=frame_counter_idx,
                    timestamp_sec=frame_timestamp_sec,
                    image_rgb=EMPTY_ARRAY if is_dup else final_rgb_numpy,
                    is_duplicate=is_dup
                )
                if not is_dup:
                    last_hash_val = current_hash_val
                    last_image_arr = final_rgb_numpy
                frame_counter_idx += 1
                next_target_sec = frame_timestamp_sec + step_sec
        except RuntimeError: pass

    def _convert_and_crop_cupy(
        self, device_frame: Any, video_height: int, video_width: int, is_16bit: bool, is_hdr: bool, roi: Roi | None, rgb_gpu_buffer: Any
    ) -> Any | None:
        try:
            interleaved_arr = None
            if hasattr(device_frame, '__dlpack__'):
                interleaved_arr = cp.from_dlpack(device_frame)
            else:
                cuda_data = device_frame.cuda()
                if hasattr(cuda_data, '__dlpack__'): interleaved_arr = cp.from_dlpack(cuda_data)
                elif hasattr(cuda_data, '__cuda_array_interface__'): interleaved_arr = cp.asarray(cuda_data)
                elif isinstance(cuda_data, (list, tuple)):
                    y_plane = _wrap_cuda_array(cuda_data[0]) if len(cuda_data) >= 2 else None
                    uv_plane = _wrap_cuda_array(cuda_data[1]) if len(cuda_data) >= 2 else None

                    if len(cuda_data) == 1:
                        plane_obj = cuda_data[0]
                        if hasattr(plane_obj, '__dlpack__'): interleaved_arr = cp.from_dlpack(plane_obj)
                        elif hasattr(plane_obj, '__cuda_array_interface__'): interleaved_arr = cp.asarray(plane_obj)
                        else: interleaved_arr = _wrap_cuda_array(plane_obj)
                else:
                    interleaved_arr = _wrap_cuda_array(cuda_data)

            if interleaved_arr is not None:
                if interleaved_arr.ndim == 1:
                    pitch = interleaved_arr.size // (video_height + video_height // 2)
                    interleaved_arr = interleaved_arr.reshape((video_height + video_height // 2, pitch))
                elif interleaved_arr.ndim == 3:
                    interleaved_arr = interleaved_arr.squeeze()

                y_plane = interleaved_arr[:video_height]
                uv_plane = interleaved_arr[video_height:video_height + video_height//2]

            if y_plane is None or uv_plane is None: return None

            if is_16bit:
                # [v3.23.23] 10-bit: lấy 8 bit cao. HDR (PQ/BT.2020) được xử lý ở
                # nhánh riêng (uỷ quyền PyAV tonemap) TRƯỚC khi vào đây, nên ở đây chỉ
                # còn SDR 10-bit — dịch bit là đúng.
                y_plane = (y_plane.view(cp.uint16) >> 8).astype(cp.uint8)
                uv_plane = (uv_plane.view(cp.uint16) >> 8).astype(cp.uint8)

            pitch = y_plane.shape[1]
            block, grid = (32, 32), (math.ceil(video_width / 32), math.ceil(video_height / 32))

            kernel = _get_cuda_nv12_to_rgb_kernel()
            if kernel: kernel(grid, block, (y_plane, uv_plane, rgb_gpu_buffer, video_width, video_height, pitch))

            if roi is None:
                return rgb_gpu_buffer.copy()
            clipped = roi.clip_to(video_width, video_height)
            return rgb_gpu_buffer[clipped.y : clipped.y2, clipped.x : clipped.x2].copy()
        except (RuntimeError, ValueError, AttributeError, MemoryError) as exc:
            logger.debug("CuPy Kernel chuyển đổi màu lỗi: %s.", exc)
            return None

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

    def _convert_and_crop_cpu_fallback(
        self, device_frame: Any, video_height: int, video_width: int, is_16bit: bool, is_hdr: bool, roi: Roi | None
    ) -> np.ndarray | None:
        try:
            host_buffer = None
            if hasattr(device_frame, 'to_ndarray'):
                try:
                    return device_frame.to_ndarray()
                except (RuntimeError, ValueError, AttributeError) as conv_exc:
                    logger.debug("to_ndarray fallback fail: %s.", conv_exc)

            if hasattr(np, 'from_dlpack') and hasattr(device_frame, '__dlpack__'):
                host_buffer = np.from_dlpack(device_frame)
            elif hasattr(device_frame, '__array__'):
                host_buffer = np.asarray(device_frame)

            if host_buffer is None:
                cuda_data = device_frame.cuda()
                plane_obj = cuda_data[0] if isinstance(cuda_data, (list, tuple)) else cuda_data

                if hasattr(np, 'from_dlpack') and hasattr(plane_obj, '__dlpack__'): host_buffer = np.from_dlpack(plane_obj)
                elif hasattr(plane_obj, '__array__'): host_buffer = np.asarray(plane_obj)

                if host_buffer is None and _memcpy_func is not None:
                    cuda_interface = getattr(plane_obj, "__cuda_array_interface__", {})
                    if "data" in cuda_interface:
                        strides, shape = cuda_interface.get("strides"), cuda_interface.get("shape")
                        pitch_bytes = strides[0] if strides else shape[1] * (2 if is_16bit else 1)
                        nv12_height = video_height + video_height // 2

                        host_buffer_bytes = np.empty((nv12_height, pitch_bytes), dtype=np.uint8)
                        mem_copy_result = _memcpy_func(
                            ctypes.c_void_p(host_buffer_bytes.ctypes.data),
                            ctypes.c_void_p(int(cuda_interface["data"][0])),
                            ctypes.c_size_t(host_buffer_bytes.nbytes), ctypes.c_int(_CUDA_D2H_FLAG)
                        )
                        if mem_copy_result == 0:
                            if is_16bit:
                                host_buffer = (host_buffer_bytes.view(np.uint16) >> 8).astype(np.uint8)
                            else:
                                host_buffer = host_buffer_bytes

            if host_buffer is None: return None

            if host_buffer.ndim == 1:
                pitch = host_buffer.size // (video_height + video_height // 2)
                host_buffer = host_buffer.reshape((video_height + video_height // 2, pitch))
            elif host_buffer.ndim == 3:
                host_buffer = host_buffer.squeeze()

            if host_buffer.shape[1] > video_width: host_buffer = host_buffer[:, :video_width]
            if is_16bit: host_buffer = (host_buffer.view(np.uint16) >> 8).astype(np.uint8)

            rgb_image = cv2.cvtColor(np.ascontiguousarray(host_buffer), cv2.COLOR_YUV2RGB_NV12)
            if roi is None: return rgb_image

            clipped = roi.clip_to(video_width, video_height)
            return rgb_image[clipped.y : clipped.y2, clipped.x : clipped.x2].copy()
        except (RuntimeError, ValueError, AttributeError, MemoryError, IndexError) as exc:
            logger.debug("CPU Fallback lỗi: %s.", exc)
            return None

    @staticmethod
    def _should_keep_frame(
        image_rgb: np.ndarray, current_hash: int, last_hash: int | None, last_image: np.ndarray | None, config: FrameSamplingConfig
    ) -> bool:
        if last_hash is None or last_image is None: return True
        if hamming_distance(current_hash, last_hash) > config.phash_distance_threshold: return True
        if pixel_diff_ratio(image_rgb, last_image) > config.pixel_diff_threshold: return True
        return False

__all__ = ["PyNvVideoCodecFrameSampler"]
