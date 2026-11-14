"""Tiền xử lý ảnh trên VRAM (GPU) bằng CuPy — cùng kết quả với nhánh CPU (``image_filters``).

[v3.23.723] Mọi hàm viết theo mô-đun mảng ``xp`` (CuPy trên GPU; NumPy chỉ để kiểm đối chiếu với OpenCV trong test):
cùng một mã chạy trên VRAM ở ứng dụng thật và được so với ``cv2`` trên CPU. Đo trên khung thật phim 86 tập:

* ``nv12_roi_to_rgb`` (NVDEC → RGB, chỉ vùng ROI) ↔ PyAV ``to_ndarray("rgb24")``: trùng khít (chép phép tính swscale);
* ``sharpen`` ↔ ``cv2.addWeighted(GaussianBlur σ1)``: tối đa ±1; ``contrast`` ↔ ``cv2.convertScaleAbs``: trùng khít;
* ``clahe_u8`` ↔ ``cv2.createCLAHE``, ``rgb_to_lab`` / ``lab_to_rgb`` ↔ ``cv2`` 8 bit: trùng khít (16,7 triệu màu);
* ``separable_resize`` ↔ ``cv2.resize`` bicubic / song tuyến: tối đa ±1.

Thứ tự pipeline giống hệt CPU: phóng → CLAHE → làm nét → tương phản → viền.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import warnings
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _tolerant_dll_directories() -> Iterator[None]:
    r"""[v3.23.724] Trong lúc ``import cupy``: ``os.add_dll_directory`` với thư mục KHÔNG tồn tại thì bỏ qua thay vì ném.

    CuPy 14 suy thư mục CUDA từ vị trí ``nvrtc`` mà ``cuda-pathfinder`` tìm được rồi thêm ``<thư mục>/bin``. Bản đóng gói
    3.23.723 chép DLL CUDA vào GỐC gói (``%TEMP%\_MEIxxxx``) → CuPy tính ra ``%TEMP%\bin`` (không tồn tại) →
    ``FileNotFoundError`` ngay khi import → bấm Trích xuất là sập (log 14/11/2026).
    """
    original = getattr(os, "add_dll_directory", None)
    if original is None:
        yield
        return

    def tolerant(path: str):  # type: ignore[no-untyped-def]
        if not os.path.isdir(path):
            logger.info("CuPy muốn thêm thư mục DLL không tồn tại %s — bỏ qua.", path)
            return contextlib.nullcontext()
        return original(path)

    os.add_dll_directory = tolerant  # type: ignore[attr-defined]
    try:
        yield
    finally:
        os.add_dll_directory = original  # type: ignore[attr-defined]


def _missing_dependency(exc: BaseException) -> str:
    """[v3.23.725] Tên mô-đun KHÔNG thuộc CuPy bị thiếu trong chuỗi ngoại lệ (CuPy bọc lỗi gốc bằng ``ImportError``).

    Rỗng nếu chính CuPy chưa cài (``No module named 'cupy'``) hoặc lỗi không phải thiếu mô-đun.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = getattr(current, "name", None) if isinstance(current, ModuleNotFoundError) else None
        if name and name.split(".")[0] not in ("cupy", "cupy_backends", "cupyx"):
            return str(name)
        current = current.__cause__ or current.__context__
    return ""


def _load_cupy() -> tuple[Any, bool, str]:
    """[v3.23.724] Nạp CuPy MỘT lần cho cả ứng dụng: (mô-đun hoặc ``None``, GPU dùng được, lý do nếu không).

    Mọi lỗi (thiếu gói, thiếu DLL CUDA/NVRTC, lỗi đường dẫn, driver) đều bị bắt — trước đây chỉ bắt ``ImportError`` nên
    ``FileNotFoundError`` của CuPy làm sập lượt trích xuất. Không dùng được thì nhánh NVDEC tự lùi PyAV + tiền xử lý CPU.
    """
    try:
        with warnings.catch_warnings(), _tolerant_dll_directories():
            warnings.simplefilter("ignore")    # "CUDA path could not be detected" khi máy không có CUDA
            import cupy as module  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        missing = _missing_dependency(exc)
        if missing:
            # [v3.23.725] Có CuPy nhưng thiếu mô-đun nó cần (bản 3.23.724 thiếu ``graphlib``): trước đây báo nhầm
            # "chưa cài CuPy" kèm cả khối chẩn đoán dài của CuPy.
            reason = f"CuPy có nhưng thiếu mô-đun '{missing}' (bản đóng gói sót mô-đun này)"
            logger.warning("Không nạp được CuPy: %s — tiền xử lý / NVDEC dùng CPU.", reason)
            return None, False, reason
        return None, False, "chưa cài CuPy"
    except Exception as exc:  # noqa: BLE001 — CuPy ném nhiều loại lỗi khi khởi tạo môi trường CUDA
        logger.warning("Không nạp được CuPy (%s: %s) — tiền xử lý / NVDEC dùng CPU.", type(exc).__name__, exc)
        return None, False, f"{type(exc).__name__}: {exc}"
    try:
        usable = bool(module.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 — lỗi runtime CUDA đa dạng
        return module, False, f"CUDA không dùng được ({exc})"
    if not usable:
        logger.info("CuPy có nhưng không truy cập được GPU/CUDA — dùng CPU cho tiền xử lý ảnh OCR.")
        return module, False, "CuPy không thấy GPU CUDA"
    return module, True, ""


cp, HAS_CUPY, CUPY_UNAVAILABLE_REASON = _load_cupy()


# ==============================================================================
# PHÓNG ẢNH TRÊN VRAM (khớp cv2.resize)
# ==============================================================================

#: Hệ số nội suy bicubic của OpenCV (``INTER_CUBIC``).
_CUBIC_A: float = -0.75


def resize_tables(src_len: int, dst_len: int, *, cubic: bool) -> tuple[Any, Any]:
    """[v3.23.722] Bảng chỉ số + trọng số nội suy một chiều, khớp ``cv2.resize`` (``INTER_CUBIC`` / ``INTER_LINEAR``).

    Ánh xạ tâm điểm ảnh ``(d + 0,5) · src/dst − 0,5`` và kẹp chỉ số ở mép như OpenCV. Trả về mảng NumPy
    (``int32`` [dst, taps], ``float32`` [dst, taps]); nhánh VRAM chép lên GPU một lần rồi giữ trong ``vram_buffers``.
    """
    import numpy as np

    position = (np.arange(dst_len, dtype=np.float64) + 0.5) * (src_len / dst_len) - 0.5
    base = np.floor(position).astype(np.int64)
    frac = position - base
    if cubic:
        offsets = np.array([-1, 0, 1, 2], dtype=np.int64)
        dist = np.abs(frac[:, None] - offsets[None, :])
        a = _CUBIC_A
        near = ((a + 2.0) * dist - (a + 3.0)) * dist * dist + 1.0
        far = ((a * dist - 5.0 * a) * dist + 8.0 * a) * dist - 4.0 * a
        weights = np.where(dist <= 1.0, near, np.where(dist < 2.0, far, 0.0))
    else:
        offsets = np.array([0, 1], dtype=np.int64)
        weights = np.stack([1.0 - frac, frac], axis=1)
    index = np.clip(base[:, None] + offsets[None, :], 0, src_len - 1)
    return index.astype(np.int32), weights.astype(np.float32)


def separable_resize(image: Any, dst_w: int, dst_h: int, *, cubic: bool, xp: Any, tables: dict[str, Any] | None = None) -> Any:
    """[v3.23.722] Phóng ảnh uint8 H×W×C bằng hai lượt nội suy một chiều, chạy trên mảng của ``xp`` (CuPy: toàn bộ trên
    VRAM, không chép về RAM; NumPy: cùng mã, dùng để kiểm đối chiếu với ``cv2.resize``).

    ``tables``: bộ nhớ đệm bảng theo kích thước (truyền ``vram_buffers``) — không dựng lại bảng mỗi khung.
    """
    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    key = f"resize_tab_{src_w}x{src_h}_{dst_w}x{dst_h}_{'c' if cubic else 'l'}"
    cached = tables.get(key) if tables is not None else None
    if cached is None:
        ix, wx = resize_tables(src_w, dst_w, cubic=cubic)
        iy, wy = resize_tables(src_h, dst_h, cubic=cubic)
        cached = (xp.asarray(ix), xp.asarray(wx), xp.asarray(iy), xp.asarray(wy))
        if tables is not None:
            tables[key] = cached
    ix, wx, iy, wy = cached
    work = image.astype(xp.float32)
    rows = work[:, ix[:, 0], :] * wx[None, :, 0, None]
    for tap in range(1, ix.shape[1]):
        rows += work[:, ix[:, tap], :] * wx[None, :, tap, None]
    out = rows[iy[:, 0], :, :] * wy[:, 0, None, None]
    for tap in range(1, iy.shape[1]):
        out += rows[iy[:, tap], :, :] * wy[:, tap, None, None]
    return xp.clip(xp.rint(out), 0, 255).astype(xp.uint8)


# ==============================================================================
# NVDEC NV12 → RGB (chỉ vùng ROI), khớp PyAV / swscale
# ==============================================================================

#: ``ff_yuv2rgb_coeffs`` của FFmpeg (crv, cbu, cgu, cgv) theo ma trận màu — đúng bảng swscale dùng khi PyAV đổi màu.
_YUV2RGB_COEFFS: dict[str, tuple[int, int, int, int]] = {
    "bt601": (104597, 132201, 25675, 53279),
    "bt709": (117489, 138438, 13975, 34925),
    "fcc": (104448, 132798, 24759, 53109),
    "smpte240m": (117579, 136230, 16907, 35559),
    "bt2020": (110013, 140363, 12277, 42626),
}


def color_matrix_for(colorspace: Any) -> str:
    """[v3.23.723] Ma trận YCbCr mà PyAV/swscale dùng cho luồng có nhãn ``colorspace`` (mã AVColorSpace hoặc tên).

    Theo chỉ số bảng ``ff_yuv2rgb_coeffs``: 1 BT.709, 4 FCC, 7 SMPTE-240M, 9/10 BT.2020; mọi nhãn khác (kể cả "không
    ghi") → BT.601.
    """
    name = str(getattr(colorspace, "name", colorspace) or "").lower()
    if name in {"1", "bt709", "itu709"}:
        return "bt709"
    if name in {"4", "fcc"}:
        return "fcc"
    if name in {"7", "smpte240m"}:
        return "smpte240m"
    if name in {"9", "10", "bt2020", "bt2020nc", "bt2020_ncl", "bt2020c", "bt2020_cl"}:
        return "bt2020"
    return "bt601"


def is_full_range(color_range: Any, pix_fmt: str = "") -> bool:
    """[v3.23.723] Luồng dải đầy (0–255, "pc"/JPEG) hay dải TV (16–235)? Định dạng ``yuvj*`` luôn là dải đầy."""
    name = str(getattr(color_range, "name", color_range) or "").lower()
    return name in {"2", "jpeg", "pc", "full"} or str(pix_fmt or "").lower().startswith("yuvj")


def _c_div(numerator: int, denominator: int) -> int:
    """Chia số nguyên kiểu C (cắt về 0)."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator >= 0) == (denominator >= 0) else -quotient


def _round_int16(value: int) -> int:
    return max(-32768, min(32767, (value + (1 << 15)) >> 16))


@functools.lru_cache(maxsize=16)
def _swscale_coeffs(matrix: str, full_range: bool) -> tuple[int, int, int, int, int, int]:
    """Hệ số 16 bit của đường SIMD ``yuv420 → rgb24`` của swscale (``ff_yuv2rgb_c_init_tables``, độ sáng / tương phản /
    bão hoà mặc định) — (y, v→r, u→b, v→g, u→g, độ lệch y)."""
    crv, cbu, cgu, cgv = _YUV2RGB_COEFFS.get(matrix, _YUV2RGB_COEFFS["bt601"])
    cgu, cgv = -cgu, -cgv
    cy, oy = 1 << 16, 0
    if full_range:
        crv, cbu, cgu, cgv = (_c_div(c * 224, 255) for c in (crv, cbu, cgu, cgv))
    else:
        cy, oy = (cy * 255) // 219, 16 << 16
    return (_round_int16(cy * 8192), _round_int16(crv * 8192), _round_int16(cbu * 8192),
            _round_int16(cgv * 8192), _round_int16(cgu * 8192), _round_int16(oy * 8))


def nv12_roi_to_rgb(
    y_plane: Any, uv_plane: Any, x0: int, y0: int, x1: int, y1: int, *, matrix: str, full_range: bool, xp: Any,
) -> Any:
    """[v3.23.723] Đổi vùng ``[y0:y1, x0:x1]`` của khung NV12 sang RGB uint8 — TRÙNG KHÍT PyAV ``to_ndarray("rgb24")``.

    Chép đúng phép tính 16 bit (``pmulhw``, cộng bão hoà) của swscale ``yuv420 → rgb24``: đo trên 12 khung thật phim 86
    tập và trên khung ngẫu nhiên gắn nhãn BT.601 / 709 / 2020, dải TV / đầy, bề rộng 320 / 720 / 326 → lệch 0.

    Nhân CUDA cũ đổi CẢ khung bằng công thức dải đầy BT.601 (bỏ qua dải TV 16–235 và nhãn màu): lệch PyAV TB 7–13 mức,
    chữ trắng không bao giờ đạt "lõi" ≥ 240 (khung thật: 7 404 so với 50 391 điểm ảnh lõi) → bộ chọn khung OCR của nhánh
    GPU chạy khác hẳn nhánh CPU.
    """
    y_coeff, vr_coeff, ub_coeff, vg_coeff, ug_coeff, y_offset = _swscale_coeffs(matrix, bool(full_range))
    rows = xp.arange(y0, y1) >> 1
    cols = xp.arange(x0, x1) & ~1
    chroma = uv_plane[rows]
    cb = xp.clip((chroma[:, cols].astype(xp.int32) << 3) - 1024, -32768, 32767)
    cr = xp.clip((chroma[:, cols + 1].astype(xp.int32) << 3) - 1024, -32768, 32767)
    luma = ((y_plane[y0:y1, x0:x1].astype(xp.int32) << 3) - y_offset) * y_coeff >> 16

    def mulhi(values: Any, coeff: int) -> Any:          # pmulhw: (a·b) >> 16
        return (values * coeff) >> 16

    green_c = xp.clip(mulhi(cb, ug_coeff) + mulhi(cr, vg_coeff), -32768, 32767)
    red = xp.clip(luma + mulhi(cr, vr_coeff), -32768, 32767)
    green = xp.clip(luma + green_c, -32768, 32767)
    blue = xp.clip(luma + mulhi(cb, ub_coeff), -32768, 32767)
    return xp.clip(xp.stack([red, green, blue], axis=-1), 0, 255).astype(xp.uint8)


# ==============================================================================
# LÀM NÉT / TƯƠNG PHẢN / CLAHE (khớp image_filters trên CPU)
# ==============================================================================


_DEVICE_TABLES: dict[tuple[str, Any], Any] = {}


def _on_device(xp: Any, key: Any, build: Any) -> Any:
    """[v3.23.723] Bảng hằng (LUT LAB, nhân Gauss, chỉ số độn mép…) đưa lên thiết bị MỘT lần cho mỗi mô-đun mảng — không
    chép RAM → VRAM mỗi khung."""
    cache_key = (getattr(xp, "__name__", str(id(xp))), key)
    table = _DEVICE_TABLES.get(cache_key)
    if table is None:
        table = xp.asarray(build())
        if len(_DEVICE_TABLES) > 256:
            _DEVICE_TABLES.clear()
        _DEVICE_TABLES[cache_key] = table
    return table


def _reflect101(length: int, before: int, after: int) -> Any:
    """Chỉ số độn mép kiểu ``BORDER_REFLECT_101`` (mặc định của OpenCV), phản xạ lặp như ``cv2.borderInterpolate`` —
    đúng cả khi ảnh thấp hơn bán kính nhân. Mảng NumPy."""
    import numpy as np

    index = np.arange(-before, length + after)
    if length == 1:
        return np.zeros_like(index)
    period = 2 * (length - 1)
    index = np.abs(index) % period
    return np.where(index >= length, period - index, index)


def _gaussian_taps(sigma: float) -> Any:
    """Nhân Gauss 1 chiều như ``cv2.getGaussianKernel`` (cỡ ``round(6σ+1) | 1``) — mảng NumPy float32."""
    import numpy as np

    size = int(round(sigma * 6 + 1)) | 1
    radius = size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    taps = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    return (taps / taps.sum()).astype(np.float32)


def gaussian_blur(image: Any, sigma: float, *, xp: Any) -> Any:
    """Làm mờ Gauss tách hai chiều trên ảnh float32 H×W×C, độn mép ``REFLECT_101`` như ``cv2.GaussianBlur``."""
    taps = _gaussian_taps(sigma)
    radius = len(taps) // 2
    height, width = int(image.shape[0]), int(image.shape[1])
    cols = image[:, _on_device(xp, ("reflect", width, radius, radius), lambda: _reflect101(width, radius, radius))]
    rows = cols[:, 0:width] * xp.float32(taps[0])
    for tap in range(1, len(taps)):
        rows += cols[:, tap:tap + width] * xp.float32(taps[tap])
    padded = rows[_on_device(xp, ("reflect", height, radius, radius), lambda: _reflect101(height, radius, radius))]
    out = padded[0:height] * xp.float32(taps[0])
    for tap in range(1, len(taps)):
        out += padded[tap:tap + height] * xp.float32(taps[tap])
    return out


def sharpen(image: Any, *, xp: Any) -> Any:
    """Làm nét như ``image_filters.apply_sharpen``: 1,5·ảnh − 0,5·Gauss(σ=1) (tối đa ±1 so với cv2)."""
    work = image.astype(xp.float32)
    blurred = xp.clip(xp.rint(gaussian_blur(work, 1.0, xp=xp)), 0, 255)          # cv2.GaussianBlur trả uint8
    return xp.clip(xp.rint(work * xp.float32(1.5) - blurred * xp.float32(0.5)), 0, 255).astype(xp.uint8)


def contrast(image: Any, factor: float, *, xp: Any) -> Any:
    """Tăng tương phản như ``image_filters.apply_contrast_boost`` (``cv2.convertScaleAbs``: |ảnh·k|, làm tròn)."""
    return xp.clip(xp.rint(xp.abs(image.astype(xp.float32) * xp.float32(factor))), 0, 255).astype(xp.uint8)


# ── LAB 8 bit kiểu OpenCV (bảng gamma sRGB + bảng căn bậc ba, số nguyên) ──
_LAB_SHIFT = 12
_GAMMA_SHIFT = 3
_LAB_SHIFT2 = _LAB_SHIFT + _GAMMA_SHIFT
_SRGB_TO_XYZ = (
    (0.412453, 0.357580, 0.180423),
    (0.212671, 0.715160, 0.072169),
    (0.019334, 0.119193, 0.950227),
)
_D65_WHITE = (0.950456, 1.0, 1.088754)
_XYZ_TO_SRGB = (
    (3.240479, -1.53715, -0.498535),
    (-0.969256, 1.875991, 0.041556),
    (0.055648, -0.204043, 1.057311),
)
_LAB_BASE = 1 << 14
_MIN_AB = -8145
_INV_GAMMA_SIZE = 1 << 12


@functools.lru_cache(maxsize=1)
def _lab_tables() -> tuple[Any, Any, Any]:
    """Bảng của ``cv2.cvtColor(RGB2LAB)`` 8 bit (``RGB2Lab_b``) — NumPy."""
    import numpy as np

    # Tính bằng float32 như OpenCV (``saturate_cast<ushort>(float)``): khớp TUYỆT ĐỐI cv2 trên cả 16,7 triệu màu (float64
    # lệch 1 ở 1 675 màu).
    f32 = np.float32

    def srgb_gamma(x: Any) -> Any:
        return np.where(x <= f32(0.04045), x / f32(12.92), ((x + f32(0.055)) / f32(1.055)) ** f32(2.4)).astype(f32)

    def lab_cbrt(x: Any) -> Any:
        return np.where(x < f32(0.008856), x * f32(7.787) + f32(16.0 / 116.0), np.cbrt(x)).astype(f32)

    gamma_tab = np.rint(f32(255.0 * (1 << _GAMMA_SHIFT)) * srgb_gamma(np.arange(256, dtype=f32) / f32(255.0))).astype(np.int64)
    cbrt_size = 256 * 3 // 2 * (1 << _GAMMA_SHIFT)
    cbrt_tab = np.rint(
        f32(1 << _LAB_SHIFT2) * lab_cbrt(np.arange(cbrt_size, dtype=f32) * f32(1.0 / (255.0 * (1 << _GAMMA_SHIFT))))
    ).astype(np.int64)
    matrix = np.array(_SRGB_TO_XYZ, dtype=np.float64)
    white = np.array(_D65_WHITE, dtype=np.float64)
    coeffs = np.rint((1 << _LAB_SHIFT) * matrix / white[:, None]).astype(np.int64)
    return gamma_tab, cbrt_tab, coeffs


def _descale(value: Any, shift: int) -> Any:
    return (value + (1 << (shift - 1))) >> shift


def rgb_to_lab(image: Any, *, xp: Any) -> Any:
    """RGB uint8 → LAB uint8 như ``cv2.cvtColor(..., COLOR_RGB2LAB)`` (đường số nguyên 8 bit; trùng khít)."""
    gamma_tab, cbrt_tab, coeffs = _lab_tables()
    gamma = _on_device(xp, "lab_gamma", lambda: gamma_tab)
    cbrt = _on_device(xp, "lab_cbrt", lambda: cbrt_tab)
    top = int(cbrt_tab.shape[0]) - 1
    red, green, blue = gamma[image[..., 0]], gamma[image[..., 1]], gamma[image[..., 2]]

    def f(row: int) -> Any:
        mixed = red * int(coeffs[row, 0]) + green * int(coeffs[row, 1]) + blue * int(coeffs[row, 2])
        return cbrt[xp.minimum(_descale(mixed, _LAB_SHIFT), top)]

    fx, fy, fz = f(0), f(1), f(2)
    l_scale = (116 * 255 + 50) // 100
    l_shift = -((16 * 255 * (1 << _LAB_SHIFT2) + 50) // 100)
    lightness = _descale(l_scale * fy + l_shift, _LAB_SHIFT2)
    a_chan = _descale(500 * (fx - fy) + 128 * (1 << _LAB_SHIFT2), _LAB_SHIFT2)
    b_chan = _descale(200 * (fy - fz) + 128 * (1 << _LAB_SHIFT2), _LAB_SHIFT2)
    return xp.clip(xp.stack([lightness, a_chan, b_chan], axis=-1), 0, 255).astype(xp.uint8)


@functools.lru_cache(maxsize=1)
def _lab_inverse_tables() -> tuple[Any, Any, Any, Any, Any]:
    """Bảng của ``Lab2RGBinteger`` (OpenCV ``color_lab.cpp``, đường 8 bit mặc định) — NumPy, tính như OpenCV."""
    import numpy as np

    f32 = np.float32
    y_tab = np.zeros(256, dtype=np.int64)
    ify_tab = np.zeros(256, dtype=np.int64)
    for level in range(256):
        if level <= 20:
            y_tab[level] = int(np.rint(f32(level * _LAB_BASE * 20 * 9) / f32(17 * 29 * 29 * 29)))
            ify_tab[level] = int(np.rint(f32(_LAB_BASE) * (f32(16) / f32(116) + f32(level * 5) / f32(3 * 17 * 29))))
        else:
            fy = f32(level * 100 * _LAB_BASE) / f32(255 * 116) + f32(16 * _LAB_BASE) / f32(116)
            ify_tab[level] = int(np.rint(fy))
            y_tab[level] = int(np.rint(fy * fy * fy / f32(_LAB_BASE * _LAB_BASE)))
    index = np.arange(_MIN_AB, _LAB_BASE * 9 // 4 + _MIN_AB, dtype=np.int64)
    low = np.where(index >= 0, (index * 108) // 841, -((-index * 108) // 841)) - (_LAB_BASE * 16 // 116 * 108 // 841)
    ab_to_xz = np.where(index <= 3390, low, (index * index // _LAB_BASE) * index // _LAB_BASE)
    white = np.array(_D65_WHITE, dtype=np.float64)
    coeffs = np.rint((1 << _LAB_SHIFT) * np.array(_XYZ_TO_SRGB, dtype=np.float64) * white[None, :]).astype(np.int64)
    linear = np.arange(_INV_GAMMA_SIZE, dtype=f32) * f32(1.0 / _INV_GAMMA_SIZE)
    wide = linear.astype(np.float64)
    inverse_gamma = np.where(wide <= 0.0031308, wide * 12.92, wide ** (1.0 / 2.4) * 1.055 - 0.055).astype(f32)
    inv_tab = np.rint(f32(255) * inverse_gamma).astype(np.int64)
    return y_tab, ify_tab, ab_to_xz, coeffs, inv_tab


def lab_to_rgb(lab: Any, *, xp: Any) -> Any:
    """LAB uint8 → RGB uint8 như ``cv2.cvtColor(..., COLOR_LAB2RGB)`` — chép ``Lab2RGBinteger`` của OpenCV: trùng khít
    trên cả 16,7 triệu giá trị LAB (bản tính float trước đó lệch ±1–2 ở 2,8 % giá trị)."""
    tables = _lab_inverse_tables()
    y_tab, ify_tab, ab_to_xz, inv_tab = (
        _on_device(xp, ("lab_inv", slot), lambda slot=slot: tables[slot]) for slot in (0, 1, 2, 4)
    )
    coeffs = tables[3]
    lightness = lab[..., 0].astype(xp.int64)
    a_chan = lab[..., 1].astype(xp.int64)
    b_chan = lab[..., 2].astype(xp.int64)
    y_val = y_tab[lightness]
    ify = ify_tab[lightness]
    a_div = ((5 * a_chan * 53687 + (1 << 7)) >> 13) - 128 * _LAB_BASE // 500
    b_div = ((b_chan * 41943 + (1 << 4)) >> 9) - 128 * _LAB_BASE // 200 + 1
    x_val = ab_to_xz[ify + a_div - _MIN_AB]
    z_val = ab_to_xz[ify - b_div - _MIN_AB]
    shift = _LAB_SHIFT + (14 - 12)
    channels = []
    for row in range(3):
        mixed = coeffs[row, 0] * x_val + coeffs[row, 1] * y_val + coeffs[row, 2] * z_val
        channels.append(inv_tab[xp.clip(_descale(mixed, shift), 0, _INV_GAMMA_SIZE - 1)])
    return xp.clip(xp.stack(channels, axis=-1), 0, 255).astype(xp.uint8)


def clahe_u8(gray: Any, clip_limit: float, tiles: int, *, xp: Any) -> Any:
    """CLAHE trên ảnh xám uint8 — trùng khít ``cv2.createCLAHE(clip_limit, (tiles, tiles)).apply``.

    Chép đúng thuật toán OpenCV: độn mép ``REFLECT_101`` cho chia hết số ô, histogram từng ô, cắt ở
    ``max(int(clip·diện tích/256), 1)`` rồi chia đều phần dư (kể cả phần lẻ theo bước), LUT tích luỹ làm tròn, nội suy
    song tuyến giữa 4 ô lân cận.
    """
    import numpy as np

    tiles = max(1, int(tiles))
    height, width = int(gray.shape[0]), int(gray.shape[1])
    pad_y = (tiles - height % tiles) % tiles
    pad_x = (tiles - width % tiles) % tiles
    source = gray
    if pad_y or pad_x:
        source = gray[_on_device(xp, ("reflect", height, 0, pad_y), lambda: _reflect101(height, 0, pad_y))][
            :, _on_device(xp, ("reflect", width, 0, pad_x), lambda: _reflect101(width, 0, pad_x))]
    full_h, full_w = int(source.shape[0]), int(source.shape[1])
    tile_h, tile_w = full_h // tiles, full_w // tiles
    area = tile_h * tile_w
    tile_id = (xp.arange(full_h)[:, None] // tile_h) * tiles + (xp.arange(full_w)[None, :] // tile_w)
    flat = (tile_id * 256 + source.astype(xp.int64)).ravel()
    hist = xp.bincount(flat, minlength=tiles * tiles * 256).reshape(tiles * tiles, 256)
    if clip_limit > 0:
        clip = max(int(clip_limit * area / 256), 1)
        excess = xp.maximum(hist - clip, 0).sum(axis=1)
        hist = xp.minimum(hist, clip)
        batch = excess // 256
        hist = hist + batch[:, None]
        residual = excess - batch * 256
        step = xp.maximum(256 // xp.maximum(residual, 1), 1)
        bins = xp.arange(256)[None, :]
        hist = hist + (((bins % step[:, None]) == 0) & ((bins // step[:, None]) < residual[:, None]))
    scale = np.float32(255.0 / area)
    lut = xp.clip(xp.rint(xp.cumsum(hist, axis=1).astype(xp.float32) * scale), 0, 255).reshape(tiles, tiles, 256)

    def axis_weights(count: int, tile: int) -> tuple[Any, Any, Any]:
        position = xp.arange(count, dtype=xp.float32) * np.float32(1.0 / tile) - np.float32(0.5)
        low = xp.floor(position).astype(xp.int64)
        frac = (position - low).astype(xp.float32)
        return xp.maximum(low, 0), xp.minimum(low + 1, tiles - 1), frac

    ty1, ty2, ya = axis_weights(height, tile_h)
    tx1, tx2, xa = axis_weights(width, tile_w)
    value = gray.astype(xp.int64)
    xa, ya = xa[None, :], ya[:, None]

    def look(ty: Any, tx: Any) -> Any:
        return lut[ty[:, None], tx[None, :], value]

    top_row = look(ty1, tx1) * (1 - xa) + look(ty1, tx2) * xa
    bottom_row = look(ty2, tx1) * (1 - xa) + look(ty2, tx2) * xa
    return xp.clip(xp.rint(top_row * (1 - ya) + bottom_row * ya), 0, 255).astype(xp.uint8)


def clahe_rgb(image: Any, clip_limit: float, tiles: int, *, xp: Any) -> Any:
    """CLAHE trên kênh L của LAB như ``image_filters.apply_clahe`` — trùng khít (LAB hai chiều + CLAHE đều bit-exact)."""
    lab = rgb_to_lab(image, xp=xp)
    lab[..., 0] = clahe_u8(xp.ascontiguousarray(lab[..., 0]), clip_limit, tiles, xp=xp)
    return lab_to_rgb(lab, xp=xp)


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
    border_thickness_px: int,
    *,
    apply_clahe: bool = False,
    clahe_clip_limit: float = 3.0,
    clahe_tile_size: int = 8,
    xp: Any = None,
) -> Any:
    """Tiền xử lý toàn bộ trên VRAM, cùng thứ tự và cùng kết quả với ``apply_cpu_preprocessing_pipeline``.

    [v3.23.723] Trước đây làm nét (Laplace 5 điểm) và tương phản (co giãn quanh 127,5) là NHÂN RIÊNG cho kết quả khác hẳn
    bản CPU (Gauss σ1; |ảnh·k|), còn CLAHE không có bản GPU → chạy sau trên CPU, SAU cả viền. Cùng một cài đặt cho ra ảnh
    khác nhau tuỳ máy có CuPy hay không. ``vram_buffers`` giữ bảng nội suy theo cỡ ảnh (không dựng lại mỗi khung).
    ``xp``: mô-đun mảng (mặc định CuPy); test truyền NumPy để so với OpenCV.
    """
    module = cp if xp is None and HAS_CUPY else xp
    if module is None:
        return gpu_array

    # 1. Phóng ảnh: bicubic khi ≤ ×2, song tuyến khi lớn hơn (như CPU) — khớp cv2.resize tới ±1.
    if upscale_small_text:
        h, w = gpu_array.shape[:2]
        # [v3.23.721] Cùng quy tắc với nhánh CPU (``upscale_size``): phóng khi ảnh thấp hơn đích.
        from subtitles_extractor.infrastructure.ocr.preprocessing.image_filters import (
            _UPSCALE_INTERPOLATION_BREAKPOINT,
            upscale_size,
        )

        size = upscale_size(h, w, upscale_target_height_px)      # [v3.23.723] trần ×4 như CPU (trước: ×3)
        if size is not None:
            target_w, target_h = size
            cubic = target_h / h <= _UPSCALE_INTERPOLATION_BREAKPOINT
            gpu_array = separable_resize(gpu_array, target_w, target_h, cubic=cubic, xp=module, tables=vram_buffers)

    # 2. CLAHE (kênh L của LAB) → 3. làm nét → 4. tương phản: đúng thứ tự CPU.
    if apply_clahe:
        gpu_array = clahe_rgb(gpu_array, clahe_clip_limit, clahe_tile_size, xp=module)
    if sharpen:
        gpu_array = _sharpen(gpu_array, xp=module)     # tham số ``sharpen`` che tên hàm cùng tên
    if abs(float(contrast_factor) - 1.0) >= 1e-4:
        gpu_array = contrast(gpu_array, contrast_factor, xp=module)

    # 5. Viền đen.
    if add_border and border_thickness_px > 0:
        b = int(border_thickness_px)
        gpu_array = module.pad(gpu_array, ((b, b), (b, b), (0, 0)), mode="constant", constant_values=0)
    return gpu_array


#: Bí danh của :func:`sharpen` cho pipeline (tham số ``sharpen`` của pipeline che tên hàm).
_sharpen = sharpen

__all__ = [
    "apply_gpu_preprocessing_pipeline", "clahe_rgb", "clahe_u8", "color_matrix_for", "contrast", "gaussian_blur",
    "is_full_range", "lab_to_rgb", "nv12_roi_to_rgb", "resize_tables", "rgb_to_lab", "separable_resize", "sharpen",
]
