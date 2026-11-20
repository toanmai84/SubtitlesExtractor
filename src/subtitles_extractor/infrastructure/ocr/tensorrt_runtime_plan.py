"""[v3.23.755] Lập kế hoạch TẢI TensorRT (thư viện + Python) để bật TensorRT cho OCR GPU *lúc chạy*.

Log 19/11 10:26 (bản 3.23.754): bật TensorRT → "TensorRT không dùng được (… No module named 'tensorrt')". PaddleX dựng mô
hình TensorRT bằng ``paddle.tensorrt`` (Paddle 3, PIR-TRT) — cần gói Python ``tensorrt`` + thư viện ``nvinfer_10.dll``;
bản paddle của ứng dụng (``paddlepaddle-gpu==3.3.1`` cu129, ``with_pip_tensorrt='OFF'``, biên dịch với TensorRT 10.5)
KHÔNG kèm TensorRT và không tự tìm TensorRT cài qua pip → ứng dụng tải về ``models/tensorrt_runtime/`` và tự đưa vào
``sys.path`` + đường dẫn DLL (bootstrap), cùng mẫu "Bật tăng tốc GPU (OCR)".

Phiên bản: gói Python TensorRT cho Python 3.13 trên Windows chỉ có từ **10.9** (10.5 chỉ tới Python 3.12) → 10.9.0.34
(gần bản paddle biên dịch nhất; mọi 10.x cùng tên ``nvinfer_10.dll``). Tải từ kho NVIDIA (``pypi.nvidia.com``):
``tensorrt-cu12-libs`` (~1,6 GB, DLL) + ``tensorrt-cu12-bindings`` (Python). Mô-đun ``tensorrt`` do gói meta
``tensorrt-cu12`` (chỉ có bản mã nguồn) tạo bằng 2 dòng — ứng dụng tự ghi đúng 2 dòng đó (Apache-2.0, NVIDIA) thay vì
dựng gói mã nguồn bằng pip.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Final

logger = logging.getLogger(__name__)

TENSORRT_VERSION: Final[str] = "10.9.0.34"
TENSORRT_RUNTIME_DIRNAME: Final[str] = "tensorrt_runtime"
TENSORRT_INDEX_URL: Final[str] = "https://pypi.nvidia.com"
TENSORRT_DOWNLOAD_GB: Final[float] = 1.6
TENSORRT_PACKAGES: Final[tuple[str, ...]] = (
    f"tensorrt-cu12-libs=={TENSORRT_VERSION}",
    f"tensorrt-cu12-bindings=={TENSORRT_VERSION}",
)
#: Đúng nội dung ``tensorrt/__init__.py`` của gói ``tensorrt-cu12`` (NVIDIA, Apache-2.0).
TENSORRT_SHIM: Final[str] = (
    "# Tạo bởi Subtitles Extractor — như gói tensorrt-cu12 (NVIDIA, Apache-2.0).\n"
    "from tensorrt_bindings import *  # noqa: F401,F403\n"
    "from tensorrt_bindings import __version__  # noqa: F401\n"
)
_KEY_DLL: Final[str] = "nvinfer_10.dll"


def build_tensorrt_install_command(python_exe: str, trt_dir: Path) -> tuple[str, ...]:
    """Lệnh pip tải TensorRT vào ``trt_dir`` — chỉ bản dựng sẵn (wheel), kho NVIDIA, không kéo phụ thuộc khác."""
    return (
        python_exe, "-m", "pip", "install",
        "--target", str(trt_dir),
        "--no-deps", "--upgrade", "--only-binary", ":all:",
        "--extra-index-url", TENSORRT_INDEX_URL,
        *TENSORRT_PACKAGES,
    )


def write_tensorrt_shim(trt_dir: Path) -> Path:
    """Ghi mô-đun ``tensorrt`` (2 dòng) cạnh ``tensorrt_bindings``; trả đường dẫn tệp."""
    package = trt_dir / "tensorrt"
    package.mkdir(parents=True, exist_ok=True)
    init = package / "__init__.py"
    init.write_text(TENSORRT_SHIM, encoding="utf-8")
    return init


def tensorrt_dll_dirs(trt_dir: Path) -> list[Path]:
    """Thư mục chứa DLL TensorRT (``tensorrt_libs``) — chỉ khi có ``nvinfer_10.dll``."""
    libs = trt_dir / "tensorrt_libs"
    return [libs] if (libs / _KEY_DLL).is_file() else []


def is_tensorrt_installed(trt_dir: Path) -> bool:
    """Đủ DLL + Python bindings + mô-đun ``tensorrt``?"""
    return bool(tensorrt_dll_dirs(trt_dir)) and (trt_dir / "tensorrt_bindings").is_dir() and (
        trt_dir / "tensorrt" / "__init__.py"
    ).is_file()


#: Tên Linux mà ``paddle/tensorrt/converter.py`` (Paddle 3.3.1, dòng 91) gọi cứng — cả trong bản Windows.
_LINUX_PLUGIN_NAME: Final[str] = "libnvinfer_plugin.so"
_WINDOWS_PLUGIN_DLL: Final[str] = "nvinfer_plugin_10.dll"
#: [v3.23.763] Tên phần C++ của Paddle (``paddle/phi/backends/dynload/tensorrt.cc``) nạp lúc CHẠY engine trên Windows.
_PADDLE_RUNTIME_PLUGIN_DLL: Final[str] = "nvinfer_plugin.dll"
#: Giữ thư viện plugin đã nạp sống suốt phiên (LoadLibrary theo tên trả đúng mô-đun đã nạp).
_LOADED_PLUGINS: list[Any] = []


def ensure_paddle_runtime_plugin(trt_dir: Path) -> Path | None:
    """[v3.23.763] Cho phần C++ của Paddle tìm được thư viện plugin TensorRT 10 lúc CHẠY engine.

    Log 19/11 14:21 (bản 3.23.761): cả bộ dò lẫn bộ nhận dạng DỰNG engine TensorRT xong, nhưng mỗi lô OCR lỗi "(NotFound)
    TensorRT is needed, but TensorRT dynamic library 'nvinfer_plugin.dll' is not found" (``dynload\\tensorrt.cc:53``) —
    tên kiểu TensorRT 8; TensorRT 10 đặt tên ``nvinfer_plugin_10.dll``. (``nvinfer_10.dll`` thì Paddle tìm thấy vì
    mô-đun ``tensorrt`` đã nạp nó: ``LoadLibrary`` theo tên trả mô-đun cùng tên đã nạp.) Sửa: tạo bí danh
    ``nvinfer_plugin.dll`` (liên kết cứng — không tốn chỗ; khác ổ đĩa thì chép), NẠP SẴN nó (đường tuyệt đối) và đưa
    thư mục vào ``PATH``. Trả đường dẫn bí danh, ``None`` nếu không làm được (không phải Windows / chưa tải TensorRT).
    """
    if sys.platform != "win32":
        return None
    libs = trt_dir / "tensorrt_libs"
    source = libs / _WINDOWS_PLUGIN_DLL
    if not source.is_file():
        return None
    alias = libs / _PADDLE_RUNTIME_PLUGIN_DLL
    try:
        if not alias.is_file() or alias.stat().st_size != source.stat().st_size:
            import os
            import shutil

            temporary = alias.with_name(alias.name + ".tmp")
            temporary.unlink(missing_ok=True)
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, alias)
            logger.info("TensorRT: tạo %s (= %s) cho Paddle lúc chạy engine.", alias.name, source.name)
    except OSError as exc:
        logger.warning("TensorRT: không tạo được %s: %s", alias.name, exc)
        return None
    try:
        import os

        path = os.environ.get("PATH", "")
        if str(libs).lower() not in path.lower().split(os.pathsep):
            os.environ["PATH"] = str(libs) + os.pathsep + path
        if not _LOADED_PLUGINS:
            loader = getattr(ctypes, "WinDLL", None) or ctypes.CDLL
            _LOADED_PLUGINS.append(loader(str(alias)))
    except OSError as exc:
        logger.warning("TensorRT: không nạp được %s: %s", alias, exc)
        return None
    return alias


class _PluginNameCtypes(ModuleType):
    """``ctypes`` cho riêng ``paddle.tensorrt.converter``: ``CDLL('libnvinfer_plugin.so')`` → DLL plugin Windows."""

    def __init__(self, plugin_dll: Path) -> None:
        super().__init__("ctypes")
        self._plugin_dll = plugin_dll

    def CDLL(self, name: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: N802 — đúng tên API ctypes
        if str(name) == _LINUX_PLUGIN_NAME:
            name = str(self._plugin_dll)
        return ctypes.CDLL(name, *args, **kwargs)

    def __getattr__(self, attribute: str) -> Any:
        return getattr(ctypes, attribute)


def patch_paddle_tensorrt_plugin_loader(trt_dir: Path) -> bool:
    """[v3.23.756] Cho đường TensorRT mới của Paddle (PIR-TRT) chạy được trên Windows.

    Log 19/11 11:15 (bản 3.23.755, đã tải TensorRT): "Failed to load dynlib/dll 'libnvinfer_plugin.so'" —
    ``paddle/tensorrt/converter.py`` (Paddle 3.3.1 cu129 Windows, đọc từ đúng gói cài) gọi cứng
    ``ctypes.CDLL('libnvinfer_plugin.so')`` (tên thư viện của Linux) rồi ``initLibNvInferPlugins``. Thay ``ctypes`` CỦA
    RIÊNG mô-đun đó bằng lớp bọc đổi tên sang ``nvinfer_plugin_10.dll`` đã tải (không đụng ``ctypes`` toàn cục / tệp của
    Paddle). ``True`` khi đã vá (Windows + có DLL + nạp được mô-đun).
    """
    if sys.platform != "win32":
        return False
    # [v3.23.763] Cùng MỘT tệp cho bước dựng (Python) và bước chạy engine (C++): một bản plugin trong tiến trình.
    plugin = ensure_paddle_runtime_plugin(trt_dir) or trt_dir / "tensorrt_libs" / _WINDOWS_PLUGIN_DLL
    if not plugin.is_file():
        return False
    try:
        import importlib

        converter = importlib.import_module("paddle.tensorrt.converter")
    except Exception as exc:  # noqa: BLE001 — để bước dựng mô hình báo lỗi thật (rồi lùi GPU thường)
        logger.debug("Không nạp được paddle.tensorrt.converter: %s", exc)
        return False
    if not isinstance(getattr(converter, "ctypes", None), _PluginNameCtypes):
        converter.ctypes = _PluginNameCtypes(plugin)
        logger.info("TensorRT: nạp plugin %s cho paddle.tensorrt (Paddle gọi tên Linux %s).", plugin.name,
                    _LINUX_PLUGIN_NAME)
    return True


#: Tham số kiểu ``Weights`` (vị trí sau ``self``, tên) của các hàm ``INetworkDefinition`` mà Paddle truyền mảng NumPy thẳng
#: vào (bindings tự chuyển — chính bước chuyển hỏng với NumPy 2).
_WEIGHT_PARAMETERS: Final[dict[str, tuple[tuple[int, str], ...]]] = {
    "add_constant": ((1, "weights"),),
    "add_scale": ((2, "shift"), (3, "scale"), (4, "power")),
    "add_scale_nd": ((2, "shift"), (3, "scale"), (4, "power")),
    "add_convolution_nd": ((3, "kernel"), (4, "bias")),
    "add_deconvolution_nd": ((3, "kernel"), (4, "bias")),
}
_NUMPY_TO_TRT: Final[dict[str, str]] = {
    "float32": "FLOAT", "float16": "HALF", "int32": "INT32", "int64": "INT64", "int8": "INT8", "uint8": "UINT8",
    "bool": "BOOL",
}
#: Mảng NumPy mà trọng số TensorRT trỏ tới (không sao chép): giữ sống tới khi dựng xong engine.
_KEEP_ALIVE: list[Any] = []
#: [v3.23.760] Lỗi TensorRT gần nhất (bộ ghi log của ứng dụng) — đưa vào log khi dựng engine lỗi.
_RECENT_ERRORS: list[str] = []
_RECENT_ERRORS_MAX: Final[int] = 20


#: Giới hạn tên lớp mạng của TensorRT là 4096 ký tự; Paddle đặt tên bằng cả biểu thức lồng nhau (log 19/11 13:30: hàng
#: loạt "ILayer::setName: … exceeds the limit (4096characters)").
_LAYER_NAME_LIMIT: Final[int] = 4000


def _short_layer_name(name: Any) -> Any:
    if not isinstance(name, str) or len(name) <= _LAYER_NAME_LIMIT:
        return name
    import hashlib

    digest = hashlib.sha1(name.encode("utf-8", "replace")).hexdigest()[:12]
    return f"{name[:3000]}…#{digest}…{name[-900:]}"


def _install_paddle_trt10_fixes(trt: Any) -> None:
    """[v3.23.761] Hai chỗ Paddle 3.3.1 (``paddle.tensorrt``) chưa khớp TensorRT 10 — log 19/11 13:30 (bản 3.23.760):

    * Bộ dò chữ dựng engine xong (68 s); bộ nhận dạng lỗi ở ``pd_op.stack``: "concat input tensors 0 and 1 have
      incompatible types Int32 and Int64" → tensor số chiều -1 → "__len__() should return >= 0". Từ TensorRT 10,
      ``add_shape`` trả kích thước kiểu **Int64**; Paddle tạo MỌI hằng kích thước kiểu Int32 (``add_1D_constant_layer``
      mặc định ``np.int32``) và tự ghi trong ``converter_utils.trt_shape``: "Many existing paddle op kernels only support
      input shape tensor as int32 … cast back to int32" — nhưng 18 chỗ gọi ``network.add_shape`` thẳng (``stack``,
      ``append_ones`` dùng cho broadcast, …) không qua bước ép đó. Sửa: ``add_shape`` trả lớp ép kiểu Int32 (đúng việc
      ``trt_shape`` làm; kích thước ảnh luôn vừa Int32).
    * Tên lớp mạng dài hơn 4096 ký tự → TensorRT báo lỗi (tên không đặt được, lỗi thật bị lấn trong log) → rút gọn
      (giữ đầu + mã băm + cuối).
    """
    network_cls = trt.INetworkDefinition
    add_shape = getattr(network_cls, "add_shape", None)
    int32, int64 = getattr(trt, "int32", None), getattr(trt, "int64", None)
    shape_fix_on = __import__("os").environ.get("SE_TRT_SHAPE_FIX", "1") != "0"     # [v3.23.770] chẩn đoán
    if shape_fix_on and add_shape is not None and int64 is not None and not getattr(add_shape, "_se_trt10_fix", False):
        import functools

        def add_shape_int32(self: Any, *args: Any, **kwargs: Any) -> Any:
            layer = add_shape(self, *args, **kwargs)
            output = layer.get_output(0) if layer is not None else None
            if output is None or output.dtype != int64:
                return layer
            if hasattr(self, "add_cast"):
                return self.add_cast(output, int32)
            cast = self.add_identity(output)
            cast.set_output_type(0, int32)
            return cast

        wrapped = functools.wraps(add_shape)(add_shape_int32)
        wrapped._se_trt10_fix = True  # type: ignore[attr-defined]
        network_cls.add_shape = wrapped
    layer_cls = getattr(trt, "ILayer", None)
    name_property = getattr(layer_cls, "__dict__", {}).get("name") if layer_cls is not None else None
    if isinstance(name_property, property) and name_property.fset is not None and not getattr(
            name_property.fset, "_se_trt10_fix", False):
        setter = name_property.fset

        def set_short_name(self: Any, value: Any) -> None:
            setter(self, _short_layer_name(value))

        set_short_name._se_trt10_fix = True  # type: ignore[attr-defined]
        assert layer_cls is not None
        layer_cls.name = property(name_property.fget, set_short_name, name_property.fdel, name_property.__doc__)


#: [v3.23.764] Lô lớn nhất engine TensorRT nhận (chiều lô của hình dạng MAX trong ``trt_dynamic_shapes`` của
#: ``inference.yml``): PP-OCRv6_small / PP-OCRv5_mobile — dò ``[1,3,4000,4000]``, nhận dạng ``[8,3,48,3200]``.
TRT_DEFAULT_DET_MAX_BATCH: Final[int] = 1
TRT_DEFAULT_REC_MAX_BATCH: Final[int] = 8


def trt_max_batch(model_config: Any, default: int) -> int:
    """[v3.23.764] Chiều lô lớn nhất của engine TensorRT theo cấu hình mô hình PaddleX (``inference.yml``).

    Log 19/11 14:46 (bản 3.23.763): engine TensorRT chạy được ("PaddleOCR chạy GPU + TensorRT") rồi SẬP tiến trình ngay
    lần suy luận đầu ("Windows fatal exception: access violation" trong ``predictor.run()`` của bộ dò): ứng dụng đẩy 64
    ảnh/lô vào bộ dò (``det_batch_size``) và 16 dòng/lô vào bộ nhận dạng, trong khi engine dựng cho lô tối đa 1 và 8
    (log 13:30: đầu vào engine ``Float[1,3,-1,-1]``). Không đọc được → ``default``.
    """
    try:
        shapes = model_config["Hpi"]["backend_configs"]["paddle_infer"]["trt_dynamic_shapes"]
        batches = [int(profile[2][0]) for profile in shapes.values()]
        return max(1, min(batches)) if batches else default
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return default


#: [v3.23.765] Hồ sơ kích thước engine TensorRT của bộ DÒ chữ (thay ``[1,3,4000,4000]`` — lô 1 — của ``inference.yml``).
#: Ảnh vào bộ dò của ứng dụng là dải phụ đề: phim 720×1280, ROI 720×118 phóng lên cao 256 px + viền 8 → ~1600×288
#: (bội số 32). Lô 8, ảnh tới 2048×512: TensorRT cấp sẵn bộ nhớ cho hình dạng LỚN NHẤT (lô × cao × rộng), nên lô / kích
#: thước lớn hơn dễ vượt 8 GB VRAM khi chạy cùng bộ nhận dạng + NVDEC. Ảnh vượt → lỗi Python → lùi GPU thường.
#: [v3.23.775] Lô 8 → 16 (chủ dự án: "Dựng engine nhận lô lớn hơn 8"): log 19/11 22:17 — TensorRT trực tiếp chạy, bộ dò
#: 64 ảnh/lô phải chia 8 lô con. Kích thước ảnh giữ nguyên (VRAM tăng theo lô; context không cấp được → GPU thường).
TRT_DET_PROFILE: Final[dict[str, list[list[int]]]] = {
    "x": [[1, 3, 32, 32], [16, 3, 288, 1600], [16, 3, 512, 2048]],
}
#: [v3.23.775] Bộ nhận dạng: ``inference.yml`` dựng lô tối đa 8 (``[8,3,48,3200]``), ứng dụng đưa 16 dòng/lô
#: (``text_recognition_batch_size``) → chia 2. Nay lô tối ưu 16, tối đa 32; chiều cao / rộng giữ của PaddleX.
TRT_REC_OPT_BATCH: Final[int] = 16
TRT_REC_MAX_BATCH: Final[int] = 32

#: [v3.23.776] Cỡ lô engine chọn trong Cài đặt → Phần cứng (chủ dự án: "TensorRT nhận lô sao không đưa vô cài đặt để
#: người dùng chọn và có kèm gợi ý"). Ghi vào biến môi trường để tiến trình tự kiểm (``spawn``) dựng đúng cỡ đó.
TRT_DET_BATCH_ENV: Final[str] = "SE_TRT_DET_BATCH"
TRT_REC_BATCH_ENV: Final[str] = "SE_TRT_REC_BATCH"
TRT_REC_OPT_ENV: Final[str] = "SE_TRT_REC_OPT"
TRT_DET_BATCH_RANGE: Final[tuple[int, int]] = (1, 64)
TRT_REC_BATCH_RANGE: Final[tuple[int, int]] = (8, 128)
#: (VRAM tối thiểu GB, lô bộ dò, lô bộ nhận dạng) — engine giữ sẵn VRAM cho lô lớn nhất (ảnh tới 2048×512 / dòng tới
#: 3200 px), chạy cùng NVDEC + bộ soát câu. 8 GB: số đã chạy được trên RTX 3070 Ti (bản 3.23.775).
_TRT_BATCH_TABLE: Final[tuple[tuple[float, int, int], ...]] = (
    (15.0, 64, 128), (11.0, 32, 64), (7.0, 16, 32), (5.0, 8, 16), (0.0, 4, 8),
)


def _env_int(name: str, default: int, bounds: tuple[int, int]) -> int:
    try:
        value = int(__import__("os").environ.get(name, "") or default)
    except ValueError:
        value = default
    return max(bounds[0], min(bounds[1], value))


def trt_batches() -> tuple[int, int, int]:
    """[v3.23.776] (lô bộ dò, lô tối đa bộ nhận dạng, lô tối ưu bộ nhận dạng) đang dùng để dựng engine."""
    det = _env_int(TRT_DET_BATCH_ENV, int(TRT_DET_PROFILE["x"][2][0]), TRT_DET_BATCH_RANGE)
    rec = _env_int(TRT_REC_BATCH_ENV, TRT_REC_MAX_BATCH, TRT_REC_BATCH_RANGE)
    opt = max(1, min(rec, _env_int(TRT_REC_OPT_ENV, TRT_REC_OPT_BATCH, (1, TRT_REC_BATCH_RANGE[1]))))
    return det, rec, opt


def set_trt_batches(det: int, rec: int, rec_opt: int) -> tuple[int, int, int]:
    """[v3.23.776] Đặt cỡ lô engine từ cài đặt (kẹp trong giới hạn); trả giá trị đã đặt."""
    import os

    os.environ[TRT_DET_BATCH_ENV] = str(max(TRT_DET_BATCH_RANGE[0], min(TRT_DET_BATCH_RANGE[1], int(det))))
    os.environ[TRT_REC_BATCH_ENV] = str(max(TRT_REC_BATCH_RANGE[0], min(TRT_REC_BATCH_RANGE[1], int(rec))))
    os.environ[TRT_REC_OPT_ENV] = str(max(1, int(rec_opt)))
    return trt_batches()


def suggest_trt_batches(vram_gb: float | None) -> tuple[int, int]:
    """[v3.23.776] Gợi ý (lô bộ dò, lô bộ nhận dạng) theo tổng VRAM; không dò được → mức của card 8 GB."""
    if vram_gb is None:
        return 16, 32
    for minimum, det, rec in _TRT_BATCH_TABLE:
        if vram_gb >= minimum:
            return det, rec
    return 4, 8


def det_profile() -> dict[str, list[list[int]]]:
    """[v3.23.776] :data:`TRT_DET_PROFILE` với chiều lô (tối ưu = tối đa) theo cài đặt."""
    batch = trt_batches()[0]
    return {name: [list(low), [batch, *opt[1:]], [batch, *high[1:]]]
            for name, (low, opt, high) in TRT_DET_PROFILE.items()}


def rec_profile(shapes: Any) -> dict[str, list[list[int]]] | None:
    """[v3.23.775] Hồ sơ bộ nhận dạng của PaddleX với chiều lô tối ưu / tối đa theo cài đặt ([v3.23.776]; mặc định
    :data:`TRT_REC_OPT_BATCH` / :data:`TRT_REC_MAX_BATCH`) — không đọc được → ``None``, để nguyên."""
    _det, maximum, optimum = trt_batches()
    try:
        result = {}
        for name, (low, opt, high) in shapes.items():
            result[str(name)] = [list(map(int, low)),
                                 [max(int(opt[0]), optimum), *map(int, opt[1:])],
                                 [max(int(high[0]), maximum), *map(int, high[1:])]]
        return result or None
    except (AttributeError, TypeError, ValueError):
        return None
_PROFILE_MARKER: Final[str] = "se_profile.json"


def _profile_signature(config: dict[str, Any]) -> str:
    import json

    return json.dumps({
        "shapes": config.get("trt_dynamic_shapes"), "run_mode": config.get("run_mode"),
        "trt_cfg": config.get("trt_cfg_setting"), "app_profile": 2,                  # [v3.23.775] lô 16 / 32
        # [v3.23.770] Engine dựng KHÔNG có bản vá Int32 (phép thử chẩn đoán) không được dùng lại cho ứng dụng.
        "shape_fix": __import__("os").environ.get("SE_TRT_SHAPE_FIX", "1"),
    }, sort_keys=True, default=str)


def _profile_limits(shapes: Any) -> dict[str, tuple[list[int], list[int]]]:
    try:
        return {str(name): (list(map(int, profile[0])), list(map(int, profile[2]))) for name, profile in shapes.items()}
    except (AttributeError, IndexError, TypeError, ValueError):
        return {}


def check_and_split_inputs(limits: dict[str, tuple[list[int], list[int]]], arrays: list[Any]) -> list[list[Any]] | None:
    """[v3.23.765] Trước khi đưa vào engine TensorRT: chia lô vượt lô tối đa thành các lô con; kích thước ngoài hồ sơ →
    ``RuntimeError`` (bộ chuyển đổi OCR lùi GPU thường) thay vì để C++ ghi tràn bộ nhớ và SẬP tiến trình (log 19/11
    14:46). ``None`` = không có hồ sơ (để nguyên)."""
    if len(limits) != 1 or len(arrays) != 1:
        return None
    (name, (low, high)), = limits.items()
    array = arrays[0]
    shape = list(getattr(array, "shape", ()))
    if len(shape) != len(high):
        raise RuntimeError(f"TensorRT: đầu vào {name} có {len(shape)} chiều, engine dựng cho {len(high)} chiều")
    for axis in range(1, len(shape)):
        if not low[axis] <= shape[axis] <= high[axis]:
            raise RuntimeError(f"TensorRT: đầu vào {name} {shape} ngoài hồ sơ engine {low}–{high} (chiều {axis})")
    step = max(1, high[0])
    return [[array[start:start + step]] for start in range(0, max(1, shape[0]), step)]


def install_paddlex_trt_profile_patch() -> bool:
    """[v3.23.765] Engine bộ dò cho lô lớn hơn + lưới an toàn hình dạng cho MỌI engine TensorRT của PaddleX.

    * ``PaddleStaticRunner._configure_trt``: bộ dò (``*_det``) dùng :data:`TRT_DET_PROFILE`; cache engine
      (``<mô hình>/.cache/paddle/trt``) gắn với hồ sơ + chế độ (``trt_fp32/fp16``) + cấu hình — khác thì xoá, dựng lại
      (PaddleX chỉ kiểm tệp có tồn tại: đổi hồ sơ / FP16↔FP32 từng dùng lại engine CŨ).
    * ``PaddleStaticRunner.__call__``: lô vượt lô tối đa → chia lô con, nối kết quả; kích thước ngoài hồ sơ → lỗi Python.
    """
    try:
        import importlib

        runner_module = importlib.import_module("paddlex.inference.models.runners.paddle_static.runner")
    except Exception as exc:  # noqa: BLE001 — chưa có PaddleX: không vá
        logger.debug("Không nạp được PaddleX runner: %s", exc)
        return False
    runner_cls = getattr(runner_module, "PaddleStaticRunner", None)
    if runner_cls is None or getattr(runner_cls, "_se_trt_profile_patch", False):
        return runner_cls is not None
    configure, call = runner_cls._configure_trt, runner_cls.__call__
    import functools
    import shutil

    import numpy as np

    def configure_trt(self: Any, model_file: Any, params_file: Any, cache_dir: Any) -> Any:
        config = self._config
        det_profile_on = __import__("os").environ.get("SE_TRT_DET_PROFILE", "1") != "0"   # [v3.23.770] chẩn đoán
        if (det_profile_on and str(getattr(self, "_model_name", "")).endswith("_det")
                and getattr(runner_module, "USE_PIR_TRT", True)):
            config["trt_dynamic_shapes"] = det_profile()                              # [v3.23.776] lô theo cài đặt
        elif (__import__("os").environ.get("SE_TRT_REC_PROFILE", "1") != "0"
              and str(getattr(self, "_model_name", "")).endswith("_rec") and getattr(runner_module, "USE_PIR_TRT", True)):
            widened = rec_profile(config.get("trt_dynamic_shapes"))
            if widened is not None:
                config["trt_dynamic_shapes"] = widened
        signature = _profile_signature(config)
        trt_dir = Path(cache_dir) / "trt"
        marker = trt_dir / _PROFILE_MARKER
        try:
            stale = trt_dir.exists() and (not marker.is_file() or marker.read_text(encoding="utf-8") != signature)
            if stale:
                logger.info("TensorRT: engine %s trong cache dựng theo hồ sơ khác — xoá để dựng lại.",
                            getattr(self, "_model_name", "?"))
                shutil.rmtree(trt_dir, ignore_errors=True)
        except OSError as exc:
            logger.debug("Không kiểm được cache TensorRT: %s", exc)
        # [v3.23.768] Ghi rõ dùng cache hay phải CHUYỂN mô hình (bước chuyển gọi ``load_inference_model`` — log 19/11 17:23:
        # tự kiểm lỗi đúng ở đó dù engine đã dựng ở lượt 3.23.765).
        prefix = str(getattr(self, "model_file_prefix", "inference"))
        cached = (trt_dir / f"{prefix}.json").is_file() and (trt_dir / f"{prefix}.pdiparams").is_file()
        logger.info("TensorRT %s: %s (%s, luồng %s).", getattr(self, "_model_name", "?"),
                    "dùng engine trong cache" if cached else "chưa có engine trong cache — chuyển mô hình",
                    trt_dir, threading.current_thread().name)
        result = configure(self, model_file, params_file, cache_dir)
        try:
            trt_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(signature, encoding="utf-8")
        except OSError as exc:
            logger.debug("Không ghi được hồ sơ cache TensorRT: %s", exc)
        self._se_trt_limits = _profile_limits(config.get("trt_dynamic_shapes"))
        logger.info("TensorRT %s: hồ sơ %s.", getattr(self, "_model_name", "?"), config.get("trt_dynamic_shapes"))
        return result

    def guarded_call(self: Any, x: Any) -> Any:
        limits = getattr(self, "_se_trt_limits", None)
        if not limits:
            return call(self, x)
        chunks = check_and_split_inputs(limits, list(x))
        if chunks is None or len(chunks) == 1:
            return call(self, x)
        outputs = [call(self, chunk) for chunk in chunks]
        return [np.concatenate([part[index] for part in outputs], axis=0) for index in range(len(outputs[0]))]

    runner_cls._configure_trt = functools.wraps(configure)(configure_trt)
    runner_cls.__call__ = functools.wraps(call)(guarded_call)
    runner_cls._se_trt_profile_patch = True
    return True


_CRASH_GUARD_NAME: Final[str] = "first_inference_pending.txt"


def crash_guard_path(trt_dir: Path) -> Path:
    """[v3.23.764] Dấu "đang chạy lần suy luận TensorRT đầu tiên" — còn lại sau khi mở lại ứng dụng = lần trước SẬP."""
    return trt_dir / _CRASH_GUARD_NAME


def crashed_on_previous_first_inference(trt_dir: Path, app_version: str) -> bool:
    """``True`` khi lần chạy TRƯỚC của CÙNG bản ứng dụng sập giữa lần suy luận TensorRT đầu tiên. Dấu của bản khác (đã
    nâng cấp — có thể đã sửa) bị xoá để thử lại."""
    marker = crash_guard_path(trt_dir)
    try:
        if not marker.is_file():
            return False
        if marker.read_text(encoding="utf-8").strip() == app_version:
            return True
        marker.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def begin_first_inference(trt_dir: Path, app_version: str) -> None:
    try:
        crash_guard_path(trt_dir).write_text(app_version, encoding="utf-8")
    except OSError as exc:
        logger.debug("Không ghi được dấu chống sập TensorRT: %s", exc)


def end_first_inference(trt_dir: Path) -> None:
    try:
        crash_guard_path(trt_dir).unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Không xoá được dấu chống sập TensorRT: %s", exc)


def tensorrt_recent_errors() -> list[str]:
    """[v3.23.760] Các thông báo lỗi TensorRT gần nhất (cũ → mới) do bộ ghi log của ứng dụng nhận."""
    return list(_RECENT_ERRORS)


def _install_app_logger(trt: Any) -> None:
    """[v3.23.760] ``trt.Logger`` → bộ ghi chuyển thông báo TensorRT vào log ứng dụng.

    Log 19/11 12:39 (bản 3.23.758): dựng TensorRT lỗi "__len__() should return >= 0" (một tensor mang số chiều -1 —
    lớp mạng phía trước dựng LỖI) nhưng lỗi gốc của TensorRT mất: Paddle tạo ``trt.Logger(trt.Logger.ERROR)`` — bộ ghi
    mặc định in ra stderr, exe cửa sổ không có stderr.
    """
    base_logger = getattr(trt, "Logger", None)
    if base_logger is None or getattr(base_logger, "_se_app_logger", False) or not hasattr(trt, "ILogger"):
        return
    severities = {name: getattr(base_logger, name) for name in ("INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "VERBOSE")
                  if hasattr(base_logger, name)}
    rank = {value: position for position, value in enumerate(severities.values())}
    errors = {severities.get("INTERNAL_ERROR"), severities.get("ERROR")}

    class Logger(trt.ILogger):  # type: ignore[misc,unused-ignore]
        """Bộ ghi log TensorRT của ứng dụng (cùng cách tạo ``Logger(mức)`` như ``tensorrt.Logger``)."""

        _se_app_logger = True

        def __init__(self, min_severity: Any = None) -> None:
            trt.ILogger.__init__(self)
            self.min_severity = severities.get("WARNING") if min_severity is None else min_severity

        def log(self, severity: Any, msg: str) -> None:
            try:
                if severity in errors:
                    _RECENT_ERRORS.append(str(msg)[:500])
                    del _RECENT_ERRORS[:-_RECENT_ERRORS_MAX]
                    logger.warning("[TensorRT] %s", msg)
                elif rank.get(severity, 99) <= rank.get(self.min_severity, 2):
                    logger.info("[TensorRT] %s", msg)
                elif severity == severities.get("WARNING"):
                    logger.debug("[TensorRT] %s", msg)
                # [v3.23.761] INFO / VERBOSE dưới mức Paddle chọn: bỏ (log 19/11 13:30: ~19 700 dòng, tệp 12,7 MB — mất phần đầu)
            except Exception:  # noqa: BLE001 — gọi lại từ C++: không được ném ngoại lệ
                pass

    for name, value in severities.items():
        setattr(Logger, name, value)
    if hasattr(base_logger, "Severity"):
        Logger.Severity = base_logger.Severity
    trt.Logger = Logger


def patch_tensorrt_numpy_bridge(module: Any | None = None) -> bool:
    """[v3.23.758] Cầu NumPy ↔ TensorRT không phụ thuộc phần pybind11 cũ của bindings TensorRT 10.x.

    Log 19/11 11:32 (bản 3.23.756, plugin đã nạp đúng): "[TRT] [E] Could not implicitly convert NumPy data type: i0 to
    TensorRT" → lùi GPU thường. Nguyên nhân gốc: bindings ``tensorrt`` 10.9 / 10.11 / 10.13 (Windows, cp313 — đọc chuỗi
    trong ``tensorrt.cp313-win_amd64.pyd``) dựng bằng pybind11 đời cũ (``numpy.core.multiarray``, chưa biết NumPy 2): đọc
    cỡ phần tử của kiểu NumPy theo bố cục NumPy 1 → NumPy 2 (ứng dụng: 2.x, Python 3.13 bắt buộc NumPy 2) ra 0 byte
    ("i0" = số nguyên 0 bit). Chiều ngược còn tệ hơn — kiểm trên bindings 10.9 thật (Linux, NumPy 2.3.5):
    ``Weights(int64, [1, 2, 3]).numpy()`` trả ``[1 1 1]`` (bước nhảy 0) KHÔNG báo lỗi; Paddle gọi ``.numpy()`` khi gộp
    BatchNorm (``paddle/tensorrt/impls/norm.py``) → engine dựng được nhưng trọng số SAI, OCR sai âm thầm.

    Sửa (không đụng tệp của Paddle / TensorRT): ``tensorrt.Weights`` thành lớp con — mảng NumPy đi qua hàm dựng
    ``Weights(DataType, con trỏ, số phần tử)`` (không qua pybind11 đọc kiểu), giữ sống mảng, ``numpy()`` trả đúng mảng
    đó; các hàm ``add_constant`` / ``add_scale(_nd)`` / ``add_(de)convolution_nd`` đổi mảng NumPy truyền thẳng vào thành
    lớp con trước khi gọi. Bộ giữ sống được dọn sau mỗi lần dựng engine. ``True`` khi đã vá (hoặc đã vá từ trước).
    """
    if module is None:
        try:
            import importlib

            module = importlib.import_module("tensorrt")
        except Exception as exc:  # noqa: BLE001 — chưa có TensorRT: bước dựng tự báo lỗi thật
            logger.debug("Chưa nạp được tensorrt: %s", exc)
            return False
    trt: Any = module
    base = trt.Weights
    try:
        _install_app_logger(trt)
    except Exception as exc:  # noqa: BLE001 — chỉ là chẩn đoán: không được chặn cầu NumPy
        logger.debug("Không cài được bộ ghi log TensorRT: %s", exc)
    try:
        _install_paddle_trt10_fixes(trt)
    except Exception as exc:  # noqa: BLE001 — vá phụ: lỗi thì bước dựng tự báo lỗi thật (rồi lùi GPU thường)
        logger.warning("Không vá được paddle.tensorrt cho TensorRT 10: %s", exc)
    if getattr(base, "_se_numpy_bridge", False):
        # [v3.23.759] Gọi trước MỖI lần dựng mô hình (không có engine nào đang dựng): trả mảng còn giữ từ lần chuyển
        # đồ thị bị lỗi giữa chừng (lỗi đó không tới bước dựng nên không được dọn).
        _KEEP_ALIVE.clear()
        return True
    import functools

    import numpy as np

    data_types = {name: getattr(trt.DataType, member) for name, member in _NUMPY_TO_TRT.items()
                  if hasattr(trt.DataType, member)}

    class Weights(base):  # type: ignore[misc,valid-type]
        """``tensorrt.Weights`` dựng từ mảng NumPy KHÔNG qua pybind11 đọc kiểu (xem :func:`patch_tensorrt_numpy_bridge`)."""

        _se_numpy_bridge = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            array = None
            if len(args) == 1 and not kwargs:
                array = args[0]
            elif not args and set(kwargs) == {"a"}:
                array = kwargs["a"]
            if isinstance(array, np.ndarray) and array.dtype.name in data_types:
                # TensorRT đọc byte theo thứ tự của máy: mảng khác thứ tự byte phải đổi trước (Paddle không tạo, phòng xa).
                contiguous = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("="))
                super().__init__(data_types[contiguous.dtype.name], int(contiguous.ctypes.data), int(contiguous.size))
                self._se_array: Any = contiguous
                _KEEP_ALIVE.append(contiguous)
            else:
                super().__init__(*args, **kwargs)
                self._se_array = None

        def numpy(self) -> Any:
            kept = getattr(self, "_se_array", None)
            return kept.reshape(-1) if kept is not None else super().numpy()

    Weights.__name__ = Weights.__qualname__ = "Weights"
    Weights.__module__ = getattr(base, "__module__", "tensorrt")

    def as_weights(value: Any) -> Any:
        return Weights(value) if isinstance(value, np.ndarray) else value

    network_cls = trt.INetworkDefinition
    for name, parameters in _WEIGHT_PARAMETERS.items():
        original = getattr(network_cls, name, None)
        if original is None or getattr(original, "_se_numpy_bridge", False):
            continue                                 # [v3.23.759] lần vá trước dừng giữa chừng: không bọc chồng

        def call(self: Any, *args: Any, _original: Any = original, _parameters: Any = parameters,
                 **kwargs: Any) -> Any:
            converted = list(args)
            for position, keyword in _parameters:
                if position < len(converted):
                    converted[position] = as_weights(converted[position])
                elif keyword in kwargs:
                    kwargs[keyword] = as_weights(kwargs[keyword])
            return _original(self, *converted, **kwargs)

        wrapped = functools.wraps(original)(call)
        wrapped._se_numpy_bridge = True  # type: ignore[attr-defined]
        setattr(network_cls, name, wrapped)

    build = getattr(trt.Builder, "build_serialized_network", None)
    if build is not None and not getattr(build, "_se_numpy_bridge", False):
        def build_and_release(self: Any, *args: Any, _build: Any = build, **kwargs: Any) -> Any:
            try:
                return _build(self, *args, **kwargs)
            finally:
                _KEEP_ALIVE.clear()                  # engine đã dựng (chép trọng số) — trả bộ nhớ các mảng

        wrapped_build = functools.wraps(build)(build_and_release)
        wrapped_build._se_numpy_bridge = True  # type: ignore[attr-defined]
        trt.Builder.build_serialized_network = wrapped_build
    trt.Weights = Weights
    bindings = sys.modules.get("tensorrt_bindings")
    if bindings is not None and getattr(bindings, "Weights", None) is base:
        bindings.Weights = Weights  # type: ignore[attr-defined]
    logger.info("TensorRT %s: dùng cầu NumPy ↔ TensorRT của ứng dụng (bindings TensorRT chưa hỗ trợ NumPy %s).",
                getattr(trt, "__version__", "?"), np.__version__)
    return True


__all__ = [
    "TENSORRT_DOWNLOAD_GB", "TENSORRT_PACKAGES", "TENSORRT_RUNTIME_DIRNAME", "TENSORRT_VERSION",
    "TRT_DEFAULT_DET_MAX_BATCH", "TRT_DEFAULT_REC_MAX_BATCH", "TRT_DET_PROFILE", "TRT_REC_MAX_BATCH",
    "TRT_REC_OPT_BATCH", "check_and_split_inputs", "det_profile", "rec_profile", "set_trt_batches",
    "suggest_trt_batches", "trt_batches",
    "install_paddlex_trt_profile_patch", "begin_first_inference", "crash_guard_path",
    "crashed_on_previous_first_inference", "end_first_inference", "trt_max_batch",
    "build_tensorrt_install_command", "ensure_paddle_runtime_plugin", "is_tensorrt_installed", "patch_paddle_tensorrt_plugin_loader",
    "patch_tensorrt_numpy_bridge", "tensorrt_recent_errors",
    "tensorrt_dll_dirs", "write_tensorrt_shim",
]
