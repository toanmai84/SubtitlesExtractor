"""[v3.23.758] Cầu NumPy ↔ TensorRT: bindings TensorRT 10.x (pybind11 cũ) chưa hỗ trợ NumPy 2.

Log 19/11 11:32 (bản 3.23.756): "[TRT] [E] Could not implicitly convert NumPy data type: i0 to TensorRT" → lùi GPU thường.
Kiểm trên bindings TensorRT 10.9 thật (Linux, NumPy 2.3.5): ``Weights(mảng float64)`` → "f0" (cùng loại lỗi), và
``Weights(...).numpy()`` trả ``[1 1 1]`` thay vì ``[1 2 3]`` (bước nhảy 0, không báo lỗi). Test giả lập bindings bằng
lớp Python có cùng 3 hàm dựng; test cuối chạy với bindings thật khi có ``SE_TEST_TRT_DIR`` (thư mục chứa
``tensorrt_bindings`` + ``tensorrt_libs``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan


class _DataType(Enum):
    FLOAT = 0
    HALF = 1
    INT8 = 2
    INT32 = 3
    BOOL = 4
    UINT8 = 5
    INT64 = 6


def _fake_trt():  # noqa: ANN202
    calls: list = []

    class Weights:                                     # như bindings hỏng: mảng NumPy → lỗi "i0", numpy() sai
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            if args and isinstance(args[0], np.ndarray) or "a" in kwargs:
                raise ValueError("[TRT] [E] Could not implicitly convert NumPy data type: i0 to TensorRT.")
            self.dtype = args[0] if args else _DataType.FLOAT
            self.ptr, self.size = (args[1], args[2]) if len(args) == 3 else (0, 0)

        def numpy(self):  # noqa: ANN202
            return "bindings-numpy"

    class Network:
        def add_constant(self, shape, weights):  # noqa: ANN001, ANN202
            calls.append(("add_constant", shape, weights))
            if isinstance(weights, np.ndarray):
                raise ValueError("[TRT] [E] Could not implicitly convert NumPy data type: i0 to TensorRT.")
            return weights

        def add_scale(self, input, mode, shift=None, scale=None, power=None):  # noqa: A002, ANN001, ANN202
            calls.append(("add_scale", shift, scale, power))
            return (shift, scale, power)

        def add_convolution_nd(self, input, num_output_maps, kernel_shape, kernel, bias=None):  # noqa: A002, ANN001, ANN202
            calls.append(("add_convolution_nd", kernel_shape, kernel, bias))
            return kernel_shape, kernel, bias

    class Builder:
        def build_serialized_network(self, network, config):  # noqa: ANN001, ANN202
            calls.append(("build", len(plan._KEEP_ALIVE)))
            return b"engine"

    trt = SimpleNamespace(Weights=Weights, DataType=_DataType, INetworkDefinition=Network, Builder=Builder,
                          __version__="10.9.0.34")
    return trt, calls


@pytest.fixture(autouse=True)
def _clear_keep_alive():  # noqa: ANN202
    plan._KEEP_ALIVE.clear()
    yield
    plan._KEEP_ALIVE.clear()


def test_numpy_weights_use_the_pointer_constructor_and_roundtrip() -> None:
    trt, _calls = _fake_trt()
    base = trt.Weights
    assert plan.patch_tensorrt_numpy_bridge(trt) is True
    array = np.arange(6, dtype=np.int64).reshape(2, 3)
    weights = trt.Weights(array)                                       # mã cũ: ValueError "i0"
    assert type(weights) == trt.Weights and isinstance(weights, base)  # Paddle so ``type(x) == trt.Weights``
    assert weights.dtype is _DataType.INT64 and weights.size == 6
    assert np.array_equal(weights.numpy(), array.reshape(-1))         # bindings: "bindings-numpy" (sai dữ liệu)
    assert weights.ptr == weights._se_array.ctypes.data
    assert trt.Weights(a=np.ones(2, np.float32)).dtype is _DataType.FLOAT
    assert trt.Weights(_DataType.HALF).dtype is _DataType.HALF and trt.Weights(_DataType.HALF).numpy() == "bindings-numpy"


def test_non_contiguous_array_is_kept_alive_and_copied_once() -> None:
    trt, _calls = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    view = np.arange(10, dtype=np.float32)[::2]
    weights = trt.Weights(view)
    assert np.array_equal(weights.numpy(), [0, 2, 4, 6, 8]) and weights._se_array.flags["C_CONTIGUOUS"]
    assert any(kept is weights._se_array for kept in plan._KEEP_ALIVE)


def test_network_methods_convert_arrays_passed_directly() -> None:
    trt, calls = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    network = trt.INetworkDefinition()
    constant = network.add_constant((1, 3), np.array([1, 2, 3], np.int64))   # mã cũ: ValueError "i0"
    assert type(constant) == trt.Weights and np.array_equal(constant.numpy(), [1, 2, 3])
    assert type(network.add_constant(shape=(1,), weights=np.array([1], np.int32))) == trt.Weights
    shift, scale, power = network.add_scale("x", "mode", np.zeros(2, np.float32), scale=np.ones(2, np.float32))
    assert type(shift) == trt.Weights and type(scale) == trt.Weights and power is None
    shape_arg = np.array([3, 3])                                     # Dims có thể là mảng: KHÔNG đổi thành Weights
    kernel_shape, kernel, bias = network.add_convolution_nd("x", 8, shape_arg, np.ones(72, np.float32))
    assert kernel_shape is shape_arg and type(kernel) == trt.Weights and bias is None
    assert trt.INetworkDefinition.add_constant.__name__ == "add_constant"


def test_keep_alive_released_after_engine_build_and_patch_is_idempotent() -> None:
    trt, calls = _fake_trt()
    assert plan.patch_tensorrt_numpy_bridge(trt) and plan.patch_tensorrt_numpy_bridge(trt)
    patched = trt.Weights
    trt.Weights(np.ones(3, np.float32))
    assert trt.Builder().build_serialized_network("net", "cfg") == b"engine"
    assert calls[-1] == ("build", 1) and plan._KEEP_ALIVE == []
    assert trt.Weights is patched and patched.__mro__[1].__name__ == "Weights"   # không bọc chồng


def test_unsupported_dtype_keeps_the_bindings_behaviour() -> None:
    trt, _calls = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    with pytest.raises(ValueError, match="NumPy data type"):
        trt.Weights(np.ones(2, np.float64))                           # TensorRT không có float64 — như bindings


def test_without_tensorrt_nothing_is_patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    assert plan.patch_tensorrt_numpy_bridge() is False


def test_adapter_applies_the_bridge_before_building() -> None:
    import inspect

    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter

    source = inspect.getsource(paddle_ocr_adapter._prepare_paddle_tensorrt)
    assert "patch_tensorrt_numpy_bridge()" in source
    assert source.index("patch_paddle_tensorrt_plugin_loader(") < source.index("patch_tensorrt_numpy_bridge()")


_TRT_DIR = os.environ.get("SE_TEST_TRT_DIR", "")


@pytest.mark.skipif(not _TRT_DIR, reason="cần SE_TEST_TRT_DIR (tensorrt_bindings + tensorrt_libs 10.9 thật)")
def test_real_bindings_roundtrip() -> None:
    script = textwrap.dedent("""
        import numpy as np, tensorrt_bindings as trt
        from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan
        base = trt.Weights
        broken = base(np.array([1, 2, 3], np.int64)).numpy().tolist()
        assert plan.patch_tensorrt_numpy_bridge(trt)
        for dt in (np.int64, np.int32, np.float32, np.float16, np.bool_, np.int8, np.uint8):
            a = np.arange(6).astype(dt).reshape(2, 3)
            w = trt.Weights(a)
            assert type(w) == trt.Weights and isinstance(w, base) and w.size == 6
            assert np.array_equal(w.numpy(), a.reshape(-1)), (dt, w.numpy())
        print("broken", broken)
    """)
    src = str(Path(__file__).resolve().parents[3] / "src")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([_TRT_DIR, src]),
               LD_LIBRARY_PATH=str(Path(_TRT_DIR) / "tensorrt_libs"))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300,
                            check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "broken [1, 1, 1]" in result.stdout or "broken [1, 2, 3]" in result.stdout


# ── [v3.23.759] lượt rà độc lập 3.23.758 ───────────────────────────────────────────────────────────────────────────
def test_non_native_byte_order_is_converted_before_handing_the_pointer() -> None:
    trt, _calls = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    swapped = np.array([1, 2, 3], dtype=np.dtype(np.int32).newbyteorder("S"))
    weights = trt.Weights(swapped)
    assert weights._se_array.dtype.isnative and weights._se_array.tolist() == [1, 2, 3]


def test_half_finished_patch_does_not_wrap_twice_and_stale_arrays_are_released() -> None:
    trt, calls = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    wrapped = trt.INetworkDefinition.add_constant
    trt.Weights = trt.Weights.__mro__[1]                     # như lần vá trước dừng trước khi thay Weights
    plan.patch_tensorrt_numpy_bridge(trt)
    assert trt.INetworkDefinition.add_constant is wrapped
    trt.Weights(np.ones(4, np.float32))                      # chuyển đồ thị lỗi trước bước dựng: mảng còn giữ
    assert len(plan._KEEP_ALIVE) == 1
    plan.patch_tensorrt_numpy_bridge(trt)                    # lần dựng mô hình sau
    assert plan._KEEP_ALIVE == []
