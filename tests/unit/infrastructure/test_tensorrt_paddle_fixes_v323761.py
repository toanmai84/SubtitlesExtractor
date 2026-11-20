"""[v3.23.761] Paddle 3.3.1 (``paddle.tensorrt``) chưa khớp TensorRT 10 — log 19/11 13:30 (bản 3.23.760).

Bộ dò chữ dựng engine xong (68 s); bộ nhận dạng lỗi ở ``pd_op.stack``: "[TensorRT] ITensor::getDimensions: … concat input
tensors 0 and 1 have incompatible types Int32 and Int64" → "__len__() should return >= 0" (``converter_utils.resize_to_1d``).
TensorRT 10 ``add_shape`` trả Int64, Paddle tạo hằng kích thước Int32 và chỉ ép kiểu trong ``trt_shape`` — 18 chỗ gọi
``network.add_shape`` thẳng. Cùng log: hàng loạt "ILayer::setName: … exceeds the limit (4096characters)", ~19 700 dòng
chi tiết TensorRT (tệp log 12,7 MB, mất phần đầu).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan


class _Tensor:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


class _Layer:
    def __init__(self, kind: str, output: _Tensor) -> None:
        self.kind, self._output, self._name = kind, output, ""

    def get_output(self, index: int) -> _Tensor:
        return self._output

    def _get_name(self) -> str:
        return self._name

    def _set_name(self, value: str) -> None:
        if len(value) > 4096:
            raise AssertionError("TensorRT: name exceeds the limit (4096characters)")
        self._name = value

    name = property(_get_name, _set_name)


def _fake_trt(shape_dtype: str = "int64", with_cast: bool = True):  # noqa: ANN202
    calls: list = []

    class Network:
        def add_shape(self, input):  # noqa: A002, ANN001, ANN202
            calls.append("add_shape")
            return _Layer("shape", _Tensor(shape_dtype))

        def add_identity(self, input):  # noqa: A002, ANN001, ANN202
            calls.append("add_identity")
            layer = _Layer("identity", _Tensor(input.dtype))
            layer.set_output_type = lambda index, dtype: setattr(layer._output, "dtype", dtype)  # type: ignore[attr-defined]
            return layer

    if with_cast:
        def add_cast(self, input, to_type):  # noqa: ANN001, ANN202
            calls.append(("add_cast", input.dtype, to_type))
            return _Layer("cast", _Tensor(to_type))

        Network.add_cast = add_cast  # type: ignore[attr-defined]

    class Weights:
        def __init__(self, *args) -> None:  # noqa: ANN002
            pass

    trt = SimpleNamespace(INetworkDefinition=Network, ILayer=_Layer, Weights=Weights, DataType=SimpleNamespace(),
                          Builder=type("Builder", (), {}), int32="int32", int64="int64")
    return trt, calls


@pytest.fixture(autouse=True)
def _restore_layer_name():  # noqa: ANN202
    original = _Layer.__dict__["name"]
    yield
    _Layer.name = original  # type: ignore[method-assign,assignment]


def test_add_shape_returns_int32_like_paddles_trt_shape() -> None:
    trt, calls = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    layer = trt.INetworkDefinition().add_shape(_Tensor("float32"))     # mã cũ: lớp shape kiểu int64
    assert layer.kind == "cast" and layer.get_output(0).dtype == "int32"
    assert calls == ["add_shape", ("add_cast", "int64", "int32")]


def test_add_shape_without_add_cast_uses_identity_and_keeps_int32_shapes() -> None:
    trt, calls = _fake_trt(with_cast=False)
    plan.patch_tensorrt_numpy_bridge(trt)
    layer = trt.INetworkDefinition().add_shape(_Tensor("float32"))
    assert layer.kind == "identity" and layer.get_output(0).dtype == "int32"
    trt32, calls32 = _fake_trt(shape_dtype="int32")
    plan.patch_tensorrt_numpy_bridge(trt32)
    assert trt32.INetworkDefinition().add_shape(_Tensor("x")).kind == "shape" and calls32 == ["add_shape"]


def test_wrappers_are_not_stacked() -> None:
    trt, calls = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    wrapped, name_prop = trt.INetworkDefinition.add_shape, _Layer.__dict__["name"]
    trt.Weights = trt.Weights.__mro__[1] if len(trt.Weights.__mro__) > 2 else trt.Weights
    plan.patch_tensorrt_numpy_bridge(trt)
    assert trt.INetworkDefinition.add_shape is wrapped and _Layer.__dict__["name"] is name_prop


def test_long_layer_names_are_shortened_but_stay_distinct() -> None:
    trt, _calls = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    long_a = "620_pd_op.stack->after_shape_tensor(" + "x" * 6000 + ")A"
    long_b = "620_pd_op.stack->after_shape_tensor(" + "x" * 6000 + ")B"
    layer_a, layer_b = _Layer("a", _Tensor("f")), _Layer("b", _Tensor("f"))
    layer_a.name, layer_b.name = long_a, long_b                       # mã cũ: lỗi "exceeds the limit"
    assert len(layer_a.name) <= 4096 and layer_a.name.startswith("620_pd_op.stack") and layer_a.name != layer_b.name
    layer_a.name = "ngắn"
    assert layer_a.name == "ngắn"


def test_logger_drops_info_and_verbose(caplog: pytest.LogCaptureFixture) -> None:
    from tests.unit.infrastructure.test_tensorrt_logger_v323760 import _fake_trt as logger_trt

    trt = logger_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    lg = trt.Logger(trt.Logger.ERROR)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=plan.logger.name):
        lg.log(trt.Logger.VERBOSE, "Layer(CaskConvolution): …")
        lg.log(trt.Logger.INFO, "[MemUsageChange] …")
        lg.log(trt.Logger.WARNING, "cảnh báo")
    forwarded = [record.getMessage() for record in caplog.records if record.getMessage().startswith("[TensorRT]")]
    assert forwarded == ["[TensorRT] cảnh báo"]


_TRT_DIR = os.environ.get("SE_TEST_TRT_DIR", "")


@pytest.mark.skipif(not _TRT_DIR, reason="cần SE_TEST_TRT_DIR (tensorrt_bindings + tensorrt_libs 10.9 thật)")
def test_real_bindings_accept_the_fixes() -> None:
    script = textwrap.dedent("""
        import tensorrt_bindings as trt
        from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan
        assert plan.patch_tensorrt_numpy_bridge(trt)
        assert trt.INetworkDefinition.add_shape._se_trt10_fix
        prop = trt.ILayer.__dict__["name"]
        assert isinstance(prop, property) and prop.fset._se_trt10_fix and prop.fget is not None
        assert "name" not in trt.IConvolutionLayer.__dict__          # lớp con dùng thuộc tính của ILayer
        print("ok")
    """)
    src = str(Path(__file__).resolve().parents[3] / "src")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([_TRT_DIR, src]),
               LD_LIBRARY_PATH=str(Path(_TRT_DIR) / "tensorrt_libs"))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300,
                            check=False)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]
