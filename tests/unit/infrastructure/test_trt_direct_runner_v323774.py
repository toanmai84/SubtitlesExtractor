"""[v3.23.774] Ứng dụng tự chạy engine TensorRT Paddle đã dựng (bỏ qua lớp chạy engine C++ của Paddle — sập với mọi
engine, chẩn đoán 19/11). Log 21:34 (bản 3.23.773): nạp thẳng engine bằng TensorRT 10.9 + bộ đệm GPU của paddle CHẠY ĐƯỢC
(bộ dò lô 8 ảnh 1600×288: 18,6 ms). Test dùng TensorRT + paddle giả lập theo đúng đường địa chỉ bộ nhớ."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from subtitles_extractor.infrastructure.ocr import trt_direct_runner as direct

_MEMORY: dict[int, np.ndarray] = {}


class _Tensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.array(array, copy=True)
        _MEMORY[id(self._array)] = self._array

    def data_ptr(self) -> int:
        return id(self._array)

    def numpy(self) -> np.ndarray:
        return self._array


def _fake_paddle():  # noqa: ANN202
    stream = SimpleNamespace(stream_base=SimpleNamespace(cuda_stream=7), synchronize=lambda: None)
    return SimpleNamespace(CUDAPlace=lambda index: ("gpu", index), to_tensor=lambda array, place: _Tensor(array),
                           device=SimpleNamespace(current_stream=lambda: stream))


class _Mode(Enum):
    INPUT = 0
    OUTPUT = 1


def _fake_trt(max_batch: int = 8, inputs: int = 1, runs: list | None = None):  # noqa: ANN202
    runs = runs if runs is not None else []

    class Context:
        def __init__(self) -> None:
            self.shape: tuple = ()
            self.addresses: dict[str, int] = {}

        def set_input_shape(self, name, shape) -> None:  # noqa: ANN001
            self.shape = tuple(shape)

        def get_tensor_shape(self, name):  # noqa: ANN001, ANN202
            n, _c, h, w = self.shape
            return (n, 1, h, w)

        def set_tensor_address(self, name, address) -> None:  # noqa: ANN001
            self.addresses[name] = address

        def execute_async_v3(self, stream) -> bool:  # noqa: ANN001
            source, target = _MEMORY[self.addresses["x"]], _MEMORY[self.addresses["out"]]
            target[...] = source[:, :1] * 2                       # "mạng": kênh đầu × 2
            runs.append((stream, source.shape[0]))
            return True

    names = ["x", "out"] + [f"extra{i}" for i in range(inputs - 1)]

    class Engine:
        num_io_tensors = len(names)

        def get_tensor_name(self, index):  # noqa: ANN001, ANN202
            return names[index]

        def get_tensor_mode(self, name):  # noqa: ANN001, ANN202
            return _Mode.OUTPUT if name == "out" else _Mode.INPUT

        def is_shape_inference_io(self, name) -> bool:  # noqa: ANN001
            return False

        def get_tensor_profile_shape(self, name, profile):  # noqa: ANN001, ANN202
            return [(1, 3, 32, 32), (max_batch, 3, 288, 1600), (max_batch, 3, 512, 2048)]

        def get_tensor_dtype(self, name):  # noqa: ANN001, ANN202
            return "float32"

        def create_execution_context(self):  # noqa: ANN202
            return Context()

    class Runtime:
        def __init__(self, logger) -> None:  # noqa: ANN001
            pass

        def deserialize_cuda_engine(self, data):  # noqa: ANN001, ANN202
            return Engine() if data else None

    class Logger:
        WARNING = 2

        def __init__(self, level) -> None:  # noqa: ANN001
            pass

    trt = SimpleNamespace(Logger=Logger, Runtime=Runtime, TensorIOMode=_Mode, init_libnvinfer_plugins=lambda *a: True,
                          nptype=lambda dtype: np.float32)
    return trt, runs


def _engine(tmp_path: Path, **kwargs) -> tuple[direct.DirectTrtEngine, list]:  # noqa: ANN003
    path = tmp_path / "engine_1_0.trt"
    path.write_bytes(b"engine")
    trt, runs = _fake_trt(**kwargs)
    return direct.DirectTrtEngine(path, trt=trt, paddle=_fake_paddle()), runs


def test_engine_runs_through_gpu_addresses_and_paddle_stream(tmp_path: Path) -> None:
    engine, runs = _engine(tmp_path)
    x = np.random.default_rng(0).random((2, 3, 288, 1600), dtype=np.float32)
    out = engine.run(x)
    assert out.shape == (2, 1, 288, 1600) and np.allclose(out, x[:, :1] * 2)
    assert runs == [(7, 2)]                                         # luồng CUDA của paddle


def test_large_batch_is_split_to_engine_max_and_concatenated_in_order(tmp_path: Path) -> None:
    engine, runs = _engine(tmp_path, max_batch=8)
    x = np.arange(20 * 3 * 32 * 32, dtype=np.float32).reshape(20, 3, 32, 32)
    out = engine.run(x)
    assert [n for _s, n in runs] == [8, 8, 4] and np.array_equal(out, x[:, :1] * 2)


@pytest.mark.parametrize("shape", [(1, 3, 600, 1600), (1, 3, 288, 4000), (1, 3, 16, 64), (1, 3, 288)])
def test_shapes_outside_profile_raise_shape_error(tmp_path: Path, shape: tuple[int, ...]) -> None:
    engine, runs = _engine(tmp_path)
    with pytest.raises(direct.TrtShapeError):
        engine.run(np.zeros(shape, np.float32))
    assert runs == []


def test_unloadable_or_multi_input_engines_are_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "engine_0_0.trt"
    empty.write_bytes(b"")
    trt, _runs = _fake_trt()
    with pytest.raises(direct.DirectTrtUnavailable, match="không nạp được"):
        direct.DirectTrtEngine(empty, trt=trt, paddle=_fake_paddle())
    with pytest.raises(direct.DirectTrtUnavailable, match="đầu vào"):
        _engine(tmp_path, inputs=2)


class _Fallback:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False
        self.extra = "thuộc tính runner Paddle"

    def __call__(self, x):  # noqa: ANN001, ANN204
        self.calls += 1
        return ["paddle"]

    def close(self) -> None:
        self.closed = True


def test_runner_uses_engine_and_falls_back_per_batch() -> None:
    class Engine:
        def __init__(self) -> None:
            self.mode = "ok"

        def run(self, array):  # noqa: ANN001, ANN202
            if self.mode == "shape":
                raise direct.TrtShapeError("ngoài hồ sơ")
            if self.mode == "boom":
                raise RuntimeError("execute lỗi")
            return array * 3

    engine, fallback = Engine(), _Fallback()
    runner = direct.DirectTrtRunner(engine, fallback, "bộ dò")
    (out,) = runner(x=[np.ones(2, np.float32)])
    assert np.array_equal(out, [3, 3]) and runner.calls == 1 and fallback.calls == 0
    engine.mode = "shape"
    assert runner(x=[np.ones(2)]) == ["paddle"] and fallback.calls == 1           # lô ngoài hồ sơ: Paddle GPU
    engine.mode = "boom"
    for _ in range(direct.DirectTrtRunner.MAX_ERRORS):
        assert runner(x=[np.ones(2)]) == ["paddle"]
    engine.mode = "ok"
    assert runner(x=[np.ones(2)]) == ["paddle"]                                     # lỗi lặp lại: tắt TensorRT
    assert runner(x=[np.ones(2), np.ones(2)]) == ["paddle"]                         # nhiều đầu vào: Paddle
    assert runner.extra == "thuộc tính runner Paddle"
    runner.close()
    assert fallback.closed


def _pipeline(tmp_path: Path, det_files: int = 1, rec_files: int = 1):  # noqa: ANN202
    preds = {}
    for attr, name, count in (("text_det_model", "PP-OCRv6_small_det", det_files),
                              ("text_rec_model", "PP-OCRv6_small_rec", rec_files)):
        model_dir = tmp_path / name
        trt_dir = model_dir / ".cache" / "paddle" / "trt" / "inference"
        trt_dir.mkdir(parents=True)
        for index in range(count):
            (trt_dir / f"engine_{name[-3:]}{index}_{index + 1}.trt").write_bytes(b"engine")
        preds[attr] = SimpleNamespace(runner=_Fallback(), model_dir=model_dir, model_name=name)
    return SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=SimpleNamespace(**preds))), preds


def _factory(loaded: list[str]):  # noqa: ANN202
    def make(path):  # noqa: ANN001, ANN202
        loaded.append(path.name)
        return SimpleNamespace(max_shape=[8, 3, 512, 2048], min_shape=[1, 3, 32, 32], run=lambda a: a)

    return make


@pytest.fixture(autouse=True)
def _fresh_engine_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(direct, "_ENGINES", {})


def test_attach_wraps_both_predictors_and_detach_restores(tmp_path: Path) -> None:
    model, preds = _pipeline(tmp_path)
    originals = {attr: pred.runner for attr, pred in preds.items()}
    loaded: list[str] = []
    described = direct.attach_direct_tensorrt(model, engine_factory=_factory(loaded))
    assert loaded == ["engine_det0_1.trt", "engine_rec0_1.trt"] and len(described) == 2
    assert all(isinstance(pred.runner, direct.DirectTrtRunner) for pred in preds.values())
    direct.detach_direct_tensorrt(model)
    assert {attr: pred.runner for attr, pred in preds.items()} == originals


def test_attach_is_all_or_nothing(tmp_path: Path) -> None:
    model, preds = _pipeline(tmp_path, rec_files=0)
    with pytest.raises(direct.DirectTrtUnavailable, match="bộ nhận dạng: chưa có"):
        direct.attach_direct_tensorrt(model, engine_factory=_factory([]))
    assert not any(isinstance(pred.runner, direct.DirectTrtRunner) for pred in preds.values())


def test_split_network_with_several_engines_is_refused(tmp_path: Path) -> None:
    """Rà độc lập 3.23.774: Paddle ghi MỘT tệp cho mỗi đồ thị con — nhiều tệp = mạng bị chia, thay runner là sai."""
    model, _preds = _pipeline(tmp_path, det_files=2)
    with pytest.raises(direct.DirectTrtUnavailable, match="2 tệp engine"):
        direct.attach_direct_tensorrt(model, engine_factory=_factory([]))


def test_engine_must_match_the_selftested_file(tmp_path: Path) -> None:
    model, preds = _pipeline(tmp_path)
    fingerprints = {attr: direct.engine_fingerprint(direct.model_engine_files(pred.model_dir)[0])
                    for attr, pred in preds.items()}
    assert len(direct.attach_direct_tensorrt(model, engine_factory=_factory([]), expected=fingerprints)) == 2
    direct.detach_direct_tensorrt(model)
    engine = direct.model_engine_files(preds["text_rec_model"].model_dir)[0]
    engine.write_bytes(b"engine khac do FP16")                                        # dựng lại khác cấu hình
    with pytest.raises(direct.DirectTrtUnavailable, match="khác lần tự kiểm"):
        direct.attach_direct_tensorrt(model, engine_factory=_factory([]), expected=fingerprints)


def test_engines_are_shared_between_adapters(tmp_path: Path) -> None:
    loaded: list[str] = []
    first, _ = _pipeline(tmp_path)
    direct.attach_direct_tensorrt(first, engine_factory=_factory(loaded))
    second = SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=SimpleNamespace(**{
        attr: SimpleNamespace(runner=_Fallback(), model_dir=pred.model_dir, model_name=pred.model_name)
        for attr, pred in vars(first.paddlex_pipeline._pipeline).items()})))
    direct.attach_direct_tensorrt(second, engine_factory=_factory(loaded))
    assert len(loaded) == 2                                                          # không nạp / giữ VRAM hai lần


def test_context_allocation_failure_is_reported(tmp_path: Path) -> None:
    trt, _runs = _fake_trt()
    engine_cls = trt.Runtime(None).deserialize_cuda_engine(b"x").__class__
    original = engine_cls.create_execution_context
    engine_cls.create_execution_context = lambda self: None                            # TensorRT thiếu VRAM: None
    try:
        path = tmp_path / "engine_1_0.trt"
        path.write_bytes(b"engine")
        with pytest.raises(direct.DirectTrtUnavailable, match="context"):
            direct.DirectTrtEngine(path, trt=trt, paddle=_fake_paddle())
    finally:
        engine_cls.create_execution_context = original


def test_stream_is_synchronized_even_when_execute_fails(tmp_path: Path) -> None:
    synced: list[bool] = []
    paddle = _fake_paddle()
    paddle.device = SimpleNamespace(current_stream=lambda: SimpleNamespace(
        stream_base=SimpleNamespace(cuda_stream=0), synchronize=lambda: synced.append(True)))
    trt, _runs = _fake_trt()
    path = tmp_path / "engine_1_0.trt"
    path.write_bytes(b"engine")
    engine = direct.DirectTrtEngine(path, trt=trt, paddle=paddle)
    engine._context.execute_async_v3 = lambda handle: (_ for _ in ()).throw(RuntimeError("lỗi giữa chừng"))
    with pytest.raises(RuntimeError):
        engine.run(np.zeros((1, 3, 32, 32), np.float32))
    assert synced == [True]                                                          # luồng số 0 vẫn là luồng hợp lệ


def test_paddlex_predictors_call_runner_by_keyword() -> None:
    """``DirectTrtRunner`` thay ``predictor.runner``: PaddleX gọi ``self.runner(x=x)`` (bộ dò + bộ nhận dạng)."""
    pytest.importorskip("paddlex")
    import inspect

    from paddlex.inference.models.text_detection import predictor as det
    from paddlex.inference.models.text_recognition import predictor as rec

    assert "self.runner(x=x)" in inspect.getsource(det) and "self.runner(x=x)" in inspect.getsource(rec)
