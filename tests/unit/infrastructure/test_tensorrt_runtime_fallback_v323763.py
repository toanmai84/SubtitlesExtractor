"""[v3.23.763] TensorRT lúc CHẠY engine.

Log 19/11 14:21 (bản 3.23.761): cả bộ dò lẫn bộ nhận dạng dựng engine TensorRT xong (hết "TensorRT không dùng được"),
nhưng mọi lô OCR lỗi "(NotFound) TensorRT is needed, but TensorRT dynamic library 'nvinfer_plugin.dll' is not found"
(``paddle\\phi\\backends\\dynload\\tensorrt.cc:53``, tên kiểu TensorRT 8; TensorRT 10: ``nvinfer_plugin_10.dll``) → 27 khung
"lỗi OCR riêng lẻ", cả phim không ra phụ đề. Sửa: bí danh ``nvinfer_plugin.dll`` nạp sẵn + chuỗi lùi GPU+TensorRT → GPU
lúc CHẠY (dựng lại mô hình không TensorRT, chạy lại lô).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.domain.exceptions import OcrInferenceError
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan

_REAL_ERROR = ("PaddleOCR lỗi: (NotFound) TensorRT is needed, but TensorRT dynamic library 'nvinfer_plugin.dll' is not "
               "found.\n  Suggestions: …")


# ── bí danh plugin cho phần C++ của Paddle ──────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def windows(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    loaded: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(plan.ctypes, "WinDLL", lambda path: loaded.append(path) or object(), raising=False)
    monkeypatch.setattr(plan, "_LOADED_PLUGINS", [])
    monkeypatch.setenv("PATH", "C:\\Windows")
    return loaded


def _trt_dir(tmp_path: Path, size: int = 64) -> Path:
    libs = tmp_path / "tensorrt_runtime" / "tensorrt_libs"
    libs.mkdir(parents=True)
    (libs / "nvinfer_plugin_10.dll").write_bytes(b"P" * size)
    return libs.parent


def test_alias_created_preloaded_and_on_path(tmp_path: Path, windows: list[str]) -> None:
    trt_dir = _trt_dir(tmp_path)
    alias = plan.ensure_paddle_runtime_plugin(trt_dir)
    libs = trt_dir / "tensorrt_libs"
    assert alias == libs / "nvinfer_plugin.dll" and alias.read_bytes() == b"P" * 64
    assert windows == [str(alias)] and os.environ["PATH"].split(os.pathsep)[0] == str(libs)
    assert plan.ensure_paddle_runtime_plugin(trt_dir) == alias and windows == [str(alias)]     # một lần mỗi phiên
    assert os.environ["PATH"].count(str(libs)) == 1


def test_stale_alias_is_replaced(tmp_path: Path, windows: list[str]) -> None:
    trt_dir = _trt_dir(tmp_path, size=128)
    (trt_dir / "tensorrt_libs" / "nvinfer_plugin.dll").write_bytes(b"old")                     # bản TensorRT cũ
    alias = plan.ensure_paddle_runtime_plugin(trt_dir)
    assert alias is not None and alias.stat().st_size == 128


def test_no_alias_off_windows_or_without_tensorrt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert plan.ensure_paddle_runtime_plugin(_trt_dir(tmp_path)) is None
    monkeypatch.setattr(sys, "platform", "win32")
    assert plan.ensure_paddle_runtime_plugin(tmp_path / "chua_tai") is None


def test_python_builder_uses_the_same_plugin_file(tmp_path: Path, windows: list[str],
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    converter = types.ModuleType("paddle.tensorrt.converter")
    converter.ctypes = plan.ctypes  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddle.tensorrt.converter", converter)
    monkeypatch.setattr("importlib.import_module", lambda name: converter)
    trt_dir = _trt_dir(tmp_path)
    assert plan.patch_paddle_tensorrt_plugin_loader(trt_dir) is True
    assert converter.ctypes._plugin_dll == trt_dir / "tensorrt_libs" / "nvinfer_plugin.dll"  # type: ignore[attr-defined]


# ── chuỗi lùi lúc chạy ──────────────────────────────────────────────────────────────────────────────────────────────
def _adapter(uses_trt: bool, errors: list[Exception]):  # noqa: ANN202
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    adapter = PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True))
    adapter._model = "trt-model"
    adapter._model_uses_trt = uses_trt
    calls: list[str] = []

    def predict(images, indices, config):  # noqa: ANN001, ANN202
        calls.append(adapter._model)
        if errors:
            raise errors.pop(0)
        return ["ok"] * len(images)

    def ensure(config) -> None:  # noqa: ANN001
        if adapter._model is None:
            adapter._model = "trt-model" if not adapter._tensorrt_unavailable else "gpu-model"
            adapter._model_uses_trt = not adapter._tensorrt_unavailable

    adapter._predict_batch = predict  # type: ignore[method-assign]
    adapter._ensure_model_for = ensure  # type: ignore[method-assign]
    adapter._release_internal = lambda: setattr(adapter, "_model", None)  # type: ignore[method-assign]
    return adapter, calls


def test_tensorrt_runtime_error_rebuilds_without_tensorrt_and_retries_the_batch() -> None:
    adapter, calls = _adapter(True, [OcrInferenceError(_REAL_ERROR)])
    images = [np.zeros((4, 4, 3), np.uint8)] * 2
    assert adapter._predict_with_tensorrt_fallback(images, [0, 1], OcrEngineConfig(use_tensorrt=True)) == ["ok", "ok"]
    assert calls == ["trt-model", "gpu-model"] and adapter._tensorrt_unavailable and not adapter._model_uses_trt
    assert adapter._predict_with_tensorrt_fallback(images, [0, 1], OcrEngineConfig()) == ["ok", "ok"]
    assert calls[-1] == "gpu-model"                                                    # cả phiên: không thử lại TRT


@pytest.mark.parametrize(("uses_trt", "error"), [
    (False, OcrInferenceError(_REAL_ERROR)),                       # mô hình không TensorRT: lỗi khác, để nguyên
    (True, OcrInferenceError("PaddleOCR lỗi: ảnh hỏng")),          # không phải lỗi TensorRT
    (True, RuntimeError("CUDA error: out of memory")),            # hết VRAM: bộ an toàn OOM xử lý
])
def test_other_errors_are_not_swallowed(uses_trt: bool, error: Exception) -> None:
    adapter, calls = _adapter(uses_trt, [error])
    with pytest.raises(type(error)):
        adapter._predict_with_tensorrt_fallback([np.zeros((4, 4, 3), np.uint8)], [0], OcrEngineConfig())
    assert calls == ["trt-model"] and not adapter._tensorrt_unavailable


def test_infer_batch_goes_through_the_fallback() -> None:
    import inspect

    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    source = inspect.getsource(PaddleOcrAdapter.infer_batch)
    assert "_predict_with_tensorrt_fallback(" in source and "self._predict_batch(" not in source
