"""[v3.23.774] Ứng dụng TỰ CHẠY engine TensorRT mà Paddle đã dựng — bỏ qua lớp chạy engine (C++) của Paddle.

Chuỗi chẩn đoán 19/11 (bản 3.23.766–773, máy chủ dự án, RTX 3070 Ti):
* mọi engine TensorRT Paddle chạy đều SẬP "access violation" ở ``predictor.run()`` — bộ dò, bộ nhận dạng, FP32/FP16, hồ
  sơ gốc / hồ sơ của ứng dụng, có / không bản vá Int32, cùng / khác luồng (log 20:31);
* nạp THẲNG chính các tệp engine đó bằng TensorRT 10.9 và chạy bằng bộ đệm GPU của paddle: CHẠY ĐƯỢC — bộ dò lô 8 ảnh
  1600×288 mất 18,6 ms (log 21:34).
→ TensorRT, driver và engine đều tốt; lỗi ở lớp Paddle C++ ↔ TensorRT (Paddle 3.3.1 biên dịch với TensorRT 10.5, thư viện
chạy là 10.9 — bản thấp nhất có gói Python cho Python 3.13).

Cách làm: giữ NGUYÊN tiền / hậu xử lý của PaddleX (đổi cỡ, chuẩn hoá, tách hộp DB, giải mã CTC) — chỉ thay bước "đưa mảng
vào mạng" (``predictor.runner``) bằng :class:`DirectTrtRunner`. Lô vượt lô tối đa của engine → chia lô con; kích thước ngoài
hồ sơ / lỗi TensorRT → lô đó chạy bằng runner Paddle GPU thường (không sập).
"""

from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DirectTrtUnavailable(RuntimeError):
    """Không gắn được TensorRT trực tiếp (thiếu engine, engine không hợp, thư viện lỗi)."""


class TrtShapeError(RuntimeError):
    """Đầu vào ngoài hồ sơ của engine — chạy bằng runner Paddle cho lô đó."""


def model_engine_files(model_dir: Path) -> list[Path]:
    """Mọi tệp engine Paddle đã dựng của mô hình (``<mô hình>/.cache/paddle/trt/**/engine_*.trt``)."""
    try:
        return sorted((Path(model_dir) / ".cache" / "paddle" / "trt").glob("**/engine_*.trt"))
    except OSError:
        return []


def find_model_engine(model_dir: Path) -> Path | None:
    """[v3.23.774] Engine của mô hình — CHỈ khi có ĐÚNG MỘT tệp. Rà độc lập: Paddle ghi một tệp cho MỖI đồ thị con
    (``converter.py``: ``engine_<băm nhóm>_<số>.trt``); nhiều tệp = mạng bị chia (thao tác TensorRT không nhận nằm giữa) →
    thay ``predictor.runner`` bằng một engine là SAI. Bước dựng xoá thư mục cũ khi hồ sơ đổi, nên sau mỗi lần dựng chỉ còn
    tệp của lần đó."""
    files = model_engine_files(model_dir)
    return files[0] if len(files) == 1 else None


def engine_fingerprint(path: Path) -> dict[str, Any]:
    """Dấu tệp engine (đường dẫn + cỡ + thời điểm) — ghi cùng kết quả tự kiểm; tệp đổi → không gắn (kiểm lại)."""
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime": round(float(stat.st_mtime), 3)}


#: Engine đã nạp, dùng CHUNG giữa các bộ OCR (trích xuất + phụ đề nhúng) — mỗi context giữ sẵn VRAM cho hồ sơ lớn nhất.
_ENGINES: dict[str, Any] = {}
_ENGINES_LOCK = threading.Lock()


def shared_engine(path: Path, factory: Any = None) -> Any:
    key = f"{path}|{engine_fingerprint(path)['size']}|{engine_fingerprint(path)['mtime']}"
    with _ENGINES_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = (factory or DirectTrtEngine)(path)
            _ENGINES[key] = engine
        return engine


class DirectTrtEngine:
    """Một engine TensorRT một đầu vào, chạy bằng bộ đệm GPU (tensor paddle) + luồng CUDA của paddle."""

    def __init__(self, path: Path, *, trt: Any | None = None, paddle: Any | None = None) -> None:
        self._trt = trt if trt is not None else importlib.import_module("tensorrt")
        self._paddle = paddle if paddle is not None else importlib.import_module("paddle")
        self.path = Path(path)
        trt_mod = self._trt
        trt_logger = trt_mod.Logger(trt_mod.Logger.WARNING)
        trt_mod.init_libnvinfer_plugins(trt_logger, "")
        self._runtime = trt_mod.Runtime(trt_logger)
        self._engine = self._runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self._engine is None:
            raise DirectTrtUnavailable(f"TensorRT không nạp được {self.path.name}")
        self._context = self._engine.create_execution_context()
        if self._context is None:                         # TensorRT 10: thiếu VRAM → None (không ném)
            raise DirectTrtUnavailable(f"TensorRT không tạo được context cho {self.path.name} (thiếu VRAM?)")
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        is_input = trt_mod.TensorIOMode.INPUT
        self.inputs = [name for name in names if self._engine.get_tensor_mode(name) == is_input]
        self.outputs = [name for name in names if self._engine.get_tensor_mode(name) != is_input]
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise DirectTrtUnavailable(f"{self.path.name}: {len(self.inputs)} đầu vào / {len(self.outputs)} đầu ra "
                                       "(chỉ hỗ trợ 1 / 1)")
        if self._engine.is_shape_inference_io(self.inputs[0]):
            raise DirectTrtUnavailable(f"{self.path.name}: đầu vào là tensor hình dạng (chưa hỗ trợ)")
        low, _opt, high = self._engine.get_tensor_profile_shape(self.inputs[0], 0)
        self.min_shape = [int(v) for v in low]
        self.max_shape = [int(v) for v in high]
        self._input_dtype = np.dtype(trt_mod.nptype(self._engine.get_tensor_dtype(self.inputs[0])))
        self._output_dtype = np.dtype(trt_mod.nptype(self._engine.get_tensor_dtype(self.outputs[0])))
        self._lock = threading.Lock()
        #: [v3.23.775] VRAM context giữ sẵn cho hồ sơ lớn nhất (MB; 0 = không đọc được) — ghi log để chỉnh cỡ lô.
        memory = getattr(self._engine, "device_memory_size_v2", None)
        if memory is None:
            memory = getattr(self._engine, "device_memory_size", 0)
        try:
            self.device_memory_mb = int(memory or 0) // (1024 * 1024)
        except (TypeError, ValueError):
            self.device_memory_mb = 0

    def _check(self, shape: tuple[int, ...]) -> None:
        if len(shape) != len(self.max_shape):
            raise TrtShapeError(f"{len(shape)} chiều, engine {len(self.max_shape)} chiều")
        for axis in range(1, len(shape)):
            if not self.min_shape[axis] <= shape[axis] <= self.max_shape[axis]:
                raise TrtShapeError(f"đầu vào {list(shape)} ngoài hồ sơ {self.min_shape}–{self.max_shape}")

    def run(self, array: np.ndarray) -> np.ndarray:
        """``array`` (lô, …) trên CPU → kết quả (lô, …) trên CPU; lô lớn hơn lô tối đa của engine được chia nhỏ."""
        array = np.ascontiguousarray(array, dtype=self._input_dtype)
        self._check(tuple(array.shape))
        step = max(1, self.max_shape[0])
        parts = [self._run_chunk(array[start:start + step]) for start in range(0, max(1, array.shape[0]), step)]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def _gpu_output(self, shape: tuple[int, ...], place: Any) -> Any:
        """Bộ đệm ra TRÊN GPU (``paddle.empty`` — không chép dữ liệu rỗng lên GPU); thiết bị hiện hành của paddle không
        phải GPU → cấp thẳng bằng ``CUDAPlace`` (TensorRT ghi vào địa chỉ RAM là sập)."""
        paddle = self._paddle
        empty = getattr(paddle, "empty", None)
        if callable(empty):
            try:
                tensor = empty(list(shape), dtype=self._output_dtype.name)
                if getattr(getattr(tensor, "place", None), "is_gpu_place", lambda: False)():
                    return tensor
            except Exception:  # noqa: BLE001 — cấp kiểu khác bên dưới
                pass
        return paddle.to_tensor(np.empty(shape, dtype=self._output_dtype), place=place)

    def _run_chunk(self, chunk: np.ndarray) -> np.ndarray:
        paddle = self._paddle
        with self._lock:                                  # một context TensorRT: một lượt chạy tại một thời điểm
            place = paddle.CUDAPlace(0)
            if self._context.set_input_shape(self.inputs[0], tuple(int(v) for v in chunk.shape)) is False:
                raise TrtShapeError(f"TensorRT từ chối hình dạng {list(chunk.shape)}")
            out_shape = tuple(int(v) for v in self._context.get_tensor_shape(self.outputs[0]))
            if any(v < 0 for v in out_shape):
                raise TrtShapeError(f"đầu ra chưa xác định hình dạng {list(out_shape)}")
            source = paddle.to_tensor(chunk, place=place)
            target = self._gpu_output(out_shape, place)
            stream: Any = paddle.device.current_stream()
            base = getattr(stream, "stream_base", stream)
            raw = getattr(base, "cuda_stream", None)
            if raw is None:
                raw = getattr(base, "raw_stream", None)
            handle = int(raw if raw is not None else stream.__cuda_stream__()[1])
            try:
                self._context.set_tensor_address(self.inputs[0], int(source.data_ptr()))
                self._context.set_tensor_address(self.outputs[0], int(target.data_ptr()))
                if not self._context.execute_async_v3(handle):
                    raise RuntimeError(f"TensorRT execute_async_v3 trả False ({self.path.name})")
            finally:
                stream.synchronize()                      # không trả bộ đệm khi GPU có thể còn ghi (kể cả đường lỗi)
            result = np.asarray(target.numpy(), dtype=np.float32 if self._output_dtype.kind == "f" else None)
            del source, target
            return result


class DirectTrtRunner:
    """Thay ``predictor.runner`` của PaddleX: ``runner(x=[mảng])`` → ``[kết quả]``; không chạy được → runner Paddle."""

    MAX_ERRORS = 3

    def __init__(self, engine: Any, fallback: Any, label: str) -> None:
        self._engine = engine
        self._fallback = fallback
        self._label = label
        self._errors = 0
        self._warned_shape = False
        self.calls = 0
        self.fallbacks = 0

    def __call__(self, x: Any) -> Any:
        arrays = list(x)
        if len(arrays) != 1 or self._errors >= self.MAX_ERRORS:
            self.fallbacks += 1
            return self._fallback(x=x)
        try:
            output = self._engine.run(arrays[0])
            self.calls += 1
            return [output]
        except TrtShapeError as exc:
            if not self._warned_shape:
                self._warned_shape = True
                logger.info("TensorRT trực tiếp %s: %s — lô này chạy Paddle GPU.", self._label, exc)
        except Exception as exc:  # noqa: BLE001 — lỗi TensorRT (Python): lô này chạy Paddle GPU
            self._errors += 1
            logger.warning("TensorRT trực tiếp %s lỗi (%d/%d): %s — lô này chạy Paddle GPU.", self._label, self._errors,
                           self.MAX_ERRORS, exc)
        self.fallbacks += 1
        return self._fallback(x=x)

    def close(self) -> None:
        close = getattr(self._fallback, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._fallback, name)


_PREDICTORS = (("text_det_model", "bộ dò"), ("text_rec_model", "bộ nhận dạng"))


def attach_direct_tensorrt(model: Any, *, engine_factory: Any = None,
                           expected: dict[str, Any] | None = None) -> list[str]:
    """Gắn :class:`DirectTrtRunner` cho bộ dò + bộ nhận dạng của mô hình PaddleOCR (GPU thường). Cả hai hoặc không gì
    (``DirectTrtUnavailable``). ``expected``: dấu tệp engine tự kiểm đã kiểm (``engine_fingerprint``) — lệch thì không gắn.
    Trả mô tả từng phần đã gắn."""
    try:
        pipeline = model.paddlex_pipeline._pipeline
    except AttributeError as exc:
        raise DirectTrtUnavailable(f"cấu trúc PaddleOCR khác dự kiến: {exc}") from exc
    planned: list[tuple[Any, Any, str, Path]] = []
    for attribute, label in _PREDICTORS:
        predictor = getattr(pipeline, attribute, None)
        runner = getattr(predictor, "runner", None)
        model_dir = getattr(predictor, "model_dir", None)
        if predictor is None or runner is None or model_dir is None:
            raise DirectTrtUnavailable(f"{label}: không có predictor / runner / thư mục mô hình")
        files = model_engine_files(Path(model_dir))
        if len(files) != 1:
            raise DirectTrtUnavailable(f"{label}: {'chưa có' if not files else f'{len(files)}'} tệp engine TensorRT trong "
                                       f"{Path(model_dir) / '.cache'} (cần đúng 1 — mạng không bị chia)")
        path = files[0]
        if expected is not None and expected.get(attribute) != engine_fingerprint(path):
            raise DirectTrtUnavailable(f"{label}: tệp engine khác lần tự kiểm ({path.name}) — cần tự kiểm lại")
        planned.append((predictor, runner, label, path))
    engines = [shared_engine(path, engine_factory) for _predictor, _runner, _label, path in planned]
    described = []
    for (predictor, runner, label, path), engine in zip(planned, engines, strict=True):
        predictor.runner = DirectTrtRunner(engine, runner, label)
        described.append(f"{label} {getattr(predictor, 'model_name', '?')}: {path.name} "
                         f"(lô ≤ {engine.max_shape[0]}, {engine.min_shape}–{engine.max_shape}"
                         + (f", VRAM {engine.device_memory_mb} MB)" if getattr(engine, "device_memory_mb", 0) else ")"))
    return described


def detach_direct_tensorrt(model: Any) -> None:
    """Gỡ :class:`DirectTrtRunner` (trả runner Paddle) — vd khi giải phóng mô hình."""
    try:
        pipeline = model.paddlex_pipeline._pipeline
    except AttributeError:
        return
    for attribute, _label in _PREDICTORS:
        predictor: Any = getattr(pipeline, attribute, None)
        runner = getattr(predictor, "runner", None)
        if predictor is not None and isinstance(runner, DirectTrtRunner):
            predictor.runner = runner._fallback


__all__ = [
    "DirectTrtEngine", "DirectTrtRunner", "DirectTrtUnavailable", "TrtShapeError", "attach_direct_tensorrt",
    "detach_direct_tensorrt", "engine_fingerprint", "find_model_engine", "model_engine_files", "shared_engine",
]
