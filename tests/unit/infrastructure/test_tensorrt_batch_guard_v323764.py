"""[v3.23.764] TensorRT sập tiến trình ở lần suy luận đầu (log 19/11 14:46, bản 3.23.763).

"Windows fatal exception: access violation" trong ``predictor.run()`` của bộ dò ngay sau "PaddleOCR chạy GPU +
TensorRT (FP32)": ứng dụng đẩy 64 ảnh/lô vào bộ dò (``det_batch_size``) và 16 dòng/lô vào bộ nhận dạng, trong khi
engine dựng theo ``trt_dynamic_shapes`` của ``inference.yml`` — lô tối đa 1 (dò, ``[1,3,4000,4000]``) và 8 (nhận dạng,
``[8,3,48,3200]``). Lỗi sập trong C++ không bắt được bằng Python → thêm dấu chống sập lặp lại.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan

_DET_YML = {"Hpi": {"backend_configs": {"paddle_infer": {"trt_dynamic_shapes": {
    "x": [[1, 3, 32, 32], [1, 3, 736, 736], [1, 3, 4000, 4000]]}}}}}
_REC_YML = {"Hpi": {"backend_configs": {"paddle_infer": {"trt_dynamic_shapes": {
    "x": [[1, 3, 48, 160], [1, 3, 48, 320], [8, 3, 48, 3200]]}}}}}


def test_max_batch_read_from_model_config() -> None:
    assert plan.trt_max_batch(_DET_YML, 99) == 1 and plan.trt_max_batch(_REC_YML, 99) == 8
    assert plan.trt_max_batch(None, 7) == 7 and plan.trt_max_batch({"Hpi": {}}, 3) == 3


def test_real_model_configs_if_present() -> None:
    import yaml

    root = Path.home() / ".paddlex" / "official_models"
    det, rec = root / "PP-OCRv6_small_det" / "inference.yml", root / "PP-OCRv6_small_rec" / "inference.yml"
    if not det.is_file() or not rec.is_file():
        pytest.skip("chưa có mô hình PP-OCRv6_small trên máy")
    assert plan.trt_max_batch(yaml.safe_load(det.read_text(encoding="utf-8")), 99) == plan.TRT_DEFAULT_DET_MAX_BATCH
    assert plan.trt_max_batch(yaml.safe_load(rec.read_text(encoding="utf-8")), 99) == plan.TRT_DEFAULT_REC_MAX_BATCH


def _fake_model(rec_batch: int = 16):  # noqa: ANN202
    det = SimpleNamespace(config=_DET_YML, batch_sampler=SimpleNamespace(batch_size=1))
    rec = SimpleNamespace(config=_REC_YML, batch_sampler=SimpleNamespace(batch_size=rec_batch))
    pipeline = SimpleNamespace(text_det_model=det, text_rec_model=rec, batch_sampler=SimpleNamespace(batch_size=1))
    return SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=pipeline)), pipeline


def _adapter(det_batch: int = 64):  # noqa: ANN202
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    return PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True, det_batch_size=det_batch, batch_size=16))


def test_tensorrt_model_batches_capped_to_engine_profile() -> None:
    adapter = _adapter()
    model, pipeline = _fake_model()
    adapter._model_uses_trt = True
    adapter._apply_det_batch_size(model)                                # mã cũ: bộ dò 64 ảnh, nhận dạng 16 dòng
    assert pipeline.text_det_model.batch_sampler.batch_size == 1 and pipeline.batch_sampler.batch_size == 1
    assert pipeline.text_rec_model.batch_sampler.batch_size == 8


def test_plain_gpu_model_keeps_large_batches() -> None:
    adapter = _adapter()
    model, pipeline = _fake_model()
    adapter._model_uses_trt = False
    adapter._apply_det_batch_size(model)
    assert pipeline.text_det_model.batch_sampler.batch_size == 64 and pipeline.text_rec_model.batch_sampler.batch_size == 16


# ── dấu chống sập lặp lại ───────────────────────────────────────────────────────────────────────────────────────────
def test_marker_lifecycle(tmp_path: Path) -> None:
    assert not plan.crashed_on_previous_first_inference(tmp_path, "3.23.764")
    plan.begin_first_inference(tmp_path, "3.23.764")
    assert plan.crashed_on_previous_first_inference(tmp_path, "3.23.764")             # tiến trình chết: dấu còn
    assert not plan.crashed_on_previous_first_inference(tmp_path, "3.23.765")         # bản mới: xoá dấu, thử lại
    assert not plan.crash_guard_path(tmp_path).exists()
    plan.begin_first_inference(tmp_path, "3.23.764")
    plan.end_first_inference(tmp_path)
    assert not plan.crashed_on_previous_first_inference(tmp_path, "3.23.764")


def test_first_tensorrt_inference_is_guarded(tmp_path: Path) -> None:
    import numpy as np

    adapter = _adapter()
    adapter._model, adapter._model_uses_trt, adapter._trt_dir = "trt", True, tmp_path
    seen: list[bool] = []

    def predict(images, indices, config):  # noqa: ANN001, ANN202
        seen.append(plan.crash_guard_path(tmp_path).exists())                  # dấu có TRONG lúc chạy
        return ["ok"]

    adapter._predict_batch = predict  # type: ignore[method-assign]
    image = [np.zeros((4, 4, 3), np.uint8)]
    assert adapter._predict_with_tensorrt_fallback(image, [0], OcrEngineConfig()) == ["ok"]
    assert seen == [True] and not plan.crash_guard_path(tmp_path).exists() and adapter._trt_first_inference_done
    adapter._predict_with_tensorrt_fallback(image, [0], OcrEngineConfig())
    assert seen == [True, False]                                                 # chỉ lần đầu của phiên


def test_crash_on_previous_run_disables_tensorrt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor import __version__
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    plan.begin_first_inference(tmp_path, __version__)                           # lần trước sập giữa chừng
    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: tmp_path)
    monkeypatch.setattr(module, "_tensorrt_thread_ok", lambda: True)                 # [v3.23.767] đang ở luồng Paddle
    adapter = _adapter()
    built: list[dict] = []
    adapter._try_create_model_with_retry = lambda cls, kwargs: built.append(kwargs) or "model"  # type: ignore[method-assign]
    assert adapter._create_model_with_device_fallback(object, use_gpu=True) == "model"
    assert "use_tensorrt" not in built[0] and adapter._tensorrt_unavailable and not adapter._model_uses_trt


def test_vram_cleanup_never_imports_paddle(monkeypatch: pytest.MonkeyPatch) -> None:
    """[v3.23.764] Luồng dọn dẹp nền từng ``import paddle`` đúng lúc test khác nạp lại paddle → mô-đun nạp dở kẹt trong
    ``sys.modules`` ("partially initialized module 'paddle' … has no attribute 'tensor'", hỏng test VL khi chạy nhóm)."""
    import builtins
    import sys

    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    imported: list[str] = []
    real_import = builtins.__import__

    def spy(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if name == "paddle" or name.startswith("paddle."):
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "paddle", raising=False)
    monkeypatch.setattr(builtins, "__import__", spy)
    PaddleOcrAdapter._empty_paddle_vram()
    assert imported == [] and "paddle" not in sys.modules
    emptied: list[bool] = []
    fake = SimpleNamespace(__version__="3.3.1", device=SimpleNamespace(cuda=SimpleNamespace(
        empty_cache=lambda: emptied.append(True))))
    monkeypatch.setitem(sys.modules, "paddle", fake)
    PaddleOcrAdapter._empty_paddle_vram()
    assert emptied == [True]
