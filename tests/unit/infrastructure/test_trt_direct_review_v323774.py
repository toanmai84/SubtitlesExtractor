"""[v3.23.774] Lượt rà độc lập phần TensorRT trực tiếp — chặn tái phát các điểm rà tìm ra.

1. Paddle sập khi chuyển bộ dò → bộ nhận dạng chưa kịp chuyển: dựng engine mỗi mô hình một tiến trình con.
2. Thiếu engine vì bước dựng sập: KHÔNG ghi nhớ "không chạy được" (lần sau dựng lại).
3. Dấu tệp engine đã kiểm được ghi cùng kết quả đạt; tệp đổi → không gắn, xoá kết quả để kiểm lại.
4. FP16 lệch nhỏ: không ghi nhớ; tiến trình con luôn ``os._exit`` (không dọn Paddle / TensorRT lúc thoát).
5. Canh "lần suy luận đầu": chỉ hết canh khi runner trực tiếp thật sự chạy ít nhất một lô.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subtitles_extractor.infrastructure.ocr import tensorrt_selftest as selftest

_KWARGS = {"device": "gpu", "use_tensorrt": True, "precision": "fp32"}


def _scripted(results: dict[str, selftest.SelfTestOutcome]):  # noqa: ANN202
    calls: list[str] = []

    def run(variant, payload, det_batch, fault_path):  # noqa: ANN001, ANN202
        calls.append(variant)
        return results[variant]

    return run, calls


def test_build_runs_one_child_per_model_and_worker_uses_single_model_classes(tmp_path: Path) -> None:
    run, calls = _scripted({v: selftest.SelfTestOutcome(True, v) for v in
                            (selftest.VARIANT_BUILD_DET, selftest.VARIANT_BUILD_REC, selftest.VARIANT_DIRECT_APP)})
    selftest.ensure_tensorrt_selftest(tmp_path, "v", _KWARGS, 8, runner=run)
    assert calls == [selftest.VARIANT_BUILD_DET, selftest.VARIANT_BUILD_REC, selftest.VARIANT_DIRECT_APP]
    source = inspect.getsource(selftest._build_engines_worker)
    assert "paddleocr.TextDetection" in source and "paddleocr.TextRecognition" in source
    assert "os._exit(0)" in source.split("except BaseException")[-1]                 # thoát cả nhánh lỗi


def test_missing_engine_after_build_crash_is_not_remembered(tmp_path: Path) -> None:
    run, calls = _scripted({
        selftest.VARIANT_BUILD_DET: selftest.SelfTestOutcome(False, "dựng engine bộ dò: SẬP"),
        selftest.VARIANT_BUILD_REC: selftest.SelfTestOutcome(True, "ok"),
        selftest.VARIANT_DIRECT_APP: selftest.SelfTestOutcome(
            False, "TensorRT trực tiếp: DirectTrtUnavailable: bộ dò: chưa có tệp engine TensorRT trong …"),
    })
    outcome = selftest.ensure_tensorrt_selftest(tmp_path, "v", _KWARGS, 8, runner=run)
    assert not outcome.ok and not outcome.definitive and not (tmp_path / selftest.RESULT_FILE).exists()


def test_fp16_mismatch_is_not_remembered(tmp_path: Path) -> None:
    run, _calls = _scripted({
        selftest.VARIANT_BUILD_DET: selftest.SelfTestOutcome(True, "ok"),
        selftest.VARIANT_BUILD_REC: selftest.SelfTestOutcome(True, "ok"),
        selftest.VARIANT_DIRECT_APP: selftest.SelfTestOutcome(False, "kết quả KHÁC Paddle GPU … (FP16 — lệch nhỏ …)"),
    })
    outcome = selftest.ensure_tensorrt_selftest(tmp_path, "v", {**_KWARGS, "precision": "fp16"}, 8, runner=run)
    assert not outcome.ok and not outcome.definitive and not (tmp_path / selftest.RESULT_FILE).exists()


def test_engine_fingerprints_are_stored_cached_and_forgettable(tmp_path: Path) -> None:
    prints = {"engines": {"text_det_model": {"path": "a.trt", "size": 1, "mtime": 2.0}}}
    run, calls = _scripted({
        selftest.VARIANT_BUILD_DET: selftest.SelfTestOutcome(True, "ok"),
        selftest.VARIANT_BUILD_REC: selftest.SelfTestOutcome(True, "ok"),
        selftest.VARIANT_DIRECT_APP: selftest.SelfTestOutcome(True, "GIỐNG", data=prints),
    })
    first = selftest.ensure_tensorrt_selftest(tmp_path, "v", _KWARGS, 8, runner=run)
    cached = selftest.ensure_tensorrt_selftest(tmp_path, "v", _KWARGS, 8, runner=run)
    assert first.ok and first.data == prints and cached.cached and cached.data == prints and len(calls) == 3
    selftest.forget_tensorrt_selftest(tmp_path, "v", _KWARGS, 8)
    again = selftest.ensure_tensorrt_selftest(tmp_path, "v", _KWARGS, 8, runner=run)
    assert not again.cached and len(calls) == 6                                    # đã quên → kiểm lại


def test_child_message_with_data_is_passed_back(tmp_path: Path) -> None:
    from tests.unit.infrastructure import trt_fake_workers as fakes

    outcome = selftest.run_child_selftest(selftest.VARIANT_DIRECT_APP, {"init_kwargs": _KWARGS}, 8,
                                          tmp_path / "f.log", worker=fakes.ok_with_data_worker, prepare=False,
                                          timeout=120)
    assert outcome.ok and outcome.data == {"engines": {"x": 1}}


def test_direct_app_worker_requires_no_fallback_and_close_scores() -> None:
    source = inspect.getsource(selftest._direct_app_worker)
    assert "fallbacks" in source and "not used or fallbacks" in source
    assert "score_gap > 0.02" in source and "box_gap > 3.0" in source
    assert source.rstrip().endswith("os._exit(0)                                                     "
                                    "# không dọn Paddle / TensorRT lúc thoát (có thể treo)")


# ── bộ chuyển đổi ───────────────────────────────────────────────────────────────────────────────────────────────────
def _adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: dict[str, Any] | None):  # noqa: ANN202
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: tmp_path)
    monkeypatch.setattr(module, "_tensorrt_thread_ok", lambda: True)
    monkeypatch.setattr(module, "_tensorrt_crashed_last_time", lambda trt_dir: False)
    monkeypatch.setattr(module, "_tensorrt_selftest",
                        lambda *a, **k: selftest.SelfTestOutcome(True, "giả lập", data=data))
    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True, det_batch_size=8))
    adapter._try_create_model_with_retry = lambda cls, kwargs: "model"  # type: ignore[method-assign]
    return adapter


def test_adapter_passes_expected_fingerprints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.ocr import trt_direct_runner

    prints = {"text_det_model": {"path": "a"}}
    adapter = _adapter(monkeypatch, tmp_path, {"engines": prints})
    seen: list[Any] = []
    monkeypatch.setattr(trt_direct_runner, "attach_direct_tensorrt",
                        lambda model, expected=None: seen.append(expected) or ["ok"])
    adapter._create_model_with_device_fallback(object, use_gpu=True)
    assert seen == [prints] and adapter._model_uses_direct_trt


def test_changed_engine_file_forgets_the_selftest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.ocr import trt_direct_runner

    adapter = _adapter(monkeypatch, tmp_path, {"engines": {}})
    forgot: list[tuple] = []
    monkeypatch.setattr(selftest, "forget_tensorrt_selftest", lambda *a: forgot.append(a))

    def changed(model, expected=None):  # noqa: ANN001, ANN202
        raise trt_direct_runner.DirectTrtUnavailable("bộ dò: tệp engine khác lần tự kiểm (a.trt) — cần tự kiểm lại")

    monkeypatch.setattr(trt_direct_runner, "attach_direct_tensorrt", changed)
    adapter._create_model_with_device_fallback(object, use_gpu=True)
    assert adapter._tensorrt_unavailable and not adapter._model_uses_direct_trt
    assert len(forgot) == 1 and forgot[0][0] == tmp_path and forgot[0][3] == 8


def test_first_inference_guard_stays_until_direct_runner_ran(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True))
    adapter._model_uses_trt = False
    adapter._model_uses_direct_trt = True
    adapter._trt_first_inference_done = False
    adapter._trt_dir = tmp_path
    calls = {"n": 0}
    monkeypatch.setattr(adapter, "_direct_trt_calls", lambda: calls["n"])
    monkeypatch.setattr(adapter, "_predict_batch", lambda images, indices, config: ["ok"])
    image = [np.zeros((4, 4, 3), np.uint8)]
    adapter._predict_with_tensorrt_fallback(image, [0], OcrEngineConfig())
    assert adapter._trt_first_inference_done is False                    # lô vừa rồi lùi Paddle: vẫn canh lô sau
    calls["n"] = 1
    adapter._predict_with_tensorrt_fallback(image, [0], OcrEngineConfig())
    assert adapter._trt_first_inference_done is True


def test_direct_trt_calls_counts_runner_batches() -> None:
    from types import SimpleNamespace

    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    adapter = module.PaddleOcrAdapter(OcrEngineConfig())
    assert adapter._direct_trt_calls() == 0                              # chưa có mô hình
    pipeline = SimpleNamespace(text_det_model=SimpleNamespace(runner=SimpleNamespace(calls=2)),
                               text_rec_model=SimpleNamespace(runner=SimpleNamespace(calls=3)))
    adapter._model = SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=pipeline))
    assert adapter._direct_trt_calls() == 5
