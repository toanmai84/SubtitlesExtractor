"""[v3.23.766] Tự kiểm TensorRT ở tiến trình riêng — lỗi sập C++ của TensorRT không còn giết ứng dụng.

Log + ``fault.log`` 19/11 14:46 (3.23.763) và 15:46 (3.23.765): "Windows fatal exception: access violation" trong
``predictor.run()`` của bộ dò, ngay lần suy luận đầu — sau cả khi lô đã vừa hồ sơ engine (Paddle C++ tự kiểm hình dạng
theo hồ sơ, nên sập nằm trong ``enqueueV3``). Test dưới chạy tiến trình con THẬT (``spawn``), kể cả sập bộ nhớ thật.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.ocr import tensorrt_selftest as selftest
from tests.unit.infrastructure import trt_fake_workers as fakes

_KWARGS = {"device": "gpu", "use_tensorrt": True, "precision": "fp32", "text_detection_model_name": "PP-OCRv6_small_det"}
_PAYLOAD = {"init_kwargs": _KWARGS, "config": None}
#: [v3.23.774] Chuỗi tự kiểm: dựng engine bộ dò → bộ nhận dạng (tiến trình riêng từng mô hình) → TensorRT trực tiếp.
_CHAIN = [selftest.VARIANT_BUILD_DET, selftest.VARIANT_BUILD_REC, selftest.VARIANT_DIRECT_APP]


def _run(worker, tmp_path: Path, variant: str = selftest.VARIANT_CROSS_THREAD, timeout: float = 120.0):  # noqa: ANN001, ANN202
    return selftest.run_child_selftest(variant, _PAYLOAD, 8, tmp_path / "fault.log", worker=worker, prepare=False,
                                       timeout=timeout)


def test_real_memory_crash_in_child_is_contained_and_reported(tmp_path: Path) -> None:
    outcome = _run(fakes.crash_worker, tmp_path)                      # tiến trình chính (pytest) vẫn sống
    assert not outcome.ok and "SẬP" in outcome.reason and "mã thoát" in outcome.reason
    assert "Segmentation fault" in outcome.reason or "access violation" in outcome.reason or "Fatal" in outcome.reason


def test_ok_error_and_hang(tmp_path: Path) -> None:
    ok = _run(fakes.ok_worker, tmp_path)
    assert ok.ok and "8 ảnh, fp32" in ok.reason
    error = _run(fakes.error_worker, tmp_path)
    assert not error.ok and "dựng engine lỗi" in error.reason
    hang = _run(fakes.hang_worker, tmp_path, timeout=3)
    assert not hang.ok and "treo" in hang.reason


def _runner(worker):  # noqa: ANN001, ANN202
    calls: list[str] = []

    def run(variant, init_kwargs, det_batch, fault_path):  # noqa: ANN001, ANN202
        calls.append(variant)
        return selftest.run_child_selftest(variant, init_kwargs, det_batch, fault_path, worker=worker, prepare=False,
                                           timeout=120)

    return run, calls


def test_failure_is_cached(tmp_path: Path) -> None:
    run, calls = _runner(fakes.crash_worker)
    outcome = selftest.ensure_tensorrt_selftest(tmp_path, "3.23.774", _KWARGS, 64, runner=run)
    # [v3.23.774] Chuỗi mới: dựng engine từng mô hình → TensorRT trực tiếp (ứng dụng chạy engine, so kết quả).
    assert not outcome.ok and calls == _CHAIN
    again = selftest.ensure_tensorrt_selftest(tmp_path, "3.23.774", _KWARGS, 64, runner=run)
    assert again.cached and not again.ok and len(calls) == 3                                      # nhớ: không chạy lại


def test_full_diagnostics_on_request_are_not_remembered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SE_TRT_FULL_DIAG", "1")
    run, calls = _runner(fakes.crash_worker)
    outcome = selftest.ensure_tensorrt_selftest(tmp_path, "3.23.774", _KWARGS, 64, runner=run)
    components = [name for name, *_rest in selftest.COMPONENT_VARIANTS]
    assert calls == [selftest.VARIANT_DEDICATED_THREAD, selftest.VARIANT_DIRECT_ENGINE, selftest.VARIANT_CROSS_THREAD,
                     *components]
    assert not outcome.ok and not (tmp_path / selftest.RESULT_FILE).exists()


def test_paddle_build_crash_after_writing_engines_still_allows_direct(tmp_path: Path) -> None:
    """Log 21:12: Paddle sập ở bước dọn SAU khi đã ghi tệp engine (``converter.py:641``) — chạy trực tiếp vẫn được."""
    run, calls = _runner(fakes.crash_on_build_ok_direct)
    outcome = selftest.ensure_tensorrt_selftest(tmp_path, "3.23.774", _KWARGS, 64, runner=run)
    assert outcome.ok and calls == _CHAIN
    assert "SẬP" in outcome.reason and "chạy được" in outcome.reason


def test_new_version_or_config_retests(tmp_path: Path) -> None:
    run, calls = _runner(fakes.ok_worker)
    assert selftest.ensure_tensorrt_selftest(tmp_path, "3.23.766", _KWARGS, 64, runner=run).ok
    selftest.ensure_tensorrt_selftest(tmp_path, "3.23.767", _KWARGS, 64, runner=run)               # bản mới
    selftest.ensure_tensorrt_selftest(tmp_path, "3.23.767", {**_KWARGS, "precision": "fp16"}, 64, runner=run)
    selftest.ensure_tensorrt_selftest(tmp_path, "3.23.767", {**_KWARGS, "precision": "fp16"}, 8, runner=run)
    assert len(calls) == 12                                                                       # 4 cấu hình × 3 bước
    selftest.ensure_tensorrt_selftest(tmp_path, "3.23.767", {**_KWARGS, "precision": "fp16"}, 8, runner=run)
    assert len(calls) == 12                                                                       # nhớ kết quả


def test_env_switch_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SE_TRT_SELFTEST", "0")
    run, calls = _runner(fakes.crash_worker)
    assert selftest.ensure_tensorrt_selftest(tmp_path, "x", _KWARGS, 8, runner=run).ok and calls == []


def test_synthetic_images_match_subtitle_band() -> None:
    pytest.importorskip("cv2")
    images = selftest.synthetic_images(3)
    assert len(images) == 3 and images[0].shape == (272, 1578, 3) and images[0].max() == 255


# ── bộ chuyển đổi OCR ───────────────────────────────────────────────────────────────────────────────────────────────
def _adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ok: bool):  # noqa: ANN202
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: tmp_path)
    monkeypatch.setattr(module, "_tensorrt_thread_ok", lambda: True)                 # [v3.23.767] đang ở luồng Paddle
    seen: list = []
    monkeypatch.setattr(module, "_tensorrt_selftest", lambda trt_dir, kwargs, det, config=None: seen.append((kwargs, det)) or
                        selftest.SelfTestOutcome(ok, "khác luồng: tiến trình tự kiểm SẬP (mã thoát 0xc0000005)"))
    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True, det_batch_size=64))
    built: list[dict] = []
    adapter._try_create_model_with_retry = lambda cls, kwargs: built.append(kwargs) or "model"  # type: ignore[method-assign]
    return adapter, built, seen


def test_adapter_runs_selftest_and_falls_back_when_it_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter, built, seen = _adapter(monkeypatch, tmp_path, ok=False)
    assert adapter._create_model_with_device_fallback(object, use_gpu=True) == "model"
    assert seen[0][0]["use_tensorrt"] is True and seen[0][1] == 64                 # đúng cấu hình + lô của ứng dụng
    assert "use_tensorrt" not in built[0] and adapter._tensorrt_unavailable and not adapter._model_uses_trt


def test_adapter_uses_direct_tensorrt_when_selftest_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """[v3.23.774] Tự kiểm đạt → KHÔNG dựng predictor TensorRT của Paddle (sập); mô hình GPU thường + runner trực tiếp."""
    from subtitles_extractor.infrastructure.ocr import trt_direct_runner

    adapter, built, _seen = _adapter(monkeypatch, tmp_path, ok=True)
    attached: list = []
    monkeypatch.setattr(trt_direct_runner, "attach_direct_tensorrt", lambda model, **_kw: attached.append(model) or ["bộ dò …"])
    adapter._create_model_with_device_fallback(object, use_gpu=True)
    assert "use_tensorrt" not in built[0] and attached == ["model"]
    assert adapter._model_uses_direct_trt and not adapter._model_uses_trt


def test_direct_attach_failure_falls_back_to_plain_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.ocr import trt_direct_runner

    adapter, built, _seen = _adapter(monkeypatch, tmp_path, ok=True)

    def fail(model, **_kw):  # noqa: ANN001, ANN003, ANN202
        raise trt_direct_runner.DirectTrtUnavailable("chưa có engine")

    monkeypatch.setattr(trt_direct_runner, "attach_direct_tensorrt", fail)
    assert adapter._create_model_with_device_fallback(object, use_gpu=True) == "model"
    assert len(built) == 1 and not adapter._model_uses_direct_trt and adapter._tensorrt_unavailable


def test_selftest_worker_is_importable_for_spawn() -> None:
    assert selftest._selftest_worker.__module__ == "subtitles_extractor.infrastructure.ocr.tensorrt_selftest"
    assert sys.modules[selftest._selftest_worker.__module__] is selftest


# ── [v3.23.766] lượt rà độc lập ─────────────────────────────────────────────────────────────────────────────────────
def test_transient_failures_are_not_remembered(tmp_path: Path) -> None:
    calls: list[str] = []

    def short(variant, payload, det_batch, fault_path):  # noqa: ANN001, ANN202
        calls.append(variant)
        return selftest.run_child_selftest(variant, payload, det_batch, fault_path, worker=fakes.hang_worker,
                                           prepare=False, timeout=2)

    outcome = selftest.ensure_tensorrt_selftest(tmp_path, "3.23.774", _KWARGS, 8, runner=short)
    assert not outcome.ok and not outcome.definitive and len(calls) == 3
    assert not (tmp_path / selftest.RESULT_FILE).exists()                                         # lần sau kiểm lại
    oom = selftest.run_child_selftest("khác luồng", _PAYLOAD, 8, tmp_path / "f.log", worker=fakes.oom_worker,
                                      prepare=False, timeout=60)
    assert not oom.ok and not oom.definitive


def test_signature_ignores_thresholds_and_keeps_entries_per_config(tmp_path: Path) -> None:
    base = selftest.selftest_signature("v", {**_KWARGS, "text_det_thresh": 0.3, "cpu_threads": 16}, 8)
    assert base == selftest.selftest_signature("v", {**_KWARGS, "text_det_thresh": 0.5, "cpu_threads": 4}, 8)
    assert base != selftest.selftest_signature("v", {**_KWARGS, "precision": "fp16"}, 8)
    run, calls = _runner(fakes.ok_worker)
    other = {**_KWARGS, "lang": "en"}                                              # bộ OCR phụ đề nhúng (ngôn ngữ khác)
    for _ in range(2):
        selftest.ensure_tensorrt_selftest(tmp_path, "v", _KWARGS, 8, runner=run)
        selftest.ensure_tensorrt_selftest(tmp_path, "v", other, 8, runner=run)
    assert len(calls) == 6                                                          # 2 cấu hình × 3 bước, không đè


def test_selftest_machinery_error_falls_back_without_blocking_ocr(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: tmp_path)
    monkeypatch.setattr(module, "_tensorrt_thread_ok", lambda: True)

    def broken(*_args):  # noqa: ANN002, ANN202
        raise BrokenPipeError("PeekNamedPipe")

    monkeypatch.setattr(module, "_tensorrt_selftest", broken)
    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True))
    built: list[dict] = []
    adapter._try_create_model_with_retry = lambda cls, kwargs: built.append(kwargs) or "model"  # type: ignore[method-assign]
    assert adapter._create_model_with_device_fallback(object, use_gpu=True) == "model"
    assert "use_tensorrt" not in built[0]


def test_unprepared_tensorrt_is_never_used_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: None)
    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True))
    built: list[dict] = []
    adapter._try_create_model_with_retry = lambda cls, kwargs: built.append(kwargs) or "model"  # type: ignore[method-assign]
    adapter._create_model_with_device_fallback(object, use_gpu=True)
    assert "use_tensorrt" not in built[0] and adapter._tensorrt_unavailable


def test_every_new_tensorrt_model_gets_a_first_inference_guard(monkeypatch: pytest.MonkeyPatch,
                                                               tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.ocr import trt_direct_runner

    adapter, _built, _seen = _adapter(monkeypatch, tmp_path, ok=True)
    monkeypatch.setattr(trt_direct_runner, "attach_direct_tensorrt", lambda model, **_kw: ["bộ dò …"])
    adapter._trt_first_inference_done = True                                       # mô hình trước trong phiên
    adapter._create_model_with_device_fallback(object, use_gpu=True)
    assert adapter._model_uses_direct_trt and adapter._trt_first_inference_done is False


def test_killed_child_is_not_a_crash(tmp_path: Path) -> None:
    outcome = selftest.run_child_selftest("khác luồng", _PAYLOAD, 8, tmp_path / "f.log", worker=fakes.terminated_worker,
                                          prepare=False, timeout=60)
    assert not outcome.ok and not outcome.definitive and "bị dừng" in outcome.reason


# ── [v3.23.767] luồng Paddle cố định cho TensorRT ───────────────────────────────────────────────────────────────────
def _fresh_paddle(monkeypatch: pytest.MonkeyPatch, module):  # noqa: ANN001, ANN202
    """paddle CHƯA nạp; ``_import_paddle_modules`` giả (đưa mô-đun paddle giả vào sys.modules) — ghi luồng nạp."""
    import threading
    import types

    seen: dict[str, str] = {}
    monkeypatch.delitem(sys.modules, "paddle", raising=False)
    monkeypatch.setattr(module, "_PADDLE_IMPORT_THREAD", None)

    def fake_import() -> tuple:
        seen["import"] = threading.current_thread().name
        fake = types.ModuleType("paddle")
        monkeypatch.setitem(sys.modules, "paddle", fake)            # tự gỡ sau test (không rò mô-đun giả)
        return fake, object

    monkeypatch.setattr(module.PaddleOcrAdapter, "_import_paddle_modules", staticmethod(fake_import))
    return seen


def test_tensorrt_import_and_build_on_the_paddle_thread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import threading

    import numpy as np

    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.domain.value_objects.device_kind import DeviceKind
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module
    from subtitles_extractor.infrastructure.ocr import trt_direct_runner

    seen = _fresh_paddle(monkeypatch, module)
    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: tmp_path)
    monkeypatch.setattr(module, "_tensorrt_selftest", lambda *a: selftest.SelfTestOutcome(True, "giả lập"))
    monkeypatch.setattr(trt_direct_runner, "attach_direct_tensorrt", lambda model, **_kw: ["bộ dò …"])
    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True, device=DeviceKind.GPU))
    monkeypatch.setattr(adapter, "_resolve_use_gpu", lambda paddle_module: True)
    monkeypatch.setattr(adapter, "_try_create_model_with_retry", lambda cls, kwargs: (
        seen.__setitem__("build", threading.current_thread().name), seen.__setitem__("trt", kwargs.get("use_tensorrt")),
        "gpu-model")[-1])
    try:
        adapter.initialize()                                           # gọi từ luồng test (≈ luồng nạp sẵn)
        assert seen["import"].startswith("paddle-tensorrt") and seen["build"].startswith("paddle-tensorrt")
        assert seen["trt"] is None and adapter._model_uses_direct_trt
        adapter._trt_dir = None
        monkeypatch.setattr(adapter, "_predict_batch", lambda images, indices, config: ["ok"])
        assert adapter._predict_with_tensorrt_fallback([np.zeros((4, 4, 3), np.uint8)], [0], OcrEngineConfig()) == ["ok"]
    finally:
        monkeypatch.setattr(module, "_PADDLE_IMPORT_THREAD", None)


def test_paddle_imported_elsewhere_disables_tensorrt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    monkeypatch.setattr(module, "_PADDLE_IMPORT_THREAD", -1)                 # paddle nạp ở luồng khác từ trước
    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: tmp_path)
    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True))
    built: list[dict] = []
    adapter._try_create_model_with_retry = lambda cls, kwargs: built.append(kwargs) or "model"  # type: ignore[method-assign]
    module._run_on_paddle_thread(adapter._create_model_with_device_fallback, object, use_gpu=True)
    assert "use_tensorrt" not in built[0] and adapter._tensorrt_unavailable


def test_plain_gpu_does_not_use_the_paddle_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.domain.value_objects.device_kind import DeviceKind
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=False, device=DeviceKind.GPU))
    where: list[str] = []
    monkeypatch.setattr(adapter, "_initialize_model", lambda: (where.append(threading.current_thread().name),
                                                              setattr(adapter, "_model", "m")))
    adapter.initialize()
    assert where == [threading.current_thread().name]


def test_run_on_paddle_thread_is_reentrant() -> None:
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    assert module._run_on_paddle_thread(lambda: module._run_on_paddle_thread(lambda: 7)) == 7      # không tự khoá chết


# ── [v3.23.767] rà độc lập: huỷ mô hình TensorRT trên ĐÚNG luồng Paddle; luồng Paddle không giữ tiến trình sống ─────────
class _TrtModel:
    def __init__(self, sink: list[str], order: list[str]) -> None:
        self._sink, self._order = sink, order

    def __del__(self) -> None:
        import threading

        self._sink.append(threading.current_thread().name)
        self._order.append("del")


def _trt_adapter(monkeypatch: pytest.MonkeyPatch, sink: list[str], order: list[str]):  # noqa: ANN202
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True))
    adapter._model = _TrtModel(sink, order)
    adapter._model_uses_trt = True
    monkeypatch.setattr(adapter, "_empty_paddle_vram", lambda: order.append("empty_cache"))
    monkeypatch.setattr(adapter, "_empty_cupy_pool", lambda: None)
    return adapter, module


def test_release_internal_destroys_trt_model_on_paddle_thread_before_freeing_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: list[str] = []
    order: list[str] = []
    adapter, _module = _trt_adapter(monkeypatch, sink, order)
    adapter._release_internal()
    assert sink and sink[0].startswith("paddle-tensorrt") and order[:2] == ["del", "empty_cache"]


def test_background_release_destroys_trt_model_on_paddle_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: list[str] = []
    order: list[str] = []
    adapter, module = _trt_adapter(monkeypatch, sink, order)
    adapter._release_locked()
    module._run_on_paddle_thread(lambda: None)                         # chờ việc đã xếp hàng xong
    assert sink and sink[0].startswith("paddle-tensorrt") and order[:2] == ["del", "empty_cache"]


def test_runtime_fallback_destroys_trt_model_on_paddle_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    import numpy as np

    from subtitles_extractor.domain.exceptions import OcrInferenceError
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig

    sink: list[str] = []
    order: list[str] = []
    adapter, module = _trt_adapter(monkeypatch, sink, order)
    adapter._trt_dir = None

    def predict(images, indices, config):  # noqa: ANN001, ANN202
        model = adapter._model                                          # khung giữ tham chiếu như predict của Paddle
        if model is not None and isinstance(model, _TrtModel):
            raise OcrInferenceError("TensorRT is needed, but TensorRT dynamic library 'nvinfer_plugin.dll' is not found")
        return ["ok"]

    def ensure(config) -> None:  # noqa: ANN001
        adapter._model = "gpu-model"

    monkeypatch.setattr(adapter, "_predict_batch", predict)
    monkeypatch.setattr(adapter, "_ensure_model_for", ensure)
    out: dict = {}
    worker = threading.Thread(target=lambda: out.setdefault("r", adapter._predict_with_tensorrt_fallback(
        [np.zeros((4, 4, 3), np.uint8)], [0], OcrEngineConfig())), name="extract")
    worker.start()
    worker.join()
    assert out["r"] == ["ok"] and sink and sink[0].startswith("paddle-tensorrt"), sink


def test_paddle_thread_does_not_keep_the_process_alive(tmp_path: Path) -> None:
    import subprocess
    import time

    script = (
        "import threading, time\n"
        "from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as m\n"
        "m._paddle_thread().submit(time.sleep, 30)\n"                   # vd tự kiểm / dựng engine còn chạy lúc đóng
        "print('đóng', flush=True)\n"
    )
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False,
                            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")})
    assert "đóng" in result.stdout and time.monotonic() - started < 25, result.stderr[-500:]


# ── [v3.23.768] tiến trình con chạy ĐÚNG đường mã của ứng dụng (PaddleOcrAdapter thật) ──────────────────────────────
@pytest.mark.skipif(not (Path.home() / ".paddlex" / "official_models" / "PP-OCRv6_small_det").is_dir(),
                    reason="cần mô hình PP-OCRv6_small trên máy (paddle thật)")
def test_real_child_runs_the_app_adapter_path_and_reports_diagnostics(tmp_path: Path) -> None:
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.domain.value_objects.device_kind import DeviceKind

    config = OcrEngineConfig(use_tensorrt=True, device=DeviceKind.GPU, detection_model_name="PP-OCRv6_small_det",
                             recognition_model_name="PP-OCRv6_small_rec", det_batch_size=2)
    payload = {"init_kwargs": {}, "config": config}
    dedicated = selftest.run_child_selftest(selftest.VARIANT_DEDICATED_THREAD, payload, 2, tmp_path / "f.log",
                                            prepare=False, timeout=600)
    # Máy kiểm thử không có GPU/TensorRT: bộ chuyển đổi THẬT lùi GPU thường → con báo đúng, kèm nhật ký của nó.
    assert not dedicated.ok and "KHÔNG dựng được bằng TensorRT" in dedicated.reason
    assert "luồng nạp paddle = luồng Paddle: True" in dedicated.reason and "Khởi tạo PaddleOCR" in dedicated.reason
    legacy = selftest.run_child_selftest(selftest.VARIANT_CROSS_THREAD, payload, 2, tmp_path / "f.log",
                                         prepare=False, timeout=600)
    assert "luồng nạp paddle = luồng Paddle: False" in legacy.reason                 # bố cục cũ: đúng là khác luồng
    assert not (tmp_path / "first_inference_pending.txt").exists()


@pytest.mark.skipif(not (Path.home() / ".paddlex" / "official_models" / "PP-OCRv6_small_det").is_dir(),
                    reason="cần mô hình PP-OCRv6_small trên máy (paddle thật)")
def test_real_child_with_paddle_preloaded_on_main_thread_runs_layout_there(tmp_path: Path) -> None:
    """[v3.23.769] Log 19/11 18:47 (Windows): "paddle đã nạp trước khi dựng: True" → bộ chuyển đổi tắt TensorRT ("paddle
    đã được nạp ở luồng khác") ở CẢ hai bố cục. Nay con chạy bố cục trên luồng chính (luồng đã nạp paddle)."""
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.domain.value_objects.device_kind import DeviceKind

    config = OcrEngineConfig(use_tensorrt=True, device=DeviceKind.GPU, detection_model_name="PP-OCRv6_small_det",
                             recognition_model_name="PP-OCRv6_small_rec", det_batch_size=2)
    outcome = selftest.run_child_selftest(selftest.VARIANT_DEDICATED_THREAD, {"init_kwargs": {}, "config": config}, 2,
                                          tmp_path / "f.log", worker=fakes.preload_paddle_then_real, prepare=False,
                                          timeout=600)
    assert "paddle đã nạp trước khi dựng: True (lúc con bắt đầu: True)" in outcome.reason
    assert "chạy bố cục trên luồng chính" in outcome.reason
    assert "paddle đã được nạp ở luồng khác" not in outcome.reason               # mã cũ: TensorRT bị tắt vì lý do này
    assert "KHÔNG dựng được bằng TensorRT" in outcome.reason                    # máy kiểm thử không GPU: lùi đúng


def test_import_origin_is_recorded(tmp_path: Path) -> None:
    pytest.importorskip("paddle")
    outcome = selftest.run_child_selftest(selftest.VARIANT_DEDICATED_THREAD, {"init_kwargs": {}, "config": None}, 2,
                                          tmp_path / "f.log", worker=fakes.import_paddle_in_prepare, prepare=False,
                                          timeout=600)
    assert "nạp bởi MainThread" in outcome.reason and "trt_fake_workers" in outcome.reason



def test_build_error_still_tries_direct(tmp_path: Path) -> None:
    run, calls = _runner(fakes.error_worker)
    selftest.ensure_tensorrt_selftest(tmp_path, "3.23.774", _KWARGS, 8, runner=run)
    assert calls == _CHAIN


@pytest.mark.skipif(not (Path.home() / ".paddlex" / "official_models" / "PP-OCRv6_small_rec").is_dir(),
                    reason="cần mô hình PP-OCRv6_small trên máy (paddle thật)")
@pytest.mark.parametrize("variant", ["chỉ bộ dò", "chỉ bộ nhận dạng"])
def test_real_component_child_runs(tmp_path: Path, variant: str) -> None:
    """[v3.23.770] Tiến trình con phép thử thành phần chạy hết đường mã (máy kiểm thử không GPU/TensorRT: PaddleX tự
    chuyển CPU — đủ để bắt lỗi trong chính mã phép thử)."""
    outcome = selftest.run_child_selftest(variant, {"init_kwargs": {}, "config": None}, 2, tmp_path / "f.log",
                                          prepare=False, timeout=600)
    assert "SẬP" not in outcome.reason and "treo" not in outcome.reason, outcome.reason
    assert outcome.ok or "không chuẩn bị được TensorRT" in outcome.reason, outcome.reason



# ── [v3.23.771] phép thử engine trực tiếp ───────────────────────────────────────────────────────────────────────────
def test_find_engine_files_newest_first(tmp_path: Path) -> None:
    import os
    import time

    det = tmp_path / "official_models" / "PP-OCRv6_small_det" / ".cache" / "paddle" / "trt" / "inference"
    rec = tmp_path / "official_models" / "PP-OCRv6_small_rec" / ".cache" / "paddle" / "trt" / "inference"
    det.mkdir(parents=True)
    rec.mkdir(parents=True)
    (det / "engine_1_0.trt").write_bytes(b"a")
    (det / "engine_1_0.pir").write_text("x")
    (rec / "engine_2_0.trt").write_bytes(b"b")
    now = time.time()
    os.utime(det / "engine_1_0.trt", (now - 100, now - 100))
    files = selftest.find_engine_files([tmp_path / "official_models", tmp_path / "missing"])
    assert [f.name for f in files] == ["engine_2_0.trt", "engine_1_0.trt"]
    assert selftest._engine_model_name(files[0]) == "PP-OCRv6_small_rec"


def test_direct_engine_variant_uses_its_worker(tmp_path: Path) -> None:
    outcome = selftest.run_child_selftest(selftest.VARIANT_DIRECT_ENGINE, _PAYLOAD, 2, tmp_path / "f.log",
                                          prepare=False, timeout=600)
    # Máy kiểm thử không GPU / chưa có TensorRT: con của phép thử engine trực tiếp báo đúng lỗi Python (không sập).
    # [v3.23.772] CuPy nạp qua bộ nạp của ứng dụng TRƯỚC paddle (log 20:53: "No module named 'cuda.pathfinder'").
    assert not outcome.ok and "không chuẩn bị được TensorRT" in outcome.reason, outcome.reason
    assert "cuda.pathfinder" not in outcome.reason and "SẬP" not in outcome.reason


def test_direct_engine_worker_needs_no_cupy() -> None:
    """[v3.23.773] Tiến trình con không nạp được CuPy (log 21:12: thiếu 'cuda.pathfinder') → bộ đệm GPU bằng tensor
    paddle (``data_ptr``) + luồng CUDA của paddle."""
    import inspect

    source = inspect.getsource(selftest._direct_engine_worker)
    code = [line.strip() for line in source.splitlines() if not line.strip().startswith("#")]
    assert not any("cupy" in line or "gpu_image_filters" in line for line in code)
    assert "data_ptr()" in source and "stream_base" in source


def test_paddle_gpu_tensor_pointer_api_exists() -> None:
    paddle = pytest.importorskip("paddle")
    tensor = paddle.to_tensor([1.0, 2.0])
    assert isinstance(tensor.data_ptr(), int) and tensor.data_ptr() != 0
    assert hasattr(paddle.device, "current_stream") and hasattr(paddle.device, "set_device")
