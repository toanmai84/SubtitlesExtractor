"""[v3.23.765] Engine TensorRT bộ dò cho lô 8 (thay lô 1 của ``inference.yml``) + lưới an toàn hình dạng.

Chủ dự án (19/11): "Sao không dựng lại engine bộ dò cho lô lớn hơn". Ảnh vào bộ dò = dải phụ đề (phim 720×1280: ROI
720×118 phóng cao 256 + viền 8 → ~1600×288). PaddleX chỉ kiểm cache engine có tồn tại (``<mô hình>/.cache/paddle/trt``):
đổi hồ sơ / FP16↔FP32 từng dùng lại engine CŨ — lô lớn đưa vào engine lô 1 làm SẬP tiến trình (log 19/11 14:46).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan

_LIMITS = {"x": ([1, 3, 32, 32], [8, 3, 512, 2048])}


def test_large_batch_split_into_engine_sized_chunks() -> None:
    batch = np.zeros((20, 3, 288, 1600), np.float32)
    chunks = plan.check_and_split_inputs(_LIMITS, [batch])
    assert chunks is not None and [chunk[0].shape[0] for chunk in chunks] == [8, 8, 4]
    assert plan.check_and_split_inputs(_LIMITS, [np.zeros((8, 3, 288, 1600))]) is not None


@pytest.mark.parametrize("shape", [(1, 3, 288, 3277), (1, 3, 16, 640), (1, 3, 600, 1600), (1, 3, 288)])
def test_out_of_profile_shapes_raise_python_error(shape: tuple[int, ...]) -> None:
    with pytest.raises(RuntimeError, match="TensorRT"):                  # bộ chuyển đổi OCR lùi GPU thường
        plan.check_and_split_inputs(_LIMITS, [np.zeros(shape, np.float32)])


def test_no_profile_or_multi_input_left_alone() -> None:
    assert plan.check_and_split_inputs({}, [np.zeros((2, 3, 32, 32))]) is None
    assert plan.check_and_split_inputs(_LIMITS, [np.zeros((1, 3, 32, 32)), np.zeros(1)]) is None


def test_det_profile_fits_the_real_subtitle_band() -> None:
    (low, _opt, high), = plan.TRT_DET_PROFILE.values()
    real = [8, 3, 288, 1600]                                              # phim 大乾悍刀行: ROI 720×118 → ~1600×288
    assert all(low[i] <= real[i] <= high[i] for i in range(4)) and high[0] == 16          # [v3.23.775] lô 8 → 16


# ── vá PaddleStaticRunner (giả lập đúng chữ ký PaddleX 3.7.2) ────────────────────────────────────────────────────────
def _fake_runner_module():  # noqa: ANN202
    module = types.ModuleType("paddlex.inference.models.runners.paddle_static.runner")
    module.USE_PIR_TRT = True  # type: ignore[attr-defined]
    calls: list = []

    class PaddleStaticRunner:
        def __init__(self, model_name: str, config: dict) -> None:
            self._model_name, self._config = model_name, config

        def _configure_trt(self, model_file, params_file, cache_dir):  # noqa: ANN001, ANN202
            trt_dir = Path(cache_dir) / "trt"
            calls.append(("configure", self._config["trt_dynamic_shapes"], (trt_dir / "det.json").exists()))
            trt_dir.mkdir(parents=True, exist_ok=True)
            (trt_dir / "det.json").write_text("engine")
            return "config"

        def __call__(self, x):  # noqa: ANN001, ANN204
            calls.append(("call", x[0].shape))
            return [x[0][:, :1] * 2]

    module.PaddleStaticRunner = PaddleStaticRunner  # type: ignore[attr-defined]
    return module, calls


@pytest.fixture()
def runner(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    module, calls = _fake_runner_module()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("importlib.import_module", lambda name: module)
    assert plan.install_paddlex_trt_profile_patch() and plan.install_paddlex_trt_profile_patch()   # không vá chồng
    return module, calls


def _det(module, run_mode: str = "trt_fp32"):  # noqa: ANN001, ANN202
    return module.PaddleStaticRunner("PP-OCRv6_small_det", {
        "run_mode": run_mode, "trt_dynamic_shapes": {"x": [[1, 3, 32, 32], [1, 3, 736, 736], [1, 3, 4000, 4000]]}})


def test_det_engine_uses_batch8_profile_and_stale_cache_is_rebuilt(runner, tmp_path: Path) -> None:  # noqa: ANN001
    module, calls = runner
    (tmp_path / "trt").mkdir()
    (tmp_path / "trt" / "det.json").write_text("engine lô 1 (bản cũ, không có dấu hồ sơ)")
    _det(module)._configure_trt("m", "p", tmp_path)
    assert calls[-1] == ("configure", plan.TRT_DET_PROFILE, False)      # cache cũ đã xoá → dựng lại theo hồ sơ ứng dụng
    _det(module)._configure_trt("m", "p", tmp_path)
    assert calls[-1][2] is True                                          # cùng hồ sơ: dùng lại cache
    _det(module, "trt_fp16")._configure_trt("m", "p", tmp_path)
    assert calls[-1][2] is False                                         # FP32 → FP16: dựng lại


def test_rec_keeps_its_sizes_gets_bigger_batch_and_split(runner, tmp_path: Path) -> None:  # noqa: ANN001
    module, calls = runner
    rec = module.PaddleStaticRunner("PP-OCRv6_small_rec", {
        "run_mode": "trt_fp32", "trt_dynamic_shapes": {"x": [[1, 3, 48, 160], [1, 3, 48, 320], [8, 3, 48, 3200]]}})
    rec._configure_trt("m", "p", tmp_path)
    # [v3.23.775] Lô tối ưu 16 / tối đa 32, chiều cao + rộng giữ của PaddleX.
    assert calls[-1][1]["x"] == [[1, 3, 48, 160], [16, 3, 48, 320], [32, 3, 48, 3200]]
    batch = np.arange(70 * 3 * 48 * 320, dtype=np.float32).reshape(70, 3, 48, 320)
    (out,) = rec([batch])
    assert out.shape == (70, 1, 48, 320) and np.array_equal(out, batch[:, :1] * 2)   # nối đúng thứ tự
    assert [c[1][0] for c in calls if c[0] == "call"] == [32, 32, 6]


def test_runner_without_trt_untouched(runner) -> None:  # noqa: ANN001
    module, calls = runner
    plain = module.PaddleStaticRunner("PP-OCRv6_small_det", {"run_mode": "paddle"})
    plain([np.zeros((64, 3, 288, 1600), np.float32)])
    assert [c for c in calls if c[0] == "call"] == [("call", (64, 3, 288, 1600))]


def test_adapter_installs_the_patch_and_skips_hard_caps() -> None:
    import inspect

    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter

    assert "install_paddlex_trt_profile_patch()" in inspect.getsource(paddle_ocr_adapter._prepare_paddle_tensorrt)
    assert "not _TRT_PROFILE_PATCHED" in inspect.getsource(paddle_ocr_adapter.PaddleOcrAdapter._apply_det_batch_size)
