"""[v3.23.775] Engine TensorRT nhận lô lớn hơn 8 (chủ dự án: "Dựng engine nhận lô lớn hơn 8").

Log 19/11 22:17 (bản 3.23.774): TensorRT trực tiếp chạy (×1,24 so với Paddle GPU) nhưng engine bộ dò / bộ nhận dạng chỉ
nhận lô ≤ 8 — bộ dò 64 ảnh/lô chia 8 lần, bộ nhận dạng 16 dòng/lô chia 2. Nay bộ dò 16, bộ nhận dạng tối ưu 16 / tối đa 32.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan
from subtitles_extractor.infrastructure.ocr import tensorrt_selftest as selftest
from subtitles_extractor.infrastructure.ocr import trt_direct_runner as direct


def test_profiles_take_more_than_eight() -> None:
    (low, opt, high), = plan.TRT_DET_PROFILE.values()
    assert opt[0] == high[0] == 16 and high[2:] == [512, 2048]                 # kích thước ảnh giữ nguyên
    paddlex = {"x": [[1, 3, 48, 160], [1, 3, 48, 320], [8, 3, 48, 3200]]}      # inference.yml PP-OCRv6_small_rec
    assert plan.rec_profile(paddlex) == {"x": [[1, 3, 48, 160], [16, 3, 48, 320], [32, 3, 48, 3200]]}
    assert plan.rec_profile({"x": [[1, 3, 48, 160], [64, 3, 48, 320], [64, 3, 48, 3200]]})["x"][2][0] == 64  # không hạ
    assert plan.rec_profile(None) is None and plan.rec_profile({"x": [[1], [2]]}) is None


def test_old_batch8_engines_are_rebuilt() -> None:
    signature = json.loads(plan._profile_signature({"trt_dynamic_shapes": plan.TRT_DET_PROFILE}))
    assert signature["app_profile"] == 2


def test_rec_profile_switch_off_keeps_paddlex(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    from tests.unit.infrastructure.test_tensorrt_det_profile_v323765 import (
        _fake_runner_module,
    )

    module, calls = _fake_runner_module()
    monkeypatch.setattr("importlib.import_module", lambda name: module)
    assert plan.install_paddlex_trt_profile_patch()
    monkeypatch.setenv("SE_TRT_REC_PROFILE", "0")
    shapes = {"x": [[1, 3, 48, 160], [1, 3, 48, 320], [8, 3, 48, 3200]]}
    module.PaddleStaticRunner("PP-OCRv6_small_rec", {"run_mode": "trt_fp32", "trt_dynamic_shapes": shapes}) \
        ._configure_trt("m", "p", tmp_path)
    assert calls[-1][1] == shapes


def test_engine_vram_is_reported() -> None:
    engine = SimpleNamespace(max_shape=[16, 3, 512, 2048], min_shape=[1, 3, 32, 32], device_memory_mb=1234)
    pipeline = SimpleNamespace()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        for name in ("text_det_model", "text_rec_model"):
            model_dir = Path(tmp) / name
            trt_dir = model_dir / ".cache" / "paddle" / "trt" / "inference"
            trt_dir.mkdir(parents=True)
            (trt_dir / "engine_1_1.trt").write_bytes(b"x")
            setattr(pipeline, name, SimpleNamespace(runner=lambda x: x, model_dir=str(model_dir), model_name=name))
        model = SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=pipeline))
        direct._ENGINES.clear()
        try:
            described = direct.attach_direct_tensorrt(model, engine_factory=lambda path: engine)
        finally:
            direct._ENGINES.clear()
    assert all("lô ≤ 16" in text and "VRAM 1234 MB" in text for text in described)


def test_out_of_vram_selftest_is_not_remembered(tmp_path) -> None:  # noqa: ANN001
    results = {selftest.VARIANT_BUILD_DET: selftest.SelfTestOutcome(True, "ok"),
               selftest.VARIANT_BUILD_REC: selftest.SelfTestOutcome(True, "ok"),
               selftest.VARIANT_DIRECT_APP: selftest.SelfTestOutcome(
                   False, "DirectTrtUnavailable: TensorRT không tạo được context cho engine_1.trt (thiếu VRAM?)")}
    outcome = selftest.ensure_tensorrt_selftest(tmp_path, "v", {"device": "gpu", "use_tensorrt": True}, 64,
                                                runner=lambda variant, *a: results[variant])
    assert not outcome.ok and not outcome.definitive and not (tmp_path / selftest.RESULT_FILE).exists()
