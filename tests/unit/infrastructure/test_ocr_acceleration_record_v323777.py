"""[v3.23.777] Tệp mô tả lượt chạy ghi rõ đường OCR THẬT SỰ chạy (CLAUDE.md: "ghi rõ đường đã chạy vào log / tệp mô tả").

Lượt cả bộ 20/11 (bản 3.23.776, 228 675 khung, OCR 1532 s): ``seraw`` chỉ ghi ``"ocr_engine": "PaddleOcrAdapter"`` và
``use_tensorrt: true`` (cài đặt) — không biết TensorRT có chạy thật hay đã lùi GPU thường, nên không so được tốc độ với
lượt không TensorRT (3.23.728: 1538 s). Nay ``environment.ocr_acceleration`` ghi đường chạy + số lô TensorRT / lùi Paddle.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.run_provenance import collect_run_info
from subtitles_extractor.domain.ports.ocr_engine_port import (
    OcrEngineConfig,
    PrecisionMode,
)
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter


def _adapter(**config) -> PaddleOcrAdapter:  # noqa: ANN003
    return PaddleOcrAdapter(OcrEngineConfig(**config))


def _with_runners(adapter: PaddleOcrAdapter, det: tuple[int, int, int], rec: tuple[int, int, int]) -> None:
    def runner(limit: int, calls: int, fallbacks: int) -> SimpleNamespace:
        return SimpleNamespace(_engine=SimpleNamespace(max_shape=[limit, 3, 48, 3200]), calls=calls, fallbacks=fallbacks)

    pipeline = SimpleNamespace(text_det_model=SimpleNamespace(runner=runner(*det)),
                               text_rec_model=SimpleNamespace(runner=runner(*rec)))
    adapter._model = SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=pipeline))
    adapter._active_device = DeviceKind.GPU


def test_direct_tensorrt_path_with_batch_counts() -> None:
    adapter = _adapter(use_tensorrt=True, precision=PrecisionMode.FP16)
    _with_runners(adapter, (8, 28584, 0), (8, 40211, 3))
    adapter._model_uses_direct_trt = True
    text = adapter.acceleration_description
    assert text.startswith("gpu+tensorrt_direct fp16")
    assert "bộ dò ≤8: 28584 lô TensorRT, 0 lô lùi Paddle GPU" in text
    assert "bộ nhận dạng ≤8: 40211 lô TensorRT, 3 lô lùi Paddle GPU" in text


def test_plain_paths() -> None:
    assert _adapter().acceleration_description == "chưa khởi tạo"
    adapter = _adapter(use_tensorrt=True)
    _with_runners(adapter, (8, 0, 0), (8, 0, 0))
    adapter._tensorrt_unavailable = True
    assert adapter.acceleration_description.startswith("gpu (TensorRT đã bật nhưng không dùng được")
    plain = _adapter()
    _with_runners(plain, (8, 0, 0), (8, 0, 0))
    assert plain.acceleration_description == "gpu"
    plain._active_device = DeviceKind.CPU
    assert plain.acceleration_description == "cpu"


def test_run_info_records_the_path() -> None:
    adapter = _adapter(use_tensorrt=True)
    _with_runners(adapter, (16, 5, 0), (32, 7, 0))
    adapter._model_uses_direct_trt = True
    request = SimpleNamespace(video_path="x.mp4")
    info = collect_run_info(run_id="r", app_version="v", request=request, metadata=None, builder_config=None,
                            ocr_engine=adapter)
    assert info["environment"]["ocr_acceleration"].startswith("gpu+tensorrt_direct fp32 (bộ dò ≤16: 5 lô TensorRT")
    other = collect_run_info(run_id="r", app_version="v", request=request, metadata=None, builder_config=None,
                             ocr_engine=SimpleNamespace())
    assert other["environment"]["ocr_acceleration"] is None and "collect_error" not in other
