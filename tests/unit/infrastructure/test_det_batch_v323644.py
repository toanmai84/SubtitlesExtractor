"""[v3.23.644] Batch bộ dò chữ PaddleOCR: áp dụng đúng chỗ, lỗi cấu trúc chỉ cảnh báo."""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter


def _fake_model():
    sampler = lambda: SimpleNamespace(batch_size=1)          # noqa: E731
    pipeline = SimpleNamespace(batch_sampler=sampler(), text_det_model=SimpleNamespace(batch_sampler=sampler()))
    return SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=pipeline)), pipeline


def test_det_batch_size_is_set_on_both_samplers() -> None:
    model, pipeline = _fake_model()
    PaddleOcrAdapter(OcrEngineConfig(det_batch_size=8))._apply_det_batch_size(model)
    assert pipeline.batch_sampler.batch_size == 8
    assert pipeline.text_det_model.batch_sampler.batch_size == 8


def test_default_keeps_the_old_behaviour() -> None:
    model, pipeline = _fake_model()
    PaddleOcrAdapter(OcrEngineConfig())._apply_det_batch_size(model)
    assert pipeline.batch_sampler.batch_size == 1 and pipeline.text_det_model.batch_sampler.batch_size == 1


def test_unexpected_paddle_structure_does_not_raise() -> None:
    PaddleOcrAdapter(OcrEngineConfig(det_batch_size=8))._apply_det_batch_size(SimpleNamespace())
    PaddleOcrAdapter(OcrEngineConfig(det_batch_size=8))._apply_det_batch_size(None)


def test_changing_det_batch_size_reloads_the_model() -> None:
    adapter = PaddleOcrAdapter(OcrEngineConfig(det_batch_size=1))
    assert adapter._needs_model_reload(OcrEngineConfig(det_batch_size=8))
    assert not adapter._needs_model_reload(OcrEngineConfig(det_batch_size=1))
