"""[v3.23.656] B1/B2: các lượt OCR dùng chung một engine không được đổi cấu hình / giải phóng model của nhau.

Trước đây ``ExtractSubtitlesUseCase.execute`` bắt đầu bằng ``apply_config(request.ocr)`` lên engine dùng chung: Re-OCR
(có tinh chỉnh riêng) chạy giữa lúc trích xuất âm thầm đổi ngưỡng/tiền xử lý của lượt trích xuất (đo trên tập 22:
17 → 4 ô chữ trên cùng 40 khung); đổi model thì giải phóng model giữa chừng (quét giao diện: cả tập trích xuất lỗi
``TypeError`` trong PaddleX).
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

import numpy as np

from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter


class _FakeModel:
    def __init__(self, name: str, delay: float = 0.0) -> None:
        self.name = name
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self.released = False

    def predict(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        time.sleep(self.delay)
        assert not self.released, "model bị giải phóng khi đang suy luận"
        return [{"rec_texts": [], "rec_scores": [], "rec_polys": []} for _ in kwargs["input"]]


def _adapter(monkeypatch, base: OcrEngineConfig, delay: float = 0.0) -> tuple[PaddleOcrAdapter, list[_FakeModel]]:
    adapter = PaddleOcrAdapter(base)
    built: list[_FakeModel] = []

    def fake_initialize() -> None:
        if adapter._model is None:
            model = _FakeModel(adapter._config.detection_model_name or "?", delay)
            built.append(model)
            adapter._model = model

    def fake_release() -> None:
        if adapter._model is not None:
            adapter._model.released = True
            adapter._model = None

    monkeypatch.setattr(adapter, "initialize", fake_initialize)
    monkeypatch.setattr(adapter, "_release_internal", fake_release)
    return adapter, built


IMG = [np.zeros((40, 200, 3), np.uint8)]


def test_run_config_is_not_hijacked_by_another_run(monkeypatch) -> None:
    run_a = OcrEngineConfig(score_threshold=0.5)
    adapter, built = _adapter(monkeypatch, run_a)
    adapter.infer_batch(IMG, [0], [0.0], config=run_a)
    adapter.apply_config(dataclasses.replace(run_a, score_threshold=0.9999))   # Re-OCR / Cài đặt đổi cấu hình
    adapter.infer_batch(IMG, [1], [0.04], config=run_a)                         # lượt A chạy tiếp
    assert [c["text_rec_score_thresh"] for c in built[0].calls] == [0.5, 0.5]
    adapter.infer_batch(IMG, [2], [0.08])                                       # lượt không truyền cấu hình: mặc định mới
    assert built[0].calls[-1]["text_rec_score_thresh"] == 0.9999


def test_model_change_waits_for_running_batch(monkeypatch) -> None:
    run_a = OcrEngineConfig(detection_model_name="det_A")
    adapter, built = _adapter(monkeypatch, run_a, delay=0.3)
    adapter.infer_batch(IMG, [0], [0.0], config=run_a)
    errors: list[BaseException] = []

    def long_batch() -> None:
        try:
            adapter.infer_batch(IMG, [1], [0.04], config=run_a)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=long_batch)
    worker.start()
    time.sleep(0.05)                                                            # lô của lượt A đang chạy
    run_b = dataclasses.replace(run_a, detection_model_name="det_B")
    adapter.apply_config(run_b)                                                 # đổi model trong Cài đặt
    adapter.infer_batch(IMG, [2], [0.08], config=run_b)                         # lượt B cần model khác
    worker.join()
    assert not errors
    assert [m.name for m in built] == ["det_A", "det_B"]                        # nạp lại SAU khi lô của A xong


def test_threshold_change_does_not_reload_model(monkeypatch) -> None:
    run_a = OcrEngineConfig(det_thresh=0.3, limit_side_len=960)
    adapter, built = _adapter(monkeypatch, run_a)
    adapter.infer_batch(IMG, [0], [0.0], config=run_a)
    adapter.infer_batch(IMG, [1], [0.0], config=dataclasses.replace(run_a, det_thresh=0.5, limit_side_len=0))
    adapter.infer_batch(IMG, [2], [0.0], config=run_a)
    assert len(built) == 1                                                      # ngưỡng đi theo từng lần predict
    assert [c["text_det_thresh"] for c in built[0].calls] == [0.3, 0.5, 0.3]


def test_extract_use_case_passes_run_config_instead_of_apply(monkeypatch) -> None:
    from subtitles_extractor.application.use_cases import extract_subtitles as module

    class Engine:
        applied: list = []
        seen: list = []

        def apply_config(self, cfg) -> None:
            self.applied.append(cfg)

        def infer_batch(self, images_rgb, frame_indices, timestamps_sec, config=None):
            self.seen.append(config)
            return []

    engine = Engine()
    cfg = OcrEngineConfig(score_threshold=0.42)
    module._infer_batch(engine, cfg, images_rgb=[], frame_indices=[], timestamps_sec=[])
    assert engine.seen == [cfg] and module._accepts_config(engine)

    class OldEngine:
        def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
            return ["ok"]

    assert module._infer_batch(OldEngine(), cfg, images_rgb=[], frame_indices=[], timestamps_sec=[]) == ["ok"]


def test_no_config_call_reuses_loaded_model(monkeypatch) -> None:
    """Tự dò ROI / OCR phụ đề nhúng (không truyền cấu hình) dùng model đang nạp — không nạp lại qua lại."""
    settings = OcrEngineConfig(language="ch")
    adapter, built = _adapter(monkeypatch, settings)
    run = dataclasses.replace(settings, language="japan")
    adapter.infer_batch(IMG, [0], [0.0], config=run)     # trích xuất với ngôn ngữ ghi đè
    adapter.infer_batch(IMG, [1], [0.0])                  # tự dò ROI
    adapter.infer_batch(IMG, [2], [0.0], config=run)
    assert len(built) == 1


def test_apply_config_does_not_wait_for_model_load(monkeypatch) -> None:
    adapter, _built = _adapter(monkeypatch, OcrEngineConfig())
    done = threading.Event()
    with adapter._init_lock:                              # một lượt đang nạp model
        threading.Thread(target=lambda: (adapter.apply_config(OcrEngineConfig(score_threshold=0.7)), done.set()),
                         daemon=True).start()
        assert done.wait(1.0)                             # luồng giao diện không bị đứng


def test_release_during_batch_is_deferred(monkeypatch) -> None:
    adapter, built = _adapter(monkeypatch, OcrEngineConfig(), delay=0.3)
    released_inside: list[bool] = []
    worker = threading.Thread(target=lambda: adapter.infer_batch(IMG, [0], [0.0]))
    worker.start()
    time.sleep(0.1)
    adapter.release()                                     # đóng app giữa lô
    released_inside.append(built[0].released)
    worker.join()
    assert released_inside == [False]                     # không giải phóng dưới chân lô đang chạy
    assert adapter._model is None                          # nhưng được giải phóng ngay khi lô xong


def test_prepare_loads_the_run_model_once(monkeypatch) -> None:
    adapter, built = _adapter(monkeypatch, OcrEngineConfig(detection_model_name="settings_det"))
    run = OcrEngineConfig(detection_model_name="run_det")
    adapter.prepare(run)
    adapter.infer_batch(IMG, [0], [0.0], config=run)
    assert [m.name for m in built] == ["run_det"]
