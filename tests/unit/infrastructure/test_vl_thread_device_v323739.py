"""[v3.23.739] PaddleOCR-VL KHÔNG chạy ở mọi phiên trên máy chủ dự án (log 17–18/11: 19:19, 23:05, 05:28, 05:54).

``PaddleOCR-VL không chạy được (is_bfloat16_supported(): incompatible function arguments … Place(undefined:0))``: PaddleX
gọi ``paddle.amp.is_bfloat16_supported()`` không truyền thiết bị → Paddle dùng "nơi chạy mong đợi" của LUỒNG HIỆN TẠI,
chỉ có ở luồng đã ``paddle.set_device``. Lượt trích xuất chạy trong luồng nền → ``undefined`` → ứng dụng âm thầm lùi về
PP-OCR: lượt soát câu "bằng VL" và gói kiểm định ghi nhãn "PaddleOCR-VL" thật ra là PP-OCR đọc lại.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.infrastructure.ocr import paddle_vl_reader as vl


class _FakePaddle(types.ModuleType):
    """Paddle giả: "nơi chạy" theo TỪNG LUỒNG như bản GPU thật (luồng mới = undefined)."""

    def __init__(self) -> None:
        super().__init__("paddle")
        self._local = threading.local()
        self.calls: list[tuple[str, str]] = []

    def set_device(self, device: str) -> None:
        self._local.place = device
        self.calls.append((threading.current_thread().name, device))

    def place(self) -> str:
        return getattr(self._local, "place", "undefined")


def test_reader_binds_device_in_the_calling_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePaddle()
    monkeypatch.setitem(sys.modules, "paddle", fake)
    seen: list[str] = []

    class _Model:
        def predict(self, payload, max_new_tokens):  # noqa: ANN001, ANN201
            seen.append(fake.place())          # PaddleX hỏi nơi chạy NGAY trong luồng đang dự đoán
            if fake.place() == "undefined":
                raise TypeError("is_bfloat16_supported(): incompatible function arguments … Place(undefined:0)")
            return [{"res": {"content": "玄昊师兄"}}]

    reader = vl.PaddleVlLineReader()
    monkeypatch.setattr(reader, "_ensure", lambda device: _Model())
    out: list[object] = []
    worker = threading.Thread(
        target=lambda: out.append(reader.read([np.zeros((40, 200, 3), np.uint8)], device="gpu")), name="ocr-worker",
    )
    worker.start()
    worker.join(30)
    assert out and not isinstance(out[0], Exception)
    # [v3.23.747] Mọi lần nạp / đọc VL chạy trên luồng VL RIÊNG ("paddle-vl") — gán thiết bị ngay trên luồng đó.
    assert seen == ["gpu"] and ("paddle-vl", "gpu") in fake.calls


def test_auditor_reports_real_reader_when_vl_falls_back(tmp_path: Path) -> None:
    """VL lỗi → gói kiểm định ghi rõ bộ đọc THẬT và lý do, không mang nhãn "PaddleOCR-VL"."""
    import gzip
    import json

    from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.infrastructure.quality.quality_auditor import (
        VlQualityAuditor,
    )

    class _Engine:
        vl_error = "is_bfloat16_supported(): Place(undefined:0)"

        def infer_batch(self, images, frame_indices, timestamps, config=None):  # noqa: ANN001, ANN201
            return [OcrFrameResult(i, t, []) for i, t in zip(frame_indices, timestamps, strict=True)]

    class _Reader:
        def __init__(self, *_a: object) -> None: ...
        def frame_at(self, _t: float) -> np.ndarray: return np.zeros((10, 10, 3), np.uint8)
        def close(self) -> None: ...

    clip, side = tmp_path / "p.roi.mp4", tmp_path / "p.roi.json"
    clip.write_bytes(b"x")
    side.write_text("{}", encoding="utf-8")
    import subtitles_extractor.infrastructure.video.roi_clip_reader as reader_mod

    original = reader_mod.RoiClipReader
    reader_mod.RoiClipReader = _Reader  # type: ignore[misc]
    try:
        out = VlQualityAuditor(_Engine()).audit(
            [SubtitleEvent(index=1, text="字", interval=TimeInterval(0, 1))], clip, side, tmp_path / "p.qa.json.gz",
            config=OcrEngineConfig(recognition_model_name="PaddleOCR-VL-1.6-0.9B"),
        )
    finally:
        reader_mod.RoiClipReader = original  # type: ignore[misc]
    payload = json.loads(gzip.open(out, "rt", encoding="utf-8").read())  # type: ignore[arg-type]
    assert payload["model"] != "PaddleOCR-VL-1.6-0.9B" and "PP-OCR" in payload["model"]
    assert "undefined" in payload["vl_error"]


def test_auxiliary_models_released_in_worker_thread_at_end_of_run() -> None:
    """Văng 18/11 05:48: 2 s sau khi mpv trang Chỉnh sửa mở (ngay sau lượt kiểm định 152 s đọc 6393 ảnh). Mô hình phụ
    (VL, bộ cứu chữ) + bộ đệm VRAM phải được trả NGAY trong luồng chạy lượt, không để giao diện / bộ thu rác dọn sau."""
    import inspect

    from subtitles_extractor.application.use_cases.extract_subtitles import (
        ExtractSubtitlesUseCase,
    )
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    assert callable(getattr(PaddleOcrAdapter, "release_auxiliary_models", None))
    source = inspect.getsource(ExtractSubtitlesUseCase._execute)  # [v3.23.746] thân lượt chạy (execute bọc đồng hồ canh treo)
    assert "_release_auxiliary_models" in source


def test_release_auxiliary_models_keeps_main_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    adapter = PaddleOcrAdapter(OcrEngineConfig())
    main_model = object()
    adapter._model = main_model
    adapter._rescue_model = object()
    adapter._vl_cache[("字", 0, 0, 1, 1)] = "字"
    released: list[bool] = []
    monkeypatch.setattr(adapter._vl_reader, "release", lambda: released.append(True))
    monkeypatch.setattr(adapter, "_empty_paddle_vram", lambda: None)
    monkeypatch.setattr(adapter, "_empty_cupy_pool", lambda: None)
    adapter.release_auxiliary_models()
    assert adapter._model is main_model                       # mô hình chính giữ nguyên (lượt sau không nạp lại)
    assert adapter._rescue_model is None and not adapter._vl_cache and released == [True]
