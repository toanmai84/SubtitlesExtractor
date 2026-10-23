"""[v3.23.619] Cài đặt "Giới hạn cạnh" (limit_side_len / limit_type) phải tới được PaddleOCR.

Trước đây adapter luôn truyền ``None`` cho ``text_det_limit_side_len`` và
``text_det_limit_type`` (cả lúc khởi tạo lẫn mỗi lần predict): cài đặt hiện ở 3 trang
(Cài đặt, Biên tập, Debug) không có tác dụng, đổi giá trị chỉ làm nạp lại model vô ích.
0 vẫn nghĩa là "để PaddleOCR tự quyết định" (hành vi mặc định không đổi).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return [{"rec_texts": [], "rec_scores": [], "rec_polys": []} for _ in kwargs["input"]]


@pytest.mark.parametrize(
    ("side", "kind", "expected"),
    [(960, "max", (960, "max")), (736, "min", (736, "min")), (0, "max", (None, None))],
)
def test_limit_side_reaches_paddle(side: int, kind: str, expected: tuple) -> None:
    adapter = PaddleOcrAdapter(OcrEngineConfig(limit_side_len=side, limit_type=kind))
    init = adapter._build_init_kwargs(use_gpu=False, use_trt=False)
    assert (init["text_det_limit_side_len"], init["text_det_limit_type"]) == expected

    model = _FakeModel()
    adapter._model = model
    adapter._predict_batch([np.zeros((32, 32, 3), np.uint8)], [0])
    call = model.calls[0]
    assert (call["text_det_limit_side_len"], call["text_det_limit_type"]) == expected
