"""[v3.23.710] Toạ độ hộp chữ trả về đúng hệ toạ độ ảnh gốc khi tiền xử lý có thêm viền.

Gốc lỗi: tỉ lệ thu-phóng được tính bằng (chiều cao sau khi cộng 2 viền) / (chiều cao gốc) → vùng chữ cao 136 px với viền
8 px cho tỉ lệ 1,118 dù ảnh không hề phóng; mọi hộp chữ bị co 0,895 lần. Phim mẫu: tâm phụ đề thật 325 px trong vùng
chữ, dữ liệu thô ghi 290 px → bộ lọc căn giữa (dung sai 26 px cho câu một chữ tin cậy thấp) loại oan câu "哼".
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig, PreprocessConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter


def _adapter(preprocess: PreprocessConfig) -> PaddleOcrAdapter:
    return PaddleOcrAdapter(OcrEngineConfig(preprocess=preprocess))


def _round_trip(height: int, preprocess: PreprocessConfig, point: tuple[int, int]) -> tuple[int, int]:
    adapter = _adapter(preprocess)
    image = np.zeros((height, 644, 3), np.uint8)
    prepared = adapter._prepare_image(image)
    # điểm ``point`` của ảnh gốc nằm ở đâu trong ảnh đã tiền xử lý
    real_scale = (prepared.image_bgr.shape[0] - 2 * prepared.border_px) / height
    in_prepared = (round(point[0] * real_scale) + prepared.border_px, round(point[1] * real_scale) + prepared.border_px)
    result = OcrFrameResult(0, 0.0, [OcrTextBox(text="哼", confidence=Confidence(0.9), polygon=[in_prepared])])
    return adapter._scale_back_result(result, prepared.scale, prepared.border_px).text_boxes[0].polygon[0]


@pytest.mark.parametrize("height", [136, 148, 60, 40])
def test_border_does_not_shrink_polygons(height: int) -> None:
    pre = PreprocessConfig(
        upscale_small_text=True, upscale_target_height_px=256, add_white_border=True, border_thickness_px=8,
        apply_clahe=False,
    )
    x, y = _round_trip(height, pre, (325, height // 2))
    assert abs(x - 325) <= 1 and abs(y - height // 2) <= 1, (x, y)      # mã cũ (cao 136): x = 291


def test_no_border_is_unchanged() -> None:
    pre = PreprocessConfig(upscale_small_text=False, add_white_border=False, apply_clahe=False)
    assert _round_trip(136, pre, (325, 68)) == (325, 68)
