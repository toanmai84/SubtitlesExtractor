"""[v3.23.727] Lọc chữ sáng ở mức hộp chữ: bông hoa hồng tím bị OCR đọc thành "福" (tin cậy 0,73) không được lên SRT.

Lượt thật 3.23.726 (phim nối 86 tập, model medium, tập 72 giây 114,6): hộp "福" có tỉ lệ điểm sáng 0,0005, hộp phụ đề
thật thấp nhất 0,134 (483 hộp, 10 tập). Phát lại cả phim khi bỏ các hộp tối + tin cậy thấp: chỉ câu "福" mất.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.application.services.bright_text_check import (
    MIN_BRIGHT_RATIO,
    bright_ratio,
    drop_dark_boxes,
)
from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig, SampledFrame
from subtitles_extractor.domain.value_objects.confidence import Confidence


def _scene(height: int = 136, width: int = 720) -> np.ndarray:
    """Vùng phụ đề: nền xám tối; chữ trắng viền đen ở giữa (x 300–420); bông hoa hồng tím ở x 372–421 khung khác."""
    return np.full((height, width, 3), 70, np.uint8)


def _subtitle(image: np.ndarray, x0: int = 300, x1: int = 420, y0: int = 66, y1: int = 133) -> np.ndarray:
    image = image.copy()
    image[y0:y1, x0:x1] = (20, 20, 20)                         # viền đen
    image[y0 + 6:y1 - 6:3, x0 + 6:x1 - 6] = (250, 250, 250)    # nét chữ trắng (khoảng 1/3 diện tích)
    return image


def _flower(image: np.ndarray, x0: int = 372, x1: int = 421, y0: int = 59, y1: int = 123) -> np.ndarray:
    image = image.copy()
    image[y0:y1, x0:x1] = (190, 60, 170)                       # hồng tím (độ chói ≈ 114)
    return image


def _box(text: str, conf: float, x0: int, x1: int, y0: int, y1: int) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _result(*boxes: OcrTextBox) -> OcrFrameResult:
    return OcrFrameResult(frame_index=0, timestamp_sec=7172.16, text_boxes=list(boxes))


def test_flower_read_as_fu_is_dropped() -> None:
    result, dropped = drop_dark_boxes(_result(_box("福", 0.731, 372, 421, 59, 123)), _flower(_scene()), 200)
    assert dropped == 1 and result.text_boxes == []


def test_real_white_subtitle_is_kept_even_with_low_confidence() -> None:
    box = _box("瑶瑶", 0.588, 300, 420, 66, 133)
    image = _subtitle(_scene())
    assert bright_ratio(image.mean(axis=2) >= 200, box.polygon) > 4 * MIN_BRIGHT_RATIO
    result, dropped = drop_dark_boxes(_result(box), image, 200)
    assert dropped == 0 and result.text_boxes == [box]


def test_confident_dark_read_is_kept() -> None:
    """Khung đầu câu hiện dần: chữ còn mờ (ít điểm sáng) nhưng OCR đọc chắc tay → giữ (không xén mốc bắt đầu).

    [v3.23.730] Chữ mờ thật vẫn có điểm sáng (đo 7196 câu của hai phim: khung đầu tối nhất 0,036); hộp KHÔNG có điểm
    sáng nào thì bỏ dù tin cậy cao — xem ``test_dark_scene_text_v323730.py``.
    """
    image = _flower(_scene())
    image[80:83, 380:410] = (240, 240, 240)                    # vài nét đã sáng: tỉ lệ ≈ 0,029
    box = _box("师叔", 0.99, 372, 421, 59, 123)
    assert drop_dark_boxes(_result(box), image, 200)[1] == 0


def test_only_the_dark_box_is_dropped_in_a_mixed_frame() -> None:
    image = _flower(_subtitle(_scene(), x0=100, x1=220))
    real, fake = _box("你好", 0.95, 100, 220, 66, 133), _box("福", 0.73, 372, 421, 59, 123)
    result, dropped = drop_dark_boxes(_result(real, fake), image, 200)
    assert dropped == 1 and result.text_boxes == [real]


def test_white_border_added_by_preprocessing_is_not_counted() -> None:
    """Ảnh đã thêm viền trắng 8 px (nhánh NVDEC): hộp chạm mép không được tính điểm sáng của viền."""
    image = np.full((256 + 16, 720 * 2 + 16, 3), 70, np.uint8)
    image[:8], image[-8:], image[:, :8], image[:, -8:] = 255, 255, 255, 255
    box = _box("福", 0.6, 0, 60, 0, 120)
    assert drop_dark_boxes(_result(box), image, 200, border_px=0)[1] == 0       # viền sáng làm hộp "sáng" giả
    assert drop_dark_boxes(_result(box), image, 200, border_px=8)[1] == 1


def test_degenerate_inputs_are_left_alone() -> None:
    box = _box("福", 0.5, 372, 421, 59, 123)
    assert drop_dark_boxes(_result(box), None, 200)[1] == 0
    assert drop_dark_boxes(_result(box), np.zeros((0, 0, 3), np.uint8), 200)[1] == 0
    off_image = _box("福", 0.5, 900, 950, 200, 260)
    assert drop_dark_boxes(_result(off_image), _scene(), 200)[1] == 0
    no_polygon = OcrTextBox(text="福", confidence=Confidence(0.5), polygon=[])
    assert drop_dark_boxes(_result(no_polygon), _scene(), 200)[1] == 0


def _request(gate: bool, border: bool = False):  # noqa: ANN202
    from types import SimpleNamespace

    sampling = FrameSamplingConfig(sample_step_sec=0.04, white_text_gate=gate, white_text_luma=200,
                                   vram_add_border=border, vram_border_thickness_px=8)
    return SimpleNamespace(sampling=sampling)


def test_use_case_applies_filter_only_when_gate_enabled() -> None:
    frame = SampledFrame(frame_index=0, timestamp_sec=7172.16, image_rgb=_flower(_scene()))
    results = [_result(_box("福", 0.731, 372, 421, 59, 123))]
    kept = ExtractSubtitlesUseCase._drop_dark_boxes(results, [frame], _request(gate=False))
    assert [b.text for b in kept[0].text_boxes] == ["福"]          # tắt "Lọc chữ sáng": giữ như trước
    dropped = ExtractSubtitlesUseCase._drop_dark_boxes(results, [frame], _request(gate=True))
    assert dropped[0].text_boxes == []
