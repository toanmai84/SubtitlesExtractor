"""[v3.23.732] Cứu chữ "诶" — chữ không có trong bảng chữ của mọi mô hình nhận dạng PaddleOCR (v6 small/medium, v5
mobile/server). Bộ dò v6 vẫn thấy hộp chữ nhưng nhận dạng ra rỗng (hộp bị lọc khỏi ``rec_polys``) hoặc "误" 0,37–0,51;
bộ nhận dạng phụ PP-OCRv5 mobile đọc chính hộp đó thành "埃" (đo 6 khung thật phim "[全集]": 5/6 khung cứu được khi đọc
cả ảnh đã tiền xử lý lẫn ảnh gốc; mỗi câu "诶" dài ~10 khung nên cả 3 câu đều có chữ).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind
from subtitles_extractor.infrastructure.ocr.unreadable_rescue import (
    apply_rescues,
    best_reading,
    crop_rect,
    detected_polygons,
    find_rescue_candidates,
)

SQUARE = [(804, 50), (949, 50), (949, 192), (804, 192)]


def _box(text: str, conf: float, polygon=SQUARE) -> OcrTextBox:  # noqa: ANN001
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=list(polygon))


def _frame(*boxes: OcrTextBox) -> OcrFrameResult:
    return OcrFrameResult(frame_index=0, timestamp_sec=3257.2, text_boxes=list(boxes))


# ── hàm thuần ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_unreadable_detected_box_and_weak_single_char_are_candidates() -> None:
    # Hộp dò thấy, nhận dạng rỗng (không có trong kết quả) → thêm mới.
    (only,) = find_rescue_candidates([SQUARE], [])
    assert only.replaces is None
    # "误" 0,51 một chữ → thay; câu nhiều chữ hay chữ đọc chắc thì không đụng.
    weak = find_rescue_candidates([SQUARE], [_box("误", 0.51)])
    assert [c.replaces for c in weak] == [0]
    assert find_rescue_candidates([SQUARE], [_box("啊", 1.0)]) == []
    assert find_rescue_candidates([SQUARE], [_box("你好", 0.40)]) == []
    assert find_rescue_candidates([SQUARE], [_box("V", 0.40)]) == []


def test_only_mapped_readings_are_accepted() -> None:
    candidates = find_rescue_candidates([SQUARE], [])
    rescued, count = apply_rescues(_frame(), candidates, [("埃", 0.85)])
    assert count == 1 and [(b.text, round(float(b.confidence), 2)) for b in rescued.text_boxes] == [("诶", 0.85)]
    # [v3.23.742] ngưỡng nhận bản đổi 埃→诶 hạ 0,6 → 0,3 (đo cả phim nối 86 tập: 埃 chỉ ra ở chỗ 诶 thật)
    assert apply_rescues(_frame(), candidates, [("埃", 0.25)])[1] == 0         # dưới ngưỡng tin cậy
    assert apply_rescues(_frame(), candidates, [("误", 0.95)])[1] == 0         # không có trong bảng quy đổi
    weak = find_rescue_candidates([SQUARE], [_box("误", 0.51)])
    replaced, _ = apply_rescues(_frame(_box("误", 0.51)), weak, [("埃", 0.94)])
    assert [b.text for b in replaced.text_boxes] == ["诶"]


def test_best_reading_prefers_mapped_text() -> None:
    assert best_reading([("误", 0.57), ("埃", 0.67)]) == ("埃", 0.67)
    assert best_reading([("埃", 0.49), ("埃", 0.78)]) == ("埃", 0.78)
    assert best_reading([("别", 0.52), ("別", 0.51)]) == ("别", 0.52)
    assert best_reading([]) == ("", 0.0)


def test_detected_polygons_reads_paddlex_result_shapes() -> None:
    poly = np.array(SQUARE, dtype=np.int16)
    assert detected_polygons({"res": {"dt_polys": [poly]}}) == [[(804.0, 50.0), (949.0, 50.0), (949.0, 192.0), (804.0, 192.0)]]
    assert detected_polygons(SimpleNamespace(res={"dt_polys": [poly]}))[0][0] == (804.0, 50.0)
    assert detected_polygons({"rec_texts": []}) == []
    assert detected_polygons(None) == []


def test_crop_rect_is_clamped() -> None:
    assert crop_rect(SQUARE, 1000, 200) == (800, 46, 953, 196)
    assert crop_rect([(0, 0), (2, 0), (2, 2), (0, 2)], 1000, 200, pad=0) is None


# ── tích hợp trong bộ chuyển đổi PaddleOCR (bộ nhận dạng phụ giả lập) ────────────────────────────────────────────────
class _FakeRecognizer:
    def __init__(self, readings: list[tuple[str, float]]) -> None:
        self.readings = readings
        self.batches: list[int] = []

    def predict(self, input, batch_size=1):  # noqa: A002, ANN001, ANN201
        self.batches.append(len(input))
        return [{"res": {"rec_text": t, "rec_score": s}} for t, s in self.readings[: len(input)]]


def _adapter(language: str = "ch"):  # noqa: ANN202
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
        PaddleOcrAdapter,
        _PreparedImage,
    )

    config = OcrEngineConfig(device=DeviceKind.CPU, language=language)
    adapter = PaddleOcrAdapter(config)
    prepared = _PreparedImage(image_bgr=np.full((272, 1456, 3), 90, np.uint8), scale=2.0, border_px=8)
    original = np.full((128, 720, 3), 90, np.uint8)
    return adapter, config, prepared, original


def test_adapter_rescues_unreadable_box_on_both_crops() -> None:
    adapter, config, prepared, original = _adapter()
    fake = _FakeRecognizer([("误", 0.57), ("埃", 0.85)])
    adapter._get_rescue_model = lambda: fake  # type: ignore[method-assign]
    raw = {"res": {"dt_polys": [np.array(SQUARE)], "rec_texts": [], "rec_scores": [], "rec_polys": []}}
    (result,) = adapter._rescue_unreadable([_frame()], [raw], [prepared], [original], config)
    assert fake.batches == [2]                                   # ảnh đã tiền xử lý + ảnh gốc
    assert [(b.text, round(float(b.confidence), 2)) for b in result.text_boxes] == [("诶", 0.85)]


@pytest.mark.parametrize("language", ["en", "japan"])
def test_adapter_skips_non_chinese_and_frames_without_candidates(language: str) -> None:
    adapter, config, prepared, original = _adapter(language)
    adapter._get_rescue_model = lambda: pytest.fail("không được nạp mô hình phụ")  # type: ignore[method-assign]
    raw = {"res": {"dt_polys": [np.array(SQUARE)]}}
    assert adapter._rescue_unreadable([_frame()], [raw], [prepared], [original], config)[0].text_boxes == []
    adapter, config, prepared, original = _adapter("ch")
    adapter._get_rescue_model = lambda: pytest.fail("không có hộp cần cứu")  # type: ignore[method-assign]
    clean = _frame(_box("啊", 1.0))
    assert adapter._rescue_unreadable([clean], [{"res": {"dt_polys": [np.array(SQUARE)]}}], [prepared], [original],
                                      config) == [clean]


def test_adapter_rescue_failure_never_breaks_ocr() -> None:
    adapter, config, prepared, original = _adapter()

    class _Broken:
        def predict(self, input, batch_size=1):  # noqa: A002, ANN001, ANN201
            raise RuntimeError("CUDA lỗi")

    adapter._get_rescue_model = lambda: _Broken()  # type: ignore[method-assign]
    raw = {"res": {"dt_polys": [np.array(SQUARE)]}}
    assert adapter._rescue_unreadable([_frame()], [raw], [prepared], [original], config)[0].text_boxes == []
    assert adapter._rescue_unavailable is True
