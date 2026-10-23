"""[v3.23.619] Auto-ROI OCR trên khung thu nhỏ, toạ độ quy về khung gốc.

Đo trên phim mẫu (720x1280, CPU 4 nhân): OCR khung nguyên cỡ 5.05 s/khung → bước dò vùng
phụ đề của một tập 3 phút mất ~30 phút. Bước này chỉ cần toạ độ ô chữ.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.ports.frame_sampler_port import SampledFrame
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
    OcrBasedAutoRoiDetector,
)

W, H = 720, 1280
#: Ô phụ đề thật trên khung gốc (giống tập mẫu: ~64-73% chiều cao).
BOX = (150, 830, 570, 900)


class _FakeOcr:
    """Trả một ô chữ ở đúng vị trí BOX theo tỉ lệ của ảnh được đưa vào."""

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []
        self.is_initialized = True

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
        results = []
        for img, idx, ts in zip(images_rgb, frame_indices, timestamps_sec, strict=True):
            h, w = img.shape[:2]
            self.sizes.append((w, h))
            k = w / W
            x0, y0, x1, y1 = (round(v * k) for v in BOX)
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            box = OcrTextBox(text="你还是天骄", confidence=Confidence(0.99), polygon=poly)
            results.append(OcrFrameResult(frame_index=idx, timestamp_sec=ts, text_boxes=[box]))
        return results


class _FakeSampler:
    def iter_frames(self, metadata, roi=None, config=None):
        for i in range(4):
            yield SampledFrame(frame_index=i, timestamp_sec=float(i),
                               image_rgb=np.zeros((H, W, 3), np.uint8),
                               is_duplicate=False, is_error=False)


def _boxes(max_side):
    ocr = _FakeOcr()
    detector = OcrBasedAutoRoiDetector(ocr, _FakeSampler(), show_review_ui=False,
                                       analysis_max_side=max_side)
    meta = SimpleNamespace(duration_sec=4.0, width=W, height=H, fps=25.0, path=None)
    _top, raw = detector._sample_and_ocr(meta, step_ms=1000, batch_size=8)
    return ocr.sizes, raw


def test_full_size_when_no_limit() -> None:
    sizes, raw = _boxes(None)
    assert set(sizes) == {(W, H)}
    assert (raw[0].coord_x_min, raw[0].coord_y_min) == (150, 830)


def test_downscaled_ocr_boxes_map_back_to_original_frame() -> None:
    sizes, raw = _boxes(960)
    assert set(sizes) == {(540, 960)}
    for box in raw:
        got = (box.coord_x_min, box.coord_y_min, box.coord_x_max, box.coord_y_max)
        assert all(abs(a - b) <= 2 for a, b in zip(got, BOX, strict=True)), got
