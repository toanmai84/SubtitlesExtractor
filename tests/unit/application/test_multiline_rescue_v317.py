"""Test rescue phụ đề NHIỀU DÒNG cho bộ lọc Y-outlier cross-frame (v3.17).

Bối cảnh: ``filter_cross_frame_spatial_outliers`` dùng density-Y theo SỐ LƯỢNG box
toàn cục. Phụ đề hai dòng có dòng-1 nằm TRÊN và dòng-2 nằm DƯỚI tâm băng đơn-dòng
khổng lồ, nên không dòng nào rơi vào băng dày đặc → cả hai bị xoá → mất nguyên cue.
Hàm rescue khôi phục một *chồng dọc* (x chồng, y kề) khi tâm tổ hợp nằm trong băng
hợp lệ và frame đó vốn sẽ bị xoá sạch — đồng thời KHÔNG đụng frame đã có dòng chính.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.spatial_filters import (
    filter_cross_frame_spatial_outliers,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment


def _box(text: str, x_min: int, y_min: int, x_max: int, y_max: int, conf: float = 0.95) -> OcrTextBox:
    return OcrTextBox(
        text=text,
        confidence=Confidence(conf),
        polygon=[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)],
    )


def _single_line_frame(idx: int, ts: float, text: str) -> OcrFrameResult:
    # Phụ đề đơn-dòng tại băng chính: box cao ~60px, center_y ≈ 65 (y 35..95).
    return OcrFrameResult(
        frame_index=idx, timestamp_sec=ts,
        text_boxes=[_box(text, 60, 35, 560, 95)],
    )


def _two_line_frame(idx: int, ts: float, line1: str, line2: str) -> OcrFrameResult:
    # Phụ đề hai dòng (box cao ~60px như OCR thật): dòng-1 center_y≈35 (y 5..66),
    # dòng-2 center_y≈96 (y 66..127) — straddle băng đơn-dòng, khe dọc ≈ 0.
    return OcrFrameResult(
        frame_index=idx, timestamp_sec=ts,
        text_boxes=[
            _box(line1, 40, 5, 580, 66),     # center_y ≈ 35.5
            _box(line2, 250, 66, 360, 127),  # center_y ≈ 96.5, chồng X với dòng-1
        ],
    )


_ROI = Roi(x=0, y=780, width=620, height=130, alignment=TextAlignment.CENTER)


class TestMultilineRescue:
    def test_two_line_cue_survives_among_dominant_single_line_band(self) -> None:
        # 200 frame đơn-dòng (băng dày đặc) + 10 frame hai dòng (cue thiểu số).
        frames = [_single_line_frame(i, i * 0.04, "主线字幕内容长句") for i in range(200)]
        frames += [
            _two_line_frame(200 + i, (200 + i) * 0.04, "我爹为我们两个立下婚", "约")
            for i in range(10)
        ]
        purified = filter_cross_frame_spatial_outliers(frames, roi=_ROI)
        # Cue hai dòng phải còn cả hai dòng trong ít nhất một frame.
        two_line_kept = any(
            len(f.text_boxes) == 2 and any("约" == b.text for b in f.text_boxes)
            for f in purified
        )
        assert two_line_kept, "Phụ đề hai dòng bị xoá nhầm — rescue thất bại."

    def test_isolated_corner_noise_still_dropped(self) -> None:
        # Băng chính + nhiễu góc trên (Y≈10) KHÔNG chồng dọc với dòng nào.
        frames = [_single_line_frame(i, i * 0.04, "主线字幕内容长句") for i in range(200)]
        # 5 frame có thêm box nhiễu cô lập ở góc, xa băng chính, không tạo chồng.
        noisy = []
        for i in range(5):
            noisy.append(
                OcrFrameResult(
                    frame_index=300 + i, timestamp_sec=(300 + i) * 0.04,
                    text_boxes=[
                        _box("主线字幕内容长句", 60, 50, 560, 80),
                        _box("LOGO", 0, 5, 40, 18),  # nhiễu góc, không chồng X
                    ],
                )
            )
        frames += noisy
        purified = filter_cross_frame_spatial_outliers(frames, roi=_ROI)
        # Nhiễu 'LOGO' không được giữ (frame có dòng chính → cổng chặn rescue).
        logo_kept = any(
            any(b.text == "LOGO" for b in f.text_boxes) for f in purified
        )
        assert not logo_kept, "Nhiễu góc bị cứu nhầm — cổng rescue quá lỏng."

    def test_clean_single_line_corpus_unchanged(self) -> None:
        # Corpus đơn-dòng sạch: rescue không được làm thay đổi gì.
        frames = [_single_line_frame(i, i * 0.04, "稳定的单行字幕") for i in range(120)]
        purified = filter_cross_frame_spatial_outliers(frames, roi=_ROI)
        assert len(purified) == len(frames)
        assert all(len(f.text_boxes) == 1 for f in purified)
