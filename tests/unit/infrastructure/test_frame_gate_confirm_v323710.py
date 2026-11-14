"""[v3.23.710] Lọc khung chữ trắng: mỗi câu được OCR nhiều hơn MỘT lần (khung xác nhận + đọc lại định kỳ).

Gốc lỗi: từ 3.23.697 một câu đứng yên chỉ được OCR ở khung đầu tiên (lúc chữ vừa hiện), các khung sau chép kết quả.
Khung đầu đọc hỏng thì cả câu mất. Lượt chạy thật phim 86 tập bằng bản 3.23.708 sót 7 câu một chữ, trong đó 4 câu
("是", "你", "哎", "好") có mọi khung rỗng dù bản 3.23.690 đọc được với độ tin cậy 1,00.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.infrastructure.video.frame_dedupe import (
    CONFIRM_DELAY_SEC,
    TEXT_REOCR_SEC,
    FrameDeduper,
    FrameVerdict,
)

H, W = 120, 600
STEP = 0.04


def _frame(text_w: int = 0) -> np.ndarray:
    g = np.full((H, W), 40, np.uint8)
    if text_w:
        g[45:75, 300 - text_w // 2:300 + text_w // 2] = 255
    return g


def _run(frames: list[np.ndarray]) -> list[str]:
    dd = FrameDeduper(FrameSamplingConfig(white_text_gate=True, white_text_luma=200))
    out = []
    for i, g in enumerate(frames):
        t = i * STEP
        v = dd.verdict(g, t)
        if v is FrameVerdict.OCR:
            dd.accept(g, t)
        out.append(v.value)
    return out


def test_static_line_gets_a_confirmation_frame_after_onset() -> None:
    verdicts = _run([_frame(60)] * 8)
    confirm_index = round(CONFIRM_DELAY_SEC / STEP)
    assert verdicts[0] == "ocr"
    assert verdicts[confirm_index] == "ocr", verdicts          # trên mã cũ: toàn "duplicate"
    assert verdicts[1:confirm_index] == ["duplicate"] * (confirm_index - 1)


def test_long_static_line_is_reread_periodically() -> None:
    count = round(2.0 / STEP)
    verdicts = _run([_frame(200)] * count)
    reads = [i for i, v in enumerate(verdicts) if v == "ocr"]
    assert len(reads) >= 5, reads                              # trên mã cũ: 2 (khung đầu + mốc MAX_GAP)
    gaps = [(b - a) * STEP for a, b in zip(reads[1:], reads[2:], strict=False)]
    assert all(gap <= TEXT_REOCR_SEC + STEP + 1e-6 for gap in gaps), gaps
    # không OCR dồn dập: phần lớn khung vẫn chép kết quả
    assert verdicts.count("duplicate") >= 0.75 * count


def test_confirmation_is_dropped_when_text_disappears() -> None:
    verdicts = _run([_frame(60)] + [_frame(0)] * 6)
    assert verdicts[0] == "ocr" and "ocr" not in verdicts[1:], verdicts


def test_every_text_change_schedules_its_own_confirmation() -> None:
    verdicts = _run([_frame(60)] * 6 + [_frame(300)] * 6)
    assert verdicts[6] == "ocr" and verdicts[6 + round(CONFIRM_DELAY_SEC / STEP)] == "ocr", verdicts
