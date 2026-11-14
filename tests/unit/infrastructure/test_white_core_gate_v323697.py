"""[v3.23.697] Lọc khung chữ trắng so trên LÕI sáng (≥ 240): nền sáng chuyển động không còn làm OCR lại câu đứng yên.

Số đo tập 1–5 phim mẫu (7927 khung mẫu, 429 dòng đáp án): khung đi OCR 56,8 % → 26,1 %, chất lượng 426/429 như cũ.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.infrastructure.video.frame_dedupe import (
    NOCORE_REOCR_SEC,
    WHITE_WARMUP_FRAMES,
    FrameDeduper,
    FrameVerdict,
)

H, W = 120, 600
STEP = 0.12


def _frame(text_luma: int = 0, text_x: int = 200, text_w: int = 200, bg_x: int | None = None) -> np.ndarray:
    g = np.full((H, W), 40, np.uint8)
    if bg_x is not None:                              # mảng nền sáng 220 (trời, áo trắng…) — không phải chữ
        g[10:110, bg_x:bg_x + 80] = 220
    if text_luma:
        g[45:75, text_x:text_x + text_w] = text_luma
    return g


def _run(dd: FrameDeduper, frames: list[np.ndarray], t0: float = 0.0) -> list[str]:
    out = []
    for i, g in enumerate(frames):
        t = t0 + i * STEP
        v = dd.verdict(g, t)
        if v is FrameVerdict.OCR:
            dd.accept(g, t)
        out.append(v.value)
    return out


def _gate(luma: int = 200) -> FrameDeduper:
    return FrameDeduper(FrameSamplingConfig(white_text_gate=True, white_text_luma=luma))


def test_static_white_text_over_moving_bright_background_is_ocr_once() -> None:
    frames = [_frame(255, bg_x=20 + 9 * (i % 6)) for i in range(10)]          # nền sáng dịch qua lại, chữ đứng yên
    # [v3.23.710] Lịch đọc lại của câu đứng yên (bước 0,12 s): khung xác nhận ngay sau, rồi định kỳ mỗi ~0,4 s.
    schedule = ["ocr", "ocr", "duplicate", "duplicate", "duplicate", "ocr", "duplicate", "duplicate", "duplicate", "ocr"]
    assert _run(_gate(), frames) == schedule
    assert _run(_gate(), [_frame(255, bg_x=20)] * 10) == schedule             # nền chuyển động không thêm lượt OCR nào


def test_white_text_change_is_still_detected() -> None:
    frames = [_frame(255, bg_x=20)] * 4 + [_frame(255, text_x=150, text_w=320, bg_x=20)] * 3
    # [v3.23.710] mỗi lần chữ đổi kèm một khung xác nhận ngay sau
    assert _run(_gate(), frames) == ["ocr", "ocr", "duplicate", "duplicate", "ocr", "ocr", "duplicate"]


def test_grey_subtitles_keep_the_old_rule() -> None:
    """Phụ đề xám (215, không có lõi ≥ 240): luật như trước — chữ đổi thì OCR lại, không bao giờ coi là trống."""
    dd = _gate()
    warm = _run(dd, [_frame(215)] * (WHITE_WARMUP_FRAMES + 5))
    assert "blank" not in warm and warm.count("ocr") >= 1
    after = _run(dd, [_frame(215, text_x=100, text_w=380)] * 3, t0=10.0)
    assert after[0] == "ocr"


def test_in_a_white_film_frames_without_core_are_reocr_at_most_every_second() -> None:
    dd = _gate()
    _run(dd, [_frame(255)] * (WHITE_WARMUP_FRAMES + 2))                         # phim đã được nhận là phụ đề trắng
    t0 = 20.0
    seq = _run(dd, [_frame(0, bg_x=100 + 3 * i) for i in range(20)], t0=t0)    # chỉ nền sáng 220, không lõi
    ocr_times = [t0 + i * STEP for i, v in enumerate(seq) if v == "ocr"]
    assert seq[0] == "ocr" and len(ocr_times) == 3                             # 0 s, ~1,08 s, ~2,16 s
    assert all(b - a >= NOCORE_REOCR_SEC - 1e-9 for a, b in zip(ocr_times, ocr_times[1:], strict=False))
    # Câu trắng trở lại sau đoạn không lõi → OCR ngay.
    assert _run(dd, [_frame(255)], t0=t0 + 20 * STEP) == ["ocr"]


def test_user_threshold_at_or_above_core_disables_core_mode() -> None:
    dd = _gate(luma=240)
    frames = [_frame(255, bg_x=20 + 9 * (i % 6)) for i in range(5)]
    assert _run(dd, frames)[0] == "ocr"
    assert dd._use_core is False
