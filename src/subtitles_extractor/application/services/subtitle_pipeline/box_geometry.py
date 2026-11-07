"""[v3.23.682] Lọc hộp chữ có hình dạng không thể là chữ phụ đề (so với cỡ chữ của chính phim).

Đo trên phim nối 86 tập (2 giờ 19 phút, 3657 câu đáp án): 6 câu "thừa" đều là OCR đọc nhầm vật trong cảnh, với độ tin
cậy 0,74–0,99 nên mọi bộ lọc theo độ tin cậy đều để lọt:
* vòng đai lưng → "00" (mỗi ký tự rộng 24 px, cao 71 px), hoa văn vải → "88"/"83" (cao 16–20 px);
* bàn tay / vạt áo → "入", "一", "小" (một ký tự rộng 265–285 px);
* chữ khắc dọc trên bia → "气气宗" (hộp đứng 49 × 119 px).
Chữ phụ đề thật của phim đó cao ~68 px, mỗi chữ Hán rộng xấp xỉ bằng chiều cao.

Luật (H = chiều cao trung vị của hộp ≥ 3 chữ Hán, độ tin cậy ≥ 0,9 — tức cỡ chữ phụ đề của chính phim):
* hộp thấp hơn 0,55·H → không phải chữ phụ đề;
* mỗi ký tự hẹp hơn 0,4 × min(chiều cao hộp, H) → chữ dọc / vật thể mảnh;
* mỗi ký tự rộng hơn 2·H VÀ độ tin cậy < 0,97 → vệt sáng bị đọc thành một chữ. Có ngưỡng tin cậy vì hộp của chữ thật đôi
  khi phình sang nền sáng (đo: "哦" tin cậy 1,0 có hộp rộng 209 px khi đứng trên thắt lưng trắng).
Chữ Latin/số tính 0,55 chữ (nửa bề rộng). Không đủ 30 mẫu để biết H, hoặc phụ đề dọc → không lọc.
Chốt an toàn: chính các câu thoại chắc chắn mà > 5% bị chấm "sai hình dạng" (toạ độ hộp lệch đơn vị, cỡ chữ lạ…) → tắt
bộ lọc cho cả lượt, không được xoá nhầm phụ đề hàng loạt.
"""

from __future__ import annotations

import bisect
import re
import statistics
from collections.abc import Sequence

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent

_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_MIN_SAMPLES = 30
_SAMPLE_MIN_CJK = 3
_SAMPLE_MIN_CONF = 0.9
MIN_HEIGHT_RATIO = 0.55
MIN_CHAR_WIDTH_RATIO = 0.4
MAX_CHAR_WIDTH_RATIO = 2.0
WIDE_KEEP_CONFIDENCE = 0.97
_LATIN_WIDTH = 0.55
#: Tỉ lệ câu thoại chắc chắn bị chấm sai hình dạng tối đa để còn tin bộ lọc.
MAX_SAMPLE_REJECT_RATIO = 0.05


def _size(box: OcrTextBox) -> tuple[float, float] | None:
    if not box.polygon:
        return None
    xs = [p[0] for p in box.polygon]
    ys = [p[1] for p in box.polygon]
    return float(max(xs) - min(xs)), float(max(ys) - min(ys))


def _sample_boxes(frames: Sequence[OcrFrameResult]) -> list[OcrTextBox]:
    """Hộp chắc chắn là câu thoại: ≥ 3 chữ Hán, tin cậy cao, có toạ độ."""
    return [
        box for frame in frames for box in frame.text_boxes
        if _size(box) and len(_CJK.findall(box.text)) >= _SAMPLE_MIN_CJK and float(box.confidence) >= _SAMPLE_MIN_CONF
    ]


def typical_glyph_height(frames: Sequence[OcrFrameResult]) -> float | None:
    """Cỡ chữ phụ đề của phim: chiều cao trung vị của các hộp chắc chắn là câu thoại (≥ 3 chữ Hán, tin cậy cao)."""
    heights = [size[1] for box in _sample_boxes(frames) if (size := _size(box))]
    return statistics.median(heights) if len(heights) >= _MIN_SAMPLES else None


def is_plausible_subtitle_box(box: OcrTextBox, glyph_height: float) -> bool:
    """Hộp có thể là một dòng phụ đề ngang có cỡ chữ ``glyph_height`` không."""
    size = _size(box)
    if size is None or glyph_height <= 0:
        return True
    width, height = size
    text = "".join(box.text.split())
    units = sum(1.0 if _CJK.match(ch) else _LATIN_WIDTH for ch in text) or 1.0
    per_char = width / units
    if height < MIN_HEIGHT_RATIO * glyph_height:
        return False
    if per_char < MIN_CHAR_WIDTH_RATIO * min(height, glyph_height):
        return False
    return not (per_char > MAX_CHAR_WIDTH_RATIO * glyph_height and float(box.confidence) < WIDE_KEEP_CONFIDENCE)


def drop_implausible_boxes(frames: Sequence[OcrFrameResult]) -> tuple[list[OcrFrameResult], int]:
    """Bỏ hộp sai hình dạng; khung không còn hộp nào thì bỏ khung. Trả (khung, số hộp đã bỏ)."""
    glyph_height = typical_glyph_height(frames)
    if glyph_height is None:
        return list(frames), 0
    samples = _sample_boxes(frames)
    rejected = sum(1 for box in samples if not is_plausible_subtitle_box(box, glyph_height))
    if rejected > MAX_SAMPLE_REJECT_RATIO * len(samples):
        return list(frames), 0
    out: list[OcrFrameResult] = []
    dropped = 0
    for frame in frames:
        kept = [box for box in frame.text_boxes if is_plausible_subtitle_box(box, glyph_height)]
        dropped += len(frame.text_boxes) - len(kept)
        if len(kept) == len(frame.text_boxes):
            out.append(frame)
        elif kept:
            out.append(OcrFrameResult(frame_index=frame.frame_index, timestamp_sec=frame.timestamp_sec, text_boxes=kept))
    return out, dropped


#: [v3.23.685] Câu bị loại khi hộp chữ trôi một chiều ≥ bấy nhiêu lần bề rộng hộp trong thời gian hiện.
MAX_DRIFT_WIDTH_RATIO = 0.4
#: ...và chuyển động đều (độ lệch quanh đường thẳng ≤ bấy nhiêu phần quãng trôi) — rung hộp OCR thì không đều.
MAX_DRIFT_RESIDUAL_RATIO = 0.5
_DRIFT_MIN_FRAMES = 3
#: Khung được tính khi các hộp khớp phủ ≥ bấy nhiêu phần chữ của câu (phim 86 tập: mảnh "心" hộp rộng 210 px ngay trước
#: "叶然小心" từng làm câu thật bị coi là trôi).
_DRIFT_MIN_COVERAGE = 0.8


def _letters(text: str) -> str:
    """Chỉ giữ chữ/số (bỏ dấu, khoảng trắng, xuống dòng) để "-00" vẫn khớp câu "00"."""
    return "".join(ch for ch in text if ch.isalnum())


def _event_box_track(
    text: str, start: float, end: float, frames: Sequence[OcrFrameResult], times: Sequence[float],
    margin: float = 0.0,
) -> list[tuple[float, float, float]]:
    """(thời điểm, tâm x, bề rộng) các hộp mang chữ của câu trong khoảng hiện của câu (nới ``margin`` hai đầu).

    Dùng khung OCR THÔ: mốc câu ở bước này là mốc nhóm khung và các hộp méo đã bị lọc hình dạng — chính những khung đó
    cho thấy chuyển động.
    """
    flat = _letters(text)
    lo = bisect.bisect_left(times, start - margin - 1e-3)
    hi = bisect.bisect_right(times, end + margin + 1e-3)
    track: list[tuple[float, float, float]] = []
    for frame in frames[lo:hi]:
        boxes = [b for b in frame.text_boxes if (t := _letters(b.text)) and t in flat]
        # Khung chỉ đọc được một mảnh câu (chữ đang hiện dần, hộp phình) không cho biết vị trí của câu.
        if not boxes or sum(len(_letters(b.text)) for b in boxes) < _DRIFT_MIN_COVERAGE * len(flat):
            continue
        xs = [x for b in boxes for x, _y in b.polygon]
        if not xs:
            continue
        x0, x1 = min(xs), max(xs)
        track.append((frame.timestamp_sec, (x0 + x1) / 2.0, float(x1 - x0)))
    return track


def drift_ratio(track: Sequence[tuple[float, float, float]]) -> float:
    """Quãng trôi ngang (theo đường thẳng khớp bình phương nhỏ nhất) chia bề rộng trung vị; 0 nếu rung không đều."""
    if len(track) < _DRIFT_MIN_FRAMES:
        return 0.0
    ts = [t for t, _x, _w in track]
    xs = [x for _t, x, _w in track]
    mt, mx = statistics.fmean(ts), statistics.fmean(xs)
    den = sum((t - mt) ** 2 for t in ts)
    if den <= 0:
        return 0.0
    slope = sum((t - mt) * (x - mx) for t, x in zip(ts, xs, strict=True)) / den
    drift = abs(slope) * (ts[-1] - ts[0])
    residual = statistics.pstdev([x - (mx + slope * (t - mt)) for t, x in zip(ts, xs, strict=True)])
    width = statistics.median(w for _t, _x, w in track)
    if drift <= 0 or width <= 0 or residual > MAX_DRIFT_RESIDUAL_RATIO * drift:
        return 0.0
    return drift / width


def drop_drifting_events(
    events: Sequence[SubtitleEvent], frames: Sequence[OcrFrameResult], sample_step_sec: float = 0.0,
) -> tuple[list[SubtitleEvent], int]:
    """[v3.23.685] Bỏ câu mà hộp chữ TRÔI ngang đều qua khung hình — chữ phụ đề đứng yên, vật trong cảnh thì chuyển động.

    Phim nối 86 tập: "00" đọc từ vòng đai lưng (45 s) trôi 32 px trong 0,7 s = 0,69 lần bề rộng hộp; trong 3645 câu thật,
    câu trôi nhiều nhất chỉ 0,15 lần ("好"), câu dài rung hộp tới 44 px nhưng chỉ 0,08 lần bề rộng 527 px.
    """
    times = [f.timestamp_sec for f in frames]
    kept: list[SubtitleEvent] = []
    for event in events:
        track = _event_box_track(event.text, event.start_sec, event.end_sec, frames, times, sample_step_sec)
        if drift_ratio(track) >= MAX_DRIFT_WIDTH_RATIO:
            continue
        kept.append(event)
    return kept, len(events) - len(kept)


__all__ = [
    "MAX_DRIFT_WIDTH_RATIO",
    "drift_ratio",
    "drop_drifting_events",
    "drop_implausible_boxes",
    "is_plausible_subtitle_box",
    "typical_glyph_height",
]
