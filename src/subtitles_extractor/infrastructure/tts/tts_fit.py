"""[v3.23.629] Dòng nào của bản dịch KHÔNG vừa khung thời gian để TTS đọc (hàm thuần).

Chiều ngược của mối quan hệ dịch ↔ TTS: khâu dịch đã gửi ngân sách âm tiết cho từng dòng
(``timing_math.readable_syllable_budget``), nhưng bản dịch vẫn có phần đuôi dòng quá dài — và dòng
người dùng tự sửa / thêm / gộp / tách ở trang TTS thì không hề có ngân sách. Mô-đun này tìm các dòng
đó, TRƯỚC khi tổng hợp (dự đoán) lẫn SAU khi tổng hợp (đo thật), để bước rút gọn bằng AI chỉ động
vào đúng chỗ cần.

Mô hình thời lượng đọc tự nhiên (tốc độ 1,0) của VieNeu v3 Turbo, hồi quy trên 141 dòng thật (tập
2 bản dịch Việt, giọng "Hải Đăng", thời lượng sau nén × hệ số nén; 20/10/2026)::

    do_dai_tu_nhien = 0,312 + 0,177 x so_am_tiet     (sai số chuẩn 0,09 s)

Đối chiếu trên cùng dữ liệu: ngưỡng dự đoán nén > 1,3× gắn cờ 26 dòng, thực tế engine phải nén
≥ 1,25× ở 24 dòng (trúng 16).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.infrastructure.tts.text_prep import dem_am_tiet
from subtitles_extractor.infrastructure.tts.timing_math import effective_available_seconds

#: Mô hình thời lượng đọc tự nhiên của VieNeu v3 Turbo (xem docstring mô-đun).
NATURAL_BASE_S = 0.312
NATURAL_PER_SYLLABLE_S = 0.177
#: Nén tới mức này vẫn nghe tự nhiên (đo: trung vị 1,07, p90 1,25).
COMFORT_SPEED = 1.3
#: Trên mức này giọng thều thào / tràn khung — đáng rút gọn câu.
PAINFUL_SPEED = 1.5
#: Ngưỡng cho DỰ ĐOÁN (trước khi tổng hợp): cao hơn vì bộ lập lịch còn dời câu sau và mượn khoảng
#: lặng — đo tập 2: dự đoán > 1,5× gắn 9 dòng nhưng thực tế chỉ 1 dòng bị nén ≥ 1,5×.
PREDICTED_SPEED = 1.8

#: Không bao giờ yêu cầu cắt quá 40% số âm tiết của một dòng (mất nghĩa) và không đụng dòng
#: dưới ``MIN_SYLLABLES_TO_SHORTEN`` âm tiết — khung quá chật thì để TTS dời câu / nén thay vì cắt
#: nghĩa (nguyên tắc: NGHĨA quan trọng hơn độ mượt, xem ``timing_math.TRANSLATION_REF_SPEED``).
MIN_KEEP_RATIO = 0.6
MIN_SYLLABLES_TO_SHORTEN = 4

REASON_PREDICTED = "predicted"
REASON_FAST = "fast"
REASON_OVERLAP = "overlap"
REASON_TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class LineFit:
    """Một dòng không vừa khung.

    Attributes:
        position: Vị trí dòng (0-based).
        syllables: Số âm tiết hiện tại.
        target_syllables: Số âm tiết nên rút còn (để đọc kịp ở tốc độ thoải mái).
        available_s: Khung hiệu dụng của dòng (giây).
        reason: ``REASON_*``.
        speed: Hệ số nén cần/đã dùng (1,0 = đọc tự nhiên).
    """

    position: int
    syllables: int
    target_syllables: int
    available_s: float
    reason: str
    speed: float = 1.0


def natural_seconds(syllables: int) -> float:
    """Thời lượng đọc tự nhiên (giây) của ``syllables`` âm tiết."""
    return NATURAL_BASE_S + NATURAL_PER_SYLLABLE_S * max(0, syllables)


def target_syllables_for(available_s: float, comfort_speed: float = COMFORT_SPEED) -> int:
    """Số âm tiết tối đa đọc kịp trong ``available_s`` giây ở tốc độ ``comfort_speed``."""
    speakable = available_s * max(0.5, comfort_speed) - NATURAL_BASE_S
    return max(1, int(speakable / NATURAL_PER_SYLLABLE_S))


def _clamped_target(syllables: int, target: int) -> int:
    """Mục tiêu âm tiết đã kẹp theo :data:`MIN_KEEP_RATIO`; 0 nếu dòng không đáng rút gọn."""
    if syllables < MIN_SYLLABLES_TO_SHORTEN:
        return 0
    floor = -(-int(syllables * MIN_KEEP_RATIO * 10) // 10)  # ceil, tránh lỗi float
    clamped = max(target, floor, 1)
    return clamped if clamped < syllables else 0


def _event_text(event: Any) -> str:
    return str(getattr(event, "text", "") or "")


def _window(events: Sequence[Any], position: int) -> float:
    event = events[position]
    nxt = float(events[position + 1].start_sec) if position + 1 < len(events) else None
    return effective_available_seconds(float(event.start_sec), float(event.end_sec), nxt)


def predict_misfits(
    events: Sequence[Any], comfort_speed: float = PREDICTED_SPEED
) -> list[LineFit]:
    """Dòng được DỰ ĐOÁN phải nén quá ``comfort_speed`` (chưa cần tổng hợp).

    Args:
        events: Dòng phụ đề (có ``start_sec``, ``end_sec``, ``text``).
        comfort_speed: Ngưỡng nén coi là đáng rút gọn (mặc định :data:`PAINFUL_SPEED`).

    Returns:
        Danh sách :class:`LineFit` (``target_syllables`` tính ở tốc độ :data:`COMFORT_SPEED`).
    """
    fits: list[LineFit] = []
    for position, event in enumerate(events):
        text = _event_text(event)
        syllables = dem_am_tiet(text.strip())
        if syllables <= 0:
            continue
        available = _window(events, position)
        if available <= 0:
            continue
        needed = natural_seconds(syllables) / available
        target = _clamped_target(syllables, target_syllables_for(available))
        if needed > comfort_speed and target:
            fits.append(LineFit(
                position, syllables, target, available, REASON_PREDICTED, round(needed, 2),
            ))
    return fits


def measured_misfits(
    events: Sequence[Any], results: Sequence[Any], speed_threshold: float = PAINFUL_SPEED
) -> list[LineFit]:
    """Dòng mà lần tổng hợp THẬT phải nén ≥ ngưỡng, bị cắt, hoặc chồng tiếng sang câu sau."""
    fits: list[LineFit] = []
    count = len(events)
    for result in results:
        position = int(result.event_index)
        if not (0 <= position < count) or getattr(result, "was_skipped", False):
            continue
        text = _event_text(events[position])
        syllables = dem_am_tiet(text.strip())
        speed = float(getattr(result, "speed_used", 1.0) or 1.0)
        if getattr(result, "was_truncated", False):
            reason = REASON_TRUNCATED
        elif getattr(result, "has_audible_overlap", False) and float(
            getattr(result, "overlap_s", 0.0)
        ) > 0.3:
            reason = REASON_OVERLAP
        elif speed >= speed_threshold:
            reason = REASON_FAST
        else:
            continue
        available = _window(events, position)
        target = target_syllables_for(available)
        if reason == REASON_FAST:
            # Số đo thật thắng ước lượng khung: lịch (dời câu / cụm dày) có thể nén dù khung rộng.
            target = min(target, int(syllables * COMFORT_SPEED / speed))
        else:
            # Chồng tiếng / bị cắt: số giây TRÀN đo thật quy ra số âm tiết phải bớt.
            excess = max(
                float(getattr(result, "overlap_s", 0.0) or 0.0),
                float(getattr(result, "cut_amount_ms", 0.0) or 0.0) / 1000.0,
            )
            if excess > 0:
                target = min(target, syllables - max(1, math.ceil(excess / NATURAL_PER_SYLLABLE_S)))
        target = _clamped_target(syllables, target)
        if not target:
            continue
        fits.append(LineFit(position, syllables, target, available, reason, round(speed, 2)))
    return fits


def merge_misfits(*groups: Sequence[LineFit]) -> list[LineFit]:
    """Gộp các danh sách, mỗi vị trí một mục (ưu tiên số đo thật), theo thứ tự dòng."""
    by_position: dict[int, LineFit] = {}
    for group in groups:
        for fit in group:
            by_position[fit.position] = fit
    return [by_position[p] for p in sorted(by_position)]


__all__ = [
    "COMFORT_SPEED",
    "NATURAL_BASE_S",
    "NATURAL_PER_SYLLABLE_S",
    "MIN_KEEP_RATIO",
    "MIN_SYLLABLES_TO_SHORTEN",
    "PAINFUL_SPEED",
    "PREDICTED_SPEED",
    "REASON_FAST",
    "REASON_OVERLAP",
    "REASON_PREDICTED",
    "REASON_TRUNCATED",
    "LineFit",
    "measured_misfits",
    "merge_misfits",
    "natural_seconds",
    "predict_misfits",
    "target_syllables_for",
]
