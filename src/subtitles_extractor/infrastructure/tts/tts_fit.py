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

[v3.23.698] Khớp lại trên lượt lồng tiếng THẬT cả phim 86 tập của chủ dự án (VieNeu, 14 giọng, 3664 dòng, 10/11/2026)::

    do_dai_tu_nhien = 0,235 + 0,247 x so_am_tiet     (sai số chuẩn 0,23 s)

Mô hình cũ (một giọng, 141 dòng) ngắn hơn thực tế 19 % (trung vị; theo giọng 0,89–1,38 lần — "Kim Thanh" chậm nhất),
nên dự đoán trước khi tổng hợp chỉ bắt 66/251 dòng thật phải nén ≥ 1,8× (mô hình mới: 168/251; F1 0,40 → 0,62), và
quy đổi số giây lấn sang số âm tiết phải bớt bị thiếu.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.infrastructure.tts.text_prep import dem_am_tiet
from subtitles_extractor.infrastructure.tts.timing_math import effective_available_seconds

#: Mô hình thời lượng đọc tự nhiên của VieNeu v3 Turbo (xem docstring mô-đun).
NATURAL_BASE_S = 0.235
NATURAL_PER_SYLLABLE_S = 0.247
#: Nén tới mức này vẫn nghe tự nhiên (đo: trung vị 1,07, p90 1,25).
COMFORT_SPEED = 1.3
#: Trên mức này giọng thều thào / tràn khung — đáng rút gọn câu.
PAINFUL_SPEED = 1.5
#: [v3.23.702] Sàn ngân sách dịch: câu quá ngắn vẫn được tối thiểu ngần này tiếng (cắt nữa chỉ mất nghĩa).
TRANSLATION_MIN_WORDS = 2


def is_vietnamese_target(target_lang: str) -> bool:
    """[v3.23.702] Ngôn ngữ đích là tiếng Việt — ngân sách "số tiếng" theo nhịp VieNeu chỉ đúng cho tiếng Việt."""
    name = (target_lang or "").strip().casefold().replace("_", "-")
    return name in {"vi", "vi-vn", "vie"} or name.startswith(("vietnam", "tiếng việt", "tieng viet", "việt"))


def translation_word_budget(available_s: float, ref_speed: float = PAINFUL_SPEED) -> int:
    """[v3.23.702] Số TIẾNG tối đa của câu dịch để VieNeu đọc kịp khung ở ≤ ``ref_speed`` × nhịp tự nhiên.

    Trước đây ngân sách dịch tính theo nhịp NHANH NHẤT của Gemini TTS (0,23 + 0,22 s/âm tiết) ở 1,9× rồi quy ra ký
    tự: câu "đạt ngân sách" vẫn bắt VieNeu đọc ~2× — lượt chạy thật phim 86 tập có 500/3664 dòng phải nén > 1,5×.
    """
    if available_s <= 0.0:
        return TRANSLATION_MIN_WORDS
    words = int((available_s * max(0.1, ref_speed) - NATURAL_BASE_S) / NATURAL_PER_SYLLABLE_S)
    return max(TRANSLATION_MIN_WORDS, words)


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


# ── [v3.23.705] Đọc liền cụm câu: gộp các MẢNH của một câu bị phụ đề gốc cắt thành nhiều dòng ──────────────────
#: Hai dòng cách nhau quá ngần này giây thì không còn là một hơi nói.
FRAGMENT_MAX_GAP_S = 0.12
#: Cụm gộp tối đa ngần này dòng / tiếng / giây — dài hơn thì phụ đề khó đọc và giọng hụt hơi.
FRAGMENT_MAX_LINES = 4
FRAGMENT_MAX_SYLLABLES = 26
FRAGMENT_MAX_DURATION_S = 7.0
#: Gộp mà vẫn chưa về ≤ 1,5× thì chỉ nhận khi tốc độ cần giảm ít nhất ngần này (tỉ lệ).
FRAGMENT_MIN_GAIN = 0.15
_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
_CLOSERS = "\"'”’»)]"


def _ends_sentence(text: str) -> bool:
    return text.strip().rstrip(_CLOSERS).rstrip().endswith(_SENTENCE_END)


def _required_speed(syllables: int, available_s: float) -> float:
    return natural_seconds(syllables) / max(available_s, 0.05)


def plan_fragment_merges(
    events: Sequence[Any], speakers: Sequence[str] = (), painful_speed: float = PAINFUL_SPEED,
) -> list[tuple[int, int]]:
    """[v3.23.705] Các cụm dòng ``(đầu, cuối)`` (tính cả hai đầu) nên đọc LIỀN một hơi (hàm thuần).

    Đo trên lượt chạy thật 3664 dòng: 464/670 dòng phải đọc > 1,5× là MẢNH của một câu bị phụ đề gốc cắt ra các
    dòng liền nhau ("Mới lần hai" 0,36 s + "mà đã giáng năm mươi đạo cùng lúc sao?"). Dịch ngắn nữa cũng không cứu
    được; đọc liền cả cụm thì mảnh ngắn mượn được thời gian của mảnh dài.

    Chỉ gộp khi: dòng liền nhau (khe ≤ :data:`FRAGMENT_MAX_GAP_S`), CÙNG người nói, dòng trước chưa kết câu, và
    trong cụm có dòng cần > ``painful_speed``. Chọn cụm NHỎ NHẤT đưa tốc độ cần về ≤ ``painful_speed``; không có thì
    cụm giảm tốc nhiều nhất (≥ :data:`FRAGMENT_MIN_GAIN`). Dòng đã ổn và không cạnh dòng nhanh thì không đụng.

    Args:
        events: Dòng phụ đề (``start_sec``, ``end_sec``, ``text``), theo thứ tự thời gian.
        speakers: Người nói theo vị trí (rỗng/thiếu = chưa phân vai: mọi dòng một giọng đọc).
        painful_speed: Ngưỡng tốc độ cần coi là quá nhanh.

    Returns:
        Các cụm không chồng nhau, tăng dần.
    """
    count = len(events)
    texts = [_event_text(event).strip() for event in events]
    syllables = [dem_am_tiet(text) for text in texts]

    def speaker(position: int) -> str:
        return str(speakers[position] or "") if position < len(speakers) else ""

    def linked(position: int) -> bool:
        """Dòng ``position`` nối tiếp được dòng ``position - 1``?"""
        prev, cur = events[position - 1], events[position]
        gap = float(cur.start_sec) - float(prev.end_sec)
        return bool(
            texts[position - 1] and texts[position] and -0.05 <= gap <= FRAGMENT_MAX_GAP_S
            and speaker(position - 1) == speaker(position) and not _ends_sentence(texts[position - 1])
        )

    def span_speed(first: int, last: int) -> float:
        nxt = float(events[last + 1].start_sec) if last + 1 < count else None
        window = effective_available_seconds(float(events[first].start_sec), float(events[last].end_sec), nxt)
        return _required_speed(sum(syllables[first:last + 1]), window)

    def allowed(first: int, last: int) -> bool:
        return (
            sum(syllables[first:last + 1]) <= FRAGMENT_MAX_SYLLABLES
            and float(events[last].end_sec) - float(events[first].start_sec) <= FRAGMENT_MAX_DURATION_S
        )

    plans: list[tuple[int, int]] = []
    floor = 0  # vị trí nhỏ nhất còn tự do (sau cụm đã chọn)
    for position in range(count):
        if position < floor or not texts[position]:
            continue
        own = _required_speed(syllables[position], _window(events, position))
        if own <= painful_speed:
            continue
        low = position
        while low > floor and position - low < FRAGMENT_MAX_LINES - 1 and linked(low):
            low -= 1
        high = position
        while high + 1 < count and high - position < FRAGMENT_MAX_LINES - 1 and linked(high + 1):
            high += 1
        best: tuple[int, float, int, int] | None = None  # (không đạt, tốc độ|số dòng, đầu, cuối)
        for first in range(low, position + 1):
            for last in range(position, high + 1):
                size = last - first + 1
                if size < 2 or size > FRAGMENT_MAX_LINES or not allowed(first, last):
                    continue
                speed = span_speed(first, last)
                if speed > own * (1.0 - FRAGMENT_MIN_GAIN) and speed > painful_speed:
                    continue
                fits = speed <= painful_speed
                key = (0, float(size) + speed / 100.0, first, last) if fits else (1, speed, first, last)
                if best is None or key < best:
                    best = key
        if best is not None:
            plans.append((best[2], best[3]))
            floor = best[3] + 1
    return plans


__all__ = [
    "COMFORT_SPEED",
    "NATURAL_BASE_S",
    "NATURAL_PER_SYLLABLE_S",
    "MIN_KEEP_RATIO",
    "MIN_SYLLABLES_TO_SHORTEN",
    "PAINFUL_SPEED",
    "PREDICTED_SPEED",
    "TRANSLATION_MIN_WORDS",
    "REASON_FAST",
    "REASON_OVERLAP",
    "REASON_PREDICTED",
    "REASON_TRUNCATED",
    "LineFit",
    "translation_word_budget",
    "is_vietnamese_target",
    "measured_misfits",
    "merge_misfits",
    "natural_seconds",
    "plan_fragment_merges",
    "predict_misfits",
    "target_syllables_for",
]
