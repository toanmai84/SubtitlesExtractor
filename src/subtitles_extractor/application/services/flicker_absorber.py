"""Module hấp thụ "câu chớp nhoáng" (flicker) và phân loại rác OCR.

CẢI TIẾN (Chất lượng Phụ đề chuẩn):
    -[LOGIC FIX]: Áp dụng Sự khoan hồng rớt chữ (Flicker Forgiveness) cho các câu
      chỉ tồn tại trong 1-2 khung hình, cho phép chúng bypass Particle Guard để
      được gộp gọn gàng vào câu chính, xóa sổ rác chớp nháy.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter

import rapidfuzz.fuzz as fuzz
from loguru import logger

from subtitles_extractor.application.services.cjk_utils import (
    cjk_char_count,
    contains_cjk,
    effective_text_length,
)
from subtitles_extractor.application.services.text_similarity import text_similarity
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

# Rác tuyệt đối nếu xuất hiện ≤ 2 frame ở video 25fps (0.04*2 = 0.08s)
JUNK_DURATION_SEC: float = 0.085

_SHORT_DROP_RATIO: float = 1.0 / 5.0
_CJK_SHORT_ABSORB_THRESHOLD: float = 0.55
_CJK_SHORT_CHAR_LIMIT: int = 6
_HIGH_CONFIDENCE_THRESHOLD: float = 0.70

_VALID_TEXT_RE: re.Pattern[str] = re.compile(r'[^\W_]', flags=re.UNICODE)


def absorb_flickers(
    events: list[SubtitleEvent],
    min_duration_sec: float,
    similarity_threshold: float,
    merge_gap_sec: float,
    max_duration_sec: float | None = None,
) -> list[SubtitleEvent]:
    """Hấp thụ khung chớp; dòng ngắn hơn ``min_duration_sec`` được kéo dài tới mức đó.

    [v3.23.674] ``max_duration_sec``: CHỈ phần kéo dài không vượt "Thời lượng tối đa" (khi người dùng đặt tối thiểu > tối
    đa thì tối đa thắng). Việc phân loại khung chớp / ngưỡng loại vẫn theo ``min_duration_sec`` như cũ.
    """
    extend_to_sec = min_duration_sec
    if max_duration_sec is not None and max_duration_sec > 0:
        extend_to_sec = min(min_duration_sec, max_duration_sec)
    if not events or min_duration_sec <= 0:
        return list(events)

    working_events = list(events)
    # [v3.23.720] Phiếu bầu chữ theo SỐ KHUNG của mọi câu đã hấp thụ vào một câu (qua nhiều lần hấp thụ nối tiếp).
    # Trước đây câu gộp luôn mang chữ của câu "chủ": OCR thật test3 3304,12 s đọc "我来了" 16 khung xen "我来" 4 khung,
    # viterbi chia 5 nhóm xen kẽ, câu gộp ra "我来" (giữ id câu → giữ cả đối tượng để id không bị tái dùng).
    text_votes: dict[int, tuple[SubtitleEvent, Counter[str]]] = {}

    def votes_of(event: SubtitleEvent) -> Counter[str]:
        entry = text_votes.get(id(event))
        if entry is not None and entry[0] is event:
            return entry[1]
        return Counter({event.text: max(1, event.frame_count)})

    def absorb(host: SubtitleEvent, flicker: SubtitleEvent, absorb_after: bool) -> SubtitleEvent:
        tally = votes_of(host) + votes_of(flicker)
        merged = _absorb_into(host, flicker, absorb_after=absorb_after)
        best = max(tally.items(), key=lambda item: (item[1], item[0] == host.text))[0]
        if best != merged.text:
            merged = dataclasses.replace(merged, text=best)
        text_votes[id(merged)] = (merged, tally)
        return merged

    absorbed_count = 0
    extended_count = 0
    dropped_flicker_count = 0
    dropped_junk_count = 0

    result: list[SubtitleEvent] = []
    i = 0
    n = len(working_events)

    while i < n:
        current = working_events[i]
        dur = current.duration_sec

        is_high_confidence = float(current.confidence) >= _HIGH_CONFIDENCE_THRESHOLD
        is_meaningful = bool(_VALID_TEXT_RE.search(current.text))

        if not is_meaningful:
            dropped_junk_count += 1
            i += 1
            continue

        if dur >= min_duration_sec:
            result.append(current)
            i += 1
            continue

        prev_event = result[-1] if result else None
        next_event = working_events[i + 1] if i + 1 < n else None

        clean_text = current.text.replace("\n", "").replace(" ", "")
        eff_len = effective_text_length(clean_text)

        # [CRITICAL FIX]: Gỡ bỏ án tử hình cho câu chớp nháy (1-2 Frames) bằng cách bypass CPS Threshold.
        if dur < 0.1:
            is_readable_flicker = True
        else:
            cps = eff_len / dur if dur > 0 else 0
            is_readable_flicker = cps <= 35.0

        is_protected = is_high_confidence and is_readable_flicker

        if dur <= JUNK_DURATION_SEC and eff_len <= 3 and not is_protected:
            dropped_junk_count += 1
            logger.debug(
                "Rác OCR cực ngắn drop: '{}' (dur={:.3f}s).", current.text[:20], dur
            )
            i += 1
            continue

        # Thẩm định hàng xóm, nới lỏng cho các frame siêu mỏng (Flicker Forgiveness)
        prev_score = _evaluate_neighbor(
            current, prev_event, similarity_threshold, merge_gap_sec, is_prev=True, is_flicker_duration=(dur <= 0.10)
        )
        next_score = _evaluate_neighbor(
            current, next_event, similarity_threshold, merge_gap_sec, is_prev=False, is_flicker_duration=(dur <= 0.10)
        )

        if prev_score is not None or next_score is not None:
            absorb_to_prev = _should_absorb_to_prev(prev_score, next_score)

            if absorb_to_prev and prev_event is not None:
                result[-1] = absorb(prev_event, current, absorb_after=True)
                absorbed_count += 1
                i += 1
                continue
            if next_event is not None:
                extended_next = absorb(next_event, current, absorb_after=False)
                working_events[i + 1] = extended_next
                absorbed_count += 1
                i += 1
                continue

        short_drop_threshold = min_duration_sec * _SHORT_DROP_RATIO

        if dur < short_drop_threshold and not is_protected:
            dropped_flicker_count += 1
            logger.debug(
                "Flicker drop: '{}' (dur={:.3f}s < min_drop={:.3f}s).",
                current.text[:20], dur, short_drop_threshold
            )
            i += 1
            continue

        extended_event = _extend_duration(current, extend_to_sec, next_event)

        if extended_event.duration_sec > 0:
            result.append(extended_event)
            extended_count += 1
        else:
            dropped_flicker_count += 1

        i += 1

    # [v3.23.720] Câu sinh ra do HẤP THỤ cũng phải đạt thời lượng tối thiểu như mọi câu ngắn: trước đây câu gộp không bao
    # giờ được xét lại. Câu lẻ 5,0–5,2 s bị chia hai mảnh (5,0–5,1 / 5,2): mảnh đầu kéo dài bị chặn ở mốc mảnh sau, mảnh
    # sau bị hấp thụ ngược → câu chỉ dài 0,2 s thay vì 0,5 s (trước bản này lỗi bị đệm đuôi 0,05 s che đi).
    for position, event in enumerate(result):
        if id(event) in text_votes and event.duration_sec < extend_to_sec:
            following = result[position + 1] if position + 1 < len(result) else None
            result[position] = _extend_duration(event, extend_to_sec, following)

    total_dropped = dropped_flicker_count + dropped_junk_count
    if total_dropped or absorbed_count or extended_count:
        logger.info(
            "Flicker absorption: hấp thụ {}, kéo dài {}, "
            "drop flicker {}, drop rác OCR {}.",
            absorbed_count, extended_count,
            dropped_flicker_count, dropped_junk_count,
        )
    return result


def _evaluate_neighbor(
    current: SubtitleEvent,
    neighbor: SubtitleEvent | None,
    similarity_threshold: float,
    merge_gap_sec: float,
    is_prev: bool,
    is_flicker_duration: bool = False,
) -> float | None:
    if neighbor is None:
        return None

    gap = (
        current.start_sec - neighbor.end_sec
        if is_prev
        else neighbor.start_sec - current.end_sec
    )

    if gap > merge_gap_sec:
        return None

    # Mặc định lấy theo Particle Guard
    score = text_similarity(current.text, neighbor.text)

    # ĐẶC ÂN KHOAN HỒNG (Flicker Forgiveness):
    # Dành cho frame cực ngắn (1-2 khung hình), bypass Particle Guard nếu nó giống y hệt gốc từ
    if is_flicker_duration and score == 0.0:
        raw_ratio = fuzz.ratio(current.text, neighbor.text) / 100.0
        if raw_ratio > 0.80:
            score = raw_ratio

    is_short_cjk = (
        contains_cjk(current.text)
        and cjk_char_count(current.text) <= _CJK_SHORT_CHAR_LIMIT
    )

    effective_threshold = (
        min(similarity_threshold, _CJK_SHORT_ABSORB_THRESHOLD)
        if is_short_cjk
        else similarity_threshold
    )

    if score < effective_threshold:
        return None

    return score


def _should_absorb_to_prev(
    prev_score: float | None, next_score: float | None
) -> bool:
    if prev_score is not None and next_score is None: return True
    if next_score is not None and prev_score is None: return False
    if prev_score is None or next_score is None: return True
    return prev_score >= next_score


def _absorb_into(
    host: SubtitleEvent, flicker: SubtitleEvent, absorb_after: bool
) -> SubtitleEvent:
    if absorb_after:
        new_start = host.start_sec
        new_end = max(host.end_sec, flicker.end_sec)
    else:
        new_start = min(host.start_sec, flicker.start_sec)
        new_end = host.end_sec

    total_frames = host.frame_count + flicker.frame_count
    if total_frames > 0:
        weighted_conf = (
            float(host.confidence) * host.frame_count
            + float(flicker.confidence) * flicker.frame_count
        ) / total_frames
    else:
        weighted_conf = float(host.confidence)

    return SubtitleEvent(
        index=host.index,
        text=host.text,
        interval=TimeInterval(new_start, new_end),
        confidence=Confidence(weighted_conf),
        frame_count=total_frames,
        position=host.position,
        bounding_box=host.bounding_box,
        uid=host.uid,
    )


def _extend_duration(
    event: SubtitleEvent,
    min_duration_sec: float,
    next_event: SubtitleEvent | None,
) -> SubtitleEvent:
    desired_end = event.start_sec + min_duration_sec

    if next_event is not None:
        desired_end = min(desired_end, next_event.start_sec - 0.001)

    desired_end = max(desired_end, event.end_sec)

    if desired_end <= event.start_sec:
        desired_end = event.start_sec + 0.05

    return SubtitleEvent(
        index=event.index,
        text=event.text,
        interval=TimeInterval(event.start_sec, desired_end),
        confidence=event.confidence,
        frame_count=event.frame_count,
        position=event.position,
        bounding_box=event.bounding_box,
        uid=event.uid,
    )

__all__ = ["JUNK_DURATION_SEC", "absorb_flickers"]
