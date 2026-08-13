"""Tinh lọc event cuối pipeline: watermark Latin + mảnh flash OCR.

Các filter ở đây chạy SAU toàn bộ pipeline dựng phụ đề, xử lý các loại rác đặc thù
quan sát được từ dữ liệu thực tế (phim CJK có watermark động kiểu ``CIACA``/
``HOME``/``BOR`` trôi qua vùng phụ đề):

* Event THUẦN Latin viết hoa ngắn giữa biển phụ đề CJK → watermark.
* Đuôi Latin dính cuối câu CJK (``嗯不对OVONET``) → cắt đuôi.
* Mảnh flash cực ngắn (< 0.18s — phụ đề thật ngắn nhất đo được ~0.21s) chỉ drop
  khi có thêm bằng chứng rác: 1 ký tự CJK, chứa dòng toàn chữ số, hoặc là echo
  (na ná event lân cận).
"""

from __future__ import annotations

import dataclasses
import re

from rapidfuzz import fuzz

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent

#: Từ Latin viết hoa hợp lệ thường gặp trong thoại CJK — không bao giờ drop.
_LATIN_WHITELIST: frozenset[str] = frozenset(
    {"OK", "KTV", "VIP", "NO", "YES", "SOS", "GPS", "ATM", "DNA", "CEO",
     "WIFI", "APP", "AI", "NBA", "MBA", "DJ", "PPT", "GDP", "CP", "BGM"}
)

#: Event thuần Latin dài hơn ngưỡng này không bị coi là watermark (câu thật).
_WATERMARK_MAX_LEN: int = 12

#: Tỉ lệ event CJK tối thiểu của cả phim để kích hoạt watermark filter.
_CJK_DOMINANT_RATIO: float = 0.80

#: Mảnh flash: ngắn hơn ngưỡng này cần thêm bằng chứng rác mới bị drop.
_FLASH_MAX_DURATION_SEC: float = 0.18

#: Echo: similarity tối thiểu với event lân cận để coi mảnh flash là echo.
_FLASH_ECHO_SIMILARITY: float = 0.66

#: Cửa sổ thời gian (s) tìm event lân cận khi kiểm echo.
_FLASH_ECHO_WINDOW_SEC: float = 4.0

_PURE_LATIN_REGEX = re.compile(r"^[A-Za-z]+$")
_TRAILING_LATIN_REGEX = re.compile(r"[A-Z]{2,8}$")
_DIGIT_ONLY_LINE_REGEX = re.compile(r"^\d+$")


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


def _cjk_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _is_cjk_char(c)) / len(chars)


def _events_are_cjk_dominant(events: list[SubtitleEvent]) -> bool:
    if not events:
        return False
    cjk_count = sum(1 for e in events if _cjk_ratio(e.text) >= 0.5)
    return cjk_count / len(events) >= _CJK_DOMINANT_RATIO


def strip_latin_suffix_from_cjk_events(
    events: list[SubtitleEvent],
) -> list[SubtitleEvent]:
    """Cắt đuôi Latin viết hoa dính cuối câu CJK (watermark trôi vào).

    Ví dụ thực tế: ``嗯不对OVONET`` → ``嗯不对``; ``还在公司里泡小姑娘BOR`` →
    ``还在公司里泡小姑娘``. Chỉ cắt khi phần CJK còn lại >= 2 ký tự và đuôi Latin
    không thuộc whitelist (OK/VIP…).

    Args:
        events: Events đã dựng.

    Returns:
        Events với text đã cắt đuôi watermark (giữ nguyên timing).
    """
    if not _events_are_cjk_dominant(events):
        return events
    refined: list[SubtitleEvent] = []
    for event in events:
        text = event.text
        match = _TRAILING_LATIN_REGEX.search(text)
        if match and match.group() not in _LATIN_WHITELIST:
            remaining = text[: match.start()].rstrip()
            if sum(1 for c in remaining if _is_cjk_char(c)) >= 2:
                refined.append(dataclasses.replace(event, text=remaining))
                continue
        refined.append(event)
    return refined


def drop_pure_latin_watermark_events(
    events: list[SubtitleEvent],
) -> list[SubtitleEvent]:
    """Drop event THUẦN Latin viết hoa ngắn trong phim phụ đề CJK chủ đạo.

    Watermark động (``ALENCIACA``/``CIACA``/``HTU``/``HOME``) trôi qua ROI sinh
    event thuần Latin; trong phim >= 80% event CJK, các event này gần như chắc
    chắn là watermark. Whitelist các từ hợp lệ (OK/VIP/KTV…) được giữ.

    Args:
        events: Events đã dựng.

    Returns:
        Events đã loại watermark thuần Latin.
    """
    if not _events_are_cjk_dominant(events):
        return events
    kept: list[SubtitleEvent] = []
    for event in events:
        stripped = event.text.strip()
        if (
            _PURE_LATIN_REGEX.match(stripped)
            and stripped.upper() == stripped
            and 2 <= len(stripped) <= _WATERMARK_MAX_LEN
            and stripped.upper() not in _LATIN_WHITELIST
        ):
            continue
        kept.append(event)
    return kept


def drop_flash_fragments(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    """Drop mảnh flash cực ngắn (< 0.18s) CÓ THÊM bằng chứng rác.

    Phụ đề thật ngắn nhất quan sát được ~0.21s, nhưng OCR có thể bắt thiếu frame
    nên KHÔNG drop mù theo duration. Mảnh < 0.18s chỉ bị drop khi:
      * chỉ còn 1 ký tự CJK, hoặc
      * chứa dòng toàn chữ số (mã rác OCR như ``216``), hoặc
      * là ECHO: na ná (>= 0.66) một event lân cận trong ±4s (mảnh vỡ sớm/muộn
        của câu thật đã có event riêng).

    Args:
        events: Events đã dựng (sort theo thời gian).

    Returns:
        Events đã loại mảnh flash rác.
    """
    kept: list[SubtitleEvent] = []
    for idx, event in enumerate(events):
        duration = event.end_sec - event.start_sec
        if duration >= _FLASH_MAX_DURATION_SEC:
            kept.append(event)
            continue

        stripped = "".join(event.text.split())
        cjk_chars = sum(1 for c in stripped if _is_cjk_char(c))

        is_single_cjk = cjk_chars <= 1 and len(stripped) <= 2
        has_digit_line = any(
            _DIGIT_ONLY_LINE_REGEX.match(line.strip())
            for line in event.text.splitlines()
        )
        is_echo = False
        if not (is_single_cjk or has_digit_line):
            for other in events[max(0, idx - 3): idx + 4]:
                if other is event:
                    continue
                if (
                    abs(other.start_sec - event.start_sec) <= _FLASH_ECHO_WINDOW_SEC
                    and (other.end_sec - other.start_sec) > duration
                    and fuzz.ratio(event.text, other.text) / 100.0
                    >= _FLASH_ECHO_SIMILARITY
                ):
                    is_echo = True
                    break

        if is_single_cjk or has_digit_line or is_echo:
            continue
        kept.append(event)
    return kept


def refine_final_events(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    """Pipeline tinh lọc cuối: suffix watermark → pure-Latin watermark → flash."""
    refined = strip_latin_suffix_from_cjk_events(events)
    refined = drop_pure_latin_watermark_events(refined)
    return drop_flash_fragments(refined)
