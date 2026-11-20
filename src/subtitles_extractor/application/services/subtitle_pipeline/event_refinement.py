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
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

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

#: [v3.23.729] Câu MỘT chữ Hán ngắn hơn ngưỡng flash vẫn là phụ đề thật khi có từ chừng này khung đọc trở lên...
_FLASH_SINGLE_MIN_FRAMES: int = 3
#: ...và tin cậy trung bình từ mức này. Phim "[全集]" (98 phút, bước 0,04 s): 7 câu cảm thán thật hiện 3–4 khung
#: (0,12–0,16 s: 啊, 啊, 嘿, 哼, 哎, 哎, 额; tin cậy 0,88–1,00) bị xoá; rác đã gặp đều 1–2 khung hoặc tin cậy ≤ 0,81.
_FLASH_SINGLE_MIN_CONFIDENCE: float = 0.85

#: Echo: similarity tối thiểu với event lân cận để coi mảnh flash là echo.
_FLASH_ECHO_SIMILARITY: float = 0.66

#: Cửa sổ thời gian (s) tìm event lân cận khi kiểm echo.
_FLASH_ECHO_WINDOW_SEC: float = 4.0

#: [v3.23.616] Event chỉ có chữ số/dấu câu dài tối đa chừng này ký tự (bỏ khoảng trắng)...
_DIGIT_GARBAGE_MAX_LEN: int = 4
#: ...và conf dưới mức này là rác OCR trong phim phụ đề CJK.
_DIGIT_GARBAGE_MAX_CONFIDENCE: float = 0.90

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


def _is_digit_garbage(text: str, confidence: float) -> bool:
    """Chuỗi chỉ gồm chữ số + dấu câu, ngắn, conf chưa cao (hàm thuần)."""
    compact = "".join(text.split())
    return (
        0 < len(compact) <= _DIGIT_GARBAGE_MAX_LEN
        and any(ch.isdigit() for ch in compact)
        and not any(ch.isalpha() for ch in compact)
        and confidence < _DIGIT_GARBAGE_MAX_CONFIDENCE
    )


def drop_digit_garbage_events(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    """[v3.23.616] Drop event chỉ có chữ số/dấu câu, ngắn, conf thấp trong phim CJK.

    OCR thật 14/10 (phim "天骄女友退婚" tập 1, cả hai lần trích): khoen thắt lưng và hoạ
    tiết áo lọt vùng phụ đề sinh "00" (7 khung, 0.32s, conf 0.70) và "0.." (conf 0.83).
    Hai event này dài hơn ngưỡng mảnh flash nên lọt mọi bộ lọc; cả hai lần người dùng
    phải xoá tay trong trình sửa. Số thật trong thoại CJK được viết kèm chữ ("二十万",
    "20万") hoặc được OCR đọc rõ (conf >= 0.90) nên không bị đụng.

    Args:
        events: Events đã dựng.

    Returns:
        Events đã loại rác chữ số; nguyên danh sách nếu phim không phải CJK chủ đạo.
    """
    if not _events_are_cjk_dominant(events):
        return events
    return [
        event for event in events
        if not _is_digit_garbage(event.text, float(event.confidence))
    ]


#: [v3.23.672] Rác "chữ + số" Latin (hoạ tiết lọt vùng phụ đề): ngưỡng tin cậy và thời lượng.
_ALNUM_GARBAGE_MAX_CONFIDENCE = 0.75
_ALNUM_GARBAGE_MAX_DURATION_SEC = 0.50
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _is_alnum_garbage(text: str, confidence: float, duration_sec: float) -> bool:
    """Chuỗi Latin LẪN chữ số, không có chữ CJK, conf thấp, hiện rất ngắn (hàm thuần).

    Ca thật (tập 5 phim mẫu, bước 0,04s): "200 2a1s wses" — 0,19s, conf ~0,57, do hoạ tiết
    quần áo lướt qua vùng phụ đề. Bộ lọc watermark chỉ bắt Latin VIẾT HOA, bộ lọc chữ số chỉ
    bắt chuỗi KHÔNG có chữ cái. Chữ Latin thật trong thoại CJK ("OK", "iPhone 15") được OCR đọc
    rõ (conf >= 0,90) hoặc hiện lâu, nên không bị đụng.
    """
    compact = "".join(text.split())
    return (
        bool(compact)
        and not _CJK_CHAR_RE.search(compact)
        and any(ch.isdigit() for ch in compact)
        and any(ch.isascii() and ch.isalpha() for ch in compact)
        and confidence < _ALNUM_GARBAGE_MAX_CONFIDENCE
        and duration_sec < _ALNUM_GARBAGE_MAX_DURATION_SEC
    )


def drop_alnum_garbage_events(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    """[v3.23.672] Drop event rác "chữ Latin + số" ngắn, conf thấp trong phim CJK chủ đạo."""
    if not _events_are_cjk_dominant(events):
        return events
    return [
        event for event in events
        if not _is_alnum_garbage(event.text, float(event.confidence), event.duration_sec)
    ]


def drop_flash_fragments(events: list[SubtitleEvent], sample_step_sec: float = 0.0) -> list[SubtitleEvent]:
    """Drop mảnh flash cực ngắn (< 0.18s) CÓ THÊM bằng chứng rác.

    Phụ đề thật ngắn nhất quan sát được ~0.21s, nhưng OCR có thể bắt thiếu frame
    nên KHÔNG drop mù theo duration. Mảnh < 0.18s chỉ bị drop khi:
      * chỉ còn 1 ký tự CJK, hoặc
      * chứa dòng toàn chữ số (mã rác OCR như ``216``), hoặc
      * là ECHO: na ná (>= 0.66) một event lân cận trong ±4s (mảnh vỡ sớm/muộn
        của câu thật đã có event riêng).

    [v3.23.682] Thời lượng tính theo BẰNG CHỨNG: ≥ số khung × bước lấy mẫu. Mốc cuối của câu là khung mẫu cuối (chưa cộng
    phần giữ khung), nên câu 2 khung ở bước 0,12 s chỉ "dài" 0,15 s dù hiện ≥ 0,24 s — trước đây bị coi là mảnh chớp:
    phim nối 86 tập mất 6 câu một chữ có thật (我, 当, 你, 切, 你, 来; tin cậy 1,0). Ở bước 0,04 s, 2 khung = 0,08 s → như cũ.
    Chỉ áp cho câu có chữ Hán: mảnh Latin lẻ ("V" — tin cậy 0,99, 2 khung) vẫn là rác như trước.

    Args:
        events: Events đã dựng (sort theo thời gian).
        sample_step_sec: Bước lấy mẫu OCR (0 = không rõ → chỉ dùng mốc thời gian như cũ).

    Returns:
        Events đã loại mảnh flash rác.
    """
    kept: list[SubtitleEvent] = []
    for idx, event in enumerate(events):
        duration = event.end_sec - event.start_sec
        if any(_is_cjk_char(c) for c in event.text):   # chữ Latin/số lẻ loi (vd "V") vẫn tính như cũ
            duration = max(duration, event.frame_count * max(0.0, sample_step_sec))
        if duration >= _FLASH_MAX_DURATION_SEC:
            kept.append(event)
            continue

        stripped = "".join(event.text.split())
        cjk_chars = sum(1 for c in stripped if _is_cjk_char(c))

        is_single_cjk = cjk_chars <= 1 and len(stripped) <= 2
        # [v3.23.729] Chữ Hán đọc rõ qua ≥ 3 khung là câu cảm thán hiện chớp nhoáng có thật, không phải rác chớp.
        if (
            is_single_cjk and cjk_chars == 1 and sample_step_sec > 0
            and event.frame_count >= _FLASH_SINGLE_MIN_FRAMES
            and float(event.confidence) >= _FLASH_SINGLE_MIN_CONFIDENCE
        ):
            kept.append(event)
            continue
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


#: [v3.23.654] Khoảng hở tối đa (giây) giữa hai câu để coi câu sau là bản "hiện dần" của câu trước.
PROGRESSIVE_MAX_GAP_SEC: float = 0.35


def _progressive_key(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def merge_progressive_events(
    events: list[SubtitleEvent], max_gap_sec: float = PROGRESSIVE_MAX_GAP_SEC,
) -> tuple[list[SubtitleEvent], int]:
    """[v3.23.654] Gộp phụ đề "hiện dần": ``我`` rồi ``我只是`` liền nhau → một câu ``我只是`` từ mốc của câu đầu.

    Phụ đề kiểu chữ chạy hiện từng phần; bộ dựng tách mỗi trạng thái thành một câu nên TTS đọc lặp ("Ta… Ta chỉ là…")
    và bản dịch có hai dòng. Điều kiện: chữ câu trước (bỏ khoảng trắng) là TIỀN TỐ thật sự của câu sau và câu sau bắt
    đầu không quá ``max_gap_sec`` sau khi câu trước tắt. Chuỗi nhiều bước (``师`` → ``师叔`` → ``师叔你``) gộp hết.

    Returns:
        ``(danh sách mới, số câu đã gộp vào câu sau)``.
    """
    merged: list[SubtitleEvent] = []
    absorbed = 0
    for event in events:
        if merged:
            previous = merged[-1]
            before, after = _progressive_key(previous.text), _progressive_key(event.text)
            gap = float(event.start_sec) - float(previous.end_sec)
            if before and len(after) > len(before) and after.startswith(before) and -0.05 <= gap <= max_gap_sec:
                merged[-1] = dataclasses.replace(
                    event,
                    interval=dataclasses.replace(event.interval, start_sec=previous.start_sec),
                    frame_count=int(getattr(previous, "frame_count", 0)) + int(getattr(event, "frame_count", 0)),
                )
                absorbed += 1
                continue
        merged.append(event)
    return merged, absorbed


#: [v3.23.734] Mảnh đọc thiếu chữ của câu liền kề: dài tối đa chừng này giây...
TRUNCATED_FRAGMENT_MAX_SEC = 0.30
#: ...và câu đầy đủ dài ít nhất chừng này lần mảnh.
TRUNCATED_FRAGMENT_NEIGHBOR_RATIO = 1.5
_HAN_ONLY = re.compile(r"[\u3400-\u9fff]+")


def _absorbs(full: SubtitleEvent, fragment: SubtitleEvent, gap: float) -> bool:
    """``fragment`` là ``full`` bị đọc thiếu đoạn đầu/đuôi, hiện sát ``full`` và ngắn hẳn."""
    a, b = "".join(full.text.split()), "".join(fragment.text.split())
    return (
        bool(b) and len(a) > len(b) and _HAN_ONLY.fullmatch(a) is not None and _HAN_ONLY.fullmatch(b) is not None
        and (a.startswith(b) or a.endswith(b))
        and fragment.duration_sec <= TRUNCATED_FRAGMENT_MAX_SEC
        and full.duration_sec >= TRUNCATED_FRAGMENT_NEIGHBOR_RATIO * fragment.duration_sec
        and gap >= -1e-3
    )


def absorb_truncated_fragments(events: list[SubtitleEvent], sample_step_sec: float = 0.0) -> list[SubtitleEvent]:
    """[v3.23.734] Gộp mảnh ĐỌC THIẾU chữ (≤ 0,3 s) vào câu đầy đủ liền kề.

    Phim "反派" (83 tập): trên nền sáng chuyển động bộ lọc khung coi mỗi khung là "chữ đổi", OCR đọc vài khung thiếu chữ
    cuối/đầu và khâu dựng tách thành câu riêng: "懂什么" | "懂什" (0,20 s), "走火了" | "走火" (0,24 s), "魔了" | "魔" |
    "魔了" | "魔" (một câu "入魔了"), "小翠" | "翠" | "小翠" — ảnh từng khung cho thấy phụ đề KHÔNG đổi. Câu thật nối tiếp
    ("你" | "你要干什么", "只当我不是你嫂嫂" | "嫂嫂") dài từ 0,56 s trở lên nên không bị đụng.
    Mảnh chỉ được gộp khi đứng SAU câu đầy đủ hoặc KẸP GIỮA hai lần cùng câu: phụ đề hiện dần thật luôn đặt phần ngắn
    TRƯỚC ("啊" | "这这全部啊", "这" | "这不可能" — ảnh xác nhận). Hai câu cùng chữ đứng liền nhau chỉ được nối khi một trong
    hai vừa nuốt mảnh; câu lặp lại thật có khoảng trống giữa ("冷静" | "冷静") giữ nguyên. Khoảng hở cho phép: 1,5 bước lấy mẫu (không rõ bước → 0,05 s).
    """
    if len(events) < 2:
        return events
    max_gap = 1.5 * sample_step_sec if sample_step_sec > 0 else 0.05
    out: list[SubtitleEvent] = []
    absorbed: list[bool] = []
    for event in events:
        if out and event.start_sec - out[-1].end_sec <= max_gap:
            previous = out[-1]
            gap = event.start_sec - previous.end_sec
            keep: SubtitleEvent | None = None
            if _absorbs(previous, event, gap):                    # "懂什么" | "懂什" → câu trước kéo dài
                keep = dataclasses.replace(
                    previous, interval=TimeInterval(previous.start_sec, max(previous.end_sec, event.end_sec)),
                    frame_count=previous.frame_count + event.frame_count,
                )
            elif (
                len(out) >= 2 and _absorbs(event, previous, gap)
                and "".join(out[-2].text.split()) == "".join(event.text.split())
                and previous.start_sec - out[-2].end_sec <= max_gap
            ):                                                    # "小翠" | "翠" | "小翠": mảnh kẹp giữa → một câu
                keep = dataclasses.replace(
                    out[-2], interval=TimeInterval(out[-2].start_sec, event.end_sec),
                    frame_count=out[-2].frame_count + previous.frame_count + event.frame_count,
                )
                out.pop()
                absorbed.pop()
            elif absorbed[-1] and "".join(previous.text.split()) == "".join(event.text.split()):
                keep = dataclasses.replace(                        # "魔了"(+"魔") | "魔了" → một câu
                    previous, interval=TimeInterval(previous.start_sec, max(previous.end_sec, event.end_sec)),
                    frame_count=previous.frame_count + event.frame_count,
                )
            if keep is not None:
                out[-1] = keep
                absorbed[-1] = True
                continue
        out.append(event)
        absorbed.append(False)
    return out


def refine_final_events(events: list[SubtitleEvent], sample_step_sec: float = 0.0) -> list[SubtitleEvent]:
    """Tinh lọc cuối: mảnh đọc thiếu chữ → đuôi watermark → watermark Latin → rác chữ số → mảnh flash."""
    refined = absorb_truncated_fragments(events, sample_step_sec)
    refined = strip_latin_suffix_from_cjk_events(refined)
    refined = drop_pure_latin_watermark_events(refined)
    refined = drop_digit_garbage_events(refined)
    refined = drop_alnum_garbage_events(refined)
    return drop_flash_fragments(refined, sample_step_sec)
