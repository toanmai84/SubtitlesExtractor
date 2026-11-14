"""[v3.23.708] Soát chữ OCR bằng HÌNH HỌC hộp chữ: câu đọc thiếu / thừa một chữ (hàm thuần).

Phụ đề cứng dùng một cỡ chữ và căn giữa, nên bề rộng hộp chữ gần như tuyến tính theo số chữ và tâm hộp trùng tâm phụ
đề. Câu OCR đọc SÓT một chữ để lại dấu vết mà độ tin cậy không cho thấy (PP-OCRv6 vẫn trả ~1,0):

* sót chữ ở GIỮA ("其实我直都在" — thiếu "一"): hộp rộng hơn số chữ đọc được khoảng một chữ;
* sót chữ ở ĐẦU / CUỐI ("个万年炼气的人", "任务"): hộp lệch tâm nửa chữ về phía còn lại;
* đọc THỪA một chữ ("成功解除余一部分"): hộp hẹp hơn số chữ khoảng một chữ.

Đo trên lượt chạy thật phim nối 86 tập (3455 câu một dòng toàn chữ Hán, đáp án đã kiểm bằng ảnh): bề rộng ≈
49,8 × số chữ + 17,4 px. Câu ĐÚNG lệch tối đa +0,52 / −0,63 chữ và lệch tâm tối đa 0,34 chữ; 5 câu SAI còn lại lệch
+1,18, +1,09, −0,98 chữ hoặc lệch tâm 0,46, 0,49 chữ — bắt đủ 5, không gắn nhầm câu nào.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Ghi chú ``SubtitleEvent.ocr_notes``.
OCR_NOTE_MISSING_CHAR = "missing_char"
OCR_NOTE_EXTRA_CHAR = "extra_char"

#: Hộp rộng hơn số chữ đọc được quá ngần này chữ → nghi sót chữ ở giữa (câu đúng: tối đa +0,52).
MISSING_WIDTH_CHARS = 0.75
#: Hộp hẹp hơn số chữ quá ngần này chữ → nghi đọc thừa chữ (câu đúng: tối thiểu −0,63).
EXTRA_WIDTH_CHARS = 0.80
#: Tâm hộp lệch tâm phụ đề quá ngần này chữ → nghi sót chữ ở mép (câu đúng: tối đa 0,34).
EDGE_OFFSET_CHARS = 0.40
#: Cần ít nhất ngần này câu để ước lượng cỡ chữ và tâm phụ đề của phim.
MIN_REFERENCE_EVENTS = 30
#: [v3.23.710] Video ngắn: từ chừng này câu đo được và chừng này số chữ khác nhau thì khớp bằng Theil–Sen.
MIN_SMALL_FILM_EVENTS = 15
MIN_SMALL_FILM_CHAR_COUNTS = 4
#: Mỗi câu cần ít nhất ngần này khung đọc đúng chữ đã bầu để lấy trung vị hộp chữ.
MIN_EVENT_FRAMES = 3
#: Phụ đề không căn giữa (tâm các câu tản rộng hơn ngần này chữ ở phân vị 90) thì bỏ phép kiểm lệch tâm.
MAX_CENTER_SPREAD_CHARS = 0.25

_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class EventGeometry:
    """Hộp chữ trung vị của một câu.

    Attributes:
        position: Vị trí câu trong danh sách.
        chars: Số chữ của câu.
        width: Bề rộng hộp chữ (px).
        center: Hoành độ tâm hộp chữ (px).
    """

    position: int
    chars: int
    width: float
    center: float


def _key(text: str) -> str:
    return _SPACE_RE.sub("", text or "")


def _is_han(text: str) -> bool:
    return bool(text) and all("一" <= ch <= "鿿" or "㐀" <= ch <= "䶿" for ch in text)


def measure_events(events: Sequence[Any], frames: Sequence[Any]) -> list[EventGeometry]:
    """Hộp chữ trung vị của từng câu MỘT DÒNG toàn chữ Hán, lấy từ các khung đọc đúng chữ của câu.

    Args:
        events: Câu phụ đề (``start_sec``, ``end_sec``, ``text``), theo thứ tự thời gian.
        frames: Khung OCR (``timestamp_sec``, ``text_boxes`` có ``text`` và ``polygon``), theo thứ tự thời gian.
    """
    import bisect

    stamps = [float(frame.timestamp_sec) for frame in frames]
    measured: list[EventGeometry] = []
    for position, event in enumerate(events):
        raw = str(getattr(event, "text", "") or "")
        text = _key(raw)
        if "\n" in raw.strip() or len(text) < 2 or not _is_han(text):
            continue
        low = bisect.bisect_left(stamps, float(event.start_sec) - 1e-3)
        high = bisect.bisect_right(stamps, float(event.end_sec) + 1e-3)
        widths: list[float] = []
        centers: list[float] = []
        for frame in frames[low:high]:
            boxes = getattr(frame, "text_boxes", None) or ()
            if not boxes or _key("".join(str(box.text or "") for box in boxes)) != text:
                continue
            xs = [float(point[0]) for box in boxes for point in box.polygon]
            if not xs:
                continue
            widths.append(max(xs) - min(xs))
            centers.append((max(xs) + min(xs)) / 2.0)
        if len(widths) >= MIN_EVENT_FRAMES:
            measured.append(EventGeometry(position, len(text), statistics.median(widths), statistics.median(centers)))
    return measured


def geometry_notes(measured: Sequence[EventGeometry]) -> dict[int, str]:
    """``{vị trí câu: ghi chú}`` các câu có hộp chữ không khớp số chữ đọc được.

    Cỡ chữ và tâm phụ đề được ước lượng từ chính các câu của phim (trung vị bề rộng theo từng số chữ, khớp đường
    thẳng). Quá ít câu, hoặc bề rộng không tăng theo số chữ, thì không kết luận gì.
    """
    if len(measured) < MIN_SMALL_FILM_EVENTS:
        return {}
    by_chars: dict[int, list[float]] = {}
    for item in measured:
        by_chars.setdefault(item.chars, []).append(item.width)
    points = [(chars, statistics.median(widths)) for chars, widths in by_chars.items() if len(widths) >= 5]
    if len(measured) >= MIN_REFERENCE_EVENTS and len(points) >= 3:
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        spread = sum((x - mean_x) ** 2 for x, _ in points)
        if spread <= 0.0:
            return {}
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
        intercept = mean_y - slope * mean_x
    else:
        # [v3.23.710] Video ngắn (một tập ~30 câu): không số chữ nào có đủ 5 câu để lấy trung vị → trước đây không
        # gắn được cờ nào, câu thiếu "一" đầu câu ("眼就察觉到了吗", tập 77) không bao giờ được đọc lại. Khớp đường
        # thẳng bền với ngoại lai (Theil–Sen: trung vị độ dốc của mọi cặp câu khác số chữ) trên toàn bộ các câu.
        if len(by_chars) < MIN_SMALL_FILM_CHAR_COUNTS:
            return {}
        slopes = [
            (right.width - left.width) / (right.chars - left.chars)
            for index, left in enumerate(measured) for right in measured[index + 1:] if right.chars != left.chars
        ]
        if not slopes:
            return {}
        slope = statistics.median(slopes)
        intercept = statistics.median(item.width - slope * item.chars for item in measured)
    if slope <= 1.0:
        return {}
    center = statistics.median(item.center for item in measured)
    offsets = sorted(abs(item.center - center) / slope for item in measured)
    centered = offsets[int(0.9 * (len(offsets) - 1))] <= MAX_CENTER_SPREAD_CHARS

    notes: dict[int, str] = {}
    for item in measured:
        extra_width = (item.width - intercept) / slope - item.chars
        if extra_width > MISSING_WIDTH_CHARS:
            notes[item.position] = OCR_NOTE_MISSING_CHAR
        elif extra_width < -EXTRA_WIDTH_CHARS:
            notes[item.position] = OCR_NOTE_EXTRA_CHAR
        elif centered and abs(item.center - center) / slope > EDGE_OFFSET_CHARS:
            notes[item.position] = OCR_NOTE_MISSING_CHAR
    return notes


#: [v3.23.710] Câu KHÔNG có chữ Hán mà hộp chữ thấp hơn ngần này lần chiều cao chữ phụ đề của phim → vật trong cảnh.
SMALL_NON_HAN_HEIGHT_RATIO = 0.70
#: [v3.23.712] Câu KHÔNG có chữ Hán có tâm hộp lệch tâm phụ đề quá ngần này lần chiều cao chữ → vật trong cảnh.
OFF_CENTER_NON_HAN_HEIGHTS = 0.50
#: Chỉ áp cho phim mà câu không chữ Hán là HIẾM (nhiều hơn ngần này thì coi là phim song ngữ, giữ nguyên).
MAX_NON_HAN_EVENTS = 3
MAX_NON_HAN_SHARE = 0.02
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _median_box(event: Any, frames: Sequence[Any], stamps: Sequence[float]) -> tuple[float, float] | None:
    """``(chiều cao, tâm x)`` trung vị của hộp chữ qua các khung đọc đúng chữ của câu; không khung nào → ``None``."""
    import bisect

    text = _key(str(getattr(event, "text", "") or ""))
    low = bisect.bisect_left(stamps, float(event.start_sec) - 1e-3)
    high = bisect.bisect_right(stamps, float(event.end_sec) + 1e-3)
    heights: list[float] = []
    centers: list[float] = []
    for frame in frames[low:high]:
        boxes = getattr(frame, "text_boxes", None) or ()
        if not boxes or _key("".join(str(box.text or "") for box in boxes)) != text:
            continue
        xs = [float(point[0]) for box in boxes for point in box.polygon]
        ys = [float(point[1]) for box in boxes for point in box.polygon]
        if ys:
            heights.append(max(ys) - min(ys))
            centers.append((min(xs) + max(xs)) / 2.0)
    if not heights:
        return None
    return statistics.median(heights), statistics.median(centers)


def drop_small_non_han_events(events: Sequence[Any], frames: Sequence[Any]) -> tuple[list[Any], int]:
    """Bỏ câu KHÔNG có chữ Hán mà hình học hộp chữ không phải của phụ đề, trong phim phụ đề chữ Hán (hàm thuần).

    Rác kiểu này có độ tin cậy cao nên các bộ lọc dựa trên độ tin cậy không bắt được; hình học thì tách bạch:

    * [v3.23.710] CHỮ NHỎ — hộp thấp hơn :data:`SMALL_NON_HAN_HEIGHT_RATIO` lần chữ phụ đề. Lượt chạy 3.23.708:
      "000" 6 khung, tin cậy 0,98, hộp cao 39 px so với 61 px (0,64 lần; câu chữ Hán thấp nhất phim 0,72 lần).
    * [v3.23.712] LỆCH TÂM — phụ đề căn giữa mà tâm hộp lệch quá :data:`OFF_CENTER_NON_HAN_HEIGHTS` lần chiều cao
      chữ. Lượt chạy 3.23.711 (vùng chữ rộng cả khung, model khác): vẫn "000" đó nhưng hộp cao 54 px (0,79 lần, lọt
      luật chữ nhỏ), tâm lệch 57 px = 0,84 lần chiều cao. Trên 3648 câu của lượt chạy đó, câu ĐÚNG lệch nhiều nhất
      là 0,34 lần ("来人"), phân vị 99,9 % là 0,12 lần.

    Cần ít nhất :data:`MIN_REFERENCE_EVENTS` câu một dòng toàn chữ Hán để biết cỡ chữ / tâm của phim; không đủ thì
    giữ nguyên. Phim song ngữ (câu không chữ Hán không hiếm) cũng giữ nguyên.
    """
    stamps = [float(frame.timestamp_sec) for frame in frames]
    heights: list[float] = []
    centers: list[float] = []
    for event in events:
        raw = str(getattr(event, "text", "") or "")
        if "\n" in raw.strip() or len(_key(raw)) < 2 or not _is_han(_key(raw)):
            continue
        box = _median_box(event, frames, stamps)
        if box and box[0] > 0:
            heights.append(box[0])
            centers.append(box[1])
    if len(heights) < MIN_REFERENCE_EVENTS:
        return list(events), 0
    typical = statistics.median(heights)
    center = statistics.median(centers)
    spread = sorted(abs(value - center) for value in centers)[int(0.9 * (len(centers) - 1))]
    centered = spread <= MAX_CENTER_SPREAD_CHARS * typical
    # Phim song ngữ / phụ đề Latin: câu không chữ Hán là chuyện thường, có thể dùng cỡ chữ khác → không kết luận.
    non_han = sum(
        1 for event in events
        if _key(str(getattr(event, "text", "") or "")) and not _HAN_RE.search(str(getattr(event, "text", "") or ""))
    )
    if non_han > max(MAX_NON_HAN_EVENTS, MAX_NON_HAN_SHARE * len(events)):
        return list(events), 0
    kept: list[Any] = []
    for event in events:
        raw = str(getattr(event, "text", "") or "")
        if _key(raw) and not _HAN_RE.search(raw):
            box = _median_box(event, frames, stamps)
            if box is not None and (
                box[0] < SMALL_NON_HAN_HEIGHT_RATIO * typical
                or (centered and abs(box[1] - center) > OFF_CENTER_NON_HAN_HEIGHTS * typical)
            ):
                continue
        kept.append(event)
    return kept, len(events) - len(kept)


def mark_geometry_suspects(events: Sequence[Any], frames: Sequence[Any]) -> int:
    """Gắn ghi chú hình học vào ``ocr_notes`` của các câu nghi thiếu / thừa chữ. Trả số câu được gắn."""
    notes = geometry_notes(measure_events(events, frames))
    marked = 0
    for position, note in notes.items():
        event = events[position]
        existing = tuple(getattr(event, "ocr_notes", ()) or ())
        if note not in existing:
            event.ocr_notes = (*existing, note)
            marked += 1
    return marked


#: Bản đọc lại phải được ít nhất ngần này khung đồng thuận mới thay chữ.
REOCR_MIN_AGREEING_FRAMES = 2
#: Ghi chú gắn cho câu đã được vòng OCR lại sửa chữ (trang Chỉnh sửa hiện là "chữ được khôi phục").
OCR_NOTE_REOCR_FIXED = "restored_char"
#: [v3.23.713] Câu có hai bản đọc mâu thuẫn (lẫn chữ trên cùng một hình) — vòng tự soát đọc lại nhiều khung để phân xử.
OCR_NOTE_DISPUTED = "reads_disagree"
#: Số vị trí tối đa được khác nhau giữa bản đang giữ và bản đọc lại để coi là "cùng câu, khác cách đọc".
DISPUTED_MAX_SUBSTITUTIONS = 2
#: Bản đọc lại thắng khi có ít nhất chừng này phiếu, chiếm ít nhất chừng này phần số lần đọc ra câu, và hơn bản nhì
#: ít nhất chừng này phiếu.
DISPUTED_MIN_VOTES = 3
DISPUTED_MIN_SHARE = 0.5
DISPUTED_MIN_LEAD = 2


def _one_char_longer(longer: str, shorter: str) -> bool:
    """``longer`` = ``shorter`` chèn thêm đúng MỘT chữ Hán."""
    if len(longer) != len(shorter) + 1:
        return False
    for index, char in enumerate(longer):
        if longer[:index] + longer[index + 1:] == shorter:
            return _is_han(char)
    return False


def _only_yi_inserted(longer: str, shorter: str) -> bool:
    """``longer`` = ``shorter`` chèn thêm toàn chữ ``一`` (so khớp tham lam từ trái sang)."""
    index = 0
    for char in longer:
        if index < len(shorter) and char == shorter[index]:
            index += 1
        elif char != "一":
            return False
    return index == len(shorter)


def _closest_window(current: str, reread: str) -> str | None:
    """Đoạn dài bằng ``current`` trong ``reread`` khác ``current`` ít vị trí nhất (≤ :data:`DISPUTED_MAX_SUBSTITUTIONS`).

    Bản đọc lại trên vùng chữ rộng hay dính thêm chữ rác hai bên ("27走了", "天霄峰CHNCON") nên phải tìm câu bên trong.
    """
    size = len(current)
    best: tuple[int, str] | None = None
    for begin in range(len(reread) - size + 1):
        window = reread[begin:begin + size]
        differing = sum(1 for a, b in zip(window, current, strict=True) if a != b)
        if differing <= DISPUTED_MAX_SUBSTITUTIONS and (best is None or differing < best[0]):
            best = (differing, window)
    return best[1] if best else None


def accept_disputed_reread(current: str, reread_texts: Sequence[str]) -> str | None:
    """[v3.23.713] Phân xử câu có hai bản đọc mâu thuẫn bằng nhiều lần đọc lại trên ảnh gốc (hàm thuần).

    Trả bản đọc THẮNG phiếu (có thể chính là ``current`` = được xác nhận), hoặc ``None`` nếu chưa đủ chắc.

    Đo trên khung thật (model small, ảnh gốc không tiền xử lý, mỗi câu 12–24 khung): "做这等腌臜之事" 12 phiếu / "攒" 5;
    "这不是癞蛤蟆吗" 15 / "痴" 1; "把你狠狠踩在脚下" 17 / "狼" 7; "清清白白" 20 / "自白" 1; "白日宣淫" 16 / 0 — trong khi
    các lần đọc lúc trích xuất (có làm nét + tương phản) cho "攒" 25/36 khung và "自白" 15/21.
    """
    base = _key(current)
    if not base:
        return None
    counts: dict[str, int] = {}
    for text in reread_texts:
        window = _closest_window(base, _key(text))
        if window is not None:
            counts[window] = counts.get(window, 0) + 1
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    best, votes = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    total = sum(counts.values())
    if votes < DISPUTED_MIN_VOTES or votes < DISPUTED_MIN_SHARE * total or votes - runner_up < DISPUTED_MIN_LEAD:
        return None
    return best


def accept_reocr_text(current: str, note: str, reread_texts: Sequence[str]) -> str | None:
    """Chữ thay cho ``current`` theo các bản OCR đọc lại, hoặc ``None`` nếu không đủ chắc (hàm thuần).

    Chỉ nhận bản đọc lại khi nó giải thích đúng dấu hiệu hình học và không đổi gì khác:

    * ``missing_char``: bản mới = chữ cũ thêm đúng MỘT chữ Hán ("其实我直都在" → "其实我一直都在");
    * ``extra_char``: bản mới = chữ cũ bớt đúng MỘT chữ Hán ("成功解除余一部分" → "成功解除一部分");
    * ít nhất :data:`REOCR_MIN_AGREEING_FRAMES` khung đọc lại ra cùng bản đó và đó là bản nhiều nhất.

    Bản đọc lại khác kiểu khác (đổi chữ, thêm hai chữ, câu khác) bị bỏ — câu giữ nguyên và vẫn mang cờ nghi ngờ.
    """
    base = _key(current)
    counts: dict[str, int] = {}
    for text in reread_texts:
        key = _key(text)
        if key and "\n" not in (text or "").strip():
            counts[key] = counts.get(key, 0) + 1
    if not base or not counts:
        return None
    best, votes = max(counts.items(), key=lambda item: item[1])
    if votes < REOCR_MIN_AGREEING_FRAMES or sum(1 for count in counts.values() if count == votes) > 1:
        return None
    if note == OCR_NOTE_MISSING_CHAR and _one_char_longer(best, base):
        return best
    # [v3.23.710] Thiếu HAI chữ "一" ("百道雷劫起劈下来" → "一百道雷劫一起劈下来", lượt chạy thật): chỉ nhận khi mọi chữ thêm
    # vào đều là "一" — nét ngang mảnh là kiểu rơi chữ có hệ thống, chữ khác thì vẫn chỉ nhận đúng một.
    # Hai chữ "一" phải ở HAI CHỖ khác nhau: "一一" liền nhau là đọc lặp nét, không phải chữ của câu.
    if (
        note == OCR_NOTE_MISSING_CHAR and len(best) == len(base) + 2 and _only_yi_inserted(best, base)
        and best.count("一一") <= base.count("一一")
    ):
        return best
    if note == OCR_NOTE_EXTRA_CHAR and _one_char_longer(base, best) and _is_han(best):
        return best
    return None


__all__ = [
    "OCR_NOTE_DISPUTED",
    "OCR_NOTE_REOCR_FIXED",
    "accept_disputed_reread",
    "accept_reocr_text",
    "drop_small_non_han_events",
    "OCR_NOTE_EXTRA_CHAR",
    "OCR_NOTE_MISSING_CHAR",
    "EventGeometry",
    "geometry_notes",
    "mark_geometry_suspects",
    "measure_events",
]
