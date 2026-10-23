"""[v3.23.440] Giãn thời gian hiển thị phụ đề theo TỐC ĐỘ ĐỌC (mượn khoảng lặng sau câu).

Bổ trợ cho dịch CPS-aware (v439): thay vì CHỈ rút ngắn chữ, khi một dòng đọc không kịp (CPS cao)
mà PHÍA SAU có khoảng lặng, ta KÉO DÀI thời gian hiển thị vào khoảng lặng đó — người xem có thêm
thời gian đọc mà **KHÔNG đụng một chữ nào** (0 rủi ro mất nghĩa).

Đo thực 19 tập: 57% dòng đọc khó (CPS>22) có khoảng lặng >0.3s phía sau đủ để giãn, nhiều dòng
CPS giảm mạnh (vd 32→13). Đây là đòn bẩy an toàn nhất cho tốc độ đọc.

Ràng buộc an toàn:
- KHÔNG đụng câu kế: chừa ``guard_gap_sec`` trước mốc bắt đầu câu sau.
- KHÔNG hiển thị quá lâu: trần ``max_duration_sec`` (dòng treo quá lâu gây khó chịu/lệch ngữ cảnh).
- CHỈ giãn khi lợi ích đáng kể (``min_gain_sec``) để tránh xê dịch vụn vặt.
- Thuần tính toán — dễ test, không phụ thuộc I/O.
"""

from __future__ import annotations

from dataclasses import replace

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

#: Chừa khoảng cách tối thiểu (giây) trước câu kế để không dính/đè lên nhau.
DEFAULT_GUARD_GAP_SEC: float = 0.1
#: Trần thời lượng hiển thị một dòng (giây) — treo lâu hơn gây khó chịu.
DEFAULT_MAX_DURATION_SEC: float = 7.0
#: Chỉ giãn khi kéo dài được ít nhất ngần này (giây), tránh xê dịch vô nghĩa.
DEFAULT_MIN_GAIN_SEC: float = 0.2
#: [v3.23.604] Khung ngắn hơn mức này là khung OCR CHỚP NHOÁNG (vài khung hình), không
#: phải thời lượng thật của câu thoại — xem :func:`compute_glitch_window`.
GLITCH_WINDOW_SEC: float = 0.3
#: [v3.23.604] Thời lượng tối thiểu kéo khung chớp nhoáng tới: 5/6 giây, mức tối thiểu
#: thường dùng cho một dòng phụ đề.
GLITCH_MIN_DISPLAY_SEC: float = 0.833


def compute_reading_end(
    visible_len: int,
    start_sec: float,
    end_sec: float,
    next_start_sec: float | None,
    target_cps: float,
    guard_gap_sec: float = DEFAULT_GUARD_GAP_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    min_gain_sec: float = DEFAULT_MIN_GAIN_SEC,
) -> float:
    """Tính mốc kết thúc MỚI để dòng đọc kịp ``target_cps``, mượn khoảng lặng phía sau.

    Args:
        visible_len: Số ký tự hiển thị (đã bỏ xuống dòng/khoảng trắng thừa nếu muốn).
        start_sec: Mốc bắt đầu hiện tại.
        end_sec: Mốc kết thúc hiện tại.
        next_start_sec: Mốc bắt đầu câu KẾ TIẾP (None nếu là câu cuối).
        target_cps: Tốc độ đọc mục tiêu (ký tự/giây).
        guard_gap_sec: Khoảng chừa trước câu kế.
        max_duration_sec: Trần thời lượng hiển thị.
        min_gain_sec: Ngưỡng lợi ích tối thiểu để thực sự giãn.

    Returns:
        Mốc kết thúc mới (≥ ``end_sec``); trả nguyên ``end_sec`` nếu không nên/không thể giãn.
    """
    duration = end_sec - start_sec
    if duration <= 0.0 or visible_len <= 0 or target_cps <= 0.0:
        return end_sec
    current_cps = visible_len / duration
    if current_cps <= target_cps:
        return end_sec  # đã đọc kịp — không cần giãn

    desired_end = start_sec + visible_len / target_cps
    hard_cap = start_sec + max_duration_sec
    if next_start_sec is not None:
        hard_cap = min(hard_cap, next_start_sec - guard_gap_sec)

    new_end = min(desired_end, hard_cap)
    if new_end - end_sec < min_gain_sec:
        return end_sec  # giãn được quá ít (hoặc không có gap) → giữ nguyên
    return new_end


def compute_glitch_window(
    visible_len: int,
    start_sec: float,
    end_sec: float,
    previous_end_sec: float | None,
    next_start_sec: float | None,
    target_cps: float,
    guard_gap_sec: float = DEFAULT_GUARD_GAP_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    min_gain_sec: float = DEFAULT_MIN_GAIN_SEC,
) -> tuple[float, float]:
    """[v3.23.604] Khung OCR chớp nhoáng: kéo tới thời lượng đọc được, mượn sau rồi trước.

    OCR đôi khi chỉ bắt được 2-4 khung hình cuối của một dòng phụ đề. Chuẩn đoán thật
    11/10 (灭门惨案II): dòng 916 "我........" chỉ có 0,065 s (4081,267-4081,332) dù trước
    nó là 1,33 s lặng và ngay sau là dòng 917; lồng tiếng phải đọc "Tôi..." ở 2,5x vẫn lấn
    sang câu sau. Ở lượt đó 2/2 chỗ chồng tiếng và 3/4 dòng đọc ≥ 2,2x là khung kiểu này.
    Bản cũ chỉ giãn ĐUÔI dòng — khung chớp nhoáng mà sát câu sau thì không giãn được.

    Khung ngắn hơn :data:`GLITCH_WINDOW_SEC` được kéo tới ``max(đọc kịp target_cps,``
    :data:`GLITCH_MIN_DISPLAY_SEC` ``)``: trước hết mượn khoảng lặng PHÍA SAU (như dòng
    thường), thiếu thì mượn khoảng lặng PHÍA TRƯỚC — luôn chừa ``guard_gap_sec`` với câu
    kề. Chạy trên bản dịch cuối của 3 phim: kéo dài 19/42, 17/32, 4/6 khung chớp
    nhoáng (21 dòng mượn khoảng lặng phía trước), không tạo chồng dòng nào.

    Args:
        visible_len: Số ký tự hiển thị.
        start_sec: Mốc bắt đầu hiện tại.
        end_sec: Mốc kết thúc hiện tại.
        previous_end_sec: Mốc kết thúc câu TRƯỚC (đã giãn nếu có); None nếu là câu đầu.
        next_start_sec: Mốc bắt đầu câu KẾ TIẾP; None nếu là câu cuối.
        target_cps: Tốc độ đọc mục tiêu (ký tự/giây).
        guard_gap_sec: Khoảng chừa với câu kề.
        max_duration_sec: Trần thời lượng hiển thị.
        min_gain_sec: Ngưỡng lợi ích tối thiểu để thực sự giãn.

    Returns:
        ``(mốc đầu, mốc cuối)`` mới; nguyên ``(start_sec, end_sec)`` nếu không phải khung
        chớp nhoáng hoặc không giãn được.
    """
    duration = end_sec - start_sec
    if duration < 0.0 or duration >= GLITCH_WINDOW_SEC or visible_len <= 0:
        return start_sec, end_sec
    reading = visible_len / target_cps if target_cps > 0.0 else 0.0
    wanted = min(max(reading, GLITCH_MIN_DISPLAY_SEC), max_duration_sec)
    new_end = start_sec + wanted
    if next_start_sec is not None:
        new_end = min(new_end, next_start_sec - guard_gap_sec)
    new_end = max(new_end, end_sec)
    new_start = start_sec
    if new_end - start_sec < wanted:
        floor = (
            previous_end_sec + guard_gap_sec
            if previous_end_sec is not None
            else new_end - wanted
        )
        new_start = min(start_sec, max(floor, new_end - wanted, 0.0))
    if (new_end - new_start) - duration < min_gain_sec:
        return start_sec, end_sec
    return new_start, new_end


def _visible_length(text: str) -> int:
    """Số ký tự người xem thực đọc (gộp xuống dòng thành khoảng trắng, bỏ 2 đầu)."""
    return len(text.replace("\n", " ").strip())


def extend_events_for_reading(
    events: list[SubtitleEvent],
    target_cps: float,
    guard_gap_sec: float = DEFAULT_GUARD_GAP_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    min_gain_sec: float = DEFAULT_MIN_GAIN_SEC,
) -> tuple[list[SubtitleEvent], int]:
    """Giãn thời gian hiển thị các dòng đọc không kịp vào khoảng lặng phía sau.

    Args:
        events: Danh sách event (giả định đã sắp theo thời gian bắt đầu).
        target_cps: Tốc độ đọc mục tiêu; ``<= 0`` = tắt (trả nguyên).
        guard_gap_sec: Khoảng chừa trước câu kế.
        max_duration_sec: Trần thời lượng hiển thị.
        min_gain_sec: Ngưỡng lợi ích tối thiểu.

    Returns:
        Cặp ``(events_mới, số_dòng_đã_giãn)``. ``events`` gốc không bị thay đổi.
    """
    if target_cps <= 0.0 or len(events) == 0:
        return list(events), 0

    result: list[SubtitleEvent] = []
    extended = 0
    for position, event in enumerate(events):
        next_start = (
            events[position + 1].start_sec if position + 1 < len(events) else None
        )
        previous_end = result[-1].end_sec if result else None
        visible = _visible_length(event.text)
        new_start, new_end = compute_glitch_window(
            visible,
            event.start_sec,
            event.end_sec,
            previous_end,
            next_start,
            target_cps,
            guard_gap_sec=guard_gap_sec,
            max_duration_sec=max_duration_sec,
            min_gain_sec=min_gain_sec,
        )
        if (new_start, new_end) == (event.start_sec, event.end_sec):
            new_end = compute_reading_end(
                visible,
                event.start_sec,
                event.end_sec,
                next_start,
                target_cps,
                guard_gap_sec=guard_gap_sec,
                max_duration_sec=max_duration_sec,
                min_gain_sec=min_gain_sec,
            )
        if new_start < event.start_sec or new_end > event.end_sec:
            event = replace(event, interval=TimeInterval(new_start, new_end))
            extended += 1
        result.append(event)
    return result, extended


__all__ = [
    "DEFAULT_GUARD_GAP_SEC",
    "DEFAULT_MAX_DURATION_SEC",
    "DEFAULT_MIN_GAIN_SEC",
    "GLITCH_MIN_DISPLAY_SEC",
    "GLITCH_WINDOW_SEC",
    "compute_glitch_window",
    "compute_reading_end",
    "extend_events_for_reading",
]
