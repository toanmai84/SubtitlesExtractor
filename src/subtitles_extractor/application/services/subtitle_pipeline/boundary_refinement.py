"""[v3.23.680] Đặt mốc phụ đề đúng khung hình bằng dải ranh giới (không OCR thêm).

Đo thật (bước 0,12 s, 25 fps; "lệch > 1 khung" = lệch > 40 ms):
* Tập 1–5 so với SRT đáp án (428 câu khớp): mốc bắt đầu lệch tb 45 → 4 ms, mốc kết thúc 36 → 8 ms;
  số mốc lệch > 1 khung 358 → 19.
* Tập 6–12 (KHÔNG dùng khi chỉnh tham số), so với bản dựng bước 0,04 s (OCR mọi khung — trên tập 1–5 chỉ 8/856 mốc
  lệch > 1 khung): 326 → 14 mốc lệch > 1 khung; tb 40 → 2 ms (đầu), 33 → 5 ms (cuối).

Cách làm (hai lượt):
1. Mốc BẮT ĐẦU: từ khung mẫu đầu tiên có câu, lùi qua các khung bỏ qua phía trước khi khung đó vẫn GIỐNG khung mẫu
   (giao/hợp mặt nạ chữ ≥ ngưỡng) và đủ chữ — không lùi quá cuối câu trước.
2. Mốc KẾT THÚC: từ khung mẫu cuối có câu, tiến qua các khung bỏ qua khi còn giống; kết thúc = khung đó + 1 khung,
   không vượt mốc BẮT ĐẦU ĐÃ CHỈNH của câu sau (câu sau giống chữ câu trước, vd "天下无敌" → "什么天下无敌",
   từng làm câu trước kéo dài thêm 2 khung — 35/78 lỗi còn lại của bản thử một lượt).
Câu chữ mờ/xám (ít điểm sáng) → không đổi mốc.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import replace

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.boundary_track import (
    BoundaryProbe,
    BoundaryTrack,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

#: Ngưỡng giống nhau (giao/hợp mặt nạ chữ). Chỉnh trên tập 1–5: 0,6–0,85 cho kết quả gần như nhau.
SIMILARITY_THRESHOLD = 0.75
#: Số ô chữ sáng tối thiểu để coi khung có chữ (mặt nạ 4×4; một chữ Hán ~ 6–10 ô).
MIN_TEXT_PIXELS = 60


def refine_event_boundaries(
    events: Sequence[SubtitleEvent],
    track: BoundaryTrack | None,
    step_sec: float | None = None,
    *,
    similarity: float = SIMILARITY_THRESHOLD,
    min_pixels: int = MIN_TEXT_PIXELS,
) -> list[SubtitleEvent]:
    """Trả danh sách câu với mốc đã đặt đúng khung (thứ tự và chữ giữ nguyên).

    ``step_sec`` bỏ trống → lấy trung vị khoảng cách giữa các khung mẫu của dải (bước thật của lượt giải mã).
    """
    events = list(events)
    if track is None or track.is_empty or not events:
        return events
    times = list(track.sample_times)
    if step_sec is None:
        diffs = sorted(b - a for a, b in zip(times, times[1:], strict=False) if b - a > 1e-6)
        step_sec = diffs[len(diffs) // 2] if diffs else 0.0
    if step_sec <= track.frame_period_sec * 1.5:
        return events
    pixels = list(track.sample_pixels)
    probe_times = [p.timestamp_sec for p in track.probes]
    period = track.frame_period_sec
    eps = period / 4

    def gap(a: int, b: int) -> list[BoundaryProbe]:
        """Khung bỏ qua nằm giữa mẫu ``a`` và ``b`` (theo chỉ số)."""
        lo = times[a] if a >= 0 else float("-inf")
        hi = times[b] if b < len(times) else float("inf")
        i = bisect.bisect_right(probe_times, lo + eps)
        j = bisect.bisect_left(probe_times, hi - eps)
        return list(track.probes[i:j])

    def nearest_sample(t: float) -> int | None:
        i = bisect.bisect_left(times, t)
        best = min((k for k in (i - 1, i) if 0 <= k < len(times)), key=lambda k: abs(times[k] - t), default=None)
        if best is None or abs(times[best] - t) > step_sec / 2 + eps:
            return None
        return best

    # Lượt 1: mốc bắt đầu.
    starts = [float(e.start_sec) for e in events]
    for n, event in enumerate(events):
        anchor = nearest_sample(float(event.start_sec))
        if anchor is None or pixels[anchor] < min_pixels:
            continue
        floor = float(events[n - 1].end_sec) - step_sec if n else float("-inf")
        new_start = times[anchor]
        for probe in reversed(gap(anchor - 1, anchor)):
            if new_start - probe.timestamp_sec > period * 1.5:   # [v3.23.680] khung hụt (bộ đệm/giải mã lỗi) → dừng
                break
            if probe.timestamp_sec < floor or probe.text_pixels < min_pixels or probe.sim_next < similarity:
                break
            new_start = probe.timestamp_sec
        if new_start < times[anchor] - eps:          # chỉ LÙI mốc (chữ đã hiện từ khung bỏ qua phía trước)
            starts[n] = min(float(event.start_sec), new_start)

    # Lượt 2: mốc kết thúc (chặn bởi mốc bắt đầu đã chỉnh của câu sau).
    out: list[SubtitleEvent] = []
    for n, event in enumerate(events):
        start = starts[n]
        end = float(event.end_sec)
        next_start = starts[n + 1] if n + 1 < len(events) else float("inf")
        last = bisect.bisect_left(times, end - eps) - 1      # mẫu cuối trước mốc kết thúc
        if last >= 0 and times[last] >= start - eps and pixels[last] >= min_pixels:
            tail = times[last]
            for probe in gap(last, last + 1):
                if probe.timestamp_sec - tail > period * 1.5:    # không liền khung trước → không suy được
                    break
                if probe.text_pixels < min_pixels or probe.sim_prev < similarity:
                    break
                if probe.timestamp_sec + period > next_start + eps:
                    break
                tail = probe.timestamp_sec
            end = tail + period
        end = min(end, next_start) if next_start != float("inf") else end
        end = max(end, start + period)
        if abs(start - float(event.start_sec)) > 1e-9 or abs(end - float(event.end_sec)) > 1e-9:
            event = replace(event, interval=TimeInterval(start, end))
        out.append(event)
    return out


#: [v3.23.683] Khung mẫu "trống thật" khi số ô chữ sáng ≤ bấy nhiêu phần số ô của hai khung mẫu kề hai bên.
BLANK_PIXEL_RATIO = 0.1
#: ...và chuỗi khung trống (khung mẫu + khung bỏ qua kề nó) dài tối thiểu bấy nhiêu khung hình.
BLANK_MIN_FRAMES = 2


def pixel_confirmed_blanks(track: BoundaryTrack | None, *, min_pixels: int = MIN_TEXT_PIXELS) -> list[float]:
    """[v3.23.683] Mốc các khung mẫu mà điểm ảnh XÁC NHẬN màn hình trống chữ (đã sắp xếp).

    Ở bước 0,12 s một khung OCR trống giữa hai lần cùng câu có thể là OCR lỡ (chữ vẫn trên màn hình) hoặc khoảng trống
    thật giữa hai câu lặp. Phim nối 86 tập: ba cặp "师叔"/"散了吧" thật trống 3–4 khung hình, khung mẫu giữa có 0 ô chữ
    sáng trong khi hai bên 250–320 ô; còn khung OCR lỡ ("啊", "走", "我", câu dài) vẫn giữ nguyên số ô (143/143/143).

    Một chuỗi khung mẫu liền nhau ít chữ sáng được nhận khi: hai khung mẫu kề hai bên đều có ≥ ``min_pixels`` ô, mỗi
    khung trong chuỗi ≤ ``BLANK_PIXEL_RATIO`` × số ô ít hơn của hai bên, và tính cả khung bỏ qua kề bên (cùng ngưỡng)
    thì chuỗi trống dài ≥ ``BLANK_MIN_FRAMES`` khung hình.
    """
    # Không có khung bỏ qua (bước = mọi khung, vd 0,04 s ở 25 fps) → không có gì đối chiếu; luật cũ (≥ 3 khung trống) áp dụng.
    if track is None or track.is_empty:
        return []
    times = list(track.sample_times)
    pixels = list(track.sample_pixels)
    probe_times = [p.timestamp_sec for p in track.probes]
    eps = (track.frame_period_sec or 0.04) / 4
    out: list[float] = []
    i = 1
    while i < len(times) - 1:
        if pixels[i - 1] < min_pixels:
            i += 1
            continue
        j = i
        while j < len(times) and pixels[j] < min_pixels:
            j += 1
        if j == i or j >= len(times) or pixels[j] < min_pixels:
            i = max(j, i + 1)
            continue
        limit = BLANK_PIXEL_RATIO * min(pixels[i - 1], pixels[j])
        if all(pixels[k] <= limit for k in range(i, j)):
            # Đếm khung trống liên tiếp: các khung mẫu trong chuỗi + khung bỏ qua giữa chúng và kề hai đầu.
            lo = bisect.bisect_right(probe_times, times[i - 1] + eps)
            hi = bisect.bisect_left(probe_times, times[j] - eps)
            blank_frames = j - i
            run = list(track.probes[lo:hi])
            # Khung bỏ qua có chữ (> ngưỡng) nằm GIỮA chuỗi thì không còn là một khoảng trống liền.
            inner = [p for p in run if times[i] - eps <= p.timestamp_sec <= times[j - 1] + eps]
            if all(p.text_pixels <= limit for p in inner):
                # [v3.23.696] Khung bỏ qua kề hai đầu chỉ được cộng khi LIỀN với chuỗi trống: đi từ mép chuỗi ra ngoài,
                # gặp khung có chữ thì dừng (trước đây "trống, có chữ, [chuỗi trống]" vẫn cộng khung trống phía ngoài).
                before = [p for p in run if p.timestamp_sec < times[i] - eps]
                after = [p for p in run if p.timestamp_sec > times[j - 1] + eps]
                blank_frames += len(inner)
                blank_frames += _leading_blank(reversed(before), limit) + _leading_blank(after, limit)
                if blank_frames >= BLANK_MIN_FRAMES:
                    out.extend(times[i:j])
        i = j
    return out


def _leading_blank(probes, limit: float) -> int:
    """Số khung trống liền nhau ở đầu dãy (dừng ở khung đầu tiên có chữ)."""
    count = 0
    for probe in probes:
        if probe.text_pixels > limit:
            break
        count += 1
    return count


__all__ = [
    "BLANK_MIN_FRAMES",
    "BLANK_PIXEL_RATIO",
    "MIN_TEXT_PIXELS",
    "SIMILARITY_THRESHOLD",
    "pixel_confirmed_blanks",
    "refine_event_boundaries",
]
